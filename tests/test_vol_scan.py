"""End-to-end tests for the vol options scan track.

All LLM calls and broker API calls are mocked — the tests verify the
plumbing: data fetch → run_vol_consensus → leg submission → position
state → exit management rules. No real network calls are made.

The fixture uses AAPL as the canonical test ticker because that's the
intended live universe (liquid, high-options-volume, 45 DTE strikes
always available).
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

from analyst_layer.market_regime import DailyRegime
from unittest.mock import MagicMock, patch

import pytest

from analyst_layer.schemas import (
    Action,
    Confidence,
    GreeksRiskReview,
    IVEnvironment,
    OptionsProposal,
    RiskVerdict,
    StructureType,
    VolConsensusPayload,
    VolSignal,
)
from config.settings import Settings
from data_layer.models import OptionContract, OptionType, PriceBar, PriceSeries, VolatilitySnapshot
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.state_store import StateStore


# ── AAPL test data ────────────────────────────────────────────────────────────

_AAPL_EXP = date(2026, 8, 15)
_AAPL_DTE = 45
_AAPL_UNDERLYING = 190.0
# 1 SD ≈ 190 × 0.28 × sqrt(45/365) ≈ 18.7 → round to 210C / 172P
_SHORT_CALL_STRIKE = 210.0
_SHORT_PUT_STRIKE = 172.0
_CALL_CONTRACT_SYMBOL = "AAPL260815C00210000"
_PUT_CONTRACT_SYMBOL = "AAPL260815P00172000"


def _aapl_vol_snapshot() -> VolatilitySnapshot:
    return VolatilitySnapshot(
        symbol="AAPL",
        as_of=datetime.now(),
        iv_rank=65.0,
        iv_percentile=68.0,
        iv_30=0.28,
        hv_20=0.20,
        hv_30=0.18,
        iv_hv_spread=0.10,
        term_structure_ratio=0.92,
        put_skew=0.04,
        earnings_within_dte=False,
    )


def _aapl_chain() -> list[OptionContract]:
    """Minimal realistic AAPL chain with strikes at the expected 1-SD levels."""
    def _c(symbol, strike, option_type, bid, ask):
        return OptionContract(
            contract_symbol=symbol,
            underlying_symbol="AAPL",
            underlying_price=_AAPL_UNDERLYING,
            expiration=_AAPL_EXP,
            dte=_AAPL_DTE,
            strike=strike,
            option_type=option_type,
            bid=bid,
            ask=ask,
            implied_volatility=0.28,
            open_interest=1500,
            volume=300,
        )

    return [
        _c(_CALL_CONTRACT_SYMBOL, _SHORT_CALL_STRIKE, OptionType.CALL, bid=2.50, ask=2.60),
        _c(_PUT_CONTRACT_SYMBOL, _SHORT_PUT_STRIKE, OptionType.PUT, bid=2.20, ask=2.30),
        # Extra strikes so build_structure has something to work with in other tests
        _c("AAPL260815C00220000", 220.0, OptionType.CALL, bid=1.00, ask=1.10),
        _c("AAPL260815P00162000", 162.0, OptionType.PUT, bid=0.90, ask=1.00),
    ]


def _aapl_approved_payload() -> VolConsensusPayload:
    """APPROVED SHORT_STRANGLE on AAPL — the happy-path consensus result."""
    signal = VolSignal(
        agent_name="iv_surface_agent",
        ticker="AAPL",
        iv_environment=IVEnvironment.ELEVATED,
        recommended_structure=StructureType.SHORT_STRANGLE,
        confidence=Confidence.HIGH,
        rationale="IVR 65 — elevated, IV/HV spread +10% — variance risk premium available",
        generated_at=datetime.now(),
        flags=[],
    )
    proposal = OptionsProposal(
        ticker="AAPL",
        structure=StructureType.SHORT_STRANGLE,
        expiration=_AAPL_EXP,
        dte=_AAPL_DTE,
        quantity=1,
        short_call_strike=_SHORT_CALL_STRIKE,
        short_put_strike=_SHORT_PUT_STRIKE,
        net_credit=4.65,
    )
    review = GreeksRiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=["within portfolio delta and vega limits"],
        portfolio_delta_after=0.02,
        portfolio_vega_after=-10.5,
        portfolio_theta_after=1.2,
        position_max_loss=None,
        reviewed_at=datetime.now(),
    )
    return VolConsensusPayload(ticker="AAPL", vol_signals=[signal], proposal=proposal, risk_review=review)


_LONG_CALL_STRIKE = 220.0
_LONG_PUT_STRIKE = 162.0
_LONG_CALL_SYMBOL = "AAPL260815C00220000"
_LONG_PUT_SYMBOL = "AAPL260815P00162000"


def _aapl_iron_condor_payload() -> VolConsensusPayload:
    """APPROVED IRON_CONDOR on AAPL — tests the mleg submission path."""
    signal = VolSignal(
        agent_name="iv_surface_agent",
        ticker="AAPL",
        iv_environment=IVEnvironment.ELEVATED,
        recommended_structure=StructureType.IRON_CONDOR,
        confidence=Confidence.HIGH,
        rationale="IVR 65 — iron condor for defined risk",
        generated_at=datetime.now(),
        flags=[],
    )
    proposal = OptionsProposal(
        ticker="AAPL",
        structure=StructureType.IRON_CONDOR,
        expiration=_AAPL_EXP,
        dte=_AAPL_DTE,
        quantity=1,
        short_call_strike=_SHORT_CALL_STRIKE,
        short_put_strike=_SHORT_PUT_STRIKE,
        long_call_strike=_LONG_CALL_STRIKE,
        long_put_strike=_LONG_PUT_STRIKE,
        # net_credit = (2.50 + 2.20) - (1.10 + 1.00) = 2.60
        net_credit=2.60,
    )
    review = GreeksRiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=["within portfolio limits"],
        portfolio_delta_after=0.01,
        portfolio_vega_after=-10.0,
        portfolio_theta_after=1.0,
        position_max_loss=None,
        reviewed_at=datetime.now(),
    )
    return VolConsensusPayload(ticker="AAPL", vol_signals=[signal], proposal=proposal, risk_review=review)


def _aapl_rejected_payload() -> VolConsensusPayload:
    """REJECTED consensus — e.g. vol expansion regime or earnings."""
    signal = VolSignal(
        agent_name="vol_regime_agent",
        ticker="AAPL",
        iv_environment=IVEnvironment.ELEVATED,
        recommended_structure=StructureType.NO_TRADE,
        confidence=Confidence.HIGH,
        rationale="vol expansion regime — VIX in backwardation, no new short premium",
        generated_at=datetime.now(),
        flags=["vol_expansion_regime"],
    )
    no_trade = OptionsProposal(
        ticker="AAPL", structure=StructureType.NO_TRADE,
        expiration=_AAPL_EXP, dte=0, quantity=0,
    )
    review = GreeksRiskReview(
        verdict=RiskVerdict.REJECTED,
        reasons=["vol expansion regime"],
        portfolio_delta_after=0.0,
        portfolio_vega_after=0.0,
        portfolio_theta_after=0.0,
        position_max_loss=None,
        reviewed_at=datetime.now(),
    )
    return VolConsensusPayload(ticker="AAPL", vol_signals=[signal], proposal=no_trade, risk_review=review)


def _vix_price_series() -> PriceSeries:
    """21 bars of fake VIX history so _fetch_vix_context has something to work with."""
    today = date.today()
    bars = [
        PriceBar(
            symbol="^VIX",
            timestamp=datetime(today.year, today.month, today.day) - timedelta(days=21 - i),
            open=15.0 + i * 0.1,
            high=16.0 + i * 0.1,
            low=14.5 + i * 0.1,
            close=15.5 + i * 0.1,
            volume=0,
        )
        for i in range(21)
    ]
    return PriceSeries(symbol="^VIX", interval="1d", bars=bars)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def vol_runtime(tmp_path: Path) -> TradingRuntime:
    """TradingRuntime with vol track enabled and AAPL in watchlist.

    LLM and broker are mocked. Data client is mocked to return AAPL-realistic
    vol snapshot, option chain, and VIX history.
    """
    os.environ["VOL_OPTIONS_TRACK_ENABLED"] = "true"
    settings = Settings(_env_file=None)
    del os.environ["VOL_OPTIONS_TRACK_ENABLED"]

    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    # For a newly opened short option, Alpaca reports qty=-1, avg_entry_price=credit
    broker.get_position_detail.return_value = {
        "qty": -1.0,
        "avg_entry_price": 2.50,
        "current_price": 2.50,
        "unrealized_plpc": 0.0,
    }
    broker.submit_option_order.return_value = {
        "status": "submitted",
        "order_status": "filled",
        "filled_qty": 1,
        "filled_avg_price": 2.50,
    }
    broker.submit_spread_order.return_value = {
        "status": "submitted",
        "order_status": "filled",
        "filled_qty": 1,
        "filled_avg_price": None,
    }
    broker.get_open_orders.return_value = []

    data_client = MagicMock()
    data_client.get_volatility_snapshot.return_value = _aapl_vol_snapshot()
    data_client.get_option_chain.return_value = _aapl_chain()
    # VIX fetch in _fetch_vix_context uses get_price_history; VIX3M raises DataLayerError
    from data_layer.exceptions import DataLayerError
    def _price_history_side_effect(symbol, **kwargs):
        if symbol == "^VIX":
            return _vix_price_series()
        raise DataLayerError(f"no mock for {symbol}")
    data_client.get_price_history.side_effect = _price_history_side_effect

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())
    store = StateStore(tmp_path / "vol_test.sqlite3")

    rt = TradingRuntime(
        settings=settings,
        data_client=data_client,
        broker=broker,
        circuit_breaker=breaker,
        state_store=store,
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )
    # Pre-populate daily regime so vol track is armed without calling pre_market_scan
    rt._daily_regime = DailyRegime(
        vix_current=22.0, vix_trend="stable", market_trend="bullish",
        arm_orb_equity=True, arm_orb_options=True, arm_thesis=True, arm_vol=True,
    )
    return rt


# ── vol_options_scan_and_trade tests ─────────────────────────────────────────

@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_blocks_short_strangle_naked_shorts(mock_consensus, vol_runtime: TradingRuntime):
    """SHORT_STRANGLE requires naked shorts (Level 4) — must be blocked on this
    Level 3 account. No orders should be submitted and a blocked event recorded.
    """
    mock_consensus.return_value = _aapl_approved_payload()  # returns SHORT_STRANGLE

    vol_runtime.vol_options_scan_and_trade()

    # No individual option orders — naked shorts are blocked
    vol_runtime._broker.submit_option_order.assert_not_called()
    # A blocked event must be recorded so the issue is visible in logs/dashboard
    events = vol_runtime._state_store.get_events(event_type_like="vol_options_blocked_level4%")
    assert len(events) >= 1
    assert "short_strangle" in events[0]["detail"]


@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_aapl_positions_tagged_as_vol_short(mock_consensus, vol_runtime: TradingRuntime):
    """Iron condor legs must be stored with strategy='vol_short' so the right
    exit rules apply and they're not conflated with the ORB long options track.
    """
    mock_consensus.return_value = _aapl_iron_condor_payload()

    vol_runtime.vol_options_scan_and_trade()

    # Iron condors submit via mleg (submit_spread_order), not individual legs
    vol_runtime._broker.submit_spread_order.assert_called_once()
    # Both short legs recorded as vol_short (_CALL_CONTRACT_SYMBOL = short call, _PUT_CONTRACT_SYMBOL = short put)
    short_call_pos = vol_runtime._state_store.get_option_position(_CALL_CONTRACT_SYMBOL)
    short_put_pos = vol_runtime._state_store.get_option_position(_PUT_CONTRACT_SYMBOL)
    assert short_call_pos is not None
    assert short_put_pos is not None
    assert short_call_pos["strategy"] == "vol_short"
    assert short_put_pos["strategy"] == "vol_short"


@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_skips_when_consensus_rejected(mock_consensus, vol_runtime: TradingRuntime):
    """REJECTED payload (e.g. vol expansion) → no orders."""
    mock_consensus.return_value = _aapl_rejected_payload()

    vol_runtime.vol_options_scan_and_trade()

    vol_runtime._broker.submit_option_order.assert_not_called()


@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_deduplicates_within_same_day(mock_consensus, vol_runtime: TradingRuntime):
    """Running the scan twice should not double-submit — the per-ticker
    dedup set must prevent re-scanning AAPL on the second call.
    """
    mock_consensus.return_value = _aapl_approved_payload()

    vol_runtime.vol_options_scan_and_trade()
    vol_runtime.vol_options_scan_and_trade()  # second call — AAPL already scanned

    assert mock_consensus.call_count == 1  # consensus only ran once


@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_iron_condor_submits_single_mleg_order(mock_consensus, vol_runtime: TradingRuntime):
    """IRON_CONDOR → single atomic mleg order, not 4 individual orders.

    This is the Level 3 fix: submit_spread_order (mleg) so Alpaca evaluates
    all legs together and never sees the short legs as uncovered.
    """
    mock_consensus.return_value = _aapl_iron_condor_payload()
    # configure get_position_detail to return sensible values for each leg
    vol_runtime._broker.get_position_detail.side_effect = lambda sym: {
        _CALL_CONTRACT_SYMBOL: {"qty": -1.0, "avg_entry_price": 2.50, "current_price": 2.50, "unrealized_plpc": 0.0},
        _PUT_CONTRACT_SYMBOL:  {"qty": -1.0, "avg_entry_price": 2.20, "current_price": 2.20, "unrealized_plpc": 0.0},
        _LONG_CALL_SYMBOL:     {"qty":  1.0, "avg_entry_price": 1.10, "current_price": 1.10, "unrealized_plpc": 0.0},
        _LONG_PUT_SYMBOL:      {"qty":  1.0, "avg_entry_price": 1.00, "current_price": 1.00, "unrealized_plpc": 0.0},
    }.get(sym)

    vol_runtime.vol_options_scan_and_trade()

    # One mleg order submitted, no individual orders
    vol_runtime._broker.submit_spread_order.assert_called_once()
    vol_runtime._broker.submit_option_order.assert_not_called()

    call_kwargs = vol_runtime._broker.submit_spread_order.call_args
    legs = call_kwargs.kwargs["legs"]
    assert call_kwargs.kwargs["contracts"] == 1
    # Mid-price credit: shorts=(2.55+2.25), longs=(1.05+0.95) → 2.80
    # (Natural would be 2.60; mid-price is the submission price now)
    assert call_kwargs.kwargs["net_credit"] == pytest.approx(2.80)
    leg_symbols = {sym for sym, _ in legs}
    assert leg_symbols == {_CALL_CONTRACT_SYMBOL, _PUT_CONTRACT_SYMBOL, _LONG_CALL_SYMBOL, _LONG_PUT_SYMBOL}


@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_iron_condor_all_four_legs_recorded_as_vol_short(mock_consensus, vol_runtime: TradingRuntime):
    """All 4 legs of the iron condor are recorded with strategy='vol_short'."""
    mock_consensus.return_value = _aapl_iron_condor_payload()
    vol_runtime._broker.get_position_detail.side_effect = lambda sym: {
        _CALL_CONTRACT_SYMBOL: {"qty": -1.0, "avg_entry_price": 2.50, "current_price": 2.50, "unrealized_plpc": 0.0},
        _PUT_CONTRACT_SYMBOL:  {"qty": -1.0, "avg_entry_price": 2.20, "current_price": 2.20, "unrealized_plpc": 0.0},
        _LONG_CALL_SYMBOL:     {"qty":  1.0, "avg_entry_price": 1.10, "current_price": 1.10, "unrealized_plpc": 0.0},
        _LONG_PUT_SYMBOL:      {"qty":  1.0, "avg_entry_price": 1.00, "current_price": 1.00, "unrealized_plpc": 0.0},
    }.get(sym)

    vol_runtime.vol_options_scan_and_trade()

    for sym in (_CALL_CONTRACT_SYMBOL, _PUT_CONTRACT_SYMBOL, _LONG_CALL_SYMBOL, _LONG_PUT_SYMBOL):
        pos = vol_runtime._state_store.get_option_position(sym)
        assert pos is not None, f"{sym} not in state store"
        assert pos["strategy"] == "vol_short", f"{sym} strategy={pos['strategy']!r}"


@patch("execution_layer.runtime.run_vol_consensus")
def test_vol_scan_no_ops_when_track_disabled(mock_consensus, tmp_path: Path):
    """VOL_OPTIONS_TRACK_ENABLED=false → method returns immediately regardless of regime."""
    os.environ["VOL_OPTIONS_TRACK_ENABLED"] = "false"
    settings = Settings(_env_file=None)
    del os.environ["VOL_OPTIONS_TRACK_ENABLED"]
    assert settings.vol_options_track_enabled is False

    rt = TradingRuntime(
        settings=settings,
        data_client=MagicMock(),
        broker=MagicMock(),
        circuit_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02),
        state_store=StateStore(tmp_path / "disabled.sqlite3"),
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )
    rt._daily_regime = DailyRegime(
        vix_current=22.0, vix_trend="stable", market_trend="bullish",
        arm_orb_equity=True, arm_orb_options=True, arm_thesis=True, arm_vol=True,
    )
    rt.vol_options_scan_and_trade()

    mock_consensus.assert_not_called()


# ── _check_vol_options_exits tests ───────────────────────────────────────────

def test_vol_exits_profit_target_closes_at_50pct(vol_runtime: TradingRuntime):
    """Close when cost-to-close is 50% of the original credit."""
    expiration = (date.today() + timedelta(days=30)).isoformat()  # clear of roll DTE
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    # Cost to close = 50% of credit → exactly at profit target
    vol_runtime._broker.get_position_detail.return_value = {
        "qty": -1.0, "avg_entry_price": 2.50, "current_price": 1.25, "unrealized_plpc": 0.0,
    }
    vol_runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 1, "filled_avg_price": 1.25,
    }
    # After close, Alpaca shows qty=0
    vol_runtime._broker.get_position_detail.side_effect = [
        {"qty": -1.0, "avg_entry_price": 2.50, "current_price": 1.25, "unrealized_plpc": 0.0},
        None,  # position gone after close
    ]

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    vol_runtime._broker.submit_option_order.assert_called_once()
    _, kwargs = vol_runtime._broker.submit_option_order.call_args
    assert kwargs["side"] == Action.BUY  # buy to close the short


def test_vol_exits_profit_target_pnl_recorded_with_correct_sign(vol_runtime: TradingRuntime):
    """Profitable close of a short option must record POSITIVE realized P&L."""
    expiration = (date.today() + timedelta(days=30)).isoformat()
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    credit = 2.50
    close_cost = 1.25  # 50% profit

    vol_runtime._broker.get_position_detail.side_effect = [
        {"qty": -1.0, "avg_entry_price": credit, "current_price": close_cost, "unrealized_plpc": 0.5},
        None,
    ]
    vol_runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 1, "filled_avg_price": close_cost,
    }

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    sales = vol_runtime._state_store.get_all_realized_option_sales()
    assert len(sales) == 1
    # P&L = (credit_received - close_cost) × 1 contract × 100 shares
    assert sales[0]["realized_pnl"] == pytest.approx((credit - close_cost) * 1 * 100)
    assert sales[0]["realized_pnl"] > 0  # profitable


def test_vol_exits_loss_limit_closes_at_2x_credit(vol_runtime: TradingRuntime):
    """Close when cost-to-close reaches 2x the original credit."""
    expiration = (date.today() + timedelta(days=30)).isoformat()
    vol_runtime._state_store.upsert_option_position(
        _PUT_CONTRACT_SYMBOL, "AAPL", "put", _SHORT_PUT_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.00, strategy="vol_short",
    )
    # Cost to close = 2x credit → exactly at loss limit
    vol_runtime._broker.get_position_detail.side_effect = [
        {"qty": -1.0, "avg_entry_price": 2.00, "current_price": 4.00, "unrealized_plpc": -1.0},
        None,
    ]
    vol_runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 1, "filled_avg_price": 4.00,
    }

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    vol_runtime._broker.submit_option_order.assert_called_once()
    _, kwargs = vol_runtime._broker.submit_option_order.call_args
    assert kwargs["side"] == Action.BUY


def test_vol_exits_loss_limit_pnl_negative(vol_runtime: TradingRuntime):
    """Loss on a short option must record NEGATIVE realized P&L."""
    expiration = (date.today() + timedelta(days=30)).isoformat()
    vol_runtime._state_store.upsert_option_position(
        _PUT_CONTRACT_SYMBOL, "AAPL", "put", _SHORT_PUT_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.00, strategy="vol_short",
    )
    credit = 2.00
    close_cost = 4.00

    vol_runtime._broker.get_position_detail.side_effect = [
        {"qty": -1.0, "avg_entry_price": credit, "current_price": close_cost, "unrealized_plpc": -1.0},
        None,
    ]
    vol_runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 1, "filled_avg_price": close_cost,
    }

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    sales = vol_runtime._state_store.get_all_realized_option_sales()
    assert len(sales) == 1
    assert sales[0]["realized_pnl"] == pytest.approx((credit - close_cost) * 1 * 100)
    assert sales[0]["realized_pnl"] < 0  # loss


def test_vol_exits_dte_roll_closes_at_21d(vol_runtime: TradingRuntime):
    """Close when DTE hits the roll level (default 21d), regardless of P&L."""
    expiration = (date.today() + timedelta(days=20)).isoformat()  # inside 21d roll floor
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    # Still at full credit — not at profit target or loss limit — but DTE triggers
    vol_runtime._broker.get_position_detail.side_effect = [
        {"qty": -1.0, "avg_entry_price": 2.50, "current_price": 2.40, "unrealized_plpc": 0.04},
        None,
    ]
    vol_runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 1, "filled_avg_price": 2.40,
    }

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    vol_runtime._broker.submit_option_order.assert_called_once()


def test_vol_exits_no_action_when_healthy(vol_runtime: TradingRuntime):
    """No exit action when position is profitable but under 50%, above DTE floor."""
    expiration = (date.today() + timedelta(days=35)).isoformat()  # > 21d roll
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    # 20% profit — not yet at 50% target
    vol_runtime._broker.get_position_detail.return_value = {
        "qty": -1.0, "avg_entry_price": 2.50, "current_price": 2.00, "unrealized_plpc": 0.2,
    }

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    vol_runtime._broker.submit_option_order.assert_not_called()


def test_vol_exits_does_not_touch_orb_long_positions(vol_runtime: TradingRuntime):
    """_check_vol_options_exits must not close ORB/long positions.
    Those have their own exit path in _check_options_exits.
    """
    expiration = (date.today() + timedelta(days=3)).isoformat()  # would trigger force-close if handled here
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, expiration,
        quantity=3, avg_entry_price=1.50, strategy="orb_options",
    )
    vol_runtime._broker.get_position_detail.return_value = {
        "qty": 3.0, "avg_entry_price": 1.50, "current_price": 0.80, "unrealized_plpc": -0.47,
    }

    vol_runtime._check_vol_options_exits(equity=100_000.0)

    vol_runtime._broker.submit_option_order.assert_not_called()


def test_orb_exits_does_not_touch_vol_short_positions(vol_runtime: TradingRuntime):
    """_check_options_exits (the ORB path) must not close vol_short positions.
    Those are filtered out by the strategy check added to that method.
    """
    expiration = (date.today() + timedelta(days=1)).isoformat()  # inside 2d ORB force-close floor
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, expiration,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    vol_runtime._broker.get_position_detail.return_value = {
        "qty": -1.0, "avg_entry_price": 2.50, "current_price": 2.30, "unrealized_plpc": 0.08,
    }

    vol_runtime._check_options_exits(equity=100_000.0)

    vol_runtime._broker.submit_option_order.assert_not_called()


# ── VIX context tests ─────────────────────────────────────────────────────────

def test_fetch_vix_context_returns_stable_on_data_error(vol_runtime: TradingRuntime):
    """If VIX fetch fails, _fetch_vix_context must return a STABLE context
    rather than crashing. STABLE allows the other agents to still vote.
    """
    from data_layer.exceptions import DataLayerError
    from analyst_layer.schemas import VolRegime
    vol_runtime._data_client.get_price_history.side_effect = DataLayerError("VIX unavailable")

    ctx = vol_runtime._fetch_vix_context(date.today())

    # A VIX of 18 is STABLE (not > 30, not backwardated, not < 15)
    from analyst_layer.agents.vol_regime_agent import VixContext
    assert isinstance(ctx, VixContext)
    assert ctx.regime == VolRegime.STABLE


def test_build_portfolio_greeks_counts_only_vol_short_positions(vol_runtime: TradingRuntime):
    """Only strategy='vol_short' positions count toward portfolio Greeks."""
    exp = (date.today() + timedelta(days=30)).isoformat()
    vol_runtime._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, exp,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    vol_runtime._state_store.upsert_option_position(
        "AAPL260701C00100000", "AAPL", "call", 100.0, exp,
        quantity=3, avg_entry_price=1.50, strategy="orb_options",  # NOT counted
    )

    greeks = vol_runtime._build_portfolio_greeks(equity=100_000.0)

    # 1 vol_short leg → net_vega = -10.0, net_theta = +1.0
    assert greeks["num_open_positions"] == 1
    assert greeks["net_vega"] == pytest.approx(-10.0)
    assert greeks["net_theta"] == pytest.approx(1.0)
    assert greeks["portfolio_value"] == 100_000.0
