"""Tests for AlphaRuntime's vol_options track — the LIVE two-plane code path.

tests/test_vol_scan.py covers the same logical track but tests the legacy
execution_layer.runtime.TradingRuntime, which is not what's actually running
in production. That gap is exactly why three real bugs in
execution_layer.alpha_plane.AlphaRuntime's vol_options_scan_and_trade went
undetected: a call to a nonexistent vol_analytics.get_vol_snapshot, a
run_vol_consensus call missing required arguments, and a fabricated
_build_portfolio_greeks. All three are fixed here with real regression
coverage on the actual class that runs live.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analyst_layer.market_regime import DailyRegime
from config.settings import Settings
from data_layer.models import OptionContract, OptionType
from execution_layer.alpha_plane import AlphaRuntime
from execution_layer.guardrails import CircuitBreaker
from execution_layer.state_store import StateStore
from tests.test_vol_scan import (
    _AAPL_DTE,
    _AAPL_EXP,
    _AAPL_UNDERLYING,
    _CALL_CONTRACT_SYMBOL,
    _PUT_CONTRACT_SYMBOL,
    _SHORT_CALL_STRIKE,
    _SHORT_PUT_STRIKE,
    _aapl_chain,
    _aapl_iron_condor_payload,
    _aapl_vol_snapshot,
    _vix_price_series,
)


@pytest.fixture
def alpha(tmp_path: Path) -> AlphaRuntime:
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_position_detail.return_value = {
        "qty": -1.0, "avg_entry_price": 2.50, "current_price": 2.50, "unrealized_plpc": 0.0,
    }
    broker.submit_spread_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 1, "filled_avg_price": None,
    }
    broker.get_open_orders.return_value = []

    data_client = MagicMock()
    data_client.get_volatility_snapshot.return_value = _aapl_vol_snapshot()
    data_client.get_option_chain.return_value = _aapl_chain()
    from data_layer.exceptions import DataLayerError
    def _price_history_side_effect(symbol, **kwargs):
        if symbol == "^VIX":
            return _vix_price_series()
        raise DataLayerError(f"no mock for {symbol}")
    data_client.get_price_history.side_effect = _price_history_side_effect

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())
    store = StateStore(tmp_path / "alpha_vol_test.sqlite3")

    rt = AlphaRuntime(
        settings=settings, data_client=data_client, broker=broker,
        intraday_breaker=breaker, options_breaker=breaker, thesis_breaker=breaker, swing_breaker=breaker,
        state_store=store, anthropic_client=MagicMock(), watchlist=["AAPL"],
    )
    rt._daily_regime = DailyRegime(
        vix_current=22.0, vix_trend="stable", market_trend="bullish",
        arm_orb_equity=True, arm_orb_options=True, arm_thesis=True, arm_vol=True,
    )
    return rt


# ── Regression: vol_options_scan_and_trade called a nonexistent
# vol_analytics.get_vol_snapshot and passed run_vol_consensus a wrong keyword
# (snapshot= instead of vol_snapshot=) while omitting option_chain entirely —
# every invocation silently failed via a broad except, so the vol_options
# track never found a single real candidate.

@patch("execution_layer.alpha_plane.run_vol_consensus")
def test_vol_scan_uses_real_data_client_methods_and_correct_consensus_args(mock_consensus, alpha: AlphaRuntime):
    mock_consensus.return_value = _aapl_iron_condor_payload()

    alpha.vol_options_scan_and_trade()

    alpha._data_client.get_volatility_snapshot.assert_called_once_with("AAPL")
    alpha._data_client.get_option_chain.assert_called_once_with("AAPL")

    mock_consensus.assert_called_once()
    _, kwargs = mock_consensus.call_args
    assert "vol_snapshot" in kwargs
    assert "option_chain" in kwargs
    assert kwargs["option_chain"] == _aapl_chain()
    # The iron condor should actually get submitted — proving the whole chain
    # (data fetch -> consensus -> _open_vol_options_position with the chain
    # argument it needs to find each leg) works end to end.
    alpha._broker.submit_spread_order.assert_called_once()


@patch("execution_layer.alpha_plane.run_vol_consensus")
def test_vol_scan_skips_candidate_with_earnings_within_dte(mock_consensus, alpha: AlphaRuntime):
    snapshot = _aapl_vol_snapshot().model_copy(update={"earnings_within_dte": True})
    alpha._data_client.get_volatility_snapshot.return_value = snapshot

    alpha.vol_options_scan_and_trade()

    mock_consensus.assert_not_called()
    events = alpha._state_store.get_events(event_type_like="vol_scan_skipped_earnings%")
    assert len(events) == 1


# ── Regression: _build_portfolio_greeks fed the Greeks Risk Officer entirely
# fabricated data (net_delta hardcoded to 0.0, net_vega/theta derived from
# just a position count) instead of real per-position Black-Scholes Greeks.

def test_build_portfolio_greeks_computes_real_values_not_fabricated(alpha: AlphaRuntime):
    alpha._state_store.upsert_option_position(
        _CALL_CONTRACT_SYMBOL, "AAPL", "call", _SHORT_CALL_STRIKE, _AAPL_EXP.isoformat(),
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    alpha._data_client.get_option_chain.return_value = _aapl_chain()

    greeks = alpha._build_portfolio_greeks(equity=100_000.0)

    assert greeks["num_open_positions"] == 1
    # A short call must have NEGATIVE delta (short call = bearish-biased) and
    # NEGATIVE vega (short vol) — the opposite of the old fabricated formula's
    # signless magic numbers, and not just "-10.0"/"1.0" times a count.
    assert greeks["net_delta"] < 0
    assert greeks["net_vega"] < 0
    assert greeks["net_vega"] != pytest.approx(-10.0)
    assert greeks["net_theta"] != pytest.approx(1.0)


def test_build_portfolio_greeks_only_counts_vol_short(alpha: AlphaRuntime):
    exp = (date.today() + timedelta(days=_AAPL_DTE)).isoformat()
    alpha._state_store.upsert_option_position(
        "AAPL260701C00100000", "AAPL", "call", 100.0, exp,
        quantity=3, avg_entry_price=1.50, strategy="orb_options",
    )
    alpha._data_client.get_option_chain.return_value = _aapl_chain()

    greeks = alpha._build_portfolio_greeks(equity=100_000.0)

    assert greeks["num_open_positions"] == 0
    assert greeks["net_delta"] == 0.0
    assert greeks["net_vega"] == 0.0
    assert greeks["net_theta"] == 0.0


def test_build_portfolio_greeks_missing_chain_quote_excluded_not_crashed(alpha: AlphaRuntime):
    """A leg whose contract can't be found in the live chain (e.g. a stale
    strike) must be skipped, not crash the whole Greeks computation."""
    exp = (date.today() + timedelta(days=_AAPL_DTE)).isoformat()
    alpha._state_store.upsert_option_position(
        "AAPL260815C99999000", "AAPL", "call", 9999.0, exp,
        quantity=-1, avg_entry_price=2.50, strategy="vol_short",
    )
    alpha._data_client.get_option_chain.return_value = _aapl_chain()  # no 9999 strike in here

    greeks = alpha._build_portfolio_greeks(equity=100_000.0)  # must not raise

    assert greeks["net_delta"] == 0.0
    assert greeks["net_vega"] == 0.0
    assert greeks["net_theta"] == 0.0
