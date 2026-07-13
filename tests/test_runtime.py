from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analyst_layer.schemas import (
    Action,
    AgentSignal,
    Confidence,
    ConsensusPayload,
    RiskReview,
    RiskVerdict,
    TradeProposal,
)
from config.settings import Settings
from data_layer.models import OptionContract, OptionType, PriceSeries
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.state_store import StateStore


def _buy_payload(ticker: str, quantity: int = 5, price: float = 100.0) -> ConsensusPayload:
    from datetime import datetime

    signal = AgentSignal(
        agent_name="x", ticker=ticker, stance=Action.BUY, confidence=Confidence.HIGH, rationale="r", generated_at=datetime.utcnow()
    )
    proposal = TradeProposal(ticker=ticker, action=Action.BUY, quantity=quantity, limit_price=price)
    review = RiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=["ok"],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    return ConsensusPayload(ticker=ticker, signals=[signal], proposal=proposal, risk_review=review)


@pytest.fixture
def runtime(tmp_path: Path) -> TradingRuntime:
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_position_shares.return_value = 5
    broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 100.0, "current_price": 100.0, "unrealized_plpc": 0.0,
    }
    broker.submit_order.return_value = {"status": "submitted", "order_id": "abc"}
    broker.get_open_orders.return_value = []

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    store = StateStore(tmp_path / "runtime_test.sqlite3")

    rt = TradingRuntime(
        settings=settings,
        data_client=MagicMock(),
        broker=broker,
        intraday_breaker=breaker, options_breaker=breaker, thesis_breaker=breaker, swing_breaker=breaker,
        state_store=store,
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )
    return rt


def test_market_open_execution_submits_when_no_wash_sale_history(runtime: TradingRuntime):
    runtime._pending_payloads = {"AAPL": _buy_payload("AAPL")}
    runtime.market_open_execution()
    runtime._broker.submit_order.assert_called_once()


def test_market_open_execution_blocks_buy_on_wash_sale(runtime: TradingRuntime):
    runtime._state_store.record_realized_sale(
        "AAPL", sale_date=date.today(), quantity=5, sale_price=90.0, cost_basis=100.0
    )
    runtime._pending_payloads = {"AAPL": _buy_payload("AAPL")}

    runtime.market_open_execution()

    runtime._broker.submit_order.assert_not_called()


def test_market_open_execution_records_realized_sale_on_sell(runtime: TradingRuntime):
    from datetime import datetime

    runtime._state_store.upsert_position("AAPL", quantity=10, avg_entry_price=120.0, last_buy_at=date.today().isoformat())

    sell_signal = AgentSignal(
        agent_name="x", ticker="AAPL", stance=Action.SELL, confidence=Confidence.HIGH, rationale="r", generated_at=datetime.utcnow()
    )
    proposal = TradeProposal(ticker="AAPL", action=Action.SELL, quantity=10, limit_price=100.0)
    review = RiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=["ok"],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    payload = ConsensusPayload(ticker="AAPL", signals=[sell_signal], proposal=proposal, risk_review=review)
    runtime._pending_payloads = {"AAPL": payload}

    runtime.market_open_execution()

    losses = runtime._state_store.get_recent_loss_sales("AAPL", since=date.today())
    assert len(losses) == 1
    assert losses[0]["realized_pnl"] == pytest.approx(-200.0)


# ---- Position reconciliation against the broker's real state ----

def test_reconcile_positions_corrects_a_fill_that_arrived_late(runtime: TradingRuntime):
    """Regression test for the exact bug this caught live: a limit order
    correctly recorded as 0 shares at submission time, which later filled
    without anything ever re-checking it.
    """
    runtime._state_store.upsert_position("AZTA", quantity=0, avg_entry_price=22.47, strategy="thesis")
    runtime._broker.get_position_detail.return_value = {
        "qty": 1.0, "avg_entry_price": 22.47, "current_price": 22.50, "unrealized_plpc": 0.001,
    }

    runtime._reconcile_positions()

    position = runtime._state_store.get_position("AZTA")
    assert position["quantity"] == 1


def test_reconcile_positions_leaves_already_correct_positions_alone(runtime: TradingRuntime):
    runtime._state_store.upsert_position("WWW", quantity=50, avg_entry_price=17.17, strategy="thesis")
    runtime._broker.get_position_detail.return_value = {
        "qty": 50.0, "avg_entry_price": 17.17, "current_price": 17.20, "unrealized_plpc": 0.002,
    }

    runtime._reconcile_positions()

    position = runtime._state_store.get_position("WWW")
    assert position["quantity"] == 50
    assert position["strategy"] == "thesis"  # untouched, not reset to the upsert_position default


def test_reconcile_positions_handles_a_position_that_no_longer_exists_on_the_broker(runtime: TradingRuntime):
    runtime._state_store.upsert_position("CLOSED", quantity=10, avg_entry_price=5.0)
    runtime._broker.get_position_detail.return_value = None  # broker reports no open position at all

    runtime._reconcile_positions()

    # Position with zero broker quantity is deleted from the DB (not set to 0)
    position = runtime._state_store.get_position("CLOSED")
    assert position is None


# ---- _record_fill: averaging into an existing position ----

def test_record_fill_buy_uses_broker_blended_quantity_and_cost_basis(runtime: TradingRuntime):
    """Regression test for a real bug: a second BUY on an existing position
    used to record only THIS fill's quantity and price, discarding the
    broker's correctly blended running total across both fills.
    """
    runtime._state_store.upsert_position("WWW", quantity=100, avg_entry_price=17.15, strategy="thesis")
    runtime._broker.get_position_detail.return_value = {
        "qty": 150.0, "avg_entry_price": 17.03, "current_price": 16.77, "unrealized_plpc": -0.015,
    }
    proposal = TradeProposal(ticker="WWW", action=Action.BUY, quantity=50, limit_price=16.80)

    runtime._record_fill("WWW", proposal, date.today(), strategy="thesis")

    position = runtime._state_store.get_position("WWW")
    assert position["quantity"] == 150
    assert position["avg_entry_price"] == 17.03


def test_record_fill_buy_never_lowers_high_water_mark(runtime: TradingRuntime):
    """A fill below the existing peak (averaging into a dip) must not pull
    the trailing-stop's high-water mark down with it — only a new high
    should move it.
    """
    runtime._state_store.upsert_position(
        "WWW", quantity=100, avg_entry_price=17.15, high_water_mark=17.32, strategy="thesis"
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 150.0, "avg_entry_price": 17.03, "current_price": 16.77, "unrealized_plpc": -0.015,
    }
    proposal = TradeProposal(ticker="WWW", action=Action.BUY, quantity=50, limit_price=16.80)

    runtime._record_fill("WWW", proposal, date.today(), strategy="thesis")

    position = runtime._state_store.get_position("WWW")
    assert position["high_water_mark"] == 17.32


def test_record_fill_buy_raises_high_water_mark_on_a_new_high(runtime: TradingRuntime):
    runtime._state_store.upsert_position(
        "WWW", quantity=50, avg_entry_price=17.15, high_water_mark=17.15, strategy="thesis"
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 100.0, "avg_entry_price": 17.46, "current_price": 17.80, "unrealized_plpc": 0.02,
    }
    proposal = TradeProposal(ticker="WWW", action=Action.BUY, quantity=50, limit_price=17.80)

    runtime._record_fill("WWW", proposal, date.today(), strategy="thesis")

    position = runtime._state_store.get_position("WWW")
    assert position["high_water_mark"] == 17.80


# ---- Options track ----

def _hold_payload(ticker: str) -> ConsensusPayload:
    from datetime import datetime

    signal = AgentSignal(
        agent_name="x", ticker=ticker, stance=Action.HOLD, confidence=Confidence.LOW, rationale="r", generated_at=datetime.utcnow()
    )
    proposal = TradeProposal(ticker=ticker, action=Action.HOLD, quantity=0, limit_price=100.0)
    review = RiskReview(
        verdict=RiskVerdict.APPROVED, reasons=["no signals"], max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02, reviewed_at=datetime.utcnow(),
    )
    return ConsensusPayload(ticker=ticker, signals=[signal], proposal=proposal, risk_review=review)


def _option_contract(
    option_type: OptionType = OptionType.CALL, dte: int = 3, strike: float = 100.0,
    underlying_price: float = 100.0, bid: float = 1.0, ask: float = 1.10,
) -> OptionContract:
    return OptionContract(
        contract_symbol="TEST260701C00100000",
        underlying_symbol="TEST",
        underlying_price=underlying_price,
        expiration=date.today() + timedelta(days=dte),
        dte=dte,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        implied_volatility=0.3,
        open_interest=100,
        volume=10,
    )


def _orb_bars(direction: str | None) -> list:
    """3 quiet range bars, then optionally a confirmed, volume-backed
    breakout in the requested direction. direction=None -> no breakout.
    """
    from data_layer.models import PriceBar

    range_bars = [
        PriceBar(symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 30) + timedelta(minutes=5 * i),
                  open=100, high=101, low=99, close=100, volume=10_000)
        for i in range(3)
    ]
    if direction is None:
        return range_bars + [
            PriceBar(symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 45), open=100, high=100.5, low=99.5, close=100, volume=10_000)
        ]
    if direction == "long":
        breakout = PriceBar(symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 45), open=100, high=102, low=100, close=101.5, volume=20_000)
    else:
        breakout = PriceBar(symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 45), open=99, high=99, low=97, close=97.5, volume=20_000)
    return range_bars + [breakout]


def test_scan_and_trade_options_orb_skips_when_no_breakout(runtime: TradingRuntime):
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars(None))
    runtime._scan_and_trade_options_orb(["AAPL"], date.today(), equity=100_000.0)
    runtime._broker.submit_option_order.assert_not_called()


def test_scan_and_trade_options_orb_opens_call_on_long_breakout(runtime: TradingRuntime):
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars("long"))
    runtime._data_client.get_option_chain.return_value = [_option_contract()]
    runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 9, "filled_avg_price": 1.10,
    }
    runtime._broker.get_position_detail.return_value = {
        "qty": 9.0, "avg_entry_price": 1.10, "current_price": 1.10, "unrealized_plpc": 0.0,
    }

    runtime._scan_and_trade_options_orb(["AAPL"], date.today(), equity=100_000.0)

    runtime._broker.submit_option_order.assert_called_once()
    _, kwargs = runtime._broker.submit_option_order.call_args
    assert kwargs["side"] == Action.BUY  # a long breakout -> a call


def test_scan_and_trade_options_orb_opens_put_on_short_breakdown(runtime: TradingRuntime):
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars("short"))
    runtime._data_client.get_option_chain.return_value = [_option_contract(option_type=OptionType.PUT)]
    runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 9, "filled_avg_price": 1.10,
    }
    runtime._broker.get_position_detail.return_value = {
        "qty": 9.0, "avg_entry_price": 1.10, "current_price": 1.10, "unrealized_plpc": 0.0,
    }

    runtime._scan_and_trade_options_orb(["AAPL"], date.today(), equity=100_000.0)

    runtime._broker.submit_option_order.assert_called_once()
    args, kwargs = runtime._broker.submit_option_order.call_args
    assert kwargs["side"] == Action.BUY  # always BUY to open -- a put, but still a long position in that put
    assert args[0] == "TEST260701C00100000"  # the PUT contract matched from the mocked chain, not a call


def test_scan_and_trade_options_orb_records_summary_event(runtime: TradingRuntime):
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars(None))
    runtime._scan_and_trade_options_orb(["AAPL", "MSFT"], date.today(), equity=100_000.0)
    events = runtime._state_store.get_events(event_type_like="options_orb_scan_summary")
    assert len(events) == 1
    assert "0 ORB signal(s) found across 2 candidates" in events[0]["detail"]


def test_open_option_position_skips_when_risk_budget_insufficient(runtime: TradingRuntime):
    """options_max_risk_pct default is 1% of equity; a $50 equity account
    can't afford even 1 contract of a $1.10 option (x100 = $110/contract).
    """
    runtime._data_client.get_option_chain.return_value = [_option_contract(ask=1.10)]
    runtime._open_option_position("AAPL", Action.BUY, equity=50.0, today=date.today())
    runtime._broker.submit_option_order.assert_not_called()


def test_open_option_position_skips_when_no_contract_selected(runtime: TradingRuntime):
    runtime._data_client.get_option_chain.return_value = []  # empty chain
    runtime._open_option_position("AAPL", Action.BUY, equity=100_000.0, today=date.today())
    runtime._broker.submit_option_order.assert_not_called()


def test_check_options_exits_force_closes_near_expiration(runtime: TradingRuntime):
    """Must fire even with healthy P&L — this is the rule that actually
    keeps the track out of 0-1 DTE-style risk, not the stop-loss.
    """
    expiration = (date.today() + timedelta(days=0)).isoformat()  # inside the 0-day force-close floor
    runtime._state_store.upsert_option_position(
        "TEST260701C00100000", "TEST", "call", 100.0, expiration, quantity=5, avg_entry_price=1.10,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 1.10, "current_price": 1.50, "unrealized_plpc": 0.36,  # up, not down
    }
    runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 5, "filled_avg_price": 1.50,
    }

    runtime._check_options_exits(equity=100_000.0)

    runtime._broker.submit_option_order.assert_called_once()
    _, kwargs = runtime._broker.submit_option_order.call_args
    assert kwargs["side"] == Action.SELL


def test_check_options_exits_stop_loss_triggers(runtime: TradingRuntime):
    expiration = (date.today() + timedelta(days=15)).isoformat()  # well clear of the 7-day force-close floor
    runtime._state_store.upsert_option_position(
        "TEST260701C00100000", "TEST", "call", 100.0, expiration, quantity=5, avg_entry_price=2.00,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 2.00, "current_price": 1.00, "unrealized_plpc": -0.5,  # -50%, past the 40% stop
    }
    runtime._broker.submit_option_order.return_value = {
        "status": "submitted", "order_status": "filled", "filled_qty": 5, "filled_avg_price": 1.00,
    }

    runtime._check_options_exits(equity=100_000.0)

    runtime._broker.submit_option_order.assert_called_once()


def test_check_options_exits_no_action_when_healthy(runtime: TradingRuntime):
    expiration = (date.today() + timedelta(days=15)).isoformat()
    runtime._state_store.upsert_option_position(
        "TEST260701C00100000", "TEST", "call", 100.0, expiration, quantity=5, avg_entry_price=2.00,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 2.00, "current_price": 2.20, "unrealized_plpc": 0.1,
    }

    runtime._check_options_exits(equity=100_000.0)

    runtime._broker.submit_option_order.assert_not_called()


def test_record_option_fill_buy_accumulates_via_broker_detail(runtime: TradingRuntime):
    runtime._broker.get_position_detail.return_value = {
        "qty": 9.0, "avg_entry_price": 1.10, "current_price": 1.10, "unrealized_plpc": 0.0,
    }
    runtime._record_option_fill(
        "TEST260701C00100000", "TEST", "call", 100.0, "2026-07-01", Action.BUY, 9, date.today(),
    )
    position = runtime._state_store.get_option_position("TEST260701C00100000")
    assert position["quantity"] == 9
    assert position["avg_entry_price"] == 1.10
    assert position["opened_at"] == date.today().isoformat()


def test_record_option_fill_sell_records_realized_pnl(runtime: TradingRuntime):
    runtime._state_store.upsert_option_position(
        "TEST260701C00100000", "TEST", "call", 100.0, "2026-07-01", quantity=5, avg_entry_price=2.00,
    )
    runtime._broker.get_position_detail.return_value = None  # fully closed
    runtime._record_option_fill(
        "TEST260701C00100000", "TEST", "call", 100.0, "2026-07-01", Action.SELL, 5, date.today(), sale_price=1.00,
    )
    sales = runtime._state_store.get_all_realized_option_sales()
    assert len(sales) == 1
    assert sales[0]["realized_pnl"] == pytest.approx((1.00 - 2.00) * 5 * 100)


def test_reconcile_option_positions_corrects_a_fill_that_arrived_late(runtime: TradingRuntime):
    """Regression test for a real bug: a slow-to-fill options order
    (verified live to lag well behind equities on the paper account)
    correctly recorded as 0 contracts at submission time, with nothing
    ever re-checking it once it eventually did fill.
    """
    runtime._state_store.upsert_option_position(
        "TEST260701C00100000", "TEST", "call", 100.0, "2026-07-01", quantity=0, avg_entry_price=0.0,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 2.0, "avg_entry_price": 4.05, "current_price": 4.10, "unrealized_plpc": 0.012,
    }

    runtime._reconcile_option_positions()

    position = runtime._state_store.get_option_position("TEST260701C00100000")
    assert position["quantity"] == 2
    assert position["avg_entry_price"] == 4.05


# ---- Circuit-breaker shutdown reconciles local state, not just the broker ----

def test_close_all_and_reconcile_records_realized_pnl_and_zeroes_local_positions(runtime: TradingRuntime):
    """Regression test for a real bug found live: a real, successful
    overnight shutdown closed every position for real on Alpaca (verified
    directly against the account), but the local DB kept showing them as
    still open and never recorded the realized P&L anywhere.
    """
    runtime._state_store.upsert_position("WWW", quantity=150, avg_entry_price=17.030533, strategy="thesis")
    runtime._state_store.upsert_option_position(
        "SOFI260702C00017500", "SOFI", "call", 17.5, "2026-07-02", quantity=17, avg_entry_price=0.56, opened_at="2026-06-23",
    )

    def _fake_last_fill_price(symbol):
        return {"WWW": 16.81, "SOFI260702C00017500": 0.90}[symbol]

    runtime._broker.get_last_fill_price.side_effect = _fake_last_fill_price

    runtime._close_all_and_reconcile("daily profit target reached", event_type="daily_profit_target_reached")

    runtime._broker.close_all_positions.assert_called_once()

    www_position = runtime._state_store.get_position("WWW")
    assert www_position["quantity"] == 0
    equity_sales = runtime._state_store.get_all_realized_sales()
    assert len(equity_sales) == 1
    assert equity_sales[0]["ticker"] == "WWW"
    assert equity_sales[0]["realized_pnl"] == pytest.approx((16.81 - 17.030533) * 150)

    sofi_position = runtime._state_store.get_option_position("SOFI260702C00017500")
    assert sofi_position["quantity"] == 0
    option_sales = runtime._state_store.get_all_realized_option_sales()
    assert len(option_sales) == 1
    assert option_sales[0]["realized_pnl"] == pytest.approx((0.90 - 0.56) * 17 * 100)


def test_close_all_and_reconcile_falls_back_to_avg_entry_price_with_no_fill_found(runtime: TradingRuntime):
    runtime._state_store.upsert_position("AZTA", quantity=12, avg_entry_price=22.2525, strategy="thesis")
    runtime._broker.get_last_fill_price.return_value = None  # no matching filled order found

    runtime._close_all_and_reconcile("circuit breaker tripped")

    sales = runtime._state_store.get_all_realized_sales()
    assert sales[0]["realized_pnl"] == pytest.approx(0.0)  # sale_price fell back to cost_basis -- no P&L claimed


# ---- Profit-target lock stops stocks only -- options keep trading (explicit instruction) ----

def test_close_stocks_and_reconcile_closes_only_equities_not_options(runtime: TradingRuntime):
    runtime._state_store.upsert_position("WWW", quantity=150, avg_entry_price=17.030533, strategy="thesis")
    runtime._state_store.upsert_option_position(
        "SOFI260702C00017500", "SOFI", "call", 17.5, "2026-07-02", quantity=17, avg_entry_price=0.56, opened_at="2026-06-23",
    )
    runtime._broker.get_last_fill_price.return_value = 16.81
    runtime._broker.get_position_detail.return_value = None  # confirms flat immediately

    runtime._close_stocks_and_reconcile("daily profit target reached, equity=100598.69")

    runtime._broker.close_position.assert_called_once_with("WWW")
    www_position = runtime._state_store.get_position("WWW")
    assert www_position["quantity"] == 0

    # The option position must be completely untouched -- not closed, not reconciled.
    sofi_position = runtime._state_store.get_option_position("SOFI260702C00017500")
    assert sofi_position["quantity"] == 17
    assert runtime._state_store.get_all_realized_option_sales() == []


def test_close_stocks_and_reconcile_only_cancels_equity_pending_orders(runtime: TradingRuntime):
    runtime._state_store.upsert_position("AZTA", quantity=10, avg_entry_price=22.20, strategy="thesis")
    runtime._broker.get_open_orders.return_value = [
        {"order_id": "eq-order-1", "symbol": "AZTA", "side": "buy", "qty": 10, "limit_price": 22.20, "status": "new", "submitted_at": None},
        {"order_id": "opt-order-1", "symbol": "RIVN260702C00015000", "side": "buy", "qty": 13, "limit_price": 0.73, "status": "accepted", "submitted_at": None},
    ]
    runtime._broker.get_position_detail.return_value = None
    runtime._broker.get_last_fill_price.return_value = 22.20

    runtime._close_stocks_and_reconcile("daily profit target reached")

    runtime._broker.cancel_order.assert_called_once_with("eq-order-1")


def test_lock_in_profit_does_not_pause_the_scheduler(runtime: TradingRuntime):
    """Regression test for the explicit behavior change: the old global
    shutdown paused the whole scheduler (stopping options too); the
    stock-only lock must not, since options_scan_and_trade needs to keep
    firing on schedule.
    """
    halt_callback = MagicMock()
    runtime._halt_callback = halt_callback
    runtime._broker.get_position_detail.return_value = None

    runtime._lock_in_profit("daily profit target reached")

    halt_callback.assert_not_called()


def test_intraday_monitoring_keeps_checking_options_after_stock_profit_lock(runtime: TradingRuntime):
    runtime._settings.options_track_enabled = True
    runtime._breaker.daily_profit_target_usd = 50.0
    runtime._broker.get_equity.return_value = 100_100.0  # +100, past the target
    # A realistic, healthy detail -- well away from the force-close/stop-loss
    # floors -- so the option position survives reconciliation untouched and
    # genuinely gets evaluated by the exit-check loop, rather than being
    # zeroed out by reconciliation before exit-checking even runs.
    runtime._broker.get_position_detail.return_value = {
        "qty": 17.0, "avg_entry_price": 0.56, "current_price": 0.60, "unrealized_plpc": 0.07,
    }
    runtime._state_store.upsert_option_position(
        "SOFI260731C00017500", "SOFI", "call", 17.5, "2026-07-31", quantity=17, avg_entry_price=0.56, opened_at="2026-06-23",
    )

    runtime.intraday_monitoring()

    assert runtime._breaker.is_stock_halted is True
    assert runtime._breaker.is_options_halted is False
    # _check_options_exits must still have run (read this option position's broker detail), not been skipped.
    runtime._broker.get_position_detail.assert_any_call("SOFI260731C00017500")
    # And it must still be open afterward -- nothing forced it closed.
    runtime._broker.submit_option_order.assert_not_called()
    position = runtime._state_store.get_option_position("SOFI260731C00017500")
    assert position["quantity"] == 17


# ---- Equity ORB track: long-only, fixed-price stop/target, same-day-only ----

def test_scan_and_trade_orb_equities_skips_short_signals(runtime: TradingRuntime):
    """No shorting support in this system -- a confirmed ORB breakdown
    must be ignored on the equity side (options already expresses it
    safely via buying puts).
    """
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars("short"))
    runtime._scan_and_trade_orb_equities(["AAPL"], date.today(), equity=100_000.0)
    runtime._broker.submit_order.assert_not_called()


def test_scan_and_trade_orb_equities_skips_no_signal(runtime: TradingRuntime):
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars(None))
    runtime._scan_and_trade_orb_equities(["AAPL"], date.today(), equity=100_000.0)
    runtime._broker.submit_order.assert_not_called()


def test_scan_and_trade_orb_equities_opens_long_position_with_stop_and_target(runtime: TradingRuntime):
    runtime._data_client.get_price_history.return_value = PriceSeries(symbol="AAPL", interval="5m", bars=_orb_bars("long"))
    runtime._broker.submit_order.return_value = {"status": "submitted", "order_id": "abc", "order_status": "filled", "filled_qty": 5, "filled_avg_price": 101.0}
    runtime._broker.get_position_shares.return_value = 5
    # Disable quality filters — this test focuses on order placement + stop/target logic
    runtime._settings.orb_require_spy_positive = False
    runtime._settings.orb_min_gap_pct = 0.0

    runtime._scan_and_trade_orb_equities(["AAPL"], date.today(), equity=100_000.0)

    runtime._broker.submit_order.assert_called_once()
    proposal = runtime._broker.submit_order.call_args[0][0]
    assert proposal.action == Action.BUY

    position = runtime._state_store.get_position("AAPL")
    assert position["strategy"] == "orb"
    assert position["stop_price"] == 99.0  # the opening range low from _orb_bars("long")
    assert position["target_price"] == pytest.approx(101.0 + 2 * (101.0 - 99.0))  # 2R off the breakout level


def test_check_orb_exits_force_closes_a_position_held_past_its_entry_day(runtime: TradingRuntime):
    """ORB is a day-trade by design -- anything still open from a prior
    day must force-close regardless of price, the same EOD discipline
    the backtest was built and validated around.
    """
    runtime._state_store.upsert_position(
        "AAPL", quantity=5, avg_entry_price=101.0, last_buy_at="2026-06-20", strategy="orb",
        stop_price=99.0, target_price=105.0,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 101.0, "current_price": 102.0, "unrealized_plpc": 0.01,  # healthy, between stop and target
    }
    runtime._broker.submit_order.return_value = {"status": "submitted", "order_status": "filled", "filled_qty": 5, "filled_avg_price": 102.0}

    runtime._check_orb_exits(equity=100_000.0)

    runtime._broker.submit_order.assert_called_once()
    proposal = runtime._broker.submit_order.call_args[0][0]
    assert proposal.action == Action.SELL


def test_check_orb_exits_stop_loss_triggers_same_day(runtime: TradingRuntime):
    runtime._state_store.upsert_position(
        "AAPL", quantity=5, avg_entry_price=101.0, last_buy_at=date.today().isoformat(), strategy="orb",
        stop_price=99.0, target_price=105.0,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 101.0, "current_price": 98.5, "unrealized_plpc": -0.025,
    }
    runtime._broker.submit_order.return_value = {"status": "submitted", "order_status": "filled", "filled_qty": 5, "filled_avg_price": 98.5}

    runtime._check_orb_exits(equity=100_000.0)

    runtime._broker.submit_order.assert_called_once()


def test_check_orb_exits_no_action_when_healthy_and_same_day(runtime: TradingRuntime):
    runtime._state_store.upsert_position(
        "AAPL", quantity=5, avg_entry_price=101.0, last_buy_at=date.today().isoformat(), strategy="orb",
        stop_price=99.0, target_price=105.0,
    )
    runtime._broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 101.0, "current_price": 102.0, "unrealized_plpc": 0.01,
    }

    runtime._check_orb_exits(equity=100_000.0)

    runtime._broker.submit_order.assert_not_called()


def test_check_orb_exits_ignores_non_orb_positions(runtime: TradingRuntime):
    runtime._state_store.upsert_position("WWW", quantity=150, avg_entry_price=17.0, strategy="thesis")
    runtime._broker.get_position_detail.return_value = {
        "qty": 150.0, "avg_entry_price": 17.0, "current_price": 10.0, "unrealized_plpc": -0.4,  # would be a huge stop hit if treated as ORB
    }

    runtime._check_orb_exits(equity=100_000.0)

    runtime._broker.submit_order.assert_not_called()
