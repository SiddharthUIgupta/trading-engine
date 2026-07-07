from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from execution_layer.guardrails import RobustCircuitBreaker
from execution_layer.protection_plane import ProtectionRuntime
from execution_layer.state_store import StateStore


@pytest.fixture
def protection(tmp_path: Path) -> ProtectionRuntime:
    settings = Settings(_env_file=None)

    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_open_orders.return_value = []
    broker.get_position_detail.return_value = {
        "qty": 5.0, "avg_entry_price": 100.0, "current_price": 100.0, "unrealized_plpc": 0.0,
    }
    broker.submit_order.return_value = {"status": "submitted", "order_id": "sell-1"}
    broker.submit_bracket_order.return_value = {"status": "submitted", "order_id": "buy-1", "stop_order_id": "stop-1"}

    def _breaker(name: str) -> RobustCircuitBreaker:
        b = RobustCircuitBreaker(
            max_position_size_pct=settings.max_position_size_pct,
            max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
            capital_limit_pct=0.25,
            daily_profit_target_usd=1_000.0,
            name=name,
        )
        b.start_trading_day(equity=100_000.0, today=date.today())
        return b

    store = StateStore(tmp_path / "protection_test.sqlite3")

    rt = ProtectionRuntime(
        settings=settings,
        broker=broker,
        state_store=store,
        anthropic_client=MagicMock(),
        data_client=MagicMock(),
        intraday_breaker=_breaker("intraday"),
        options_breaker=_breaker("options"),
        thesis_breaker=_breaker("thesis"),
        swing_breaker=_breaker("swing"),
    )
    return rt


# ── Bug 1: bracket stop must be cancelled before a software exit sells ──────

def test_intraday_exit_cancels_bracket_stop_before_selling(protection: ProtectionRuntime):
    protection._state_store.upsert_position(
        "AAPL", quantity=10, avg_entry_price=100.0,
        last_buy_at=date.today().isoformat(), strategy="thesis",
        stop_price=82.0, bracket_stop_order_id="stop-abc",
    )
    protection._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 80.0,  # -20% breaches thesis 18% stop
        "unrealized_plpc": -0.20,
    }

    protection._check_intraday_exits(equity=100_000.0)

    protection._broker.cancel_order.assert_called_once_with("stop-abc")
    protection._broker.submit_order.assert_called_once()
    # cancel must happen before the sell is submitted, not after
    cancel_call_order = protection._broker.method_calls.index(
        next(c for c in protection._broker.method_calls if c[0] == "cancel_order")
    )
    submit_call_order = protection._broker.method_calls.index(
        next(c for c in protection._broker.method_calls if c[0] == "submit_order")
    )
    assert cancel_call_order < submit_call_order

    positions = {p["ticker"]: p for p in protection._state_store.get_positions()}
    assert positions["AAPL"]["bracket_stop_order_id"] is None


def test_intraday_exit_without_bracket_stop_does_not_cancel(protection: ProtectionRuntime):
    protection._state_store.upsert_position(
        "MSFT", quantity=10, avg_entry_price=100.0,
        last_buy_at=date.today().isoformat(), strategy="thesis", stop_price=82.0,
    )
    protection._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 80.0, "unrealized_plpc": -0.20,
    }

    protection._check_intraday_exits(equity=100_000.0)

    protection._broker.cancel_order.assert_not_called()
    protection._broker.submit_order.assert_called_once()


def test_swing_stop_loss_exit_cancels_bracket_stop(protection: ProtectionRuntime):
    """Regression test for the scoping bug: bracket_stop_id was only assigned
    inside the `if not should_exit` branch, so a stop-loss/max-hold/news exit
    (which sets should_exit=True *before* that branch runs) would never see
    it and never cancel the resting stop.
    """
    protection._state_store.upsert_position(
        "TSLA", quantity=10, avg_entry_price=100.0,
        last_buy_at=date.today().isoformat(), strategy="swing",
        stop_price=92.0, bracket_stop_order_id="stop-swing-1",
    )
    protection._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 90.0,  # -10% >= swing 8% stop
        "unrealized_plpc": -0.10,
    }

    protection._check_swing_exits(equity=100_000.0)

    protection._broker.cancel_order.assert_called_once_with("stop-swing-1")
    protection._broker.submit_order.assert_called_once()


# ── Bug 2: breaker halted state must sync back to "false" on a new day ─────

def test_breaker_state_syncs_true_on_trip_and_false_after_reset(protection: ProtectionRuntime):
    protection._intraday_breaker.check_drawdown(-50_000.0)  # forces _tripped = True
    assert protection._intraday_breaker.is_tripped

    protection._sync_breaker_state_to_db()
    assert protection._state_store.is_breaker_halted("intraday") is True

    # New trading day — ensure_day_started resets in-memory _tripped to False
    protection._intraday_breaker.ensure_day_started(equity=100_000.0, today=date.today() + timedelta(days=1))
    assert not protection._intraday_breaker.is_tripped

    protection._sync_breaker_state_to_db()
    assert protection._state_store.is_breaker_halted("intraday") is False


def test_breaker_state_sync_folds_profit_lock_into_halted_key(protection: ProtectionRuntime):
    """Alpha's is_breaker_halted() only ever reads the 'halted' key — a
    profit-lock previously wrote a separate 'profit_locked' key that Alpha
    never checked, so profit locks never actually stopped Alpha from queuing
    new BUY intents for that bucket.
    """
    protection._thesis_breaker.check_profit_target(1_000_000.0)  # forces _profit_locked = True
    assert protection._thesis_breaker.is_stock_halted

    protection._sync_breaker_state_to_db()

    assert protection._state_store.is_breaker_halted("thesis") is True


# ── Bug 3: stale order intents must expire, never execute ──────────────────

def test_stale_order_intent_expires_and_is_not_submitted(protection: ProtectionRuntime):
    protection._state_store.write_order_intent(
        client_order_id="old-1", strategy="thesis", ticker="NVDA",
        action="BUY", quantity=5, limit_price=100.0,
    )
    # Back-date created_at to 5 hours ago (past the 4h expiry window)
    stale_ts = (datetime.utcnow() - timedelta(hours=5)).isoformat()
    with protection._state_store._connect() as conn:
        conn.execute("UPDATE order_intents SET created_at=? WHERE client_order_id=?", (stale_ts, "old-1"))
        conn.commit()

    protection.consume_order_intents(today=date.today(), equity=100_000.0)

    protection._broker.submit_bracket_order.assert_not_called()
    protection._broker.submit_order.assert_not_called()

    with protection._state_store._connect() as conn:
        status = conn.execute(
            "SELECT status FROM order_intents WHERE client_order_id=?", ("old-1",)
        ).fetchone()[0]
    assert status == "expired"


def test_fresh_order_intent_still_executes(protection: ProtectionRuntime):
    protection._state_store.write_order_intent(
        client_order_id="fresh-1", strategy="thesis", ticker="NVDA",
        action="BUY", quantity=5, limit_price=100.0,
    )

    protection.consume_order_intents(today=date.today(), equity=100_000.0)

    protection._broker.submit_bracket_order.assert_called_once()


# ── Bug: options exit checks were gated behind options_track_enabled ────────
# Real production bug: options_track_enabled (default False — deliberately off
# after a real $8,647 loss) was ALSO gating the options stop-loss exit check,
# not just new entries. A position sitting at -68% would never get exited.
# Exit checks must be unconditional (invariant #1) — track-enabled flags are
# for Alpha Plane's entry decisions, never for Protection's exit checks.

def test_options_exit_fires_even_when_options_track_disabled(protection: ProtectionRuntime):
    assert protection._settings.options_track_enabled is False  # the exact bug condition

    protection._state_store.upsert_option_position(
        contract_symbol="ACN260731P00125000", underlying_symbol="ACN", option_type="put",
        strike=125.0, expiration="2026-07-31", quantity=1, avg_entry_price=7.50,
        opened_at=date.today().isoformat(), strategy="orb_options",
    )
    protection._broker.get_position_detail.return_value = {
        "qty": 1.0, "avg_entry_price": 7.50, "current_price": 2.40, "unrealized_plpc": -0.68,
    }
    protection._broker.submit_option_order = MagicMock(
        return_value={"status": "filled", "order_id": "exit-1"}
    )

    protection.intraday_monitoring()

    protection._broker.submit_option_order.assert_called_once()
    args, kwargs = protection._broker.submit_option_order.call_args
    assert args[0] == "ACN260731P00125000" or kwargs.get("contract_symbol") == "ACN260731P00125000"


# ── Regression: reconciliation used to call broker.get_position_detail() with
# no error handling at all. Now that get_position_detail raises on a genuine
# API failure (rather than silently returning None), a single ticker's
# transient error must not crash the whole reconciliation loop — every other
# position still needs to get reconciled and protected on the same tick.

def test_reconcile_positions_one_ticker_api_failure_does_not_block_others(protection: ProtectionRuntime):
    protection._state_store.upsert_position(
        "BROKEN", quantity=10, avg_entry_price=100.0,
        last_buy_at=date.today().isoformat(), strategy="thesis",
    )
    protection._state_store.upsert_position(
        "FINE", quantity=5, avg_entry_price=50.0,
        last_buy_at=date.today().isoformat(), strategy="thesis",
    )

    def side_effect(ticker):
        if ticker == "BROKEN":
            raise RuntimeError("transient API failure")
        return {"qty": 3.0, "avg_entry_price": 50.0, "current_price": 55.0, "unrealized_plpc": 0.1}

    protection._broker.get_position_detail = MagicMock(side_effect=side_effect)

    protection._reconcile_positions()  # must not raise

    # BROKEN must NOT be deleted or altered despite the failed lookup
    positions = {p["ticker"]: p for p in protection._state_store.get_positions()}
    assert positions["BROKEN"]["quantity"] == 10
    # FINE must still be correctly reconciled to the broker's real qty
    assert positions["FINE"]["quantity"] == 3


def test_reconcile_option_positions_one_contract_api_failure_does_not_block_others(protection: ProtectionRuntime):
    protection._state_store.upsert_option_position(
        contract_symbol="BROKEN260731P00100000", underlying_symbol="BROKEN", option_type="put",
        strike=100.0, expiration="2026-07-31", quantity=2, avg_entry_price=5.0,
        opened_at=date.today().isoformat(), strategy="orb_options",
    )
    protection._state_store.upsert_option_position(
        contract_symbol="FINE260731C00050000", underlying_symbol="FINE", option_type="call",
        strike=50.0, expiration="2026-07-31", quantity=4, avg_entry_price=3.0,
        opened_at=date.today().isoformat(), strategy="orb_options",
    )

    def side_effect(contract_symbol):
        if contract_symbol == "BROKEN260731P00100000":
            raise RuntimeError("transient API failure")
        return {"qty": 1.0, "avg_entry_price": 3.0, "current_price": 4.0, "unrealized_plpc": 0.33}

    protection._broker.get_position_detail = MagicMock(side_effect=side_effect)

    protection._reconcile_option_positions()  # must not raise

    positions = {p["contract_symbol"]: p for p in protection._state_store.get_option_positions()}
    assert positions["BROKEN260731P00100000"]["quantity"] == 2
    assert positions["FINE260731C00050000"]["quantity"] == 1


# ── Regression: swing_track_enabled gated _check_swing_exits the same way
# options_track_enabled gated options exits (bug #4/#6 in the 2026-07-06
# audit). Dormant only because the flag defaults True — if it's ever
# disabled the way options_track_enabled was, swing stop-losses would
# silently stop being checked, reproducing the ACN/B incident on the swing
# book instead.

def test_swing_exit_fires_even_when_swing_track_disabled(protection: ProtectionRuntime):
    protection._settings.swing_track_enabled = False

    protection._state_store.upsert_position(
        "SWNG", quantity=10, avg_entry_price=100.0,
        last_buy_at=date.today().isoformat(), strategy="swing",
    )
    protection._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 80.0, "unrealized_plpc": -0.20,
    }

    protection.intraday_monitoring()

    protection._broker.submit_order.assert_called_once()


# ── GlobalRiskState wiring (audit finding #7) ───────────────────────────────
# Regression: GlobalRiskState (weekly/trailing drawdown halt) was never wired
# into the live two-plane system at all — main_alpha.py/main_protection.py
# constructed all 4 breakers with no global_state=, so no automatic
# full-system halt existed no matter how bad the drawdown got.

from execution_layer.guardrails import GlobalRiskState


def _protection_with_global_state(tmp_path: Path) -> ProtectionRuntime:
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_open_orders.return_value = []

    global_state = GlobalRiskState(max_weekly_drawdown_pct=0.08, max_trailing_drawdown_pct=0.20)

    def _breaker(name: str) -> RobustCircuitBreaker:
        b = RobustCircuitBreaker(
            max_position_size_pct=settings.max_position_size_pct,
            max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
            capital_limit_pct=0.25,
            daily_profit_target_usd=1_000.0,
            name=name,
            global_state=global_state,
        )
        b.start_trading_day(equity=100_000.0, today=date.today())
        return b

    store = StateStore(tmp_path / "global_risk_test.sqlite3")
    rt = ProtectionRuntime(
        settings=settings, broker=broker, state_store=store,
        anthropic_client=MagicMock(), data_client=MagicMock(),
        intraday_breaker=_breaker("intraday"), options_breaker=_breaker("options"),
        thesis_breaker=_breaker("thesis"), swing_breaker=_breaker("swing"),
        global_risk_state=global_state,
    )
    return rt


def test_trailing_drawdown_halt_persists_to_db_for_alpha_to_read(tmp_path: Path):
    rt = _protection_with_global_state(tmp_path)
    # First tick establishes the peak at 100k
    rt.intraday_monitoring()
    assert GlobalRiskState.is_halted_in_db(rt._state_store) == (False, "")

    # Equity craters 25% from peak — past the 20% trailing limit
    rt._broker.get_equity.return_value = 75_000.0
    rt.intraday_monitoring()

    halted, reason = GlobalRiskState.is_halted_in_db(rt._state_store)
    assert halted is True
    assert "TRAILING HALT" in reason


def test_globally_halted_breaker_blocks_new_intraday_positions(tmp_path: Path):
    """Once GlobalRiskState trips, Protection's own breakers (which have the
    shared instance wired in-memory) must immediately reflect the halt —
    no restart needed, since it's the same process/object."""
    rt = _protection_with_global_state(tmp_path)
    rt.intraday_monitoring()  # establish peak
    rt._broker.get_equity.return_value = 75_000.0  # -25%, past trailing limit
    rt.intraday_monitoring()

    assert rt._intraday_breaker.is_stock_halted is True
    assert rt._options_breaker.is_stock_halted is True
    assert rt._thesis_breaker.is_stock_halted is True
    assert rt._swing_breaker.is_stock_halted is True


# ── Halted state must gate entries only, never discretionary sells ─────────
# Regression: breaker.is_stock_halted was checked uniformly for BUY and SELL
# order intents in consume_order_intents. A legitimate discretionary SELL
# ("thesis broken, exit early") could be silently skipped during a halt.

def test_consume_order_intents_allows_sell_even_when_breaker_halted(protection: ProtectionRuntime):
    protection._thesis_breaker._tripped = True
    protection._state_store.write_order_intent(
        client_order_id="sell-halted-1", strategy="thesis", ticker="NVDA",
        action="SELL", quantity=5, limit_price=90.0,
    )

    protection.consume_order_intents(today=date.today(), equity=100_000.0)

    protection._broker.submit_order.assert_called_once()


def test_consume_order_intents_still_blocks_buy_when_breaker_halted(protection: ProtectionRuntime):
    protection._thesis_breaker._tripped = True
    protection._state_store.write_order_intent(
        client_order_id="buy-halted-1", strategy="thesis", ticker="NVDA",
        action="BUY", quantity=5, limit_price=100.0,
    )

    protection.consume_order_intents(today=date.today(), equity=100_000.0)

    protection._broker.submit_bracket_order.assert_not_called()
    protection._broker.submit_order.assert_not_called()


# ── Dead bearish-news swing exit (audit finding #14) ────────────────────────
# Regression: self._daily_news_ticker_signals was declared but never populated
# anywhere in this file despite the class docstring claiming it's "loaded
# from state_store events" — the adverse-news swing exit branch could never
# fire. Alpha now records a machine-readable 'daily_regime_news_signals'
# event; Protection must load it at the start of each tick.

def test_load_daily_news_signals_populates_from_alpha_event(tmp_path: Path):
    rt = _protection_with_global_state(tmp_path)
    import json
    rt._state_store.record_event(
        event_type="daily_regime_news_signals",
        detail=json.dumps([{"ticker": "XYZ", "catalyst": "guidance cut", "direction": "bearish"}]),
    )

    rt._load_daily_news_signals()

    assert rt._daily_news_ticker_signals == [{"ticker": "XYZ", "catalyst": "guidance cut", "direction": "bearish"}]


def test_swing_exit_fires_on_bearish_news_catalyst(tmp_path: Path):
    rt = _protection_with_global_state(tmp_path)
    rt._state_store.upsert_position(
        "XYZ", quantity=10, avg_entry_price=100.0,
        last_buy_at=date.today().isoformat(), strategy="swing",
    )
    rt._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 102.0, "unrealized_plpc": 0.02,
    }
    rt._daily_news_ticker_signals = [{"ticker": "XYZ", "catalyst": "guidance cut", "direction": "bearish"}]

    rt._check_swing_exits(equity=100_000.0)

    rt._broker.submit_order.assert_called_once()


# ── _compute_today_pnl silent degradation (audit finding #15) ──────────────
# Regression: a bare except swallowed any broker.get_all_positions() failure
# with only a WARNING log — a transient API blip could silently understate
# today_pnl and let a real drawdown slip past check_drawdown() with no
# visible trace.

def test_compute_today_pnl_logs_error_and_records_event_on_broker_failure(protection: ProtectionRuntime):
    protection._broker.get_all_positions.side_effect = RuntimeError("transient API failure")

    pnl = protection._compute_today_pnl(frozenset({"thesis"}), include_options=False)

    assert isinstance(pnl, float)  # must not raise — realized portion still returned
    events = protection._state_store.get_events(event_type_like="pnl_computation_degraded%")
    assert len(events) == 1
