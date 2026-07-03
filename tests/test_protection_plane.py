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
