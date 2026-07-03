"""Regression tests for the 2026-07 risk-inversion fixes.

Invariants locked in here:
  1. Stop-loss exits execute UNCONDITIONALLY — a tripped or profit-locked
     breaker must never block or skip an exit (breakers gate entries only).
  2. Hitting the daily profit target blocks new entries but NEVER liquidates
     open positions (the backtested thesis edge depends on winners running).
  3. check_profit_target fires exactly once per day (transition-only).
  4. The macro-news regime adjustment is monotonic: bearish news may raise
     effective VIX; bullish news must never lower it.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from analyst_layer.market_regime import assess_daily_regime
from config.settings import Settings
from data_layer.models import PriceBar
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.state_store import StateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_runtime(tmp_path: Path, current_price: float = 75.0) -> TradingRuntime:
    """Runtime with one thesis position entered at 100.0, now at
    `current_price`. Default 75.0 = -25%, past the 18% thesis stop.
    """
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_position_shares.return_value = 10
    broker.get_position_detail.return_value = {
        "qty": 10.0,
        "avg_entry_price": 100.0,
        "current_price": current_price,
        "unrealized_plpc": (current_price - 100.0) / 100.0,
    }
    broker.submit_order.return_value = {"status": "submitted", "order_id": "exit-1"}
    broker.get_open_orders.return_value = []

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.05)
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    store = StateStore(tmp_path / "exit_priority_test.sqlite3")
    store.upsert_position("AAPL", 10, 100.0, strategy="thesis", high_water_mark=100.0)

    return TradingRuntime(
        settings=settings,
        data_client=MagicMock(),
        broker=broker,
        intraday_breaker=breaker,
        options_breaker=breaker,
        thesis_breaker=breaker,
        swing_breaker=breaker,
        state_store=store,
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )


# ---------------------------------------------------------------------------
# 1. Exits survive breaker state
# ---------------------------------------------------------------------------

def test_stop_loss_exit_executes_while_breaker_tripped(tmp_path: Path):
    rt = _make_runtime(tmp_path)
    rt._breaker._tripped = True  # drawdown breach earlier in the day

    rt._check_intraday_exits(equity=100_000.0)

    rt._broker.submit_order.assert_called_once()
    proposal = rt._broker.submit_order.call_args[0][0]
    assert proposal.action.value in ("SELL", "sell")
    assert proposal.ticker == "AAPL"


def test_stop_loss_exit_executes_while_profit_locked(tmp_path: Path):
    rt = _make_runtime(tmp_path)
    rt._breaker._profit_locked = True

    rt._check_intraday_exits(equity=100_000.0)

    rt._broker.submit_order.assert_called_once()


def test_intraday_monitoring_runs_exit_checks_when_all_breakers_halted(tmp_path: Path):
    rt = _make_runtime(tmp_path)
    for b in (rt._intraday_breaker, rt._options_breaker, rt._thesis_breaker, rt._swing_breaker):
        b._tripped = True

    rt.intraday_monitoring()

    # The tripped breaker must not have suppressed the stop-loss sell.
    rt._broker.submit_order.assert_called_once()


def test_one_failed_exit_does_not_abort_remaining_exits(tmp_path: Path):
    rt = _make_runtime(tmp_path)
    rt._state_store.upsert_position("MSFT", 10, 100.0, strategy="thesis", high_water_mark=100.0)
    # First submit blows up, second must still be attempted.
    rt._broker.submit_order.side_effect = [RuntimeError("alpaca 500"), {"status": "submitted", "order_id": "x"}]

    rt._check_intraday_exits(equity=100_000.0)

    assert rt._broker.submit_order.call_count == 2


# ---------------------------------------------------------------------------
# 2. Profit lock never liquidates
# ---------------------------------------------------------------------------

def test_profit_lock_does_not_close_positions_or_cancel_orders(tmp_path: Path):
    rt = _make_runtime(tmp_path, current_price=110.0)  # winner, no exit due

    rt._lock_in_profit(reason="test target reached")

    rt._broker.close_position.assert_not_called()
    rt._broker.cancel_order.assert_not_called()
    # Position untouched in local state as well.
    positions = {p["ticker"]: p for p in rt._state_store.get_positions()}
    assert positions["AAPL"]["quantity"] == 10


def test_profit_lock_still_blocks_new_entries(tmp_path: Path):
    breaker = CircuitBreaker(
        max_position_size_pct=0.05, max_daily_drawdown_pct=0.05, daily_profit_target_usd=50.0
    )
    breaker.start_trading_day(equity=100_000.0, today=date.today())
    assert breaker.check_profit_target(100_100.0) is True
    assert breaker.is_stock_halted is True  # entry paths still gate on this


# ---------------------------------------------------------------------------
# 3. Transition-only profit target
# ---------------------------------------------------------------------------

def test_check_profit_target_fires_exactly_once(tmp_path: Path):
    breaker = CircuitBreaker(
        max_position_size_pct=0.05, max_daily_drawdown_pct=0.05, daily_profit_target_usd=50.0
    )
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    assert breaker.check_profit_target(100_100.0) is True
    assert breaker.check_profit_target(100_200.0) is False  # already locked — no re-fire
    assert breaker.is_profit_locked is True


# ---------------------------------------------------------------------------
# 4. Monotonic regime guard
# ---------------------------------------------------------------------------

def _vix_bars(level: float, n: int = 10) -> list[PriceBar]:
    return [
        PriceBar(
            symbol="^VIX",
            timestamp=datetime(2026, 6, 1) + timedelta(days=i),
            open=level, high=level, low=level, close=level, volume=0,
        )
        for i in range(n)
    ]


def _spy_closes(n: int = 40) -> list[float]:
    return [400.0 + i for i in range(n)]  # steady uptrend


def test_bullish_news_never_lowers_effective_vix():
    # VIX 31 disarms thesis; confidently bullish news must NOT re-arm it.
    regime = assess_daily_regime(
        _spy_closes(), _vix_bars(31.0),
        macro_sentiment="bullish", macro_confidence=0.9,
    )
    baseline = assess_daily_regime(_spy_closes(), _vix_bars(31.0))
    assert regime.arm_thesis == baseline.arm_thesis
    assert regime.arm_thesis is False


def test_bearish_news_can_raise_effective_vix_and_disarm():
    # VIX 28.5 arms thesis; strongly bearish news pushes effective VIX over 30.
    armed = assess_daily_regime(_spy_closes(), _vix_bars(28.5))
    tightened = assess_daily_regime(
        _spy_closes(), _vix_bars(28.5),
        macro_sentiment="bearish", macro_confidence=0.9,
    )
    assert armed.arm_thesis is True
    assert tightened.arm_thesis is False
