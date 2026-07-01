from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from analyst_layer.schemas import Action, TradeProposal
from execution_layer.guardrails import CircuitBreaker, CircuitBreakerTripped, execute_global_shutdown


@pytest.fixture
def breaker() -> CircuitBreaker:
    cb = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    cb.start_trading_day(equity=100_000.0, today=date(2026, 6, 18))
    return cb


def test_validate_position_size_passes_within_limit(breaker: CircuitBreaker):
    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=10, limit_price=100.0)  # 1,000 notional
    breaker.validate_position_size(proposal, equity=100_000.0)  # should not raise


def test_validate_position_size_blocks_oversized_order(breaker: CircuitBreaker):
    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=1000, limit_price=100.0)  # 100,000 notional
    with pytest.raises(CircuitBreakerTripped):
        breaker.validate_position_size(proposal, equity=100_000.0)


def test_validate_position_size_ignores_hold(breaker: CircuitBreaker):
    proposal = TradeProposal(ticker="AAPL", action=Action.HOLD, quantity=0, limit_price=100.0)
    breaker.validate_position_size(proposal, equity=100_000.0)  # should not raise


def test_validate_position_size_never_blocks_a_sell(breaker: CircuitBreaker):
    """A position that grew past the cap via price appreciation must still
    be exitable — the cap bounds new exposure (BUY), not reducing it (SELL).
    """
    proposal = TradeProposal(ticker="AAPL", action=Action.SELL, quantity=1000, limit_price=100.0)  # 100,000 notional
    breaker.validate_position_size(proposal, equity=100_000.0)  # should not raise despite exceeding the 5% cap


def test_check_drawdown_trips_at_threshold(breaker: CircuitBreaker):
    # -$2,000 = 2% of $100,000 day-start equity — exactly at the 2% limit
    assert breaker.check_drawdown(-2_000.0) is True
    assert breaker.is_tripped is True


def test_check_drawdown_does_not_trip_below_threshold(breaker: CircuitBreaker):
    # -$1,000 = 1% loss — below the 2% threshold
    assert breaker.check_drawdown(-1_000.0) is False
    assert breaker.is_tripped is False


def test_check_drawdown_does_not_trip_on_profit(breaker: CircuitBreaker):
    assert breaker.check_drawdown(500.0) is False
    assert breaker.is_tripped is False


def test_check_drawdown_requires_day_started():
    cb = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    with pytest.raises(RuntimeError):
        cb.check_drawdown(-2_000.0)


def test_assert_not_tripped_raises_after_breach(breaker: CircuitBreaker):
    breaker.check_drawdown(-3_000.0)
    with pytest.raises(CircuitBreakerTripped):
        breaker.assert_not_tripped()


def test_execute_global_shutdown_closes_positions_and_records_event():
    broker = MagicMock()
    state_store = MagicMock()

    execute_global_shutdown(broker, state_store, reason="test breach")

    broker.close_all_positions.assert_called_once_with(cancel_orders=True)
    state_store.record_event.assert_called_once()
    assert state_store.record_event.call_args.kwargs["event_type"] == "circuit_breaker_shutdown"


def test_invalid_limits_rejected_at_construction():
    with pytest.raises(ValueError):
        CircuitBreaker(max_position_size_pct=0.0, max_daily_drawdown_pct=0.02)


# ---- Track-scoped halting: profit target stops stocks only, options keep running ----

@pytest.fixture
def breaker_with_profit_target() -> CircuitBreaker:
    cb = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02, daily_profit_target_usd=50.0)
    cb.start_trading_day(equity=100_000.0, today=date(2026, 6, 18))
    return cb


def test_profit_locked_halts_stocks_but_not_options(breaker_with_profit_target: CircuitBreaker):
    breaker_with_profit_target.check_profit_target(100_100.0)  # +100, past the $50 target

    assert breaker_with_profit_target.is_stock_halted is True
    assert breaker_with_profit_target.is_options_halted is False


def test_real_drawdown_breach_halts_both_stocks_and_options(breaker: CircuitBreaker):
    breaker.check_drawdown(-3_000.0)  # -3% loss — exceeds 2% limit

    assert breaker.is_stock_halted is True
    assert breaker.is_options_halted is True


def test_assert_options_trading_allowed_does_not_raise_on_profit_lock(breaker_with_profit_target: CircuitBreaker):
    breaker_with_profit_target.check_profit_target(100_100.0)
    breaker_with_profit_target.assert_options_trading_allowed()  # must not raise -- options keep going

    with pytest.raises(CircuitBreakerTripped):
        breaker_with_profit_target.assert_not_tripped()  # stock-side check still raises


def test_assert_options_trading_allowed_raises_on_real_breach(breaker: CircuitBreaker):
    breaker.check_drawdown(-3_000.0)
    with pytest.raises(CircuitBreakerTripped):
        breaker.assert_options_trading_allowed()
    with pytest.raises(ValueError):
        CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=1.5)
