from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from analyst_layer.agents.intraday_exit_agent import IntradayExitAgent
from analyst_layer.schemas import Action, TradeProposal
from config.settings import Settings
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.state_store import StateStore


def _fake_tool_use_response(tool_name: str, input_payload: dict):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=input_payload)
    return SimpleNamespace(content=[block], usage=None)


# ---- IntradayExitAgent ----

def test_intraday_exit_agent_returns_sell_with_full_quantity():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_decision", {"ticker": "AAPL", "action": "SELL", "quantity": 999, "order_type": "LIMIT", "limit_price": 1.0}
    )
    agent = IntradayExitAgent(client, model="claude-haiku-4-5-20251001")

    proposal = agent.review(
        ticker="AAPL", quantity=10, avg_entry_price=100.0, current_price=105.0, unrealized_plpc=0.05
    )

    assert proposal.action == Action.SELL
    assert proposal.quantity == 10  # always the full held position, not whatever the model said
    assert proposal.limit_price == 105.0


def test_intraday_exit_agent_returns_hold():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_decision", {"ticker": "AAPL", "action": "HOLD", "quantity": 0, "order_type": "LIMIT", "limit_price": 1.0}
    )
    agent = IntradayExitAgent(client, model="claude-haiku-4-5-20251001")

    proposal = agent.review(
        ticker="AAPL", quantity=10, avg_entry_price=100.0, current_price=101.0, unrealized_plpc=0.01
    )

    assert proposal.action == Action.HOLD
    assert proposal.quantity == 0


def test_intraday_exit_agent_forces_hold_if_model_returns_buy():
    """This agent has no BUY path by design — a stray BUY must never reach the broker."""
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_decision", {"ticker": "AAPL", "action": "BUY", "quantity": 5, "order_type": "LIMIT", "limit_price": 105.0}
    )
    agent = IntradayExitAgent(client, model="claude-haiku-4-5-20251001")

    proposal = agent.review(
        ticker="AAPL", quantity=10, avg_entry_price=100.0, current_price=105.0, unrealized_plpc=0.05
    )

    assert proposal.action == Action.HOLD
    assert proposal.quantity == 0


# ---- TradingRuntime._check_intraday_exits ----
# Default exit thresholds used throughout: stop_loss=2%, take_profit=3%, trailing_stop=1.5%

@pytest.fixture
def runtime_with_position(tmp_path: Path) -> TradingRuntime:
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_position_shares.return_value = 10
    broker.submit_order.return_value = {"status": "submitted", "order_id": "abc"}

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    store = StateStore(tmp_path / "intraday_test.sqlite3")
    store.upsert_position(
        "AAPL", quantity=10, avg_entry_price=100.0, last_buy_at=date.today().isoformat(),
        entry_regime="bullish_crossover", high_water_mark=100.0,
    )

    rt = TradingRuntime(
        settings=settings,
        data_client=MagicMock(),
        broker=broker,
        circuit_breaker=breaker,
        state_store=store,
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )
    return rt


def test_check_intraday_exits_holds_within_rule_bands_no_llm_call(runtime_with_position: TradingRuntime):
    """Within all rule thresholds AND no regime reversal -> no LLM call at all."""
    runtime_with_position._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 100.5, "unrealized_plpc": 0.005,
    }
    runtime_with_position._current_regime = MagicMock(return_value="bullish_crossover")  # unchanged from entry
    runtime_with_position._exit_agent.review = MagicMock()

    runtime_with_position._check_intraday_exits(equity=100_000.0)

    runtime_with_position._exit_agent.review.assert_not_called()
    runtime_with_position._broker.submit_order.assert_not_called()


def test_check_intraday_exits_holds_strong_gain_while_advancing(runtime_with_position: TradingRuntime):
    """A position at +10% with no pullback from peak is held — no hard take-profit
    cap anymore. The trailing stop trails 1.5% behind peak, but when the current
    price IS the peak there's zero drawdown so it doesn't fire. Winners run.
    """
    runtime_with_position._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 110.0, "unrealized_plpc": 0.10,
    }
    runtime_with_position._exit_agent.review = MagicMock()

    runtime_with_position._check_intraday_exits(equity=100_000.0)

    runtime_with_position._exit_agent.review.assert_not_called()
    runtime_with_position._broker.submit_order.assert_not_called()  # held, not sold


def test_check_intraday_exits_trailing_stop_fires_on_pullback_from_peak(runtime_with_position: TradingRuntime):
    """Trailing stop fires when price pulls >1.5% back from the recorded peak.
    Peak=$112 stored as high_water_mark, current=$110 → drawdown=-1.79% > threshold.
    """
    runtime_with_position._state_store.upsert_position(
        "AAPL", quantity=10, avg_entry_price=100.0, last_buy_at=date.today().isoformat(),
        entry_regime="bullish_crossover", high_water_mark=112.0,
    )
    runtime_with_position._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 110.0, "unrealized_plpc": 0.10,
    }
    runtime_with_position._exit_agent.review = MagicMock()

    runtime_with_position._check_intraday_exits(equity=100_000.0)

    runtime_with_position._exit_agent.review.assert_not_called()
    runtime_with_position._broker.submit_order.assert_called_once()  # trailing stop fired


def test_check_intraday_exits_escalates_to_llm_on_regime_reversal(runtime_with_position: TradingRuntime):
    """Within all rule thresholds (so rules say hold) but the regime flipped
    since entry -> escalate to the LLM exit-review agent exactly once.
    """
    runtime_with_position._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 100.5, "unrealized_plpc": 0.005,
    }
    runtime_with_position._current_regime = MagicMock(return_value="bearish_crossover")  # reversed from entry
    runtime_with_position._exit_agent.review = MagicMock(
        return_value=TradeProposal(ticker="AAPL", action=Action.SELL, quantity=10, limit_price=100.5)
    )

    runtime_with_position._check_intraday_exits(equity=100_000.0)

    runtime_with_position._exit_agent.review.assert_called_once()
    runtime_with_position._broker.submit_order.assert_called_once()


def test_check_intraday_exits_escalation_is_rate_limited_to_once_per_day(runtime_with_position: TradingRuntime):
    runtime_with_position._broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 100.5, "unrealized_plpc": 0.005,
    }
    runtime_with_position._current_regime = MagicMock(return_value="bearish_crossover")
    runtime_with_position._exit_agent.review = MagicMock(
        return_value=TradeProposal(ticker="AAPL", action=Action.HOLD, quantity=0, limit_price=100.5)
    )

    runtime_with_position._check_intraday_exits(equity=100_000.0)
    runtime_with_position._check_intraday_exits(equity=100_000.0)

    runtime_with_position._exit_agent.review.assert_called_once()  # not twice, despite two ticks


def test_check_intraday_exits_no_escalation_without_entry_regime(tmp_path: Path):
    """A position with no entry_regime on file (e.g. opened before this
    column existed) must never escalate — there's nothing to compare
    the current regime against.
    """
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_position_shares.return_value = 10
    broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 100.5, "unrealized_plpc": 0.005,
    }

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    store = StateStore(tmp_path / "no_regime_test.sqlite3")
    store.upsert_position("AAPL", quantity=10, avg_entry_price=100.0)  # no entry_regime passed

    rt = TradingRuntime(
        settings=settings, data_client=MagicMock(), broker=broker, circuit_breaker=breaker,
        state_store=store, anthropic_client=MagicMock(), watchlist=["AAPL"],
    )
    rt._current_regime = MagicMock(return_value="bearish_crossover")
    rt._exit_agent.review = MagicMock()

    rt._check_intraday_exits(equity=100_000.0)

    rt._exit_agent.review.assert_not_called()


# ---- Per-strategy exit routing (momentum vs thesis) ----

def test_exit_params_for_momentum_uses_trailing_stop_not_hard_cap(runtime_with_position: TradingRuntime):
    params = runtime_with_position._exit_params_for("momentum")
    assert params["take_profit_pct"] is None  # no hard cap — trailing stop rides winners
    assert params["stop_loss_pct"] == runtime_with_position._settings.exit_stop_loss_pct
    assert params["trailing_stop_activation_pct"] == runtime_with_position._settings.exit_trailing_stop_activation_pct


def test_exit_params_for_thesis_uses_wide_bracket_no_take_profit(runtime_with_position: TradingRuntime):
    params = runtime_with_position._exit_params_for("thesis")
    assert params["take_profit_pct"] is None
    assert params["stop_loss_pct"] == runtime_with_position._settings.thesis_stop_loss_pct
    assert params["trailing_stop_activation_pct"] == runtime_with_position._settings.thesis_trailing_stop_activation_pct


def test_thesis_position_survives_a_drop_that_would_close_a_momentum_position(tmp_path: Path):
    """Same -10% move: a momentum position (2% stop) gets sold; a thesis
    position (18% stop) does not. Proves the strategy tag actually routes
    to different exit thresholds end to end, not just in isolated unit tests.
    """
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0
    broker.get_position_shares.return_value = 10
    broker.get_position_detail.return_value = {
        "qty": 10.0, "avg_entry_price": 100.0, "current_price": 90.0, "unrealized_plpc": -0.10,
    }
    broker.submit_order.return_value = {"status": "submitted", "order_id": "abc"}

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    store = StateStore(tmp_path / "thesis_exit_test.sqlite3")
    store.upsert_position(
        "RDW", quantity=10, avg_entry_price=100.0, last_buy_at=date.today().isoformat(),
        high_water_mark=100.0, strategy="thesis",
    )

    rt = TradingRuntime(
        settings=settings, data_client=MagicMock(), broker=broker, circuit_breaker=breaker,
        state_store=store, anthropic_client=MagicMock(), watchlist=["RDW"],
    )
    rt._current_regime = MagicMock(return_value="neutral")

    rt._check_intraday_exits(equity=100_000.0)

    broker.submit_order.assert_not_called()  # -10% is within the thesis track's 18% stop
