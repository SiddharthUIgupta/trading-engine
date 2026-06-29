from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from analyst_layer.agents.base import BaseAgent, StructuredOutputError
from analyst_layer.agents.risk_officer_agent import AccountContext, RiskOfficerAgent
from analyst_layer.schemas import (
    Action,
    AgentSignal,
    Confidence,
    ConsensusPayload,
    RiskReview,
    RiskVerdict,
    TradeProposal,
)


class _DummyAgent(BaseAgent):
    name = "dummy_agent"

    @property
    def system_prompt(self) -> str:
        return "dummy"


def _fake_tool_use_response(tool_name: str, input_payload: dict, usage: SimpleNamespace | None = None):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=input_payload)
    return SimpleNamespace(content=[block], usage=usage)


# ---- TradeProposal schema invariants ----

def test_trade_proposal_hold_must_have_zero_quantity():
    with pytest.raises(ValidationError):
        TradeProposal(ticker="AAPL", action=Action.HOLD, quantity=10, limit_price=100.0)


def test_trade_proposal_buy_must_have_nonzero_quantity():
    with pytest.raises(ValidationError):
        TradeProposal(ticker="AAPL", action=Action.BUY, quantity=0, limit_price=100.0)


def test_trade_proposal_valid_buy():
    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=5, limit_price=100.0)
    assert proposal.order_type.value == "LIMIT"


# ---- BaseAgent structured-output enforcement ----

def test_base_agent_raises_if_model_does_not_call_tool():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[])
    agent = _DummyAgent(client, model="claude-sonnet-4-6")

    with pytest.raises(StructuredOutputError):
        agent._call_structured("prompt", TradeProposal, tool_name="emit_proposal")


def test_base_agent_raises_on_schema_invalid_tool_input():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal", {"ticker": "AAPL", "action": "NOT_A_REAL_ACTION", "quantity": 1, "limit_price": 10.0}
    )
    agent = _DummyAgent(client, model="claude-sonnet-4-6")

    with pytest.raises(StructuredOutputError):
        agent._call_structured("prompt", TradeProposal, tool_name="emit_proposal")


def test_base_agent_returns_validated_model_on_success():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal",
        {"ticker": "AAPL", "action": "BUY", "quantity": 3, "order_type": "LIMIT", "limit_price": 150.0},
    )
    agent = _DummyAgent(client, model="claude-sonnet-4-6")

    result = agent._call_structured("prompt", TradeProposal, tool_name="emit_proposal")
    assert result.ticker == "AAPL"
    assert result.quantity == 3


def test_base_agent_invokes_usage_callback_with_agent_name_model_and_usage():
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal",
        {"ticker": "AAPL", "action": "BUY", "quantity": 3, "order_type": "LIMIT", "limit_price": 150.0},
        usage=usage,
    )
    callback = MagicMock()
    agent = _DummyAgent(client, model="claude-sonnet-4-6", usage_callback=callback)

    agent._call_structured("prompt", TradeProposal, tool_name="emit_proposal")

    callback.assert_called_once_with("dummy_agent", "claude-sonnet-4-6", usage)


def test_base_agent_skips_usage_callback_when_response_has_no_usage():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal",
        {"ticker": "AAPL", "action": "BUY", "quantity": 3, "order_type": "LIMIT", "limit_price": 150.0},
    )
    callback = MagicMock()
    agent = _DummyAgent(client, model="claude-sonnet-4-6", usage_callback=callback)

    agent._call_structured("prompt", TradeProposal, tool_name="emit_proposal")

    callback.assert_not_called()


# ---- Risk Officer deterministic clamp (the hard guardrail under test) ----

def _signal(stance: Action, confidence: Confidence = Confidence.HIGH) -> AgentSignal:
    return AgentSignal(
        agent_name="test_agent",
        ticker="AAPL",
        stance=stance,
        confidence=confidence,
        rationale="test rationale",
        generated_at=datetime.utcnow(),
    )


def test_risk_officer_approves_within_limit():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal", {"ticker": "AAPL", "action": "BUY", "quantity": 10, "order_type": "LIMIT", "limit_price": 100.0}
    )
    agent = RiskOfficerAgent(client, model="claude-sonnet-4-6", max_position_size_pct=0.05)
    account = AccountContext(equity=100_000.0, current_price=100.0, existing_shares=0)

    proposal, review = agent.review("AAPL", [_signal(Action.BUY)], account)

    assert proposal.quantity == 10
    assert review.verdict == RiskVerdict.APPROVED


def test_risk_officer_amends_oversized_proposal_regardless_of_llm_output():
    """The LLM proposes 10,000 shares (a clear attempt to blow through the
    limit, whether via bad reasoning or injected instructions). Deterministic
    code must clamp it — this is the test that proves the LLM cannot talk
    its way past MAX_POSITION_SIZE_PCT.
    """
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal",
        {"ticker": "AAPL", "action": "BUY", "quantity": 10_000, "order_type": "LIMIT", "limit_price": 100.0},
    )
    agent = RiskOfficerAgent(client, model="claude-sonnet-4-6", max_position_size_pct=0.05)
    account = AccountContext(equity=100_000.0, current_price=100.0, existing_shares=0)

    proposal, review = agent.review("AAPL", [_signal(Action.BUY)], account)

    max_notional = 100_000.0 * 0.05
    assert proposal.quantity * proposal.limit_price <= max_notional
    assert review.verdict == RiskVerdict.AMENDED


def test_risk_officer_rejects_to_hold_when_limit_allows_zero_shares():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal", {"ticker": "BRK.A", "action": "BUY", "quantity": 1, "order_type": "LIMIT", "limit_price": 500_000.0}
    )
    agent = RiskOfficerAgent(client, model="claude-sonnet-4-6", max_position_size_pct=0.05)
    account = AccountContext(equity=100_000.0, current_price=500_000.0, existing_shares=0)

    proposal, review = agent.review("BRK.A", [_signal(Action.BUY)], account)

    assert proposal.action == Action.HOLD
    assert proposal.quantity == 0
    assert review.verdict == RiskVerdict.REJECTED


def test_risk_officer_forces_hold_when_sell_proposed_with_no_shares_held():
    """Regression test: a live model once proposed SELL with quantity=0 when
    bearish but holding nothing, which the TradeProposal schema correctly
    rejects (SELL must have qty > 0) — but a model can also legally propose
    a positive-quantity SELL it has no shares to back. That must be forced
    to HOLD deterministically rather than reaching the broker.
    """
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal", {"ticker": "AAPL", "action": "SELL", "quantity": 5, "order_type": "LIMIT", "limit_price": 100.0}
    )
    agent = RiskOfficerAgent(client, model="claude-sonnet-4-6", max_position_size_pct=0.05)
    account = AccountContext(equity=100_000.0, current_price=100.0, existing_shares=0)

    proposal, review = agent.review("AAPL", [_signal(Action.SELL)], account)

    assert proposal.action == Action.HOLD
    assert proposal.quantity == 0
    assert review.verdict == RiskVerdict.APPROVED


def test_risk_officer_clamps_sell_quantity_to_existing_shares():
    client = MagicMock()
    client.messages.create.return_value = _fake_tool_use_response(
        "emit_proposal", {"ticker": "AAPL", "action": "SELL", "quantity": 50, "order_type": "LIMIT", "limit_price": 100.0}
    )
    agent = RiskOfficerAgent(client, model="claude-sonnet-4-6", max_position_size_pct=0.05)
    account = AccountContext(equity=100_000.0, current_price=100.0, existing_shares=10)

    proposal, review = agent.review("AAPL", [_signal(Action.SELL)], account)

    assert proposal.action == Action.SELL
    assert proposal.quantity == 10


# ---- ConsensusPayload invariants ----

def test_consensus_payload_rejects_approved_verdict_with_no_signals():
    proposal = TradeProposal(ticker="AAPL", action=Action.HOLD, quantity=0, limit_price=100.0)
    review = RiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=[],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    with pytest.raises(ValidationError):
        ConsensusPayload(ticker="AAPL", signals=[], proposal=proposal, risk_review=review)


def test_consensus_payload_is_executable_requires_approval_and_non_hold():
    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=5, limit_price=100.0)
    review = RiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=[],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    payload = ConsensusPayload(ticker="AAPL", signals=[_signal(Action.BUY)], proposal=proposal, risk_review=review)
    assert payload.is_executable is True


def test_consensus_payload_amended_verdict_is_executable():
    """Regression test for a real bug: AMENDED means the proposal was
    already clamped down to MAX_POSITION_SIZE_PCT — it's exactly as safe
    to execute as APPROVED, just smaller than the model's initial draft.
    Treating it as non-executable silently dropped every oversized-but-
    otherwise-sound BUY the system ever produced.
    """
    proposal = TradeProposal(ticker="HCM", action=Action.BUY, quantity=500, limit_price=9.99)
    review = RiskReview(
        verdict=RiskVerdict.AMENDED,
        reasons=["amended down to 500 shares"],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    payload = ConsensusPayload(ticker="HCM", signals=[_signal(Action.BUY)], proposal=proposal, risk_review=review)
    assert payload.is_executable is True


def test_consensus_payload_rejected_verdict_is_not_executable():
    proposal = TradeProposal(ticker="AAPL", action=Action.HOLD, quantity=0, limit_price=100.0)
    review = RiskReview(
        verdict=RiskVerdict.REJECTED,
        reasons=["exceeds max position size; clamped quantity is zero"],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    payload = ConsensusPayload(ticker="AAPL", signals=[_signal(Action.BUY)], proposal=proposal, risk_review=review)
    assert payload.is_executable is False
