"""Integration test for the LangGraph consensus wiring itself (graph.py).

Mocks the Anthropic client at the lowest level (messages.create) so the
real StateGraph, fan-out/fan-in edges, and the risk officer gate all run
for real — only the LLM call is faked. This is the test that would catch
a broken edge or a state-merging bug that per-agent unit tests can't see.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from analyst_layer.agents.risk_officer_agent import AccountContext
from analyst_layer.graph import run_consensus
from analyst_layer.schemas import RiskVerdict


def _tool_response(tool_name: str, input_payload: dict):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=input_payload)
    return SimpleNamespace(content=[block])


def _signal_payload(stance: str) -> dict:
    return {
        "agent_name": "placeholder",
        "ticker": "AAPL",
        "stance": stance,
        "confidence": "high",
        "rationale": "mocked rationale for test",
        "generated_at": datetime.utcnow().isoformat(),
        "supporting_data_refs": [],
    }


def test_full_consensus_graph_runs_and_gates_through_risk_officer(
    sample_sentiment, sample_fundamentals, sample_filings, sample_price_series
):
    client = MagicMock()

    def fake_create(**kwargs):
        tool_name = kwargs["tool_choice"]["name"]
        if tool_name == "emit_signal":
            return _tool_response("emit_signal", _signal_payload("BUY"))
        if tool_name == "emit_proposal":
            return _tool_response(
                "emit_proposal",
                {"ticker": "AAPL", "action": "BUY", "quantity": 5, "order_type": "LIMIT", "limit_price": 191.0},
            )
        raise AssertionError(f"unexpected tool_name {tool_name}")

    client.messages.create.side_effect = fake_create

    account = AccountContext(equity=100_000.0, current_price=191.0, existing_shares=0, max_daily_drawdown_pct=0.02)

    payload = run_consensus(
        client=client,
        model="claude-sonnet-4-6",
        max_position_size_pct=0.05,
        ticker="AAPL",
        sentiment=sample_sentiment,
        fundamentals=sample_fundamentals,
        filings=sample_filings,
        price_series=sample_price_series,
        account=account,
    )

    assert len(payload.signals) == 3
    assert {s.agent_name for s in payload.signals} == {
        "macro_sentiment_agent",
        "fundamental_sec_agent",
        "technical_analysis_agent",
    }
    assert payload.risk_review.verdict == RiskVerdict.APPROVED
    assert payload.is_executable is True


def test_full_consensus_graph_clamps_oversized_proposal_end_to_end(
    sample_sentiment, sample_fundamentals, sample_filings, sample_price_series
):
    client = MagicMock()

    def fake_create(**kwargs):
        tool_name = kwargs["tool_choice"]["name"]
        if tool_name == "emit_signal":
            return _tool_response("emit_signal", _signal_payload("BUY"))
        if tool_name == "emit_proposal":
            # 10,000 shares is a deliberate attempt to blow past 5% of 100k equity at $191/share.
            return _tool_response(
                "emit_proposal",
                {"ticker": "AAPL", "action": "BUY", "quantity": 10_000, "order_type": "LIMIT", "limit_price": 191.0},
            )
        raise AssertionError(f"unexpected tool_name {tool_name}")

    client.messages.create.side_effect = fake_create

    account = AccountContext(equity=100_000.0, current_price=191.0, existing_shares=0, max_daily_drawdown_pct=0.02)

    payload = run_consensus(
        client=client,
        model="claude-sonnet-4-6",
        max_position_size_pct=0.05,
        ticker="AAPL",
        sentiment=sample_sentiment,
        fundamentals=sample_fundamentals,
        filings=sample_filings,
        price_series=sample_price_series,
        account=account,
    )

    max_notional = 100_000.0 * 0.05
    assert payload.proposal.quantity * payload.proposal.limit_price <= max_notional
    assert payload.risk_review.verdict == RiskVerdict.AMENDED


def test_partial_agent_failure_surfaces_in_final_reasons(
    sample_sentiment, sample_fundamentals, sample_filings, sample_price_series
):
    """Regression: when 1 of 3 sub-agents fails but the other 2 succeed (so
    risk_node still reaches a real verdict via risk_agent.review()), the
    failure reason used to be silently dropped — only the all-3-failed path
    included state["errors"] in the final payload. A degraded 2-of-3
    consensus must still show that an agent failed.
    """
    client = MagicMock()
    call_count = {"n": 0}

    def fake_create(**kwargs):
        tool_name = kwargs["tool_choice"]["name"]
        if tool_name == "emit_signal":
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated agent failure")
            return _tool_response("emit_signal", _signal_payload("BUY"))
        if tool_name == "emit_proposal":
            return _tool_response(
                "emit_proposal",
                {"ticker": "AAPL", "action": "BUY", "quantity": 5, "order_type": "LIMIT", "limit_price": 191.0},
            )
        raise AssertionError(f"unexpected tool_name {tool_name}")

    client.messages.create.side_effect = fake_create

    account = AccountContext(equity=100_000.0, current_price=191.0, existing_shares=0, max_daily_drawdown_pct=0.02)

    payload = run_consensus(
        client=client,
        model="claude-sonnet-4-6",
        max_position_size_pct=0.05,
        ticker="AAPL",
        sentiment=sample_sentiment,
        fundamentals=sample_fundamentals,
        filings=sample_filings,
        price_series=sample_price_series,
        account=account,
    )

    assert len(payload.signals) == 2  # only 2 of 3 agents succeeded
    assert any("failed: simulated agent failure" in r for r in payload.risk_review.reasons)
