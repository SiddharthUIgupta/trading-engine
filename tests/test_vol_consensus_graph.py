"""Integration test for the vol consensus LangGraph wiring (vol_graph.py).

Mirrors tests/test_consensus_graph.py's approach for the equity consensus
graph: mocks the Anthropic client at the lowest level so the real StateGraph
fan-out/fan-in and the Greeks Risk Officer gate all run for real.
"""
from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from analyst_layer.agents.vol_regime_agent import VixContext
from analyst_layer.schemas import RiskVerdict
from analyst_layer.vol_graph import run_vol_consensus
from tests.test_vol_scan import _aapl_chain, _aapl_vol_snapshot


def _tool_response(tool_name: str, input_payload: dict):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=input_payload)
    return SimpleNamespace(content=[block])


def _vol_signal_payload(agent_name: str) -> dict:
    return {
        "agent_name": agent_name,
        "ticker": "AAPL",
        "iv_environment": "elevated",
        "recommended_structure": "iron_condor",
        "confidence": "high",
        "rationale": "mocked rationale for test",
        "generated_at": datetime.utcnow().isoformat(),
        "flags": [],
    }


def _greeks_review_payload() -> dict:
    return {
        "verdict": "approved",
        "reasons": ["within portfolio limits"],
        "portfolio_delta_after": 0.01,
        "portfolio_vega_after": -10.0,
        "portfolio_theta_after": 1.0,
        "position_max_loss": None,
    }


_TOOL_TO_AGENT = {
    "emit_iv_surface_signal": "iv_surface_agent",
    "emit_event_risk_signal": "event_risk_agent",
    "emit_vol_regime_signal": "vol_regime_agent",
}


def _run(client) -> object:
    portfolio = {"net_delta": 0.0, "net_vega": 0.0, "net_theta": 0.0, "portfolio_value": 100_000.0, "num_open_positions": 0}
    vix_context = VixContext(vix_current=18.0, vix_1w_ago=17.5, vix_1m_ago=19.0, vix3m_current=19.5)
    return run_vol_consensus(
        client=client,
        model="claude-sonnet-4-6",
        ticker="AAPL",
        vol_snapshot=_aapl_vol_snapshot(),
        option_chain=_aapl_chain(),
        vix_context=vix_context,
        portfolio=portfolio,
        max_position_size_pct=0.05,
        allow_uncovered=False,  # exercise the iron-condor downgrade path, matching _aapl_chain's legs
    )


def test_full_vol_consensus_graph_runs_and_gates_through_greeks_officer():
    client = MagicMock()

    def fake_create(**kwargs):
        tool_name = kwargs["tool_choice"]["name"]
        if tool_name in _TOOL_TO_AGENT:
            payload = _vol_signal_payload(_TOOL_TO_AGENT[tool_name])
            if tool_name == "emit_vol_regime_signal":
                payload["vol_regime"] = "stable"
            return _tool_response(tool_name, payload)
        if tool_name == "emit_greeks_risk_review":
            return _tool_response(tool_name, _greeks_review_payload())
        raise AssertionError(f"unexpected tool_name {tool_name}")

    client.messages.create.side_effect = fake_create

    payload = _run(client)

    assert len(payload.vol_signals) == 3
    assert payload.risk_review.verdict == RiskVerdict.APPROVED


def test_partial_vol_agent_failure_surfaces_in_final_reasons():
    """Regression: when 1 of 3 vol sub-agents fails but the other 2 succeed
    (enough for greeks_risk_node to reach a real verdict via risk_officer.
    review()), the failure reason used to be silently dropped — only the
    all-failed / NO_TRADE-veto paths included state["errors"].
    """
    client = MagicMock()
    failed_once = {"done": False}

    def fake_create(**kwargs):
        tool_name = kwargs["tool_choice"]["name"]
        if tool_name == "emit_iv_surface_signal" and not failed_once["done"]:
            failed_once["done"] = True
            raise RuntimeError("simulated vol agent failure")
        if tool_name in _TOOL_TO_AGENT:
            payload = _vol_signal_payload(_TOOL_TO_AGENT[tool_name])
            if tool_name == "emit_vol_regime_signal":
                payload["vol_regime"] = "stable"
            return _tool_response(tool_name, payload)
        if tool_name == "emit_greeks_risk_review":
            return _tool_response(tool_name, _greeks_review_payload())
        raise AssertionError(f"unexpected tool_name {tool_name}")

    client.messages.create.side_effect = fake_create

    payload = _run(client)

    assert len(payload.vol_signals) == 2  # only 2 of 3 agents succeeded
    assert any("failed: simulated vol agent failure" in r for r in payload.risk_review.reasons)
