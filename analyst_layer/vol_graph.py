"""LangGraph wiring for the volatility-based consensus flow.

Fan-out: three vol agents (IV surface, event risk, vol regime) run as
independent parallel branches — each reads only its own slice of the
data so none can anchor on another's framing. Fan-in: the Greeks Risk
Officer only runs after all three complete, and is the sole node
permitted to produce the VolConsensusPayload that the execution layer
acts on.

Flow:
  START → [iv_surface, event_risk, vol_regime] → greeks_risk_officer → END

The structure builder (options_structurer.build_structure) runs inside
the Greeks Risk Officer node — it builds the OptionsProposal from the
agents' structure recommendation before submitting it for Greeks review.
This keeps the proposal and its review in the same node so the officer
always sees the actual strikes, not an abstract structure type.
"""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from anthropic import Anthropic
from langgraph.graph import END, START, StateGraph

from analyst_layer.agents.event_risk_agent import EventRiskAgent
from analyst_layer.agents.greeks_risk_officer import GreeksRiskOfficer, PortfolioGreeks
from analyst_layer.agents.iv_surface_agent import IVSurfaceAgent
from analyst_layer.agents.vol_regime_agent import VixContext, VolRegimeAgent
from analyst_layer.options_structurer import build_structure
from analyst_layer.schemas import (
    GreeksRiskReview,
    OptionsProposal,
    RiskVerdict,
    StructureType,
    VolConsensusPayload,
    VolSignal,
)
from data_layer.models import OptionContract, VolatilitySnapshot


class VolConsensusState(TypedDict):
    ticker: str
    vol_snapshot: VolatilitySnapshot
    option_chain: list[OptionContract]
    vix_context: VixContext
    portfolio: PortfolioGreeks
    vol_signals: Annotated[list[VolSignal], operator.add]
    proposal: OptionsProposal | None
    risk_review: GreeksRiskReview | None
    errors: Annotated[list[str], operator.add]


def build_vol_consensus_graph(
    client: Anthropic,
    model: str,
    max_position_size_pct: float = 0.05,
    allow_uncovered: bool = True,
    subagent_model: str | None = None,
    usage_callback=None,
):
    subagent_model = subagent_model or model

    # IV surface agent uses the main (Sonnet) model — options structure selection
    # (iron condor vs strangle vs spread) requires genuine vol market reasoning,
    # unlike the simpler classification the other sub-agents do.
    iv_agent = IVSurfaceAgent(client, model, usage_callback=usage_callback)
    event_agent = EventRiskAgent(client, subagent_model, usage_callback=usage_callback)
    regime_agent = VolRegimeAgent(client, subagent_model, usage_callback=usage_callback)
    risk_officer = GreeksRiskOfficer(
        client, model,
        max_position_size_pct=max_position_size_pct,
        usage_callback=usage_callback,
    )

    def iv_surface_node(state: VolConsensusState) -> dict:
        try:
            signal = iv_agent.analyze(state["ticker"], state["vol_snapshot"])
            return {"vol_signals": [signal]}
        except Exception as exc:  # noqa: BLE001
            return {"errors": [f"iv_surface_agent failed: {exc}"]}

    def event_risk_node(state: VolConsensusState) -> dict:
        try:
            # Use the first available signal's recommended structure as context,
            # or default to SHORT_STRANGLE if no signals yet (parallel execution)
            proposed = StructureType.SHORT_STRANGLE
            signal = event_agent.analyze(state["ticker"], state["vol_snapshot"], proposed)
            return {"vol_signals": [signal]}
        except Exception as exc:  # noqa: BLE001
            return {"errors": [f"event_risk_agent failed: {exc}"]}

    def vol_regime_node(state: VolConsensusState) -> dict:
        try:
            signal = regime_agent.analyze(state["ticker"], state["vix_context"])
            return {"vol_signals": [signal]}
        except Exception as exc:  # noqa: BLE001
            return {"errors": [f"vol_regime_agent failed: {exc}"]}

    def greeks_risk_node(state: VolConsensusState) -> dict:
        now = datetime.now()
        signals = state["vol_signals"]

        def _reject(reason: str) -> dict:
            proposal = OptionsProposal(
                ticker=state["ticker"],
                structure=StructureType.NO_TRADE,
                expiration=datetime.now().date(),
                dte=0,
                quantity=0,
            )
            review = GreeksRiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=[reason] + state["errors"],
                portfolio_delta_after=state["portfolio"]["net_delta"],
                portfolio_vega_after=state["portfolio"]["net_vega"],
                portfolio_theta_after=state["portfolio"]["net_theta"],
                position_max_loss=None,
                reviewed_at=now,
            )
            return {"proposal": proposal, "risk_review": review}

        if not signals:
            return _reject("no vol agent signals available — cannot proceed")

        # Consensus: take the most conservative structure among agents
        # NO_TRADE vetoes; otherwise use the most restricted non-NO_TRADE recommendation
        _conservatism_rank = {
            StructureType.NO_TRADE: 0,
            StructureType.IRON_CONDOR: 1,
            StructureType.SHORT_PUT_SPREAD: 2,
            StructureType.SHORT_CALL_SPREAD: 2,
            StructureType.SHORT_PUT: 3,
            StructureType.SHORT_CALL: 3,
            StructureType.CALENDAR: 4,
            StructureType.SHORT_STRANGLE: 5,
        }
        all_no_trade = all(s.recommended_structure == StructureType.NO_TRADE for s in signals)
        if all_no_trade:
            return _reject("all three vol agents recommend NO_TRADE")

        consensus_structure = min(
            [s.recommended_structure for s in signals],
            key=lambda st: _conservatism_rank.get(st, 0),
        )
        if consensus_structure == StructureType.NO_TRADE:
            return _reject("one or more vol agents vetoed — conservative consensus is NO_TRADE")

        # Downgrade naked short positions when the account/settings don't allow them.
        # A strangle at IVR > 50 is still the correct premium-selling instinct;
        # the iron condor expresses the same thesis with defined risk.
        if not allow_uncovered and consensus_structure == StructureType.SHORT_STRANGLE:
            consensus_structure = StructureType.IRON_CONDOR

        # Build the actual proposal from the options chain
        vol = state["vol_snapshot"]
        structure_result = build_structure(
            ticker=state["ticker"],
            structure_type=consensus_structure,
            chain=state["option_chain"],
            iv_30=vol.iv_30,
            quantity=1,
        )
        if not structure_result.selected or structure_result.proposal is None:
            return _reject(f"structure builder could not build {consensus_structure.value}: {structure_result.reasons}")

        # Greeks risk officer review
        try:
            review = risk_officer.review(
                ticker=state["ticker"],
                vol_signals=signals,
                proposal=structure_result.proposal,
                portfolio=state["portfolio"],
            )
            if state["errors"]:
                # Partial failure: enough agents succeeded to reach a real
                # verdict, but the ones that failed must stay visible — a
                # degraded 2-of-3 consensus is not the same as a healthy one.
                review = review.model_copy(update={"reasons": [*review.reasons, *state["errors"]]})
            return {"proposal": structure_result.proposal, "risk_review": review}
        except Exception as exc:  # noqa: BLE001
            return _reject(f"greeks_risk_officer failed: {exc}")

    graph = StateGraph(VolConsensusState)
    graph.add_node("iv_surface", iv_surface_node)
    graph.add_node("event_risk", event_risk_node)
    graph.add_node("vol_regime", vol_regime_node)
    graph.add_node("greeks_risk_officer", greeks_risk_node)

    graph.add_edge(START, "iv_surface")
    graph.add_edge(START, "event_risk")
    graph.add_edge(START, "vol_regime")
    graph.add_edge("iv_surface", "greeks_risk_officer")
    graph.add_edge("event_risk", "greeks_risk_officer")
    graph.add_edge("vol_regime", "greeks_risk_officer")
    graph.add_edge("greeks_risk_officer", END)

    return graph.compile()


def run_vol_consensus(
    client: Anthropic,
    model: str,
    ticker: str,
    vol_snapshot: VolatilitySnapshot,
    option_chain: list[OptionContract],
    vix_context: VixContext,
    portfolio: PortfolioGreeks,
    max_position_size_pct: float = 0.05,
    allow_uncovered: bool = True,
    subagent_model: str | None = None,
    usage_callback=None,
) -> VolConsensusPayload:
    app = build_vol_consensus_graph(
        client, model,
        max_position_size_pct=max_position_size_pct,
        allow_uncovered=allow_uncovered,
        subagent_model=subagent_model,
        usage_callback=usage_callback,
    )
    initial_state: VolConsensusState = {
        "ticker": ticker,
        "vol_snapshot": vol_snapshot,
        "option_chain": option_chain,
        "vix_context": vix_context,
        "portfolio": portfolio,
        "vol_signals": [],
        "proposal": None,
        "risk_review": None,
        "errors": [],
    }
    final_state = app.invoke(initial_state)
    return VolConsensusPayload(
        ticker=ticker,
        vol_signals=final_state["vol_signals"],
        proposal=final_state["proposal"],
        risk_review=final_state["risk_review"],
    )
