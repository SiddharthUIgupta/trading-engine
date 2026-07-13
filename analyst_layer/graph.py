"""LangGraph wiring for the Layer-2 consensus flow.

Fan-out: the three narrow-scope analysts (macro/sentiment, fundamental/SEC,
technical) run as independent branches off START — they don't see each
other's output, by design, so none of them can anchor on another agent's
framing. Fan-in: the Risk Compliance Officer node only runs once all three
have completed, and it is the sole node allowed to produce the
ConsensusPayload that the execution layer is permitted to act on.
"""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from anthropic import Anthropic
from langgraph.graph import END, START, StateGraph

from analyst_layer.agents.fundamental_agent import FundamentalAgent
from analyst_layer.agents.macro_sentiment_agent import MacroSentimentAgent
from analyst_layer.agents.risk_officer_agent import AccountContext, RiskOfficerAgent
from analyst_layer.agents.technical_agent import TechnicalAgent
from analyst_layer.schemas import AgentSignal, ConsensusPayload, RiskReview, TradeProposal
from analyst_layer import vibe_data
from data_layer.models import FilingSummary, FundamentalsSnapshot, PriceSeries, SentimentSnapshot


class ConsensusState(TypedDict):
    ticker: str
    sentiment: SentimentSnapshot
    fundamentals: FundamentalsSnapshot
    filings: list[FilingSummary]
    price_series: PriceSeries
    account: AccountContext
    lessons: str  # formatted lesson block for prompt injection; "" if none retrieved yet
    signals: Annotated[list[AgentSignal], operator.add]
    proposal: TradeProposal | None
    risk_review: RiskReview | None
    errors: Annotated[list[str], operator.add]


def build_consensus_graph(
    client: Anthropic,
    model: str,
    max_position_size_pct: float,
    subagent_model: str | None = None,
    usage_callback=None,
):
    subagent_model = subagent_model or model
    macro_agent = MacroSentimentAgent(client, subagent_model, usage_callback=usage_callback)
    fundamental_agent = FundamentalAgent(client, subagent_model, usage_callback=usage_callback)
    technical_agent = TechnicalAgent(client, subagent_model, usage_callback=usage_callback)
    risk_agent = RiskOfficerAgent(
        client, model, max_position_size_pct=max_position_size_pct, usage_callback=usage_callback
    )

    def macro_node(state: ConsensusState) -> dict:
        try:
            signal = macro_agent.analyze(state["ticker"], state["sentiment"], lessons=state.get("lessons", ""))
            return {"signals": [signal]}
        except Exception as exc:  # noqa: BLE001 — isolate one agent's failure from the others
            return {"errors": [f"macro_sentiment_agent failed: {exc}"]}

    def fundamental_node(state: ConsensusState) -> dict:
        try:
            sec_context = vibe_data.fetch_sec_context(state["ticker"])
            signal = fundamental_agent.analyze(
                state["ticker"], state["fundamentals"], state["filings"],
                lessons=state.get("lessons", ""), sec_context=sec_context,
            )
            return {"signals": [signal]}
        except Exception as exc:  # noqa: BLE001
            return {"errors": [f"fundamental_sec_agent failed: {exc}"]}

    def technical_node(state: ConsensusState) -> dict:
        try:
            extra_context = vibe_data.compute_technical_signals(state["price_series"])
            signal = technical_agent.analyze(
                state["ticker"], state["price_series"],
                lessons=state.get("lessons", ""), extra_context=extra_context,
            )
            return {"signals": [signal]}
        except Exception as exc:  # noqa: BLE001
            return {"errors": [f"technical_analysis_agent failed: {exc}"]}

    def risk_node(state: ConsensusState) -> dict:
        if not state["signals"]:
            now = datetime.utcnow()
            fallback_proposal = TradeProposal(ticker=state["ticker"], action="HOLD", quantity=0, limit_price=state["account"]["current_price"])
            fallback_review = RiskReview(
                verdict="rejected",
                reasons=["no sub-agent signals available; forcing HOLD"] + state["errors"],
                max_position_size_pct_checked=max_position_size_pct,
                max_daily_drawdown_pct_checked=state["account"].get("max_daily_drawdown_pct", 0.0),
                reviewed_at=now,
            )
            return {"proposal": fallback_proposal, "risk_review": fallback_review}

        try:
            proposal, review = risk_agent.review(state["ticker"], state["signals"], state["account"], lessons=state.get("lessons", ""))
            if state["errors"]:
                # Partial failure: enough agents succeeded to reach a real
                # verdict, but the ones that failed must stay visible — a
                # degraded 2-of-3 consensus is not the same as a healthy one.
                review = review.model_copy(update={"reasons": [*review.reasons, *state["errors"]]})
            return {"proposal": proposal, "risk_review": review}
        except Exception as exc:  # noqa: BLE001 — a malformed/failed risk officer call must never crash the run
            now = datetime.utcnow()
            fallback_proposal = TradeProposal(
                ticker=state["ticker"], action="HOLD", quantity=0, limit_price=state["account"]["current_price"]
            )
            fallback_review = RiskReview(
                verdict="rejected",
                reasons=[f"risk officer agent failed; forcing HOLD: {exc}"],
                max_position_size_pct_checked=max_position_size_pct,
                max_daily_drawdown_pct_checked=state["account"].get("max_daily_drawdown_pct", 0.0),
                reviewed_at=now,
            )
            return {"proposal": fallback_proposal, "risk_review": fallback_review}

    graph = StateGraph(ConsensusState)
    graph.add_node("macro_sentiment", macro_node)
    graph.add_node("fundamental_sec", fundamental_node)
    graph.add_node("technical_analysis", technical_node)
    graph.add_node("risk_compliance", risk_node)

    graph.add_edge(START, "macro_sentiment")
    graph.add_edge(START, "fundamental_sec")
    graph.add_edge(START, "technical_analysis")
    graph.add_edge("macro_sentiment", "risk_compliance")
    graph.add_edge("fundamental_sec", "risk_compliance")
    graph.add_edge("technical_analysis", "risk_compliance")
    graph.add_edge("risk_compliance", END)

    return graph.compile()


def run_consensus(
    client: Anthropic,
    model: str,
    max_position_size_pct: float,
    ticker: str,
    sentiment: SentimentSnapshot,
    fundamentals: FundamentalsSnapshot,
    filings: list[FilingSummary],
    price_series: PriceSeries,
    account: AccountContext,
    subagent_model: str | None = None,
    usage_callback=None,
    lessons: str = "",
) -> ConsensusPayload:
    app = build_consensus_graph(
        client, model, max_position_size_pct, subagent_model=subagent_model, usage_callback=usage_callback
    )
    initial_state: ConsensusState = {
        "ticker": ticker,
        "sentiment": sentiment,
        "fundamentals": fundamentals,
        "filings": filings,
        "price_series": price_series,
        "account": account,
        "lessons": lessons,
        "signals": [],
        "proposal": None,
        "risk_review": None,
        "errors": [],
    }
    final_state = app.invoke(initial_state)
    return ConsensusPayload(
        ticker=ticker,
        signals=final_state["signals"],
        proposal=final_state["proposal"],
        risk_review=final_state["risk_review"],
    )
