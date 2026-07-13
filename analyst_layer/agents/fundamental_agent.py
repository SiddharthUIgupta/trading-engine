from __future__ import annotations

from datetime import datetime

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import AgentSignal
from data_layer.models import FilingSummary, FundamentalsSnapshot


class FundamentalAgent(BaseAgent):
    """Ingests recent financial statements, filings, and analyst revisions
    via the Data Layer. Narrow scope: fundamentals only.
    """

    name = "fundamental_sec_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Fundamental/SEC analyst on a trading desk. You "
            "evaluate financial statement metrics, recent SEC filings, and "
            "analyst rating revisions ONLY. You do not evaluate sentiment, "
            "technicals, or compliance. Respond by calling the emit_signal "
            "tool with a stance of BUY, SELL, or HOLD grounded only in the "
            "fundamentals data provided."
        )

    def analyze(
        self,
        ticker: str,
        fundamentals: FundamentalsSnapshot,
        filings: list[FilingSummary],
        lessons: str = "",
        sec_context: str = "",
    ) -> AgentSignal:
        revisions_text = "\n".join(
            f"  - {r.firm}: {r.rating} (target {r.target_price})" for r in fundamentals.revisions
        ) or "  (none)"
        filings_text = "\n".join(
            f"  - {f.filing_type.value} filed {f.filed_on.isoformat()}: {f.summary}" for f in filings
        ) or "  (none)"

        prompt = (
            f"{lessons}"
            f"Ticker: {ticker}\n"
            f"As of: {fundamentals.as_of.isoformat()}\n"
            f"EPS: {fundamentals.eps}\n"
            f"Revenue: {fundamentals.revenue}\n"
            f"P/E: {fundamentals.pe_ratio}\n"
            f"Analyst revisions:\n{revisions_text}\n"
            f"Recent filings:\n{filings_text}\n"
            f"{sec_context}"
            "\nBased solely on this fundamentals data, emit your signal."
        )
        signal = self._call_structured(prompt, AgentSignal, tool_name="emit_signal")
        return signal.model_copy(update={"agent_name": self.name, "generated_at": datetime.utcnow()})
