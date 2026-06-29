"""General investment analyst — synthesizes the three narrow signals into
one overall qualitative view. Deliberately separate from RiskOfficerAgent:
this agent never sees equity, position limits, or anything else tied to
sizing a trade for the autonomous harness. Its only job is the same
judgment call a human analyst makes reading three independent write-ups —
weighing how strong and how fresh each signal actually is, not just
counting BUY/SELL/HOLD votes. Used by the dashboard's on-demand ticker
analysis, not by the scanners.
"""
from __future__ import annotations

from datetime import datetime

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import AgentSignal


class GeneralAnalystAgent(BaseAgent):
    name = "general_analyst_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a general investment analyst. You are given three independent "
            "narrow reads on a stock — fundamental, sentiment, and technical — each "
            "from a specialist who only saw their own slice of the data. Your job is "
            "to form your OWN overall opinion by weighing the strength, freshness, and "
            "quality of each signal against the others — not by counting votes. A deep "
            "valuation discount and a fresh, still-unconfirmed technical crossover are "
            "not equally weighted just because they're both one signal each; use "
            "judgment about which evidence should dominate. This is a general "
            "investment opinion, not a sized trade order — you are not given account "
            "equity, a position limit, or any daily P&L target, and should not act as "
            "though you have one. Respond by calling the emit_signal tool with a stance "
            "of BUY, SELL, or HOLD, a confidence level, and a rationale that explicitly "
            "addresses how you weighed the disagreement between the three reads. Keep "
            "the rationale under 1200 characters — be decisive and specific about which "
            "signal you weighted most heavily and why, not exhaustive."
        )

    def synthesize(self, ticker: str, signals: list[AgentSignal]) -> AgentSignal:
        signals_text = "\n".join(
            f"  - {s.agent_name}: {s.stance.value} (confidence={s.confidence.value}) — {s.rationale}"
            for s in signals
        )
        prompt = (
            f"Ticker: {ticker}\n\n"
            f"Independent specialist reads:\n{signals_text}\n\n"
            "Form your overall opinion."
        )
        signal = self._call_structured(prompt, AgentSignal, tool_name="emit_signal")
        return signal.model_copy(update={"agent_name": self.name, "generated_at": datetime.utcnow()})
