from __future__ import annotations

from datetime import datetime

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import AgentSignal
from data_layer.models import SentimentSnapshot


class MacroSentimentAgent(BaseAgent):
    """Parses news/social sentiment indices. Narrow scope: sentiment and
    macro tone only — no price-technical or fundamental reasoning.
    """

    name = "macro_sentiment_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Macro/Sentiment analyst on a trading desk. You read "
            "news sentiment data and macro tone indicators ONLY. You do not "
            "evaluate technicals, fundamentals, or compliance — those are other "
            "agents' jobs. You must respond by calling the emit_signal tool "
            "with a stance of BUY, SELL, or HOLD, a confidence level, and a "
            "concise rationale grounded only in the sentiment data provided."
        )

    def analyze(self, ticker: str, sentiment: SentimentSnapshot, lessons: str = "") -> AgentSignal:
        prompt = (
            f"{lessons}"
            f"Ticker: {ticker}\n"
            f"Sentiment source: {sentiment.source}\n"
            f"Sentiment score (-1 bearish .. +1 bullish): {sentiment.score:.3f}\n"
            f"Polarity label: {sentiment.polarity.value}\n"
            f"Headline count: {sentiment.headline_count}\n"
            f"As of: {sentiment.as_of.isoformat()}\n\n"
            "Based solely on this sentiment telemetry, emit your signal."
        )
        signal = self._call_structured(prompt, AgentSignal, tool_name="emit_signal")
        return signal.model_copy(update={"agent_name": self.name, "generated_at": datetime.utcnow()})
