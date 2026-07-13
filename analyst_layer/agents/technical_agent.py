from __future__ import annotations

import statistics
from datetime import datetime

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import AgentSignal
from data_layer.models import PriceSeries


def _simple_moving_average(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _realized_volatility(closes: list[float]) -> float | None:
    if len(closes) < 2:
        return None
    returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]
    return statistics.pstdev(returns)


class TechnicalAgent(BaseAgent):
    """Calculates quantitative indicators deterministically in Python
    (moving averages, realized volatility) and hands the *numbers*, not
    raw prices, to the LLM for regime interpretation. Narrow scope:
    price-technical signals only.
    """

    name = "technical_analysis_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Technical Analysis analyst on a trading desk. You "
            "interpret precomputed quantitative indicators (moving averages, "
            "realized volatility, regime classification) ONLY. You do not "
            "evaluate sentiment, fundamentals, or compliance. Respond by "
            "calling the emit_signal tool with a stance grounded only in the "
            "indicator values provided."
        )

    def analyze(self, ticker: str, series: PriceSeries, lessons: str = "", extra_context: str = "") -> AgentSignal:
        closes = [bar.close for bar in series.bars]
        sma_short = _simple_moving_average(closes, window=min(10, len(closes)))
        sma_long = _simple_moving_average(closes, window=min(30, len(closes)))
        volatility = _realized_volatility(closes)
        last_close = closes[-1]

        if sma_short is not None and sma_long is not None:
            regime = "bullish_crossover" if sma_short > sma_long else "bearish_crossover"
        else:
            regime = "insufficient_data"

        prompt = (
            f"{lessons}"
            f"Ticker: {ticker}\n"
            f"Last close: {last_close:.4f}\n"
            f"SMA(short): {sma_short}\n"
            f"SMA(long): {sma_long}\n"
            f"Realized volatility (stdev of returns): {volatility}\n"
            f"Regime classification: {regime}\n"
            f"Bars analyzed: {len(closes)}\n"
            f"{extra_context}"
            "\nBased solely on these precomputed indicators, emit your signal."
        )
        signal = self._call_structured(prompt, AgentSignal, tool_name="emit_signal")
        return signal.model_copy(update={"agent_name": self.name, "generated_at": datetime.utcnow()})
