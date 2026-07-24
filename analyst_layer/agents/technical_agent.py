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
    """Calculates quantitative indicators deterministically in Python and
    hands the numbers to the LLM for momentum confirmation. Scope: price and
    volume technicals only — momentum, trend, volatility.
    """

    name = "technical_analysis_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Technical Analysis analyst on a momentum trading desk. "
            "This stock was pre-selected by a quantitative factor screen for "
            "strong price momentum relative to the market universe. Your job is "
            "to CONFIRM or DENY that momentum using the technical indicators "
            "provided. Emit BUY if momentum is intact (price above short-term "
            "MA, positive 20d return, volume confirming). Emit SELL only if "
            "momentum has clearly broken (price below both MAs on high volume, "
            "negative 20d return). Emit HOLD only if indicators are genuinely "
            "mixed with no directional read. Do NOT use HOLD as a default — the "
            "factor screen already filtered for momentum, so HOLD should be rare. "
            "Respond by calling the emit_signal tool."
        )

    def analyze(self, ticker: str, series: PriceSeries, lessons: str = "", extra_context: str = "") -> AgentSignal:
        closes = [bar.close for bar in series.bars]
        volumes = [bar.volume for bar in series.bars if hasattr(bar, "volume") and bar.volume]
        sma_short = _simple_moving_average(closes, window=min(10, len(closes)))
        sma_long = _simple_moving_average(closes, window=min(30, len(closes)))
        volatility = _realized_volatility(closes)
        last_close = closes[-1]

        mom20 = (last_close / closes[-21] - 1) if len(closes) >= 21 else None
        mom60 = (last_close / closes[-61] - 1) if len(closes) >= 61 else None

        if volumes and len(volumes) >= 21:
            vol_ratio = sum(volumes[-5:]) / 5 / (sum(volumes[-21:]) / 21)
            vol_context = f"Volume ratio (5d avg / 21d avg): {vol_ratio:.2f}x"
        else:
            vol_context = "Volume data unavailable"

        if sma_short is not None and sma_long is not None:
            trend = "above_both_MAs" if last_close > sma_short > sma_long else (
                "between_MAs" if sma_long < last_close < sma_short or sma_short < last_close < sma_long
                else "below_both_MAs"
            )
        else:
            trend = "insufficient_data"

        prompt = (
            f"{lessons}"
            f"Ticker: {ticker}\n"
            f"Last close: {last_close:.4f}\n"
            f"SMA(10): {f'{sma_short:.4f}' if sma_short is not None else 'N/A'}\n"
            f"SMA(30): {f'{sma_long:.4f}' if sma_long is not None else 'N/A'}\n"
            f"Price vs MAs: {trend}\n"
            f"20d return: {f'{mom20:.1%}' if mom20 is not None else 'N/A'}\n"
            f"60d return: {f'{mom60:.1%}' if mom60 is not None else 'N/A'}\n"
            f"Realized volatility: {f'{volatility:.4f}' if volatility is not None else 'N/A'}\n"
            f"{vol_context}\n"
            f"Bars analyzed: {len(closes)}\n"
            f"{extra_context}"
            "\nIs the momentum intact? Emit your signal."
        )
        signal = self._call_structured(prompt, AgentSignal, tool_name="emit_signal")
        return signal.model_copy(update={"agent_name": self.name, "generated_at": datetime.utcnow()})
