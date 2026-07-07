"""Vol Regime Agent — assesses the macro volatility environment.

tastylive's most-cited risk for premium sellers: selling into a vol-expansion
regime. When the VIX is spiking (implying market fear is rising fast), selling
strangles means you're selling into increasing vol — the position moves against
you on both wings simultaneously as skew widens and the overall level rises.

Natenberg's framing: you want to sell premium when vol is high relative to what
it will be, not when it's rising toward an unknown ceiling. The VIX term
structure (VIX vs VIX3M) is the regime signal: backwardation (near > far)
signals acute fear and expansion; contango (near < far) signals stability.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import (
    Confidence,
    IVEnvironment,
    StructureType,
    VolRegime,
    VolSignal,
)


class VixContext:
    """Lightweight carrier for VIX data passed into the agent."""

    def __init__(
        self,
        vix_current: float,
        vix_1w_ago: float | None = None,
        vix_1m_ago: float | None = None,
        vix3m_current: float | None = None,
    ) -> None:
        self.vix_current = vix_current
        self.vix_1w_ago = vix_1w_ago
        self.vix_1m_ago = vix_1m_ago
        self.vix3m_current = vix3m_current

    @property
    def regime(self) -> VolRegime:
        """Deterministic regime classification before the LLM layer."""
        # VIX backwardation = acute fear = expansion regime
        if self.vix3m_current and self.vix_current > self.vix3m_current * 1.05:
            return VolRegime.EXPANSION
        # VIX spike: > 25% rise over 1 week = expansion
        if self.vix_1w_ago and self.vix_current > self.vix_1w_ago * 1.25:
            return VolRegime.EXPANSION
        # Absolute VIX > 30 = elevated fear
        if self.vix_current > 30:
            return VolRegime.EXPANSION
        # VIX < 15 = suppressed vol
        if self.vix_current < 15:
            return VolRegime.CONTRACTION
        return VolRegime.STABLE


class VolRegimeOutput(VolSignal):
    """VolRegimeAgent's structured output — VolSignal plus regime classification."""
    vol_regime: VolRegime


class VolRegimeAgent(BaseAgent):
    """Assesses the macro volatility regime using VIX level and term structure.

    Inputs:  VixContext (current VIX, 1w/1m history, VIX3M)
    Outputs: VolSignal with regime flag — EXPANSION vetoes new short-premium
             trades; STABLE and CONTRACTION permit them.
    """

    name = "vol_regime_agent"

    @property
    def system_prompt(self) -> str:
        return """You are a macro volatility regime analyst for an options premium-selling desk.

Your job: assess whether the current VIX environment is safe for opening new
short-premium positions, or whether a vol-expansion regime makes new trades dangerous.

Regime classification (deterministic thresholds are pre-applied before you receive
this prompt — your job is to add context and nuance):

EXPANSION — VIX is spiking or in backwardation (VIX > VIX3M). This is the worst
  possible environment to sell premium. tastylive research: short-premium strategies
  lose most of their annual P&L in vol-expansion periods. Veto new trades.

STABLE — VIX is in a normal range (15-30), neither spiking nor suppressed.
  The variance risk premium is collectible. Permit new trades with normal sizing.

CONTRACTION — VIX < 15, vol is suppressed. Premium is thin in absolute terms,
  but may still be worth selling if a specific underlying's IVR is elevated.
  Permit with reduced sizing (premium per contract is lower).

Key signals to weigh:
- VIX term structure: if current VIX > VIX3M (backwardation), the market is pricing
  near-term fear above future uncertainty — a classic regime-change warning.
- Rate of VIX change: a sharp spike in a week matters more than the absolute level.
- Absolute level: VIX > 30 warrants caution regardless of trend.
"""

    def analyze(self, ticker: str, vix: VixContext) -> VolSignal:
        regime = vix.regime
        flags: list[str] = []
        if regime == VolRegime.EXPANSION:
            flags.append("vol_expansion_regime")
            recommended = StructureType.NO_TRADE
        else:
            recommended = StructureType.IRON_CONDOR  # conservative default; IV surface agent drives final choice

        iv_env = IVEnvironment.ELEVATED if regime == VolRegime.STABLE else (
            IVEnvironment.MODERATE if regime == VolRegime.CONTRACTION else IVEnvironment.DEPRESSED
        )

        term_structure_note = "N/A"
        if vix.vix3m_current:
            ratio = vix.vix_current / vix.vix3m_current
            term_structure_note = f"{ratio:.2f} (VIX/VIX3M) — {'backwardation (fear spike)' if ratio > 1.0 else 'contango (normal)'}"

        vix_1w_ago_note = f"{vix.vix_1w_ago:.1f}" if vix.vix_1w_ago else "N/A"
        vix_1m_ago_note = f"{vix.vix_1m_ago:.1f}" if vix.vix_1m_ago else "N/A"
        vix3m_current_note = f"{vix.vix3m_current:.1f}" if vix.vix3m_current else "N/A"

        prompt = f"""Assess the macro volatility regime for new short-premium trades on {ticker}.

Current VIX: {vix.vix_current:.1f}
VIX 1 week ago: {vix_1w_ago_note}
VIX 1 month ago: {vix_1m_ago_note}
VIX3M (3-month implied): {vix3m_current_note}
VIX/VIX3M term structure: {term_structure_note}
Pre-classified regime: {regime.value}

Given this regime, confirm or adjust the recommended structure and confidence level.
Note any specific reasons the regime would make new premium-selling positions
particularly dangerous or particularly attractive right now.
"""
        result = self._call_structured(
            user_prompt=prompt,
            output_model=VolRegimeOutput,
            tool_name="emit_vol_regime_signal",
            max_tokens=500,
        )
        # Enforce the deterministic EXPANSION veto — LLM cannot override it
        final_structure = StructureType.NO_TRADE if regime == VolRegime.EXPANSION else result.recommended_structure

        return VolSignal(
            agent_name=self.name,
            ticker=ticker,
            iv_environment=iv_env,
            recommended_structure=final_structure,
            confidence=result.confidence,
            rationale=result.rationale,
            generated_at=datetime.now(),
            flags=flags + result.flags,
        )
