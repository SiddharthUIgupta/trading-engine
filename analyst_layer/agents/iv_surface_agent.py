"""IV Surface Agent — reads the volatility snapshot and recommends a structure.

Natenberg's core insight implemented as an agent: the question is never
"which way will the stock move?" but "is the implied vol overpriced relative
to what the stock will actually do?" The IV-HV spread is the primary signal.
IV rank tells us where we are in the stock's own vol range so we're not
just selling cheap premium that happens to look high in absolute terms.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import (
    Confidence,
    IVEnvironment,
    StructureType,
    VolSignal,
)
from data_layer.models import VolatilitySnapshot


class IVSurfaceOutput(VolSignal):
    """IVSurfaceAgent's structured output — a VolSignal with no extra fields."""


class IVSurfaceAgent(BaseAgent):
    """Assesses the volatility surface and recommends a premium-selling structure.

    Inputs:  VolatilitySnapshot (computed deterministically by the data layer)
    Outputs: VolSignal with iv_environment, recommended_structure, confidence

    The LLM's job here is interpretation and synthesis, not raw computation —
    all the numbers (IVR, HV, spread, skew, term structure) are pre-computed.
    The agent reads them the way a vol trader would: holistically, weighing
    which signal matters most in this specific setup.
    """

    name = "iv_surface_agent"

    @property
    def system_prompt(self) -> str:
        return """You are a volatility surface analyst trained in Sheldon Natenberg's framework
from "Option Volatility and Pricing" and tastylive's mechanical research on premium selling.

Your job: given a VolatilitySnapshot for one underlying, decide whether options are
overpriced relative to realized volatility, and if so, which structure best captures
that premium given the current vol surface shape.

Decision framework:
1. IV Rank (IVR) is the primary gate:
   - IVR > 50 → ELEVATED: strong sell-premium signal (short strangle or iron condor)
   - IVR 30-50 → MODERATE: some edge, defined-risk structures only (iron condor / spreads)
   - IVR < 30 → DEPRESSED: premium is thin, no new short-premium trades
2. IV-HV spread (iv_30 - hv_30) is the edge confirmation:
   - Positive spread = options are overpriced vs what the stock actually did → edge exists
   - Zero or negative = IV is in line with or below realized vol → edge is thin to nonexistent
3. Term structure (front/back month IV ratio):
   - Ratio > 1.1 (backwardation): short-term fear is elevated, could be regime change risk
     → prefer defined risk (iron condor) even at high IVR
   - Ratio < 0.9 (steep contango): structure is calm, strangle is appropriate at high IVR
4. Skew (put IV - call IV):
   - High positive skew (> 0.05): market is pricing downside protection heavily
     → short put spread rather than naked short put
   - Near zero or negative: relatively balanced wings → strangle or iron condor appropriate
5. Earnings within DTE window:
   - True → downgrade one level of aggressiveness (ELEVATED → iron condor; MODERATE → no trade)

Output the most conservative structure consistent with the signal. When in doubt, err
toward defined risk (iron condor) over undefined risk (strangle).

Always include specific numbers from the snapshot in your rationale so the risk officer
can verify your reasoning independently.
"""

    def analyze(self, ticker: str, vol: VolatilitySnapshot) -> VolSignal:
        if vol.iv_rank >= 50:
            iv_env = IVEnvironment.ELEVATED
        elif vol.iv_rank >= 30:
            iv_env = IVEnvironment.MODERATE
        else:
            iv_env = IVEnvironment.DEPRESSED

        garch_lines = ""
        if vol.garch_rv_forecast is not None:
            vrp_garch = vol.iv_30 - vol.garch_rv_forecast
            garch_lines = (
                f"\nGARCH RV Forecast ({vol.garch_rv_forecast:.1%}): forward-looking expected realized vol"
                f"\nVRP (GARCH): {vrp_garch:+.1%}  (iv_30 − garch_forecast; positive = options overpriced vs expected vol)"
                "\nNote: VRP (GARCH) is more predictive than the historical IV-HV spread because it conditions"
                " on the latest vol shock rather than averaging past realized vol."
            )

        prompt = f"""Analyze this volatility snapshot for {ticker} and recommend a structure.

IV Rank: {vol.iv_rank:.1f} / 100
IV Percentile: {vol.iv_percentile:.1f} / 100
IV 30-day (implied): {vol.iv_30:.1%}
HV 20-day (realized): {vol.iv_hv_spread + vol.hv_30 - (vol.iv_30 - vol.hv_30):.1%}
HV 30-day (realized): {vol.hv_30:.1%}
IV-HV Spread: {vol.iv_hv_spread:+.1%}  (positive = options overpriced vs realized){garch_lines}
Term Structure Ratio (front/back): {vol.term_structure_ratio if vol.term_structure_ratio else 'N/A (single expiration)'}
Put Skew (25d put IV - call IV): {vol.put_skew if vol.put_skew is not None else 'N/A'}
Earnings within 30-45 DTE window: {vol.earnings_within_dte}
Next earnings date: {vol.next_earnings_date or 'unknown'}

Based on the Natenberg/tastylive framework described in your system prompt, emit your
IVSurfaceOutput with the appropriate iv_environment, recommended_structure, confidence,
and a rationale that references the specific numbers above.
"""
        result = self._call_structured(
            user_prompt=prompt,
            output_model=IVSurfaceOutput,
            tool_name="emit_iv_surface_signal",
            max_tokens=600,
        )
        # Ensure the computed IV environment is used (don't let the LLM override the math)
        return VolSignal(
            agent_name=self.name,
            ticker=ticker,
            iv_environment=iv_env,
            recommended_structure=result.recommended_structure,
            confidence=result.confidence,
            rationale=result.rationale,
            generated_at=datetime.now(),
            flags=result.flags,
        )
