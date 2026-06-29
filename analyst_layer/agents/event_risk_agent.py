"""Event Risk Agent — flags binary events within the trade's DTE window.

McMillan's warning applied mechanically: known binary events (earnings,
FDA announcements, FOMC) inflate IV artificially. You are not selling
structural variance-risk-premium when IV is elevated purely because of an
event — you are selling event-outcome risk, which is a different game
with a much worse edge for premium sellers.

The rule: if a known event falls within the DTE window, downgrade the
structure one level of aggressiveness, or veto outright if the event
is within 7 days of expiration (gamma risk is extreme there).
"""
from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import (
    Confidence,
    IVEnvironment,
    StructureType,
    VolSignal,
)
from data_layer.models import VolatilitySnapshot


class EventRiskOutput(VolSignal):
    """EventRiskAgent's structured output."""


class EventRiskAgent(BaseAgent):
    """Assesses event risk for a proposed options trade.

    Inputs:  VolatilitySnapshot (earnings_within_dte + next_earnings_date)
    Outputs: VolSignal — either confirms no event risk, or flags the risk
             and recommends downgrading to a safer structure.

    The earnings date check is deterministic (done before the LLM call).
    The LLM's job is to synthesize the event context into a recommendation
    about structure choice and to flag any non-earnings events the
    snapshot doesn't capture (FOMC, major macro releases, etc.).
    """

    name = "event_risk_agent"

    @property
    def system_prompt(self) -> str:
        return """You are an event-risk analyst for an options premium-selling desk.

Your job: assess whether any known binary events fall within the proposed trade's
30-45 DTE window, and recommend how aggressively (or not) to structure the position.

Key principles from McMillan and Natenberg:
1. Earnings within DTE: IV is artificially inflated by event uncertainty, not by
   structural variance risk premium. Selling naked strangles into earnings is
   selling the event, not the vol — a much lower-edge game. Downgrade to iron
   condor (defined risk) or recommend NO_TRADE if earnings are within 14 days.
2. FOMC meetings: Major Fed decisions can move rates and vol simultaneously.
   If an FOMC decision falls within the DTE, note it but don't necessarily veto.
3. No known events: confirm NO_TRADE flag is absent and the structure from
   the IV surface agent is appropriate.

If earnings_within_dte is True and earnings are more than 21 days out:
  → Downgrade: SHORT_STRANGLE → IRON_CONDOR
  → Keep: IRON_CONDOR and below (already defined risk)
If earnings_within_dte is True and earnings are 7-21 days out:
  → Downgrade to NO_TRADE regardless of structure
If earnings_within_dte is True and earnings are < 7 days out:
  → NO_TRADE (pure gamma risk, not IV risk — a completely different animal)
"""

    def analyze(
        self,
        ticker: str,
        vol: VolatilitySnapshot,
        proposed_structure: StructureType,
    ) -> VolSignal:
        today = date.today()
        days_to_earnings: int | None = None
        if vol.next_earnings_date:
            days_to_earnings = (vol.next_earnings_date - today).days

        # Deterministic pre-classification before the LLM
        flags: list[str] = []
        if vol.earnings_within_dte:
            flags.append("earnings_within_dte")
        if days_to_earnings is not None and days_to_earnings <= 14:
            flags.append("earnings_imminent")

        prompt = f"""Assess event risk for a proposed {proposed_structure.value} on {ticker}.

Earnings within 30-45 DTE window: {vol.earnings_within_dte}
Next earnings date: {vol.next_earnings_date or 'unknown'}
Days to earnings from today ({today}): {days_to_earnings if days_to_earnings is not None else 'unknown'}
Currently proposed structure: {proposed_structure.value}
IV-HV spread (context): {vol.iv_hv_spread:+.1%}

Based on the event-risk rules in your system prompt:
- Should the structure be changed, and if so to what?
- Are there any other known macro events (FOMC, major economic releases) in
  the next 45 days that would affect this recommendation?

Emit EventRiskOutput with your verdict.
"""
        result = self._call_structured(
            user_prompt=prompt,
            output_model=EventRiskOutput,
            tool_name="emit_event_risk_signal",
            max_tokens=500,
        )
        # If we pre-classified earnings_imminent, enforce NO_TRADE regardless of LLM output
        final_structure = result.recommended_structure
        if days_to_earnings is not None and days_to_earnings <= 7:
            final_structure = StructureType.NO_TRADE
            flags.append("earnings_veto_gamma_risk")

        iv_env = IVEnvironment.ELEVATED if vol.iv_rank >= 50 else (
            IVEnvironment.MODERATE if vol.iv_rank >= 30 else IVEnvironment.DEPRESSED
        )

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
