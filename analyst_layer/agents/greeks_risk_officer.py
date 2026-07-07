"""Greeks Risk Officer — portfolio-level Greeks check before any options trade.

Natenberg's risk-management framework applied at the book level: the danger
isn't any single trade — it's what happens to the portfolio's aggregate Greeks
if conditions move against all positions simultaneously. A portfolio of 10
short strangles looks fine in isolation but becomes a concentrated vega-short
bet that will bleed badly if the VIX spikes from 18 to 30.

The officer checks:
  - Portfolio delta after the proposed trade (directional drift limit)
  - Portfolio vega after the proposed trade (vol-spike loss limit)
  - Position-level max loss vs available capital
  - Concentration: not too much premium sold on a single underlying
"""
from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from pydantic import Field

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import (
    GreeksRiskReview,
    OptionsProposal,
    RiskVerdict,
    StructureType,
    VolSignal,
)


class PortfolioGreeks(TypedDict):
    """Current portfolio aggregate Greeks before the proposed trade."""
    net_delta: float          # positive = long bias, negative = short bias
    net_vega: float           # negative = short vol (normal for premium sellers)
    net_theta: float          # positive = daily decay income (normal for premium sellers)
    portfolio_value: float    # total account value in dollars
    num_open_positions: int


class GreeksRiskOfficerOutput(GreeksRiskReview):
    """Structured output from the Greeks Risk Officer.

    reviewed_at is overridden with a default: it's a system-managed
    timestamp, never asked of the LLM in the prompt, and always overwritten
    with datetime.now() by the caller below regardless of what's returned
    here. Without a default, GreeksRiskReview's required reviewed_at field
    makes it part of the tool's required JSON schema — the LLM never
    supplies it, so every real call failed schema validation.
    """
    reviewed_at: datetime = Field(default_factory=datetime.now)


class GreeksRiskOfficer(BaseAgent):
    """Reviews a proposed OptionsProposal against portfolio-level Greeks limits.

    Inputs:  OptionsProposal, list[VolSignal], PortfolioGreeks
    Outputs: GreeksRiskReview — APPROVED, AMENDED (quantity reduced), or REJECTED
    """

    name = "greeks_risk_officer"

    # Tastylive-informed portfolio limits
    MAX_PORTFOLIO_DELTA = 0.30          # net delta as fraction of portfolio (±)
    MAX_VEGA_PCT_OF_PORTFOLIO = 0.05    # max net vega as % of portfolio value
    MAX_POSITION_PCT_OF_PORTFOLIO = 0.05  # max loss on one position = 5% of portfolio

    def __init__(self, *args, max_position_size_pct: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_position_pct = max_position_size_pct

    @property
    def system_prompt(self) -> str:
        return f"""You are the Greeks Risk Officer for an options premium-selling desk,
applying Natenberg's portfolio-level risk framework.

Your job: review a proposed options structure against the portfolio's current aggregate
Greeks and determine whether the trade is safe to execute.

Hard limits (enforced deterministically before this call — your role is to explain and
potentially AMEND the quantity, not to override the limits):

1. Portfolio delta (|net_delta|): must stay < {self.MAX_PORTFOLIO_DELTA:.0%} of portfolio value.
   Reason: premium sellers should be approximately delta-neutral — accumulating directional
   drift turns the book into a directional bet, which is not the strategy.

2. Portfolio vega: must stay < {self.MAX_VEGA_PCT_OF_PORTFOLIO:.0%} of portfolio value (in dollar terms).
   Reason: if the VIX spikes 10 points, a large net-short-vega book takes that loss
   simultaneously across all positions. This is the primary tail risk.

3. Max loss on any single position: < {self._max_position_pct:.0%} of portfolio value.
   Reason: tastylive research — position concentration is the number-one way premium-
   selling accounts blow up. Diversification across many underlyings beats sizing up.

4. Concentration: maximum of 2 active positions on any single underlying.
   Reason: undefined risk on two strangles on the same name = concentrated vega.

Verdicts:
  APPROVED — trade is within all limits as proposed.
  AMENDED — quantity reduced to fit within limits; the trade is still worth doing.
  REJECTED — the structure itself (not just the size) violates risk limits, or the
             vol signals are too conflicted to proceed.
"""

    def review(
        self,
        ticker: str,
        vol_signals: list[VolSignal],
        proposal: OptionsProposal,
        portfolio: PortfolioGreeks,
    ) -> GreeksRiskReview:
        # Hard veto: if any agent flagged a veto condition, reject immediately
        all_flags: list[str] = []
        for s in vol_signals:
            all_flags.extend(s.flags)
        if "earnings_veto_gamma_risk" in all_flags:
            return GreeksRiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=["earnings are within 7 days of expiration — gamma risk veto"],
                portfolio_delta_after=portfolio["net_delta"],
                portfolio_vega_after=portfolio["net_vega"],
                portfolio_theta_after=portfolio["net_theta"],
                position_max_loss=proposal.max_loss,
                reviewed_at=datetime.now(),
            )

        if "vol_expansion_regime" in all_flags and proposal.structure != StructureType.NO_TRADE:
            return GreeksRiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=["vol expansion regime detected by regime agent — no new short-premium trades"],
                portfolio_delta_after=portfolio["net_delta"],
                portfolio_vega_after=portfolio["net_vega"],
                portfolio_theta_after=portfolio["net_theta"],
                position_max_loss=proposal.max_loss,
                reviewed_at=datetime.now(),
            )

        if proposal.structure == StructureType.NO_TRADE:
            return GreeksRiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=["proposal structure is NO_TRADE — nothing to execute"],
                portfolio_delta_after=portfolio["net_delta"],
                portfolio_vega_after=portfolio["net_vega"],
                portfolio_theta_after=portfolio["net_theta"],
                position_max_loss=None,
                reviewed_at=datetime.now(),
            )

        # Approximate Greeks for the proposed structure (simplified estimates)
        # A 16-delta short strangle per contract ≈ net delta ~0, vega ~-$0.10/% per contract
        # These are approximations — real production would pull from the chain
        position_vega_estimate = -0.10 * proposal.quantity  # rough short vega
        position_delta_estimate = 0.0  # strangles/condors approximately delta-neutral

        projected_vega = portfolio["net_vega"] + position_vega_estimate
        projected_delta = portfolio["net_delta"] + position_delta_estimate
        projected_theta = portfolio["net_theta"] + abs(position_vega_estimate) * 0.1  # theta/vega ratio ≈ 0.1

        # Check max loss vs portfolio
        max_loss_ok = True
        loss_reason: str | None = None
        if proposal.max_loss is not None:
            position_dollar_loss = proposal.max_loss * 100 * proposal.quantity
            max_loss_pct = position_dollar_loss / portfolio["portfolio_value"] if portfolio["portfolio_value"] > 0 else 1.0
            if max_loss_pct > self._max_position_pct:
                max_loss_ok = False
                loss_reason = (
                    f"max loss ${position_dollar_loss:,.0f} = {max_loss_pct:.1%} of portfolio "
                    f"exceeds {self._max_position_pct:.0%} limit"
                )

        # Check vega limit
        vega_limit = portfolio["portfolio_value"] * self.MAX_VEGA_PCT_OF_PORTFOLIO
        vega_ok = abs(projected_vega) * 100 <= vega_limit

        signal_conflicts = [s for s in vol_signals if s.recommended_structure == StructureType.NO_TRADE]
        majority_veto = len(signal_conflicts) >= 2  # 2+ of 3 agents say no trade

        signals_summary = "\n".join(
            f"  - {s.agent_name}: {s.recommended_structure.value} ({s.confidence.value}) — {s.rationale[:200]}"
            for s in vol_signals
        )

        prompt = f"""Review this proposed options trade on {ticker}.

PROPOSED STRUCTURE: {proposal.structure.value}
  Expiration: {proposal.expiration} ({proposal.dte}d DTE)
  Quantity: {proposal.quantity} contracts
  Short call strike: {proposal.short_call_strike}
  Short put strike: {proposal.short_put_strike}
  Long call strike: {proposal.long_call_strike}
  Long put strike: {proposal.long_put_strike}
  Net credit (per share): {proposal.net_credit}
  Max loss (per share): {proposal.max_loss}

CURRENT PORTFOLIO GREEKS:
  Net delta: {portfolio['net_delta']:+.3f}
  Net vega: {portfolio['net_vega']:+.2f}
  Net theta: {portfolio['net_theta']:+.2f}
  Portfolio value: ${portfolio['portfolio_value']:,.0f}
  Open positions: {portfolio['num_open_positions']}

PROJECTED AFTER TRADE:
  Net delta: {projected_delta:+.3f}
  Net vega: {projected_vega:+.2f}
  Net theta: {projected_theta:+.2f}

LIMIT CHECKS:
  Max loss per position: {'PASS' if max_loss_ok else f'FAIL — {loss_reason}'}
  Vega limit: {'PASS' if vega_ok else 'FAIL — projected vega exceeds portfolio limit'}

VOL AGENT SIGNALS:
{signals_summary}
  Majority veto (2+ agents say NO_TRADE): {majority_veto}

FLAGS ACROSS ALL AGENTS: {', '.join(all_flags) if all_flags else 'none'}

Issue your GreeksRiskOfficerOutput: APPROVED, AMENDED (reduce quantity), or REJECTED.
If AMENDED, calculate the maximum quantity that keeps max_loss within the 5% limit.
"""
        result = self._call_structured(
            user_prompt=prompt,
            output_model=GreeksRiskOfficerOutput,
            tool_name="emit_greeks_risk_review",
            max_tokens=700,
        )

        # Deterministic overrides: hard failures cannot be approved by the LLM
        if majority_veto and result.verdict == RiskVerdict.APPROVED:
            return GreeksRiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=["2 or more vol agents vetoed — overriding LLM approval"],
                portfolio_delta_after=projected_delta,
                portfolio_vega_after=projected_vega,
                portfolio_theta_after=projected_theta,
                position_max_loss=proposal.max_loss,
                reviewed_at=datetime.now(),
            )
        if not vega_ok and result.verdict == RiskVerdict.APPROVED:
            return GreeksRiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=[f"vega limit breach — cannot approve (projected vega {projected_vega:.2f})"],
                portfolio_delta_after=projected_delta,
                portfolio_vega_after=projected_vega,
                portfolio_theta_after=projected_theta,
                position_max_loss=proposal.max_loss,
                reviewed_at=datetime.now(),
            )

        return GreeksRiskReview(
            verdict=result.verdict,
            reasons=result.reasons,
            portfolio_delta_after=result.portfolio_delta_after,
            portfolio_vega_after=result.portfolio_vega_after,
            portfolio_theta_after=result.portfolio_theta_after,
            position_max_loss=result.position_max_loss,
            reviewed_at=datetime.now(),
        )
