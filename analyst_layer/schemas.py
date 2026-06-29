"""Strict schemas for the Analyst & Intel Layer.

Two parallel execution contracts exist:

  TradeProposal — the original directional equity contract (BUY/SELL/HOLD).
  OptionsProposal — the volatility-based options contract (structure + strikes).

No agent output, and no LLM completion, ever reaches the broker as free text.
All agent output is forced through tool-call schemas that terminate in one of
the typed models below. The Risk Officer / Greeks Risk Officer sign-off is
required on all payloads before the execution layer is permitted to act.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """`extra="forbid"` rejects any field the LLM tool call didn't declare
    in the schema. Full pydantic `strict=True` is deliberately NOT used
    here: these models are populated from JSON tool-call input (e.g.
    action="BUY" as a plain string), and strict mode would refuse to
    coerce that string into the Action enum. The enum/Literal types
    themselves already reject any value outside the allowed set, which
    is the actual type-safety guarantee we need.
    """

    model_config = ConfigDict(extra="forbid")


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    LIMIT = "LIMIT"


class TradeProposal(StrictModel):
    """The deterministic execution contract. Matches the mandated schema
    exactly: ticker, action, quantity, order_type, limit_price.
    """

    ticker: str
    action: Action
    quantity: int = Field(ge=0)
    order_type: OrderType = OrderType.LIMIT
    limit_price: float = Field(gt=0)

    def model_post_init(self, __context) -> None:
        if self.action == Action.HOLD and self.quantity != 0:
            raise ValueError("HOLD proposals must carry quantity=0")
        if self.action != Action.HOLD and self.quantity == 0:
            raise ValueError("BUY/SELL proposals must carry quantity > 0")


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentSignal(StrictModel):
    """One sub-agent's narrow-scope read. Never executable on its own —
    these are inputs to consensus, not orders.
    """

    agent_name: str
    ticker: str
    stance: Action
    confidence: Confidence
    rationale: str = Field(min_length=1, max_length=2000)
    generated_at: datetime
    supporting_data_refs: list[str] = Field(default_factory=list)


class RiskVerdict(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    AMENDED = "amended"


class RiskReview(StrictModel):
    """The Risk Compliance Officer's explicit sign-off. A ConsensusPayload
    with verdict == REJECTED must never reach the execution layer's
    order-submission path. AMENDED is different: the proposal has already
    been clamped down to comply with MAX_POSITION_SIZE_PCT (see
    RiskOfficerAgent._clamp_to_limits) — it's exactly as safe to execute as
    an originally-correct APPROVED proposal, just smaller than the LLM's
    initial draft. Treating AMENDED as non-executable would mean the
    system only ever trades when the model's draft quantity happens to
    already fit under the cap, silently dropping every oversized-but-
    otherwise-sound proposal instead.
    """

    verdict: RiskVerdict
    reasons: list[str] = Field(default_factory=list)
    max_position_size_pct_checked: float
    max_daily_drawdown_pct_checked: float
    reviewed_at: datetime


class ConsensusPayload(StrictModel):
    """Structured output of one full deliberation round. Required explicit
    risk sign-off — `risk_review.verdict` gates everything downstream.
    """

    ticker: str
    signals: list[AgentSignal] = Field(default_factory=list)
    proposal: TradeProposal
    risk_review: RiskReview

    def model_post_init(self, __context) -> None:
        if not self.signals and self.risk_review.verdict == RiskVerdict.APPROVED:
            raise ValueError("a payload with zero sub-agent signals cannot carry an APPROVED verdict")

    @property
    def is_executable(self) -> bool:
        return (
            self.risk_review.verdict in (RiskVerdict.APPROVED, RiskVerdict.AMENDED)
            and self.proposal.action != Action.HOLD
        )


# ── Volatility-based schemas ──────────────────────────────────────────────────
# These replace the directional TradeProposal flow when the system is operating
# in premium-selling mode (Natenberg/tastylive framework).


class IVEnvironment(str, Enum):
    """Tastylive's primary trade-selection filter expressed as a code enum.

    ELEVATED (IVR > 50): options are meaningfully overpriced vs realized vol
        → sell premium, undefined or defined risk.
    MODERATE (IVR 30–50): some premium available, but thinner margin
        → defined-risk only (iron condor / spreads).
    DEPRESSED (IVR < 30): premium isn't worth the risk
        → no new short-premium trades.
    """

    ELEVATED = "elevated"
    MODERATE = "moderate"
    DEPRESSED = "depressed"


class StructureType(str, Enum):
    """Options structures ordered by aggressiveness.

    SHORT_STRANGLE — sell OTM call + OTM put, naked (undefined loss).
        Best edge when IVR is high and no binary events are near.
        Natenberg: pure vol-selling, maximally exposed to variance risk premium.
    IRON_CONDOR — strangle with long wings added to cap max loss.
        McMillan's defined-risk equivalent of the strangle.
        Preferred when IVR is moderate or account size limits undefined risk.
    SHORT_PUT — sell a single OTM put. Bullish tilt + premium income.
        Tastylive "wheel" entry leg. Assignment means you buy the stock.
    SHORT_CALL — sell a single OTM call. Bearish tilt + premium income.
    SHORT_PUT_SPREAD — sell put + buy further OTM put (defined risk).
        Bullish directional bias with limited capital at risk.
    SHORT_CALL_SPREAD — sell call + buy further OTM call (defined risk).
        Bearish directional bias with limited capital at risk.
    CALENDAR — sell front-month, buy back-month at same strike.
        Long vega play when term structure is steep (front IV > back IV).
        Profits when IV expands or realized vol is close to strike.
    NO_TRADE — screening criteria not met; no position to open.
    """

    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    SHORT_PUT_SPREAD = "short_put_spread"
    SHORT_CALL_SPREAD = "short_call_spread"
    CALENDAR = "calendar"
    NO_TRADE = "no_trade"


class VolRegime(str, Enum):
    """Macro volatility regime assessed by the VolRegimeAgent.

    EXPANSION — VIX is spiking or in backwardation; selling premium now means
        selling into rising vol (the worst possible time). tastylive research
        shows short premium strategies bleed badly in vol-expansion regimes.
    STABLE — VIX is in a normal range with no strong trend; the variance risk
        premium is collectible reliably.
    CONTRACTION — VIX is low and falling; thin premium, but still collectible
        if IVR is elevated for the specific underlying.
    """

    EXPANSION = "expansion"
    STABLE = "stable"
    CONTRACTION = "contraction"


class VolSignal(StrictModel):
    """One vol-focused agent's narrow-scope read.

    Not executable on its own — feeds into VolConsensusPayload alongside
    the other agents' signals before the Greeks Risk Officer reviews.
    """

    agent_name: str
    ticker: str
    iv_environment: IVEnvironment
    recommended_structure: StructureType
    confidence: Confidence
    rationale: str = Field(min_length=1, max_length=2000)
    generated_at: datetime
    flags: list[str] = Field(default_factory=list)


class OptionsProposal(StrictModel):
    """The deterministic execution contract for options structures.

    All legs are specified so the execution layer can submit a single
    multi-leg combo order (Alpaca supports legs=[...] on options orders).
    The net_credit and max_loss fields are per-share; multiply by 100 for
    the dollar value of one contract.
    """

    ticker: str
    structure: StructureType
    expiration: date
    dte: int = Field(ge=0)
    quantity: int = Field(ge=0)

    # Strangle / iron condor legs (set for those structures)
    short_call_strike: float | None = None
    short_put_strike: float | None = None
    long_call_strike: float | None = None
    long_put_strike: float | None = None

    # Single-leg (short put / short call / spread)
    single_strike: float | None = None

    # Economics (per share, multiply by 100 for one contract's dollar value)
    net_credit: float | None = Field(default=None, ge=0)
    max_loss: float | None = Field(default=None, ge=0)

    # Management levels (tastylive rules baked in)
    profit_target_pct: float = 0.50   # close when 50% of credit is captured
    loss_limit_pct: float = 2.00      # close or roll at 2x credit received
    roll_dte: int = 21                 # roll or close when DTE reaches 21

    def model_post_init(self, __context) -> None:
        if self.structure == StructureType.NO_TRADE:
            return
        if self.dte < 7:
            raise ValueError("non-NO_TRADE proposals must have dte >= 7")
        if self.quantity < 1:
            raise ValueError("non-NO_TRADE proposals must have quantity >= 1")
        if self.structure in (StructureType.SHORT_STRANGLE, StructureType.IRON_CONDOR):
            if self.short_call_strike is None or self.short_put_strike is None:
                raise ValueError(f"{self.structure} requires both short_call_strike and short_put_strike")
        if self.structure == StructureType.IRON_CONDOR:
            if self.long_call_strike is None or self.long_put_strike is None:
                raise ValueError("iron_condor requires long_call_strike and long_put_strike (the wings)")
        if self.structure in (StructureType.SHORT_PUT, StructureType.SHORT_CALL,
                              StructureType.SHORT_PUT_SPREAD, StructureType.SHORT_CALL_SPREAD):
            if self.single_strike is None:
                raise ValueError(f"{self.structure} requires single_strike")


class GreeksRiskReview(StrictModel):
    """The Greeks Risk Officer's sign-off for an options structure.

    Unlike the directional RiskReview which only checks position size,
    this evaluates the structure's impact on the portfolio's aggregate Greeks
    — specifically delta (directional drift), vega (vol exposure), and theta
    (daily decay income). Natenberg: the risk isn't the individual trade, it's
    the book's net exposure if everything moves against you at once.
    """

    verdict: RiskVerdict
    reasons: list[str] = Field(default_factory=list)
    portfolio_delta_after: float
    portfolio_vega_after: float
    portfolio_theta_after: float
    position_max_loss: float | None = None
    reviewed_at: datetime


class VolConsensusPayload(StrictModel):
    """Output of one full volatility-consensus deliberation.

    is_executable gates the execution layer — the layer must check this
    before submitting any options order.
    """

    ticker: str
    vol_signals: list[VolSignal] = Field(default_factory=list)
    proposal: OptionsProposal | None = None
    risk_review: GreeksRiskReview | None = None

    def model_post_init(self, __context) -> None:
        if (self.risk_review and self.risk_review.verdict == RiskVerdict.APPROVED
                and not self.vol_signals):
            raise ValueError("an APPROVED VolConsensusPayload must have at least one vol signal")

    @property
    def is_executable(self) -> bool:
        return (
            self.proposal is not None
            and self.proposal.structure != StructureType.NO_TRADE
            and self.risk_review is not None
            and self.risk_review.verdict in (RiskVerdict.APPROVED, RiskVerdict.AMENDED)
        )
