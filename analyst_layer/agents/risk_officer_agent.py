"""Risk Compliance Officer agent.

Critical distinction from the other three sub-agents: the LLM here only
ever produces a *draft* TradeProposal. Whether that draft survives is
decided by deterministic Python (`_clamp_to_limits`), not by the model's
own judgment. This is the code-level enforcement point referenced in the
mandate's "Deterministic Validation" requirement — the LLM cannot talk
its way past MAX_POSITION_SIZE_PCT no matter what it outputs.
"""
from __future__ import annotations

import math
from datetime import datetime

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import (
    Action,
    AgentSignal,
    Confidence,
    OrderType,
    RiskReview,
    RiskVerdict,
    TradeProposal,
)

_CONFIDENCE_WEIGHT = {Confidence.LOW: 1, Confidence.MEDIUM: 2, Confidence.HIGH: 3}


class AccountContext(dict):
    """Minimal shape: equity, current_price, existing_shares. Kept as a
    plain dict (not a frozen Pydantic model) since it's an internal
    call parameter, not a cross-layer contract.
    """


class RiskOfficerAgent(BaseAgent):
    name = "risk_compliance_officer_agent"

    def __init__(self, client, model: str, max_position_size_pct: float, usage_callback=None) -> None:
        super().__init__(client, model, usage_callback=usage_callback)
        self._max_position_size_pct = max_position_size_pct

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Risk Compliance Officer on a trading desk. You "
            "review the other analysts' signals and draft ONE trade proposal "
            "that reflects the weight of evidence. You do not get final say "
            "on position sizing — that is enforced separately by deterministic "
            "code regardless of what you propose. Respond by calling the "
            "emit_proposal tool with ticker, action, quantity, order_type "
            "(always LIMIT), and limit_price.\n\n"
            "Strict rules on action/quantity combinations:\n"
            "- HOLD must always have quantity=0.\n"
            "- BUY or SELL must always have quantity > 0 — these are never "
            "valid with quantity=0. If you would otherwise propose a 0-quantity "
            "BUY or SELL, propose HOLD instead.\n"
            "- You can only SELL shares that are actually held. If 'Existing "
            "shares held' is 0, you cannot propose SELL — propose HOLD instead, "
            "even if you are bearish.\n"
            "- If signals conflict, give higher priority to the technical_analysis_agent's signal. "
            "Technical price action leads fundamental/macro data. If the technical_analysis_agent "
            "proposes BUY with MEDIUM or HIGH confidence, propose BUY even if other agents propose HOLD or SELL."
        )

    def review(
        self, ticker: str, signals: list[AgentSignal], account: AccountContext, lessons: str = ""
    ) -> tuple[TradeProposal, RiskReview]:
        draft = self._draft_proposal(ticker, signals, account, lessons=lessons)
        final_proposal, verdict, reasons = self._clamp_to_limits(draft, account)
        review = RiskReview(
            verdict=verdict,
            reasons=reasons,
            max_position_size_pct_checked=self._max_position_size_pct,
            max_daily_drawdown_pct_checked=account.get("max_daily_drawdown_pct", 0.0),
            reviewed_at=datetime.utcnow(),
        )
        return final_proposal, review

    def _draft_proposal(
        self, ticker: str, signals: list[AgentSignal], account: AccountContext, lessons: str = ""
    ) -> TradeProposal:
        signals_text = "\n".join(
            f"  - {s.agent_name}: {s.stance.value} (confidence={s.confidence.value}) — {s.rationale}"
            for s in signals
        )
        prompt = (
            f"{lessons}"
            f"Ticker: {ticker}\n"
            f"Current price: {account['current_price']}\n"
            f"Account equity: {account['equity']}\n"
            f"Existing shares held: {account.get('existing_shares', 0)}\n\n"
            f"Sub-agent signals:\n{signals_text}\n\n"
            "Draft your proposal."
        )
        # The tool's input_schema (TradeProposal) already rejects a 0-quantity
        # BUY/SELL, but raises rather than coercing. A SELL beyond what's held
        # would pass that schema check (any positive quantity is schema-valid)
        # and only fail later at the broker — so it's clamped here instead.
        draft = self._call_structured(prompt, TradeProposal, tool_name="emit_proposal")
        existing_shares = int(account.get("existing_shares", 0))
        if draft.action == Action.SELL and existing_shares <= 0:
            return draft.model_copy(update={"action": Action.HOLD, "quantity": 0})
        if draft.action == Action.SELL and draft.quantity > existing_shares:
            return draft.model_copy(update={"quantity": existing_shares})
        return draft

    def _clamp_to_limits(
        self, draft: TradeProposal, account: AccountContext
    ) -> tuple[TradeProposal, RiskVerdict, list[str]]:
        """Deterministic, non-LLM enforcement of position sizing.

        Three independent checks applied in order:
        1. Correlation hard-block — reject if the new position duplicates
           existing exposure beyond the HARD_BLOCK_THRESHOLD.
        2. Kelly-adjusted size — use the data-driven fraction (already
           correlation-adjusted) instead of the flat max cap.
        3. Hard cap fallback — Kelly is always bounded by MAX_POSITION_SIZE_PCT.
        """
        reasons: list[str] = []

        if draft.action == Action.HOLD:
            return draft, RiskVerdict.APPROVED, ["no position change proposed"]

        # ── 1. Correlation hard-block (BUY only) ─────────────────────────────
        if draft.action == Action.BUY:
            corr_blocked: bool = account.get("correlation_hard_blocked", False)
            corr_reason: str = account.get("correlation_reason", "")
            if corr_blocked:
                rejected = draft.model_copy(update={"action": Action.HOLD, "quantity": 0})
                reasons.append(corr_reason)
                return rejected, RiskVerdict.REJECTED, reasons
            if corr_reason:
                reasons.append(corr_reason)

        # ── 2. Kelly-adjusted effective cap ──────────────────────────────────
        # AccountContext carries a pre-computed kelly_fraction (already
        # correlation-penalised if in the soft-reduce zone, capped at
        # max_position_size_pct). Fall back to flat cap when absent.
        effective_pct: float = account.get("kelly_fraction", self._max_position_size_pct)
        kelly_reason: str = account.get("kelly_reason", f"flat cap {self._max_position_size_pct:.1%}")
        reasons.append(kelly_reason)

        equity = account["equity"]
        price = account["current_price"]
        max_notional = equity * effective_pct
        max_shares = math.floor(max_notional / price) if price > 0 else 0

        if draft.quantity <= max_shares:
            return draft, RiskVerdict.APPROVED, reasons + [
                f"quantity {draft.quantity} within Kelly-sized limit "
                f"({effective_pct:.1%} of equity = ${max_notional:.0f})"
            ]

        if max_shares <= 0:
            rejected = draft.model_copy(update={"action": Action.HOLD, "quantity": 0})
            reasons.append(
                f"Kelly-sized limit allows 0 shares at ${price:.2f} "
                f"({effective_pct:.1%} × ${equity:.0f} = ${max_notional:.0f}); converting to HOLD"
            )
            return rejected, RiskVerdict.REJECTED, reasons

        amended = draft.model_copy(update={"quantity": max_shares})
        reasons.append(
            f"quantity {draft.quantity} → amended to {max_shares} shares "
            f"({effective_pct:.1%} of equity = ${max_notional:.0f})"
        )
        return amended, RiskVerdict.AMENDED, reasons
