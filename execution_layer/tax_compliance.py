"""Wash-sale guardrail (IRC Section 1091 / IRS Pub. 550).

Same philosophy as guardrails.CircuitBreaker: deterministic, code-level
enforcement that the LLM layer cannot reason its way around. The rule:
a loss on a sale is disallowed (deferred into the replacement shares'
cost basis instead) if you buy the same or a "substantially identical"
security within 30 days before or after the sale.

Scope and limitations (deliberately simple, not a full tax-lot engine):
  - "Substantially identical" is approximated as "same ticker." Options,
    or ETFs tracking the same underlying, are NOT detected.
  - Only trades placed through this system are known. A pre-existing
    position or a loss-sale made through another broker/account is
    invisible to this guard.
  - check_before_buy BLOCKS the trade by default — it's new, discretionary
    exposure, so refusing it is the safe default. warn_before_sell only
    WARNS — refusing to let the system close a losing position for tax
    bookkeeping reasons would itself be a risk-management hazard.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from execution_layer.state_store import StateStore

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 30


@dataclass(frozen=True)
class WashSaleViolation:
    ticker: str
    reason: str


class WashSaleGuard:
    def __init__(self, state_store: StateStore, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
        self._state_store = state_store
        self._lookback_days = lookback_days

    def check_before_buy(self, ticker: str, today: date) -> WashSaleViolation | None:
        since = today - timedelta(days=self._lookback_days)
        loss_sales = self._state_store.get_recent_loss_sales(ticker, since=since)
        if not loss_sales:
            return None
        most_recent = loss_sales[0]  # query orders by sale_date DESC
        return WashSaleViolation(
            ticker=ticker,
            reason=(
                f"buying {ticker} now would trigger a wash sale: a loss of "
                f"{most_recent['realized_pnl']:.2f} was realized on {most_recent['sale_date']} "
                f"(within the {self._lookback_days}-day lookback window)"
            ),
        )

    def warn_before_sell(self, ticker: str, proposed_sale_price: float, today: date) -> str | None:
        position = self._state_store.get_position(ticker)
        if position is None or position.get("last_buy_at") is None:
            return None
        if proposed_sale_price >= position["avg_entry_price"]:
            return None  # not a loss sale — wash sale rule only applies to losses

        last_buy_at = date.fromisoformat(position["last_buy_at"][:10])
        if (today - last_buy_at).days > self._lookback_days:
            return None

        warning = (
            f"selling {ticker} at a loss now may itself be a wash sale: the current lot was "
            f"bought {last_buy_at.isoformat()}, within the {self._lookback_days}-day window — "
            "the loss may be disallowed/deferred for tax purposes. Proceeding anyway (this is a "
            "tax-reporting consequence, not a trading risk, so the sale is not blocked)."
        )
        logger.warning(warning)
        return warning
