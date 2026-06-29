"""Intraday exit check — a deliberately cheap, narrow agent.

Runs once per held position per intraday_monitoring tick (every 15 min
during market hours), NOT the full 4-agent consensus — that would be
4 LLM calls x N positions x ~28 ticks/day, which is the kind of API-cost
blowup the user explicitly asked to avoid. This agent only ever decides
HOLD vs SELL on an existing position; it never opens new positions, so
it carries no BUY path at all.
"""
from __future__ import annotations

from analyst_layer.agents.base import BaseAgent
from analyst_layer.schemas import Action, OrderType, TradeProposal


class IntradayExitAgent(BaseAgent):
    name = "intraday_exit_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You monitor ONE already-open position during the trading day and decide "
            "only whether to exit it now or keep holding — you never open new positions "
            "and you are never asked about any ticker you don't already hold. Respond by "
            "calling the emit_decision tool with action=HOLD (quantity=0) to keep the "
            "position, or action=SELL with quantity equal to the full position size to "
            "exit it now. You only have a HOLD/SELL choice — never propose BUY."
        )

    def review(
        self,
        ticker: str,
        quantity: float,
        avg_entry_price: float,
        current_price: float,
        unrealized_plpc: float,
    ) -> TradeProposal:
        prompt = (
            f"Ticker: {ticker}\n"
            f"Shares held: {quantity}\n"
            f"Average entry price: {avg_entry_price:.2f}\n"
            f"Current price: {current_price:.2f}\n"
            f"Unrealized P&L: {unrealized_plpc:+.2%}\n\n"
            "Decide: hold this position, or exit it now?"
        )
        draft = self._call_structured(prompt, TradeProposal, tool_name="emit_decision")

        if draft.action == Action.BUY:
            # The agent has no BUY path by design — if it still returns one
            # (model error, not a trust assumption we rely on), force HOLD
            # rather than ever let a stray BUY reach the broker from here.
            return TradeProposal(ticker=ticker, action=Action.HOLD, quantity=0, limit_price=current_price)
        if draft.action == Action.SELL:
            return TradeProposal(
                ticker=ticker,
                action=Action.SELL,
                quantity=int(quantity),
                order_type=OrderType.LIMIT,
                limit_price=current_price,
            )
        return TradeProposal(ticker=ticker, action=Action.HOLD, quantity=0, limit_price=current_price)
