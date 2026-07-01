"""Alpaca broker wrapper.

Paper trading is forced by construction: `AlpacaBroker.from_settings`
is the only supported constructor, and it reads `Settings.is_live`
(config/settings.py) — which is itself only True when BOTH
TRADING_ENV=live and the literal confirmation token are set. There is
no code path here that can flip to a live endpoint from a single flag.
"""
from __future__ import annotations

import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, OptionLegRequest

from analyst_layer.schemas import Action, TradeProposal
from config.settings import Settings

logger = logging.getLogger(__name__)


class LiveTradingBlockedError(Exception):
    """Raised if anything attempts to construct a live-mode broker without
    the full explicit override. This should be unreachable in practice
    since Settings already forces paper — it exists as a defense-in-depth
    assertion, not the primary guard.
    """


class AlpacaBroker:
    def __init__(self, trading_client: TradingClient, is_live: bool) -> None:
        self._client = trading_client
        self.is_live = is_live
        if self.is_live:
            logger.warning("AlpacaBroker initialized in LIVE mode — real capital is at risk.")
        else:
            logger.info("AlpacaBroker initialized in PAPER mode.")

    @classmethod
    def from_settings(cls, settings: Settings) -> "AlpacaBroker":
        if settings.is_live and not (settings.trading_env == "live"):
            # Defense-in-depth: this branch should be unreachable given Settings'
            # own validator, but we refuse to construct a live client if the two
            # checks ever disagree.
            raise LiveTradingBlockedError("live/paper settings are inconsistent; refusing to start broker")

        client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=not settings.is_live,
        )
        return cls(client, is_live=settings.is_live)

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.equity)

    def get_position_shares(self, ticker: str) -> float:
        try:
            position = self._client.get_open_position(ticker)
            return float(position.qty)
        except Exception:  # noqa: BLE001 — Alpaca raises when there's no open position
            return 0.0

    def get_position_detail(self, ticker: str) -> dict | None:
        """Live position snapshot straight from Alpaca — authoritative for
        unrealized P&L, unlike recomputing it from our own avg_entry_price
        bookkeeping (which can drift from partial fills, splits, etc.).
        Returns None if there's no open position.
        """
        try:
            position = self._client.get_open_position(ticker)
        except Exception:  # noqa: BLE001 — Alpaca raises when there's no open position
            return None
        return {
            "qty": float(position.qty),
            "avg_entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "unrealized_plpc": float(position.unrealized_plpc),
        }

    def submit_order(self, proposal: TradeProposal, poll_for_fill_seconds: float = 3.0) -> dict:
        """Polls briefly after submission so the caller can know whether
        the order actually filled before this returns, rather than racing
        a position-state read against Alpaca's own fill processing (a
        limit order that matches near-instantly can otherwise report 0
        shares held for a brief window after `submit_order` returns).
        A genuinely slow-to-fill limit order just returns its still-open
        status after the poll window — this never blocks indefinitely.
        """
        if proposal.action == Action.HOLD:
            logger.info("HOLD proposal for %s — no order submitted.", proposal.ticker)
            return {"status": "skipped", "reason": "HOLD"}

        side = OrderSide.BUY if proposal.action == Action.BUY else OrderSide.SELL
        order_request = LimitOrderRequest(
            symbol=proposal.ticker,
            qty=proposal.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=float(f"{proposal.limit_price:.2f}"),  # guarantee exactly 2 decimal places; raw floats fail Alpaca's sub-penny check
        )
        order = self._client.submit_order(order_request)
        logger.info(
            "Submitted %s order: %s x%d @ %.2f (paper=%s)",
            proposal.action.value,
            proposal.ticker,
            proposal.quantity,
            proposal.limit_price,
            not self.is_live,
        )
        return self._poll_for_fill(order, poll_for_fill_seconds)

    def submit_option_order(
        self, contract_symbol: str, side: Action, contracts: int, limit_price: float, poll_for_fill_seconds: float = 3.0
    ) -> dict:
        """Long calls/puts only — `side` is always BUY to open a position
        (the only thing this track does) or SELL to close one. There is no
        options-equivalent of `Action.HOLD` since this is only ever called
        once a contract has already been selected; the caller decides
        whether to call at all.

        Reuses the same LimitOrderRequest the equity path uses — Alpaca's
        options orders take an OCC contract symbol exactly like an equity
        ticker, just with a 100-share multiplier baked into what Alpaca
        treats as the underlying notional. `limit_price` here is the
        per-share premium (matching how the options chain quotes
        bid/ask), not pre-multiplied by 100 — verified live against a
        paper-account fill before this was trusted.
        """
        order_side = OrderSide.BUY if side == Action.BUY else OrderSide.SELL
        order_request = LimitOrderRequest(
            symbol=contract_symbol,
            qty=contracts,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            position_intent=PositionIntent.BUY_TO_OPEN if side == Action.BUY else PositionIntent.SELL_TO_CLOSE,
        )
        order = self._client.submit_order(order_request)
        logger.info(
            "Submitted options %s order: %s x%d @ %.2f (paper=%s)",
            side.value, contract_symbol, contracts, limit_price, not self.is_live,
        )
        return self._poll_for_fill(order, poll_for_fill_seconds)

    def _poll_for_fill(self, order, poll_for_fill_seconds: float) -> dict:
        filled_qty = 0.0
        filled_avg_price = None
        order_status = order.status.value if hasattr(order.status, "value") else str(order.status)
        deadline = time.monotonic() + poll_for_fill_seconds
        while time.monotonic() < deadline:
            current = self._client.get_order_by_id(order.id)
            order_status = current.status.value if hasattr(current.status, "value") else str(current.status)
            filled_qty = float(current.filled_qty or 0)
            filled_avg_price = float(current.filled_avg_price) if current.filled_avg_price else None
            if order_status in ("filled", "canceled", "rejected", "expired"):
                break
            time.sleep(0.5)

        return {
            "status": "submitted",
            "order_id": str(order.id),
            "order_status": order_status,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
        }

    def submit_spread_order(
        self,
        legs: list[tuple[str, Action]],
        contracts: int,
        net_credit: float,
        poll_for_fill_seconds: float = 5.0,
    ) -> dict:
        """Submit a multi-leg options spread as a single atomic mleg order.

        Alpaca evaluates all legs together, so the short legs are never seen
        as uncovered even for a single-tick window. Required for defined-risk
        structures (iron condors, vertical spreads) on Level 3 accounts.

        `net_credit` is a positive number representing premium received.
        Alpaca's mleg convention: negative limit_price = credit to the account.
        """
        leg_requests = [
            OptionLegRequest(
                symbol=symbol,
                ratio_qty=1.0,
                position_intent=(
                    PositionIntent.SELL_TO_OPEN if side == Action.SELL
                    else PositionIntent.BUY_TO_OPEN
                ),
            )
            for symbol, side in legs
        ]
        order_request = LimitOrderRequest(
            order_class=OrderClass.MLEG,
            qty=contracts,
            time_in_force=TimeInForce.DAY,
            limit_price=float(f"{-net_credit:.2f}"),  # negative = credit per Alpaca mleg convention; string-format guarantees exactly 2 decimal places
            legs=leg_requests,
        )
        order = self._client.submit_order(order_request)
        logger.info(
            "Submitted mleg spread: %d legs x%d contracts net_credit=%.2f (paper=%s)",
            len(legs), contracts, net_credit, not self.is_live,
        )
        return self._poll_for_fill(order, poll_for_fill_seconds)

    def get_last_fill_price(self, symbol: str) -> float | None:
        """Most recent FILLED order's fill price for one symbol — used
        after a bulk close (e.g. circuit-breaker shutdown) where the
        position no longer exists, so there's no current_price/
        avg_entry_price left to read from `get_position_detail`.
        """
        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol], limit=5))
        for order in orders:
            status = order.status.value if hasattr(order.status, "value") else str(order.status)
            if status == "filled" and order.filled_avg_price:
                return float(order.filled_avg_price)
        return None

    def get_open_orders(self) -> list[dict]:
        """Orders submitted but not yet in a terminal state — the dashboard
        gap this exists to close: a position only shows up once it's
        actually filled, so a slow-to-fill order (seen repeatedly with
        options on the paper account) was otherwise invisible anywhere
        except the raw events log.

        mleg orders have symbol=None at the top level (each leg carries its
        own symbol). The `legs` field is populated for mleg orders so callers
        can display individual leg details rather than just "mleg".
        """
        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        result = []
        for o in orders:
            legs = None
            if o.legs:
                legs = [
                    {
                        "symbol": leg.symbol,
                        "side": leg.side.value if hasattr(leg.side, "value") else str(leg.side),
                        "position_intent": (
                            leg.position_intent.value
                            if hasattr(leg.position_intent, "value")
                            else str(leg.position_intent)
                        ),
                    }
                    for leg in o.legs
                ]
            result.append({
                "order_id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "qty": float(o.qty) if o.qty else None,
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                "submitted_at": str(o.submitted_at) if o.submitted_at else None,
                "legs": legs,
            })
        return result

    def get_all_positions(self) -> list[dict]:
        """Returns all open positions with current prices and unrealized P&L."""
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    def close_all_positions(self, cancel_orders: bool = True) -> None:
        logger.warning("Closing ALL open positions (cancel_orders=%s).", cancel_orders)
        self._client.close_all_positions(cancel_orders=cancel_orders)

    def close_position(self, symbol: str) -> None:
        """Closes exactly one position — unlike close_all_positions, this
        can target a single equity ticker without also touching open
        options positions (used for the stocks-only profit lock).
        """
        try:
            self._client.close_position(symbol_or_asset_id=symbol)
        except Exception as exc:  # noqa: BLE001 — already flat is not an error worth raising over
            logger.warning("close_position(%s) failed (likely already flat): %s", symbol, exc)

    def cancel_order(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001 — already filled/canceled is not an error worth raising over
            logger.warning("cancel_order(%s) failed (likely already resolved): %s", order_id, exc)
