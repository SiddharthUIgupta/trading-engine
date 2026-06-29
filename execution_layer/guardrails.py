"""The Circuit Breaker Rule.

Three independent, hardcoded checks live here, all enforced in plain
Python with no LLM involvement:

1. Per-trade max position size — a second, independent check of the same
   limit the Risk Officer agent already applied in analyst_layer
   (defense in depth: a bug or prompt-injection in Layer 2 cannot bypass
   this because it never reaches the broker without passing here too).
2. Max daily drawdown — cross-trade state the analyst layer has no
   visibility into, so it can only live at the execution boundary. A
   breach trips the breaker, which the runtime must treat as a hard
   stop: close open positions and halt the loop for the rest of the
   trading day.
3. Daily profit target — the conservative-by-design counterpart to (2).
   Once the day's gain reaches a fixed dollar target, trading halts and
   positions close to lock the gain in, rather than risking it on
   further trades. The preference here is explicitly to bank a modest,
   reliable gain over chasing a bigger one: "stop as soon as you're up
   $50 today" beats "give it back trying to make $200."
"""
from __future__ import annotations

import logging
from datetime import date

from analyst_layer.schemas import Action, TradeProposal

logger = logging.getLogger(__name__)


class CircuitBreakerTripped(Exception):
    """Raised when a hard risk limit is violated. Callers must treat this
    as non-recoverable for the remainder of the trading day.
    """


class CircuitBreaker:
    def __init__(
        self,
        max_position_size_pct: float,
        max_daily_drawdown_pct: float,
        daily_profit_target_usd: float | None = None,
    ) -> None:
        if not (0 < max_position_size_pct <= 1):
            raise ValueError("max_position_size_pct must be in (0, 1]")
        if not (0 < max_daily_drawdown_pct <= 1):
            raise ValueError("max_daily_drawdown_pct must be in (0, 1]")
        if daily_profit_target_usd is not None and daily_profit_target_usd <= 0:
            raise ValueError("daily_profit_target_usd must be > 0 if set")
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.daily_profit_target_usd = daily_profit_target_usd
        self._day_start_equity: float | None = None
        self._trading_day: date | None = None
        self._tripped = False
        self._profit_locked = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def is_profit_locked(self) -> bool:
        return self._profit_locked

    @property
    def is_halted(self) -> bool:
        """True if trading should stop for the rest of the day for ANY
        reason — a risk breach or a profit target already banked.
        """
        return self._tripped or self._profit_locked

    @property
    def is_stock_halted(self) -> bool:
        """Stocks stop for either reason: a real risk breach, or the
        day's profit target already banked.
        """
        return self._tripped or self._profit_locked

    @property
    def is_options_halted(self) -> bool:
        """Options only stop on a real risk breach (drawdown) — by
        design, hitting the daily profit target halts stocks but leaves
        options running on their own, already-bounded per-trade risk
        limits (stop-loss, force-close-near-expiration).
        """
        return self._tripped

    def start_trading_day(
        self, equity: float, today: date, profit_target_pct: float | None = None
    ) -> None:
        self._day_start_equity = equity
        self._trading_day = today
        self._tripped = False
        self._profit_locked = False
        if profit_target_pct is not None:
            self.daily_profit_target_usd = equity * profit_target_pct
        logger.info(
            "Trading day %s started with equity=%.2f daily_profit_target=%.2f",
            today.isoformat(), equity,
            self.daily_profit_target_usd if self.daily_profit_target_usd is not None else 0.0,
        )

    def ensure_day_started(
        self, equity: float, today: date, profit_target_pct: float | None = None
    ) -> None:
        """Idempotent guard for mid-day restarts: starts the trading day only
        if start_trading_day() was not already called today. Safe to call at
        the top of every intraday job — no-op when the day is already running.
        """
        if self._day_start_equity is None or self._trading_day != today:
            self.start_trading_day(equity=equity, today=today, profit_target_pct=profit_target_pct)

    def validate_position_size(self, proposal: TradeProposal, equity: float) -> None:
        if proposal.action != Action.BUY:
            # The cap bounds NEW exposure. A SELL only reduces exposure — even
            # if the position grew past the cap from price appreciation, the
            # guardrail must never block exiting it.
            return
        notional = proposal.quantity * proposal.limit_price
        max_notional = equity * self.max_position_size_pct
        if notional > max_notional:
            raise CircuitBreakerTripped(
                f"order notional {notional:.2f} for {proposal.ticker} exceeds hard max position "
                f"size {max_notional:.2f} ({self.max_position_size_pct:.1%} of equity={equity:.2f})"
            )

    def check_drawdown(self, current_equity: float) -> bool:
        """Returns True (and trips the breaker) if intraday drawdown from
        day-start equity has reached max_daily_drawdown_pct.
        """
        if self._day_start_equity is None:
            raise RuntimeError("start_trading_day() must be called before check_drawdown()")

        drawdown_pct = (self._day_start_equity - current_equity) / self._day_start_equity
        if drawdown_pct >= self.max_daily_drawdown_pct:
            self._tripped = True
            logger.error(
                "CIRCUIT BREAKER TRIPPED: drawdown %.2f%% >= limit %.2f%% (start=%.2f, current=%.2f)",
                drawdown_pct * 100,
                self.max_daily_drawdown_pct * 100,
                self._day_start_equity,
                current_equity,
            )
            return True
        return False

    def check_profit_target(self, current_equity: float) -> bool:
        """Returns True (and locks in for the day) once the day's gain
        reaches daily_profit_target_usd. No-op (always False) if no
        target was configured.
        """
        if self.daily_profit_target_usd is None:
            return False
        if self._day_start_equity is None:
            raise RuntimeError("start_trading_day() must be called before check_profit_target()")

        profit = current_equity - self._day_start_equity
        if profit >= self.daily_profit_target_usd:
            self._profit_locked = True
            logger.info(
                "DAILY PROFIT TARGET REACHED: profit=%.2f >= target=%.2f (start=%.2f, current=%.2f) — "
                "calling it for today",
                profit,
                self.daily_profit_target_usd,
                self._day_start_equity,
                current_equity,
            )
            return True
        return False

    def assert_not_tripped(self) -> None:
        """Stock-side check — raises on either a real risk breach or the
        profit target already being banked.
        """
        if self._tripped:
            raise CircuitBreakerTripped("circuit breaker is tripped for the remainder of the trading day")
        if self._profit_locked:
            raise CircuitBreakerTripped("daily profit target already reached — done trading for today")

    def assert_options_trading_allowed(self) -> None:
        """Options-side check — raises ONLY on a real risk breach. The
        profit-locked flag deliberately does not block options; see
        is_options_halted.
        """
        if self._tripped:
            raise CircuitBreakerTripped("circuit breaker is tripped for the remainder of the trading day")


def execute_global_shutdown(
    broker, state_store, reason: str, event_type: str = "circuit_breaker_shutdown"
) -> None:
    """The mandated response to a tripped breaker (or a banked profit
    target — same mechanics, close every open position and record the
    event) and record the shutdown event. Callers (runtime.py) are
    responsible for halting the scheduler loop afterward — this function
    only handles the broker/state side.
    """
    log_fn = logger.info if event_type == "daily_profit_target_reached" else logger.error
    log_fn("GLOBAL SHUTDOWN TRIGGERED (%s): %s", event_type, reason)
    state_store.record_event(event_type=event_type, detail=reason)
    broker.close_all_positions(cancel_orders=True)
