"""Circuit breaker risk management — two classes.

CircuitBreaker
    Original strategy-level guard. Enforces three per-strategy limits:
    1. Per-trade max position size (defense-in-depth against analyst layer bugs)
    2. Max daily drawdown — trips the breaker, halts strategy for the day
    3. Daily profit target — locks in a gain and stops trading for the day

RobustCircuitBreaker (extends CircuitBreaker)
    Five-level adaptive risk framework built on top of the original:

    Level 0  Per-trade cap            Hard block (inherited)
    Level 1  Soft brake               After N consecutive losses → halve position sizes
    Level 2  Daily halt               After X% daily drawdown → halt strategy (inherited, improved)
    Level 3  Weekly halt              After Y% weekly loss → halt ALL strategies until Monday
    Level 4  Trailing halt            After Z% drop from equity peak → full system halt
    + VIX scaling                     Multiplies position sizing 0.40–1.00× based on VIX level

GlobalRiskState
    Shared singleton across all four RobustCircuitBreaker instances.
    Owns the weekly and trailing drawdown checks so a breach from one
    strategy propagates immediately to all strategies.
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
        capital_limit_pct: float = 1.0,
        daily_profit_target_usd: float | None = None,
        name: str = "default",
    ) -> None:
        if not (0 < max_position_size_pct <= 1):
            raise ValueError("max_position_size_pct must be in (0, 1]")
        if not (0 < max_daily_drawdown_pct <= 1):
            raise ValueError("max_daily_drawdown_pct must be in (0, 1]")
        if not (0 < capital_limit_pct <= 1):
            raise ValueError("capital_limit_pct must be in (0, 1]")
        if daily_profit_target_usd is not None and daily_profit_target_usd <= 0:
            raise ValueError("daily_profit_target_usd must be > 0 if set")
        self.name = name
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.capital_limit_pct = capital_limit_pct
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
            "[%s] Trading day %s started: equity=%.2f capital_limit=%.0f%% daily_profit_target=%.2f",
            self.name, today.isoformat(), equity, self.capital_limit_pct * 100,
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

    def check_drawdown(self, today_pnl: float) -> bool:
        """Returns True (and trips the breaker) if this agent's today-only P&L
        has lost more than max_daily_drawdown_pct of day-start equity.

        today_pnl is the sum of:
          - realized P&L from positions closed today (negative = loss)
          - unrealized P&L on positions opened today (negative = loss)
        Pre-existing overnight positions are excluded — their MTM moves do not
        count against this agent's drawdown limit.
        """
        if self._day_start_equity is None:
            raise RuntimeError("start_trading_day() must be called before check_drawdown()")

        if today_pnl >= 0:
            return False

        drawdown_pct = -today_pnl / self._day_start_equity
        if drawdown_pct >= self.max_daily_drawdown_pct:
            self._tripped = True
            logger.error(
                "[%s] CIRCUIT BREAKER TRIPPED: today_pnl=%.2f (%.2f%% drawdown >= limit %.2f%%, start_equity=%.2f)",
                self.name, today_pnl, drawdown_pct * 100,
                self.max_daily_drawdown_pct * 100, self._day_start_equity,
            )
            return True
        return False

    def validate_capital_limit(
        self, new_notional: float, agent_deployed: float, total_deployed: float, total_equity: float
    ) -> None:
        """Raises CircuitBreakerTripped if this trade would exceed the agent's
        capital allocation OR the shared pool remaining across all agents.
        """
        agent_max = total_equity * self.capital_limit_pct
        if agent_deployed + new_notional > agent_max:
            raise CircuitBreakerTripped(
                f"[{self.name}] capital limit exceeded: deployed={agent_deployed:.0f} + new={new_notional:.0f} "
                f"> agent_max={agent_max:.0f} ({self.capital_limit_pct:.0%} of equity={total_equity:.0f})"
            )
        remaining_pool = total_equity - total_deployed
        if new_notional > remaining_pool:
            raise CircuitBreakerTripped(
                f"[{self.name}] shared pool exhausted: remaining={remaining_pool:.0f} < new={new_notional:.0f} "
                f"(total_deployed={total_deployed:.0f}, equity={total_equity:.0f})"
            )

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
            raise CircuitBreakerTripped(f"[{self.name}] circuit breaker is tripped for the remainder of the trading day")
        if self._profit_locked:
            raise CircuitBreakerTripped(f"[{self.name}] daily profit target already reached — done trading for today")

    def assert_options_trading_allowed(self) -> None:
        """Options-side check — raises ONLY on a real risk breach. The
        profit-locked flag deliberately does not block options; see
        is_options_halted.
        """
        if self._tripped:
            raise CircuitBreakerTripped(f"[{self.name}] circuit breaker is tripped for the remainder of the trading day")


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


# ─────────────────────────────────────────────────────────────────────────────
# GlobalRiskState — shared across all RobustCircuitBreaker instances
# ─────────────────────────────────────────────────────────────────────────────

class GlobalRiskState:
    """Tracks cross-strategy limits: weekly drawdown and trailing drawdown from peak.

    One instance is shared across all four strategy breakers. When any breaker
    detects a weekly or trailing breach it writes to this object, and every
    breaker's is_halted check reads it — so one bad strategy immediately gates
    all others from opening new positions.

    Weekly halt resets automatically on Monday. Trailing halt requires an
    explicit operator call to reset() (it means the account has lost 20%+ from
    its all-time high — that warrants a human decision before resuming).
    """

    def __init__(
        self,
        max_weekly_drawdown_pct: float = 0.08,
        max_trailing_drawdown_pct: float = 0.20,
    ) -> None:
        self.max_weekly_drawdown_pct = max_weekly_drawdown_pct
        self.max_trailing_drawdown_pct = max_trailing_drawdown_pct
        self._equity_peak: float = 0.0
        self._week_start_equity: float = 0.0
        self._week_iso: int | None = None
        self.weekly_halted: bool = False
        self.trailing_halted: bool = False
        self.halt_reason: str = ""

    @property
    def is_halted(self) -> bool:
        return self.weekly_halted or self.trailing_halted

    def update(self, equity: float, today: date) -> tuple[bool, str]:
        """Call once per pre-market scan.  Returns (newly_halted, reason)."""
        # Bootstrap peak on first call
        if self._equity_peak == 0.0:
            self._equity_peak = equity

        # Track all-time equity peak
        if equity > self._equity_peak:
            self._equity_peak = equity

        # Weekly accounting — reset at start of each ISO week
        iso_week = today.isocalendar()[1]
        if self._week_iso != iso_week:
            self._week_iso = iso_week
            self._week_start_equity = equity
            if self.weekly_halted:
                logger.info("GlobalRisk: new ISO week — weekly halt lifted")
                self.weekly_halted = False

        # Level 4 — trailing drawdown from all-time peak
        if not self.trailing_halted and self._equity_peak > 0:
            trailing_dd = (self._equity_peak - equity) / self._equity_peak
            if trailing_dd >= self.max_trailing_drawdown_pct:
                self.trailing_halted = True
                reason = (
                    f"TRAILING HALT: equity={equity:,.0f} is {trailing_dd:.1%} below "
                    f"all-time peak={self._equity_peak:,.0f} "
                    f"(limit={self.max_trailing_drawdown_pct:.0%}) — manual reset required"
                )
                self.halt_reason = reason
                logger.critical("GlobalRisk: %s", reason)
                return True, reason

        # Level 3 — weekly drawdown
        if not self.weekly_halted and self._week_start_equity > 0:
            weekly_dd = (self._week_start_equity - equity) / self._week_start_equity
            if weekly_dd >= self.max_weekly_drawdown_pct:
                self.weekly_halted = True
                reason = (
                    f"WEEKLY HALT: lost {weekly_dd:.1%} this week "
                    f"(equity={equity:,.0f}, week_start={self._week_start_equity:,.0f}, "
                    f"limit={self.max_weekly_drawdown_pct:.0%}) — resumes Monday"
                )
                self.halt_reason = reason
                logger.error("GlobalRisk: %s", reason)
                return True, reason

        return False, ""

    def reset_trailing(self) -> None:
        """Manual operator reset for trailing halt — requires deliberate human action."""
        logger.warning("GlobalRisk: trailing halt manually reset by operator")
        self.trailing_halted = False
        self.halt_reason = ""

    def status(self) -> str:
        if self.trailing_halted:
            return f"TRAILING HALT | {self.halt_reason}"
        if self.weekly_halted:
            return f"WEEKLY HALT | {self.halt_reason}"
        peak_dd = (self._equity_peak - self._week_start_equity) / self._equity_peak if self._equity_peak else 0
        return (
            f"OK | peak={self._equity_peak:,.0f} | "
            f"week_start={self._week_start_equity:,.0f} | "
            f"weekly_limit={self.max_weekly_drawdown_pct:.0%} | "
            f"trailing_limit={self.max_trailing_drawdown_pct:.0%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RobustCircuitBreaker — drop-in replacement for CircuitBreaker
# ─────────────────────────────────────────────────────────────────────────────

class RobustCircuitBreaker(CircuitBreaker):
    """Five-level adaptive circuit breaker.

    Extends CircuitBreaker with:
    - Soft brake: consecutive loss detection → halves position sizing
    - VIX scaling: automatically reduces sizing in volatile markets
    - Global halt propagation: weekly + trailing halts via GlobalRiskState

    All existing CircuitBreaker public methods are preserved so this is a
    drop-in replacement — callers only need to additionally call:
      set_vix(vix)                  — daily, after regime assessment
      record_trade_outcome(won)     — after every closed trade
      get_size_multiplier()         — multiply Kelly fraction before sizing
    """

    # VIX → position size multiplier bands
    _VIX_BANDS: tuple[tuple[float, float], ...] = (
        (35.0, 0.40),
        (30.0, 0.55),
        (25.0, 0.70),
        (20.0, 0.85),
        (0.0,  1.00),
    )

    def __init__(
        self,
        max_position_size_pct: float,
        max_daily_drawdown_pct: float,
        capital_limit_pct: float = 1.0,
        daily_profit_target_usd: float | None = None,
        name: str = "default",
        consecutive_loss_limit: int = 3,
        soft_brake_multiplier: float = 0.5,
        global_state: GlobalRiskState | None = None,
        win_rate_window: int = 20,
    ) -> None:
        super().__init__(
            max_position_size_pct=max_position_size_pct,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
            capital_limit_pct=capital_limit_pct,
            daily_profit_target_usd=daily_profit_target_usd,
            name=name,
        )
        self._consecutive_loss_limit = consecutive_loss_limit
        self._soft_brake_multiplier = soft_brake_multiplier
        self._global_state = global_state
        self._win_rate_window = win_rate_window

        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._trade_outcomes: list[int] = []
        self._current_vix: float = 18.0

    # ── New public API ────────────────────────────────────────────────────────

    def set_vix(self, vix: float) -> None:
        """Update current VIX for size scaling. Call daily after regime assessment."""
        self._current_vix = max(0.0, vix)

    def record_trade_outcome(self, won: bool) -> None:
        """Call after every closed trade to update consecutive loss counter.

        Fires the soft brake log the moment the consecutive loss limit is crossed.
        Resets consecutive loss counter on a win.
        """
        if won:
            self._consecutive_losses = 0
            self._consecutive_wins += 1
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            if self._consecutive_losses == self._consecutive_loss_limit:
                logger.warning(
                    "[%s] Soft brake engaged: %d consecutive losses — "
                    "position sizing reduced to %.0f%% until next win",
                    self.name, self._consecutive_losses,
                    self._soft_brake_multiplier * 100,
                )

        self._trade_outcomes.append(1 if won else 0)
        if len(self._trade_outcomes) > self._win_rate_window:
            self._trade_outcomes.pop(0)

    def get_size_multiplier(self) -> float:
        """Position size multiplier [0.0, 1.0] to stack on top of Kelly fraction.

        Returns 0.0 if any hard halt is active (daily, weekly, trailing).
        Otherwise stacks soft brake × VIX band multiplier.

        Usage:
            kelly_fraction = kelly_fraction * breaker.get_size_multiplier()
        """
        # Any hard halt → 0x
        if self._tripped or (self._global_state and self._global_state.is_halted):
            return 0.0

        multiplier = 1.0

        # Level 1 — soft brake from consecutive losses
        if self._consecutive_losses >= self._consecutive_loss_limit:
            multiplier *= self._soft_brake_multiplier

        # VIX scaling
        for threshold, scale in self._VIX_BANDS:
            if self._current_vix >= threshold:
                multiplier *= scale
                break

        return max(0.0, min(1.0, multiplier))

    @property
    def rolling_win_rate(self) -> float | None:
        if not self._trade_outcomes:
            return None
        return sum(self._trade_outcomes) / len(self._trade_outcomes)

    @property
    def is_soft_braked(self) -> bool:
        return self._consecutive_losses >= self._consecutive_loss_limit

    # ── Override halt properties to include global halts ─────────────────────

    @property
    def _globally_halted(self) -> bool:
        return bool(self._global_state and self._global_state.is_halted)

    @property
    def is_halted(self) -> bool:
        return super().is_halted or self._globally_halted

    @property
    def is_stock_halted(self) -> bool:
        return super().is_stock_halted or self._globally_halted

    @property
    def is_options_halted(self) -> bool:
        return super().is_options_halted or self._globally_halted

    def assert_not_tripped(self) -> None:
        if self._globally_halted:
            raise CircuitBreakerTripped(
                f"[{self.name}] globally halted: {self._global_state.halt_reason}"
            )
        super().assert_not_tripped()

    def assert_options_trading_allowed(self) -> None:
        if self._globally_halted:
            raise CircuitBreakerTripped(
                f"[{self.name}] globally halted: {self._global_state.halt_reason}"
            )
        super().assert_options_trading_allowed()

    def status_summary(self) -> str:
        """One-line status for logging."""
        if self._globally_halted:
            state = f"GLOBAL HALT ({self._global_state.halt_reason})"
        elif self._tripped:
            state = "DAILY HALT"
        elif self._profit_locked:
            state = "PROFIT LOCKED"
        elif self.is_soft_braked:
            state = f"SOFT BRAKE ({self._consecutive_losses} consecutive losses)"
        else:
            state = "OK"

        vix_mult = self.get_size_multiplier()
        wr = f"{self.rolling_win_rate:.0%}" if self.rolling_win_rate is not None else "n/a"
        return (
            f"[{self.name}] {state} | "
            f"size_mult={vix_mult:.0%} | vix={self._current_vix:.1f} | "
            f"consec_loss={self._consecutive_losses} | "
            f"win_rate={wr} ({len(self._trade_outcomes)} trades)"
        )
