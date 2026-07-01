"""Kelly Criterion position sizing.

Replaces the flat MAX_POSITION_SIZE_PCT cap with a data-driven fraction
that reflects the system's actual edge. Half-Kelly is always used — full
Kelly maximises long-run growth but produces drawdowns most accounts can't
stomach, and the theoretical derivation assumes exact knowledge of win rate
and payoff ratio, which we never have.

Formula (half-Kelly):
    f* = 0.5 * (p - q / b)
where:
    p = win rate (fraction of trades with pnl > 0)
    q = 1 - p
    b = average win / average loss  (win-to-loss payoff ratio)

The result is always clamped to [0, max_position_size_pct] so the hard
risk cap from settings is never exceeded even if Kelly suggests more.

Requires at least MIN_TRADES_FOR_KELLY closed trades before trusting the
estimate. Below that threshold the system falls back to the conservative
default (half of max_position_size_pct) until there is enough history.
"""
from __future__ import annotations

MIN_TRADES_FOR_KELLY = 15
_FALLBACK_MULTIPLIER = 0.5  # conservative: half the max cap while bootstrapping
# When Kelly comes out negative (no measured edge), allow this minimum fraction
# so the system keeps trading and the VW bandit keeps getting examples.
# Only applies while n < EXPLORATION_TRADE_THRESHOLD — after that, a 0 Kelly
# means zero (the system has enough history to know it has no edge).
EXPLORATION_MIN_PCT = 0.01        # 1% of equity per trade
EXPLORATION_TRADE_THRESHOLD = 50  # stop exploring once we have this many trades


def compute_kelly_fraction(
    win_rate: float,
    win_loss_ratio: float,
    *,
    half_kelly: bool = True,
) -> float:
    """Core Kelly formula. Returns a fraction in [0, 1]; 0 means no edge."""
    if win_rate <= 0.0 or win_loss_ratio <= 0.0:
        return 0.0
    kelly = win_rate - (1.0 - win_rate) / win_loss_ratio
    if kelly <= 0.0:
        return 0.0
    return kelly * 0.5 if half_kelly else kelly


def kelly_fraction_from_pnl_history(
    realized_pnls: list[float],
    max_position_size_pct: float,
    *,
    min_trades: int = MIN_TRADES_FOR_KELLY,
) -> tuple[float, str]:
    """Derive Kelly position fraction from the account's realized P&L history.

    Parameters
    ----------
    realized_pnls:
        List of realized P&L values (positive = win, negative/zero = loss).
        Should include all closed equity trades, most recent last.
    max_position_size_pct:
        Hard upper cap from settings — Kelly never exceeds this regardless
        of what the formula produces.
    min_trades:
        Minimum closed trades before trusting the Kelly estimate.

    Returns
    -------
    (fraction, reason)
        fraction: position size as a fraction of equity, in (0, max_position_size_pct]
        reason:   human-readable explanation for logging / risk officer prompt
    """
    n = len(realized_pnls)

    if n < min_trades:
        fallback = max_position_size_pct * _FALLBACK_MULTIPLIER
        return fallback, (
            f"bootstrapping: {n}/{min_trades} trades — using {fallback:.1%} "
            f"({_FALLBACK_MULTIPLIER:.0%} of {max_position_size_pct:.1%} cap)"
        )

    wins = [p for p in realized_pnls if p > 0]
    losses = [p for p in realized_pnls if p <= 0]

    if not wins:
        return 0.0, f"no winning trades in {n} closed — Kelly=0, no new position"

    if not losses:
        # Perfect record (unlikely with real data) — use cap conservatively
        return max_position_size_pct, (
            f"all {n} trades profitable — no loss sample, using max cap conservatively"
        )

    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    raw_kelly = compute_kelly_fraction(win_rate, win_loss_ratio)
    capped = min(raw_kelly, max_position_size_pct)

    if capped == 0.0 and n < EXPLORATION_TRADE_THRESHOLD:
        capped = EXPLORATION_MIN_PCT
        reason = (
            f"half-Kelly from {n} trades: "
            f"win_rate={win_rate:.1%}, avg_win=${avg_win:.0f}, avg_loss=${avg_loss:.0f}, "
            f"W/L_ratio={win_loss_ratio:.2f} → raw={raw_kelly:.1%} (negative edge) "
            f"→ exploration floor {EXPLORATION_MIN_PCT:.1%} applied "
            f"({n}/{EXPLORATION_TRADE_THRESHOLD} trades to exit exploration)"
        )
    else:
        reason = (
            f"half-Kelly from {n} trades: "
            f"win_rate={win_rate:.1%}, avg_win=${avg_win:.0f}, avg_loss=${avg_loss:.0f}, "
            f"W/L_ratio={win_loss_ratio:.2f} → raw={raw_kelly:.1%} → capped={capped:.1%}"
        )
    return capped, reason
