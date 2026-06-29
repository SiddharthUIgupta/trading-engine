"""Trade-log statistics shared by both backtests.

Deliberately reports per-trade edge statistics (win rate, average
win/loss, profit factor) rather than a portfolio-level return — that
would require position-sizing and concurrency assumptions (how much
capital per trade, how many trades held at once) this backtest doesn't
model. Per-trade stats answer the actual question at hand: does the
entry+exit rule combination have positive expected value, independent
of how much money you'd put behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Trade:
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    direction: str = "long"  # "long" or "short" -- only ORB trades short; thesis/momentum are long-only

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def return_pct(self) -> float | None:
        if not self.is_closed:
            return None
        raw = (self.exit_price - self.entry_price) / self.entry_price
        return raw if self.direction == "long" else -raw


@dataclass(frozen=True)
class BacktestReport:
    total_trades: int
    still_open_at_backtest_end: int
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    profit_factor: float | None
    best_trade_pct: float | None
    worst_trade_pct: float | None
    mean_return_pct: float | None
    median_return_pct: float | None
    confidence_note: str


def summarize(trades: list[Trade]) -> BacktestReport:
    closed = [t for t in trades if t.is_closed]
    still_open = len(trades) - len(closed)
    returns = [t.return_pct for t in closed]

    if not returns:
        return BacktestReport(
            total_trades=0, still_open_at_backtest_end=still_open, win_rate=None, avg_win_pct=None,
            avg_loss_pct=None, profit_factor=None, best_trade_pct=None, worst_trade_pct=None,
            mean_return_pct=None, median_return_pct=None,
            confidence_note="No closed trades — no basis for any conclusion.",
        )

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else None)
    sorted_returns = sorted(returns)
    median = sorted_returns[len(sorted_returns) // 2]

    n = len(returns)
    if n < 20:
        confidence_note = f"Only {n} closed trades — far too small a sample to draw a real conclusion from."
    elif n < 50:
        confidence_note = f"{n} closed trades — a thin sample; treat this as suggestive, not conclusive."
    else:
        confidence_note = f"{n} closed trades — a reasonable sample, though still not a substitute for live validation."

    return BacktestReport(
        total_trades=len(returns),
        still_open_at_backtest_end=still_open,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        best_trade_pct=max(returns),
        worst_trade_pct=min(returns),
        mean_return_pct=sum(returns) / n,
        median_return_pct=median,
        confidence_note=confidence_note,
    )
