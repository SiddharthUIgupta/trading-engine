"""Portfolio-level stats (exposure, drawdown) deliberately kept separate
from backtest/metrics.py — that module's own docstring is explicit that
per-trade edge statistics avoid needing position-sizing/concurrency
assumptions. Max drawdown and exposure genuinely need those assumptions,
so they get their own module rather than complicating metrics.py's
scope for every existing caller (thesis, momentum, orb).

Disclosed assumption: the equity curve here is equal-weighted — each
closed trade contributes its return_pct once, on its exit date, divided
by that date's average concurrency. This is the simplest assumption that
doesn't require picking an arbitrary capital-per-trade number; it is not
a claim about how much capital an actual live version would allocate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from backtest.metrics import Trade


@dataclass(frozen=True)
class ExposureReport:
    avg_concurrent_positions: float
    max_concurrent_positions: int
    max_drawdown_pct: float | None


def compute_exposure_and_drawdown(trades: list[Trade]) -> ExposureReport:
    closed = [t for t in trades if t.is_closed]
    if not closed:
        return ExposureReport(avg_concurrent_positions=0.0, max_concurrent_positions=0, max_drawdown_pct=None)

    start = min(t.entry_date for t in closed)
    end = max(t.exit_date for t in closed)

    daily_concurrency: dict[date, int] = {}
    day = start
    while day <= end:
        daily_concurrency[day] = 0
        day += timedelta(days=1)
    for t in closed:
        day = t.entry_date
        while day <= t.exit_date:
            daily_concurrency[day] += 1
            day += timedelta(days=1)

    concurrency_values = list(daily_concurrency.values())
    avg_concurrent = sum(concurrency_values) / len(concurrency_values)
    max_concurrent = max(concurrency_values)

    # Equal-weighted equity curve: each trade's return contributes on its
    # exit date, scaled by that date's average concurrency so overlapping
    # trades don't each count as if they used 100% of capital.
    trades_by_exit_date: dict[date, list[Trade]] = {}
    for t in closed:
        trades_by_exit_date.setdefault(t.exit_date, []).append(t)

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for day in sorted(trades_by_exit_date):
        day_concurrency = daily_concurrency.get(day, 1) or 1
        for t in trades_by_exit_date[day]:
            equity *= (1 + t.return_pct / day_concurrency)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    return ExposureReport(
        avg_concurrent_positions=avg_concurrent,
        max_concurrent_positions=max_concurrent,
        max_drawdown_pct=max_drawdown,
    )
