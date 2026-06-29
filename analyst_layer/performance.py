"""Risk-adjusted performance metrics computed from realized trade history.

All metrics are derived from the realized_sales / realized_option_sales tables
(list[dict] from StateStore) — no equity curve or benchmark needed.

Metric definitions:
  Sharpe  — mean(daily_pnl) / std(daily_pnl, ddof=1) × √252. Zero risk-free rate
             (appropriate for paper trading where the "cash" earns nothing in the sim).
  Sortino — mean(daily_pnl) / downside_std × √252, where downside_std uses the
             full-sample denominator (MAR=0, same convention as Empyrical/QuantStats).
  Calmar  — annualized_total_pnl / max_drawdown (both in $; interpretable as
             "annual $ earned per $ of max drawdown"). None if max_drawdown is 0.
  Max DD  — largest peak-to-trough decline in the cumulative realized P&L curve ($).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float       # sum(wins) / abs(sum(losses)); inf if no losses
    avg_win: float             # avg P&L of winning trades (positive number)
    avg_loss: float            # avg P&L of losing trades (negative number)
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float        # $ amount, always >= 0
    calmar_ratio: float | None
    total_pnl: float


def _aggregate_by_day(sales: list[dict]) -> list[float]:
    """Sum realized_pnl by sale_date, return a sorted list of daily totals."""
    by_day: dict[str, float] = defaultdict(float)
    for s in sales:
        by_day[s["sale_date"]] += s["realized_pnl"]
    return [by_day[d] for d in sorted(by_day)]


def _max_drawdown(daily_pnl: list[float]) -> float:
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for pnl in daily_pnl:
        cum += pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_metrics(sales: list[dict]) -> PerformanceMetrics:
    """Compute risk-adjusted metrics from a list of realized_sales dicts.

    Each dict must have `sale_date` (ISO string) and `realized_pnl` (float).
    All other keys are ignored — compatible with both equity and options sales.
    """
    if not sales:
        return PerformanceMetrics(
            total_trades=0, win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, sharpe_ratio=None,
            sortino_ratio=None, max_drawdown=0.0, calmar_ratio=None, total_pnl=0.0,
        )

    pnls = [s["realized_pnl"] for s in sales]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls)
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = (
        (sum_wins / sum_losses) if sum_losses > 0
        else (float("inf") if sum_wins > 0 else 0.0)
    )
    avg_win = sum_wins / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    total_pnl = sum(pnls)

    daily = _aggregate_by_day(sales)
    n = len(daily)

    sharpe: float | None = None
    sortino: float | None = None

    if n >= 2:
        mean_d = sum(daily) / n
        # Sample variance (ddof=1)
        var_d = sum((x - mean_d) ** 2 for x in daily) / (n - 1)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.0
        if std_d > 0:
            sharpe = (mean_d / std_d) * math.sqrt(252)

        # Sortino: full-sample denominator, MAR=0
        down_sq_sum = sum(x * x for x in daily if x < 0)
        if down_sq_sum > 0:
            down_std = math.sqrt(down_sq_sum / n)
            sortino = (mean_d / down_std) * math.sqrt(252)

    max_dd = _max_drawdown(daily)
    calmar: float | None = None
    if max_dd > 0 and n > 0:
        annualized_pnl = total_pnl * (252 / n)
        calmar = annualized_pnl / max_dd

    return PerformanceMetrics(
        total_trades=len(pnls),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        total_pnl=total_pnl,
    )
