"""Tests for analyst_layer.performance — risk-adjusted metrics."""
from __future__ import annotations

import math

import pytest

from analyst_layer.performance import PerformanceMetrics, compute_metrics


def _sale(date_str: str, pnl: float) -> dict:
    return {"sale_date": date_str, "realized_pnl": pnl}


def test_empty_sales_returns_zero_metrics():
    m = compute_metrics([])
    assert m.total_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.sharpe_ratio is None
    assert m.sortino_ratio is None
    assert m.calmar_ratio is None
    assert m.max_drawdown == 0.0
    assert m.total_pnl == 0.0


def test_win_rate_and_profit_factor():
    sales = [
        _sale("2026-01-02", 100.0),
        _sale("2026-01-03", -50.0),
        _sale("2026-01-04", 200.0),
        _sale("2026-01-05", -30.0),
    ]
    m = compute_metrics(sales)
    assert m.total_trades == 4
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(300.0 / 80.0)
    assert m.avg_win == pytest.approx(150.0)
    assert m.avg_loss == pytest.approx(-40.0)
    assert m.total_pnl == pytest.approx(220.0)


def test_all_wins_profit_factor_is_infinite():
    sales = [_sale("2026-01-02", 50.0), _sale("2026-01-03", 100.0)]
    m = compute_metrics(sales)
    assert m.profit_factor == float("inf")
    assert m.win_rate == pytest.approx(1.0)
    assert m.avg_loss == 0.0


def test_all_losses():
    sales = [_sale("2026-01-02", -50.0), _sale("2026-01-03", -30.0)]
    m = compute_metrics(sales)
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0


def test_same_day_trades_aggregated_for_sharpe():
    """Multiple trades on the same day count as one daily P&L observation."""
    sales = [
        _sale("2026-01-02", 50.0),
        _sale("2026-01-02", 30.0),   # same day → daily total = 80
        _sale("2026-01-03", -20.0),
    ]
    m = compute_metrics(sales)
    assert m.total_trades == 3
    # Only 2 daily observations → Sharpe can be computed
    assert m.sharpe_ratio is not None


def test_sharpe_none_when_all_daily_pnl_equal():
    """Equal P&L every day → std=0 → Sharpe is undefined."""
    sales = [_sale(f"2026-01-0{i}", 100.0) for i in range(2, 6)]
    m = compute_metrics(sales)
    assert m.sharpe_ratio is None


def test_sharpe_none_for_single_trading_day():
    """Cannot compute std from a single data point."""
    sales = [_sale("2026-01-02", 100.0), _sale("2026-01-02", -30.0)]
    m = compute_metrics(sales)
    assert m.sharpe_ratio is None  # only 1 unique day


def test_sharpe_positive_for_net_winning_series():
    sales = [
        _sale("2026-01-02", 50.0),
        _sale("2026-01-03", -10.0),
        _sale("2026-01-04", 60.0),
        _sale("2026-01-05", -5.0),
        _sale("2026-01-06", 80.0),
    ]
    m = compute_metrics(sales)
    assert m.sharpe_ratio is not None
    assert m.sharpe_ratio > 0.0


def test_max_drawdown_peak_trough():
    # Cumulative: 100 → 150 → 50 → 80 → max peak=150, trough=50 → dd=100
    sales = [
        _sale("2026-01-02", 100.0),
        _sale("2026-01-03", 50.0),
        _sale("2026-01-04", -100.0),
        _sale("2026-01-05", 30.0),
    ]
    m = compute_metrics(sales)
    assert m.max_drawdown == pytest.approx(100.0)


def test_max_drawdown_zero_for_monotone_gains():
    sales = [_sale(f"2026-01-0{i}", 50.0) for i in range(2, 6)]
    m = compute_metrics(sales)
    assert m.max_drawdown == pytest.approx(0.0)
    assert m.calmar_ratio is None  # max_dd=0 → undefined


def test_calmar_positive_for_profitable_series_with_drawdown():
    sales = [
        _sale("2026-01-02", 200.0),
        _sale("2026-01-03", -50.0),
        _sale("2026-01-04", 100.0),
    ]
    m = compute_metrics(sales)
    assert m.max_drawdown == pytest.approx(50.0)
    assert m.calmar_ratio is not None
    assert m.calmar_ratio > 0.0


def test_sortino_none_when_no_down_days():
    """Sortino denominator is 0 when no daily P&L is negative."""
    sales = [_sale("2026-01-02", 100.0), _sale("2026-01-03", 50.0)]
    m = compute_metrics(sales)
    assert m.sortino_ratio is None


def test_sortino_positive_with_down_days():
    sales = [
        _sale("2026-01-02", 100.0),
        _sale("2026-01-03", -30.0),
        _sale("2026-01-04", 80.0),
    ]
    m = compute_metrics(sales)
    assert m.sortino_ratio is not None
    assert m.sortino_ratio > 0.0


def test_sortino_higher_than_sharpe_for_asymmetric_returns():
    """When losses are smaller than wins, Sortino > Sharpe (downside std < total std)."""
    sales = [
        _sale("2026-01-02", 200.0),
        _sale("2026-01-03", -5.0),
        _sale("2026-01-04", 180.0),
        _sale("2026-01-05", -3.0),
        _sale("2026-01-06", 190.0),
    ]
    m = compute_metrics(sales)
    assert m.sharpe_ratio is not None and m.sortino_ratio is not None
    assert m.sortino_ratio > m.sharpe_ratio


def test_returns_performance_metrics_dataclass():
    m = compute_metrics([_sale("2026-01-02", 10.0)])
    assert isinstance(m, PerformanceMetrics)
