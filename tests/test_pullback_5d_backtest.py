from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from backtest.metrics import Trade
from backtest.portfolio_metrics import compute_exposure_and_drawdown
from backtest.pullback_5d_backtest import evaluate_pullback_5d, run_pullback_5d_backtest
from data_layer.models import PriceBar, PriceSeries


# ── Signal function unit tests (synthetic values) ───────────────────────────

def test_signal_fires_on_known_good_pullback():
    result = evaluate_pullback_5d(
        close=92, high_20d=100, sma_200=85, avg_dollar_vol_20d=30_000_000,
        pullback_min_pct=0.05, pullback_max_pct=0.10, min_dollar_vol=20_000_000,
    )
    assert result.passed is True


def test_signal_does_not_fire_below_200sma_even_with_valid_pullback():
    """A stock 8% below its 20-session high is still in the pullback band,
    but if it's ALSO below its 200-day SMA, the uptrend isn't intact — this
    is the falling-knife guard and must block entry regardless of the
    pullback math looking correct in isolation.
    """
    result = evaluate_pullback_5d(
        close=92, high_20d=100, sma_200=95, avg_dollar_vol_20d=30_000_000,
        pullback_min_pct=0.05, pullback_max_pct=0.10, min_dollar_vol=20_000_000,
    )
    assert result.passed is False
    assert any("FAIL" in r and "SMA" in r for r in result.reasons)


def test_signal_does_not_fire_when_illiquid():
    result = evaluate_pullback_5d(
        close=92, high_20d=100, sma_200=85, avg_dollar_vol_20d=5_000_000,
        pullback_min_pct=0.05, pullback_max_pct=0.10, min_dollar_vol=20_000_000,
    )
    assert result.passed is False
    assert any("FAIL" in r and "$vol" in r for r in result.reasons)


def test_signal_does_not_fire_outside_pullback_band():
    # Only 2% pullback — too shallow for the 5-10% band.
    result = evaluate_pullback_5d(
        close=98, high_20d=100, sma_200=85, avg_dollar_vol_20d=30_000_000,
        pullback_min_pct=0.05, pullback_max_pct=0.10, min_dollar_vol=20_000_000,
    )
    assert result.passed is False


# ── Walk-forward test on synthetic bars ─────────────────────────────────────

def _make_bars(closes: list[float], volume: int = 2_000_000) -> list[PriceBar]:
    bars = []
    for i, c in enumerate(closes):
        bars.append(PriceBar(
            symbol="TEST", timestamp=datetime(2023, 1, 1) + timedelta(days=i),
            open=c, high=c * 1.01, low=c * 0.99, close=c, volume=volume,
        ))
    return bars


class _FakeDataClient:
    def __init__(self, series: PriceSeries):
        self._series = series

    def get_price_history(self, ticker, start_date, end_date):
        return self._series


def test_walk_forward_enters_on_a_real_pullback_and_exits_on_time_limit():
    """220 flat bars at 100 (to satisfy the 200-SMA warmup), then a rally to
    120 (sets a new 20-session high), then a controlled 8% pullback to
    110.4 while staying comfortably above the (now-rising) 200-SMA and
    liquid — this should fire an entry, then time-exit after 5 sessions
    since price doesn't move enough to hit the stop.
    """
    closes = [100.0] * 220
    closes += [101, 103, 106, 110, 115, 120]  # rally to a new 20-session high of 120
    closes += [110.4]  # 8% pullback from 120 -> signal day
    closes += [110.4, 110.4, 110.4, 110.4, 110.4]  # flat during the hold (no stop hit)
    bars = _make_bars(closes)
    series = PriceSeries(symbol="TEST", interval="1d", bars=bars)
    client = _FakeDataClient(series)

    trades = run_pullback_5d_backtest(
        client, universe=["TEST"], years_back=1,
        pullback_min_pct=0.05, pullback_max_pct=0.10, stop_loss_pct=0.08, time_exit_sessions=5,
        min_dollar_vol=1_000_000, use_pit_universe=False,
    )

    closed = [t for t in trades if t.is_closed]
    assert len(closed) >= 1
    assert closed[0].exit_reason == "time-exit"


def test_walk_forward_produces_no_trades_when_liquidity_floor_too_high():
    closes = [100.0] * 220 + [101, 103, 106, 110, 115, 120, 110.4] + [110.4] * 5
    bars = _make_bars(closes, volume=100)  # tiny volume -> low dollar volume
    series = PriceSeries(symbol="TEST", interval="1d", bars=bars)
    client = _FakeDataClient(series)

    trades = run_pullback_5d_backtest(
        client, universe=["TEST"], years_back=1,
        pullback_min_pct=0.05, pullback_max_pct=0.10, stop_loss_pct=0.08, time_exit_sessions=5,
        min_dollar_vol=20_000_000,  # real floor, unreachable at volume=100
        use_pit_universe=False,
    )
    assert len([t for t in trades if t.is_closed]) == 0


# ── Exposure / drawdown ─────────────────────────────────────────────────────

def test_compute_exposure_and_drawdown_on_overlapping_trades():
    trades = [
        Trade("A", date(2024, 1, 1), 100, date(2024, 1, 10), 110, "time-exit"),
        Trade("B", date(2024, 1, 5), 100, date(2024, 1, 15), 90, "stop-loss"),
        Trade("C", date(2024, 2, 1), 100, date(2024, 2, 5), 105, "time-exit"),
    ]
    report = compute_exposure_and_drawdown(trades)
    assert report.max_concurrent_positions == 2
    assert report.max_drawdown_pct == pytest.approx(0.10, abs=1e-6)


def test_compute_exposure_and_drawdown_empty_trades():
    report = compute_exposure_and_drawdown([])
    assert report.max_drawdown_pct is None
    assert report.max_concurrent_positions == 0
