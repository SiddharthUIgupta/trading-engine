from __future__ import annotations

from datetime import date, datetime, timedelta

from backtest.momentum_backtest import _walk_forward
from data_layer.models import PriceBar

_KWARGS = dict(
    volume_lookback_days=10, max_float_shares=20_000_000, ema_short_period=9, ema_long_period=20,
    min_daily_gain_pct=0.10, clean_body_dominance_threshold=0.55, clean_lookback_bars=12,
    min_relative_volume=5.0, price_min=1.0, price_max=1000.0,
    stop_loss_pct=0.02, take_profit_pct=0.03, trailing_stop_pct=0.015,
)


def _daily_bar(d: date, close: float, volume: int) -> PriceBar:
    return PriceBar(symbol="TEST", timestamp=datetime(d.year, d.month, d.day, 16, 0), open=close, high=close, low=close, close=close, volume=volume)


def _signal_day_intraday(d: date, start_price: float = 100.0) -> list[PriceBar]:
    """Clean uptrend, decisive candles -- passes VWAP/EMA/clean-body checks."""
    return [
        PriceBar(
            symbol="TEST", timestamp=datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=5 * i),
            open=start_price + i, high=start_price + i + 1, low=start_price + i, close=start_price + i + 1, volume=10_000,
        )
        for i in range(25)
    ]


def _quiet_intraday(d: date, price: float) -> list[PriceBar]:
    return [
        PriceBar(
            symbol="TEST", timestamp=datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=5 * i),
            open=price, high=price, low=price, close=price, volume=1_000,
        )
        for i in range(25)
    ]


def _build_calendar(n_days: int, start: date = date(2026, 1, 5)):
    return [start + timedelta(days=i) for i in range(n_days)]


def test_no_overlapping_entries_while_a_position_is_open():
    """Regression test for the bug caught before this ever ran for real:
    a second qualifying signal day that falls within an already-open
    trade's holding period must NOT open a second, overlapping trade.
    """
    days = _build_calendar(20)
    daily_by_date = {}
    bars_by_day = {}

    # 10 quiet prior days establish the trailing average volume.
    for d in days[:10]:
        daily_by_date[d] = _daily_bar(d, close=100.0, volume=100_000)
        bars_by_day[d] = _quiet_intraday(d, 100.0)

    # Day 10: signal fires (12% gain, 6x RVOL, clean uptrend).
    signal_day = days[10]
    daily_by_date[signal_day] = _daily_bar(signal_day, close=112.0, volume=600_000)
    bars_by_day[signal_day] = _signal_day_intraday(signal_day, start_price=100.0)

    # Day 11: entry day -- price holds flat, never triggers any exit.
    entry_day = days[11]
    daily_by_date[entry_day] = _daily_bar(entry_day, close=124.0, volume=10_000)
    bars_by_day[entry_day] = _quiet_intraday(entry_day, 124.0)

    # Day 12: ALSO a qualifying signal day by the same criteria -- but the
    # day-11 trade is still open and hasn't exited, so this must be ignored.
    second_signal_day = days[12]
    daily_by_date[second_signal_day] = _daily_bar(second_signal_day, close=140.0, volume=600_000)
    bars_by_day[second_signal_day] = _signal_day_intraday(second_signal_day, start_price=124.0)

    # Remaining days: quiet, flat -- the open trade just sits there.
    for d in days[13:]:
        daily_by_date[d] = _daily_bar(d, close=124.0, volume=10_000)
        bars_by_day[d] = _quiet_intraday(d, 124.0)

    trading_days = sorted(bars_by_day.keys())
    daily_dates_sorted = sorted(daily_by_date.keys())

    trades, _ = _walk_forward(
        "TEST", trading_days, bars_by_day, daily_by_date, daily_dates_sorted, shares_float=1_000_000, **_KWARGS,
    )

    # Exactly one trade -- the day-12 signal was never evaluated as a fresh
    # entry at all (the loop was mid-exit-scan for the day-11 trade, which
    # day 12's rising prices legitimately closed via take-profit along the
    # way). That's correct: it proves no second, overlapping position opened.
    assert len(trades) == 1
    assert trades[0].entry_date == bars_by_day[entry_day][0].timestamp.date()
    assert trades[0].is_closed is True
    assert trades[0].exit_reason == "take-profit"


def test_stop_loss_fires_before_take_profit_when_both_touched_in_one_bar():
    days = _build_calendar(15)
    daily_by_date = {}
    bars_by_day = {}

    for d in days[:10]:
        daily_by_date[d] = _daily_bar(d, close=100.0, volume=100_000)
        bars_by_day[d] = _quiet_intraday(d, 100.0)

    signal_day = days[10]
    daily_by_date[signal_day] = _daily_bar(signal_day, close=112.0, volume=600_000)
    bars_by_day[signal_day] = _signal_day_intraday(signal_day, start_price=100.0)

    entry_day = days[11]
    daily_by_date[entry_day] = _daily_bar(entry_day, close=124.0, volume=10_000)
    entry_open = 124.0
    # Entry bar, then one wild bar whose range crosses BOTH the 2% stop and
    # the 3% target -- stop-loss must win this ambiguity, not take-profit.
    wild_bar = PriceBar(
        symbol="TEST", timestamp=datetime(entry_day.year, entry_day.month, entry_day.day, 9, 35),
        open=entry_open, high=entry_open * 1.05, low=entry_open * 0.95, close=entry_open, volume=5_000,
    )
    bars_by_day[entry_day] = [
        PriceBar(symbol="TEST", timestamp=datetime(entry_day.year, entry_day.month, entry_day.day, 9, 30),
                 open=entry_open, high=entry_open, low=entry_open, close=entry_open, volume=5_000),
        wild_bar,
    ]

    trading_days = sorted(bars_by_day.keys())
    daily_dates_sorted = sorted(daily_by_date.keys())

    trades, _ = _walk_forward(
        "TEST", trading_days, bars_by_day, daily_by_date, daily_dates_sorted, shares_float=1_000_000, **_KWARGS,
    )

    assert len(trades) == 1
    assert trades[0].exit_reason == "stop-loss"
