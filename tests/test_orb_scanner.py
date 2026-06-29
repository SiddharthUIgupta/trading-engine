from __future__ import annotations

from datetime import datetime, timedelta

from analyst_layer.orb_scanner import evaluate_orb
from data_layer.models import PriceBar, PriceSeries


def _bar(i: int, o: float, h: float, l: float, c: float, v: int = 10_000) -> PriceBar:
    return PriceBar(
        symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 30) + timedelta(minutes=5 * i),
        open=o, high=h, low=l, close=c, volume=v,
    )


def _series(bars) -> PriceSeries:
    return PriceSeries(symbol="TEST", interval="5m", bars=bars)


def test_no_breakout_when_price_stays_inside_the_range():
    bars = [_bar(i, 100, 101, 99, 100) for i in range(10)]  # 3 range bars + 7 flat bars, all inside
    signal = evaluate_orb(_series(bars), opening_range_minutes=15, bar_minutes=5)
    assert signal.direction == "none"


def test_long_breakout_on_a_confirmed_close_above_range():
    range_bars = [_bar(0, 100, 101, 99, 100), _bar(1, 100, 101, 99, 100), _bar(2, 100, 101, 99, 100)]
    # opening_range_high = 101. A wick above that doesn't confirm; a CLOSE above does.
    wick_only = _bar(3, 100, 102, 100, 100.5)
    confirmed_breakout = _bar(4, 100.5, 103, 100, 102.5)
    signal = evaluate_orb(_series(range_bars + [wick_only, confirmed_breakout]), opening_range_minutes=15, bar_minutes=5)
    assert signal.direction == "long"
    assert signal.breakout_bar_index == 4
    assert signal.opening_range_high == 101


def test_short_breakout_on_a_confirmed_close_below_range():
    range_bars = [_bar(0, 100, 101, 99, 100), _bar(1, 100, 101, 99, 100), _bar(2, 100, 101, 99, 100)]
    confirmed_breakdown = _bar(3, 99, 99, 97, 97.5)
    signal = evaluate_orb(_series(range_bars + [confirmed_breakdown]), opening_range_minutes=15, bar_minutes=5)
    assert signal.direction == "short"
    assert signal.opening_range_low == 99


def test_takes_the_first_confirmed_breakout_not_a_later_one():
    range_bars = [_bar(0, 100, 101, 99, 100), _bar(1, 100, 101, 99, 100), _bar(2, 100, 101, 99, 100)]
    first_breakout = _bar(3, 101, 102, 101, 101.5)  # closes above 101
    later_bar = _bar(4, 101.5, 105, 101, 104)
    signal = evaluate_orb(_series(range_bars + [first_breakout, later_bar]), opening_range_minutes=15, bar_minutes=5)
    assert signal.breakout_bar_index == 3


def test_insufficient_bars_returns_none():
    bars = [_bar(0, 100, 101, 99, 100)]  # fewer than the 3 bars needed for a 15-min range at 5-min bars
    signal = evaluate_orb(_series(bars), opening_range_minutes=15, bar_minutes=5)
    assert signal.direction == "none"


def test_volume_confirmation_rejects_a_low_volume_breakout():
    range_bars = [_bar(0, 100, 101, 99, 100, v=10_000), _bar(1, 100, 101, 99, 100, v=10_000), _bar(2, 100, 101, 99, 100, v=10_000)]
    weak_breakout = _bar(3, 100, 102, 100, 101.5, v=5_000)  # below the range's own average volume
    signal = evaluate_orb(_series(range_bars + [weak_breakout]), opening_range_minutes=15, bar_minutes=5, volume_confirmation_multiple=1.5)
    assert signal.direction == "none"


def test_volume_confirmation_accepts_a_high_volume_breakout():
    range_bars = [_bar(0, 100, 101, 99, 100, v=10_000), _bar(1, 100, 101, 99, 100, v=10_000), _bar(2, 100, 101, 99, 100, v=10_000)]
    strong_breakout = _bar(3, 100, 102, 100, 101.5, v=20_000)  # 2x the range's average volume
    signal = evaluate_orb(_series(range_bars + [strong_breakout]), opening_range_minutes=15, bar_minutes=5, volume_confirmation_multiple=1.5)
    assert signal.direction == "long"


def test_volume_confirmation_keeps_scanning_past_a_rejected_breakout():
    range_bars = [_bar(0, 100, 101, 99, 100, v=10_000), _bar(1, 100, 101, 99, 100, v=10_000), _bar(2, 100, 101, 99, 100, v=10_000)]
    weak_breakout = _bar(3, 100, 102, 100, 101.5, v=5_000)
    strong_breakout_later = _bar(4, 101.5, 103, 101, 102.5, v=25_000)
    signal = evaluate_orb(
        _series(range_bars + [weak_breakout, strong_breakout_later]), opening_range_minutes=15, bar_minutes=5, volume_confirmation_multiple=1.5,
    )
    assert signal.direction == "long"
    assert signal.breakout_bar_index == 4
