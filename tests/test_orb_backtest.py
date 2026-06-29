from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.orb_backtest import _evaluate_one_session
from data_layer.models import PriceBar

_RANGE_BARS = [
    PriceBar(symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 30) + timedelta(minutes=5 * i),
             open=100, high=101, low=99, close=100, volume=10_000)
    for i in range(3)  # opening_range_high=101, opening_range_low=99
]


def _bar(minute_offset: int, o: float, h: float, l: float, c: float) -> PriceBar:
    return PriceBar(
        symbol="TEST", timestamp=datetime(2026, 1, 5, 9, 30) + timedelta(minutes=minute_offset),
        open=o, high=h, low=l, close=c, volume=10_000,
    )


def test_long_breakout_hits_target_at_2r():
    breakout = _bar(15, 100, 102, 100, 101.5)  # closes above 101 -> long signal
    entry_bar = _bar(20, 101.5, 101.6, 101.4, 101.5)  # entry @ open = 101.5
    # risk = entry(101.5) - stop(99) = 2.5; target = 101.5 + 2*2.5 = 106.5
    rising = _bar(25, 101.5, 107.0, 101.5, 106.0)

    trade = _evaluate_one_session("TEST", _RANGE_BARS + [breakout, entry_bar, rising], opening_range_minutes=15, risk_reward_multiple=2.0)

    assert trade.direction == "long"
    assert trade.entry_price == 101.5
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(106.5)
    assert trade.return_pct > 0


def test_long_breakout_hits_stop_loss():
    breakout = _bar(15, 100, 102, 100, 101.5)
    entry_bar = _bar(20, 101.5, 101.6, 101.4, 101.5)
    falling = _bar(25, 101.0, 101.0, 98.0, 98.5)  # low breaches stop=99

    trade = _evaluate_one_session("TEST", _RANGE_BARS + [breakout, entry_bar, falling], opening_range_minutes=15, risk_reward_multiple=2.0)

    assert trade.exit_reason == "stop-loss"
    assert trade.exit_price == 99.0  # the opening range low, not the bar's worse extreme
    assert trade.return_pct < 0


def test_short_breakout_profits_when_price_falls():
    breakdown = _bar(15, 99, 99, 97, 97.5)  # closes below 99 -> short signal
    entry_bar = _bar(20, 97.5, 97.6, 97.4, 97.5)  # entry @ open = 97.5
    # risk = stop(101) - entry(97.5) = 3.5; target = 97.5 - 2*3.5 = 90.5
    falling = _bar(25, 97.5, 97.5, 89.0, 90.0)

    trade = _evaluate_one_session("TEST", _RANGE_BARS + [breakdown, entry_bar, falling], opening_range_minutes=15, risk_reward_multiple=2.0)

    assert trade.direction == "short"
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(90.5)
    assert trade.return_pct > 0  # a profitable short must show a POSITIVE return, not negative


def test_short_breakout_loses_when_price_rises_back_through_stop():
    breakdown = _bar(15, 99, 99, 97, 97.5)
    entry_bar = _bar(20, 97.5, 97.6, 97.4, 97.5)
    rising = _bar(25, 98.0, 102.0, 98.0, 101.5)  # high breaches stop=101

    trade = _evaluate_one_session("TEST", _RANGE_BARS + [breakdown, entry_bar, rising], opening_range_minutes=15, risk_reward_multiple=2.0)

    assert trade.exit_reason == "stop-loss"
    assert trade.return_pct < 0  # a losing short must show a NEGATIVE return


def test_no_breakout_means_no_trade():
    flat = [_bar(15 + 5 * i, 100, 100.5, 99.5, 100) for i in range(5)]
    trade = _evaluate_one_session("TEST", _RANGE_BARS + flat, opening_range_minutes=15, risk_reward_multiple=2.0)
    assert trade is None


def test_neither_stop_nor_target_hit_forces_flat_at_eod_close():
    breakout = _bar(15, 100, 102, 100, 101.5)
    entry_bar = _bar(20, 101.5, 101.6, 101.4, 101.5)
    quiet = _bar(25, 101.5, 101.6, 101.4, 101.5)  # never reaches stop(99) or target(106.5)

    trade = _evaluate_one_session("TEST", _RANGE_BARS + [breakout, entry_bar, quiet], opening_range_minutes=15, risk_reward_multiple=2.0)

    assert trade.exit_reason == "eod-close"
    assert trade.exit_price == 101.5  # the last bar's close
