from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.thesis_backtest import run_thesis_backtest
from data_layer.models import PriceBar

_KWARGS = dict(
    min_pullback_pct=0.20, max_pullback_pct=0.50, stop_loss_pct=0.18,
    trailing_stop_pct=0.10, trailing_stop_activation_pct=0.20,
)


def _bar(day_offset: int, open_: float, high: float, low: float, close: float) -> PriceBar:
    return PriceBar(
        symbol="TEST",
        timestamp=datetime(2024, 1, 1) + timedelta(days=day_offset),
        open=open_, high=high, low=low, close=close, volume=1_000_000,
    )


def _flat_bars(n: int, price: float, start_offset: int = 0) -> list[PriceBar]:
    return [_bar(start_offset + i, price, price, price, price) for i in range(n)]


class _FakeSeries:
    def __init__(self, bars):
        self.bars = bars


class _FakeDataClient:
    def __init__(self, bars: list[PriceBar]):
        self._bars = bars

    def get_price_history(self, ticker, start_date, end_date):
        return _FakeSeries(self._bars)


def _run(bars: list[PriceBar]):
    client = _FakeDataClient(bars)
    return run_thesis_backtest(client, universe=["TEST"], **_KWARGS)


def test_no_pullback_means_no_trades():
    bars = _flat_bars(280, 100.0)  # never pulls back
    trades = _run(bars)
    assert trades == []


def test_pullback_within_band_enters_and_stays_open_when_healthy():
    bars = _flat_bars(252, 100.0)  # establishes year_high = 100
    bars.append(_bar(252, 75.0, 75.0, 75.0, 75.0))  # 25% pullback -> passes
    # price holds steady afterward — no stop, no trailing-stop trigger
    bars.extend(_flat_bars(10, 76.0, start_offset=253))

    trades = _run(bars)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_price == 76.0  # next day's open after the signal day
    assert trade.is_closed is False  # still open at backtest end


def test_stop_loss_exits_at_the_stop_price_not_the_bars_low():
    bars = _flat_bars(252, 100.0)
    bars.append(_bar(252, 75.0, 75.0, 75.0, 75.0))  # signal day
    bars.append(_bar(253, 75.0, 75.0, 75.0, 75.0))  # entry fills here at open=75.0
    # Stop price = 75 * (1 - 0.18) = 61.5. Bar's low plunges well past it,
    # but the fill must be at the stop price, not the bar's extreme low.
    bars.append(_bar(254, 70.0, 70.0, 55.0, 65.0))
    # Recovers back near the year-high so the padding bars (needed to clear
    # run_thesis_backtest's minimum-bar floor) don't themselves qualify as a
    # fresh pullback and trigger an unintended second entry.
    bars.extend(_flat_bars(10, 99.0, start_offset=255))

    trades = _run(bars)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_price == 75.0
    assert trade.exit_price == pytest.approx(75.0 * 0.82)
    assert trade.exit_reason == "stop-loss"
    assert trade.return_pct == pytest.approx(-0.18)


def test_trailing_stop_only_engages_after_activation_threshold():
    bars = _flat_bars(252, 100.0)
    bars.append(_bar(252, 75.0, 75.0, 75.0, 75.0))  # signal day
    bars.append(_bar(253, 75.0, 75.0, 75.0, 75.0))  # entry @ 75.0
    # Rises to 95 (gain_to_peak = 26.7%, clears the 20% activation floor),
    # then drops — trailing stop = 95 * (1 - 0.10) = 85.5.
    bars.append(_bar(254, 75.0, 95.0, 75.0, 90.0))
    # A dip that does NOT breach the trailing stop yet (low=86 > 85.5)
    bars.append(_bar(255, 90.0, 90.0, 86.0, 88.0))
    # Now breaches it
    bars.append(_bar(256, 87.0, 87.0, 80.0, 82.0))
    bars.extend(_flat_bars(10, 82.0, start_offset=257))  # padding past run_thesis_backtest's minimum-bar floor

    trades = _run(bars)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "trailing-stop"
    assert trade.exit_price == pytest.approx(95.0 * 0.90)


def test_pullback_beyond_ceiling_does_not_enter():
    bars = _flat_bars(252, 100.0)
    bars.append(_bar(252, 30.0, 30.0, 30.0, 30.0))  # 70% pullback -- beyond the 50% ceiling
    bars.extend(_flat_bars(10, 30.0, start_offset=253))

    trades = _run(bars)
    assert trades == []


def test_does_not_average_into_an_existing_position():
    """One position per ticker at a time -- a second qualifying day while
    already in the trade must not open a second, overlapping trade.
    """
    bars = _flat_bars(252, 100.0)
    bars.append(_bar(252, 75.0, 75.0, 75.0, 75.0))  # first signal
    bars.append(_bar(253, 75.0, 75.0, 75.0, 75.0))  # entry @ 75.0
    # Price dips further the next day -- still a qualifying pullback by the
    # same screen, but we're already in a position on this ticker.
    bars.append(_bar(254, 70.0, 70.0, 70.0, 70.0))
    bars.extend(_flat_bars(10, 71.0, start_offset=255))

    trades = _run(bars)
    assert len(trades) == 1  # not two
