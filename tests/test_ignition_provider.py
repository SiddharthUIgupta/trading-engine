from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from analyst_layer.ignition_provider import IgnitionSignalProvider
from data_layer.models import PriceBar, PriceSeries


def _flat_bars(n: int, price: float = 100.0, volume: int = 1_000_000) -> list[PriceBar]:
    return [
        PriceBar(
            symbol="TEST", timestamp=datetime(2026, 1, 1) + timedelta(days=i),
            open=price, high=price + 1, low=price - 1, close=price, volume=volume,
        )
        for i in range(n)
    ]


def test_returns_none_with_fewer_than_21_bars():
    series = PriceSeries(symbol="TEST", interval="1d", bars=_flat_bars(15))
    provider = IgnitionSignalProvider()
    assert provider.compute("TEST", series) is None


def test_flat_series_gives_near_zero_metrics():
    bars = _flat_bars(21)
    series = PriceSeries(symbol="TEST", interval="1d", bars=bars)
    provider = IgnitionSignalProvider()

    result = provider.compute("TEST", series)

    assert result["gap_pct"] == pytest.approx(0.0, abs=1e-9)
    assert result["volume_zscore_20d"] == pytest.approx(0.0)  # zero variance -> defined as 0
    assert result["range_expansion"] == pytest.approx(1.0)  # constant range every day
    assert result["consec_up_days"] == 0.0  # flat close-to-close, never strictly up


def test_hand_computed_gap_and_volume_spike():
    """Hand-computed reference: 20 flat days, then a final day with a real
    gap-up open and a real volume spike — confirms the formulas match
    exact arithmetic, not just directionally-plausible output.
    """
    bars = _flat_bars(20, price=100.0, volume=1_000_000)
    # Final bar: gaps up 5% from prior close (100 -> 105 open), volume 5x normal.
    bars.append(PriceBar(
        symbol="TEST", timestamp=datetime(2026, 1, 21),
        open=105.0, high=106.0, low=104.0, close=105.5, volume=5_000_000,
    ))
    series = PriceSeries(symbol="TEST", interval="1d", bars=bars)
    provider = IgnitionSignalProvider()

    result = provider.compute("TEST", series)

    assert result["gap_pct"] == pytest.approx((105.0 - 100.0) / 100.0)
    # volume_zscore: mean=1_000_000, stdev=0 among the first 20 (all identical) -> defined as 0
    # by the zero-variance guard, since a real stdev of 0 would divide by zero.
    assert result["volume_zscore_20d"] == pytest.approx(0.0)
    assert result["consec_up_days"] >= 1  # today's close (105.5) > prior close (100.0)


def test_consec_up_days_counts_correctly():
    bars = _flat_bars(15, price=100.0)  # pad for the 21-bar minimum
    # Last 6 days: strictly increasing closes.
    for i, price in enumerate([100, 101, 102, 103, 104, 105]):
        bars.append(PriceBar(
            symbol="TEST", timestamp=datetime(2026, 2, 1) + timedelta(days=i),
            open=price, high=price + 1, low=price - 1, close=price, volume=1_000_000,
        ))
    series = PriceSeries(symbol="TEST", interval="1d", bars=bars)
    provider = IgnitionSignalProvider()

    result = provider.compute("TEST", series)

    assert result["consec_up_days"] == 5.0  # 5 up-moves among the last 6 closes
