from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from analyst_layer.finra_short_volume_provider import FinraShortVolumeSignalProvider
from data_layer.models import PriceBar, PriceSeries


def _snapshot(as_of: date) -> PriceSeries:
    return PriceSeries(
        symbol="TEST", interval="1d",
        bars=[PriceBar(symbol="TEST", timestamp=datetime(as_of.year, as_of.month, as_of.day), open=100, high=101, low=99, close=100, volume=1000)],
    )


def _series(n: int, base_ratio: float = 0.4, anchor: date = date(2026, 7, 2)) -> list[tuple[date, float]]:
    return [(anchor - timedelta(days=i), base_ratio + i * 0.001) for i in range(n)]


def test_compute_returns_none_with_fewer_than_20_days():
    client = MagicMock()
    client.get_short_vol_series.return_value = _series(15)
    provider = FinraShortVolumeSignalProvider(client)

    result = provider.compute("TEST", _snapshot(date(2026, 7, 2)))
    assert result is None


def test_compute_returns_correct_metrics_with_enough_history():
    client = MagicMock()
    client.get_short_vol_series.return_value = _series(25)
    provider = FinraShortVolumeSignalProvider(client)

    result = provider.compute("TEST", _snapshot(date(2026, 7, 2)))

    assert result is not None
    assert result["short_vol_ratio"] == pytest.approx(0.4)  # most recent (i=0)
    expected_5d = sum(0.4 + i * 0.001 for i in range(5)) / 5
    assert result["short_vol_ratio_5d_avg"] == pytest.approx(expected_5d)
    assert "short_vol_ratio_zscore_20d" in result


def test_get_metric_as_of_returns_most_recent_date_in_series():
    client = MagicMock()
    anchor = date(2026, 7, 2)
    client.get_short_vol_series.return_value = _series(25, anchor=anchor)
    provider = FinraShortVolumeSignalProvider(client)

    result = provider.compute("TEST", _snapshot(anchor))
    as_of = provider.get_metric_as_of("TEST", "2026-07-05", result)

    assert as_of == anchor.isoformat()


def test_compute_uses_as_of_date_from_pit_snapshot_not_today():
    """The anchor date must come from pit_snapshot (candidate_date), not
    wall-clock 'today' — this is the whole point of the PIT-clean design.
    """
    client = MagicMock()
    client.get_short_vol_series.return_value = _series(25)
    provider = FinraShortVolumeSignalProvider(client)

    old_date = date(2026, 1, 15)
    provider.compute("TEST", _snapshot(old_date))

    call_args = client.get_short_vol_series.call_args
    assert call_args.args[1] == old_date or call_args.kwargs.get("as_of_date") == old_date
