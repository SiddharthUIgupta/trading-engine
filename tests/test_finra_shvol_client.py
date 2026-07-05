from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_layer.finra_shvol_client import FinraShortVolumeClient

_FIXTURE = Path(__file__).parent / "fixtures" / "CNMSshvol_sample.txt"


def test_parse_ticker_ratio_against_real_fixture_file(tmp_path: Path):
    """Regression test against a real, checked-in FINRA file (not synthetic)
    — confirms schema handling: high-precision floats, and correctly
    ignoring the comma-separated Market column.
    """
    client = FinraShortVolumeClient(tmp_path)

    # AAPL: 12031979.218105|89177|25887108.497784 -> ratio = short/total
    ratio = client._parse_ticker_ratio(_FIXTURE, "AAPL")
    assert ratio == pytest.approx(12_031_979.218105 / 25_887_108.497784)

    # TSLA has a decimal ShortExemptVolume (75178.750000) — confirms the
    # parser doesn't choke on that column even though it's unused for the ratio.
    ratio_tsla = client._parse_ticker_ratio(_FIXTURE, "TSLA")
    assert ratio_tsla == pytest.approx(20_861_512.334783 / 33_183_356.013124)


def test_parse_ticker_ratio_returns_none_for_unknown_ticker(tmp_path: Path):
    client = FinraShortVolumeClient(tmp_path)
    assert client._parse_ticker_ratio(_FIXTURE, "NOTATICKER") is None


def test_ensure_cached_returns_none_on_403_without_raising(tmp_path: Path):
    """A 403/404 (weekend, holiday, not-yet-published) must be treated as
    routine — None, not an exception — never block the rest of the batch.
    """
    client = FinraShortVolumeClient(tmp_path)
    with patch("data_layer.finra_shvol_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=403)
        result = client._ensure_cached(date(2026, 7, 4))  # a Saturday/holiday in this scenario
    assert result is None


def test_ensure_cached_caches_to_disk_and_does_not_refetch(tmp_path: Path):
    client = FinraShortVolumeClient(tmp_path)
    fixture_content = _FIXTURE.read_bytes()
    with patch("data_layer.finra_shvol_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, content=fixture_content)
        path1 = client._ensure_cached(date(2026, 7, 2))
        assert mock_get.call_count == 1
        assert path1 is not None
        assert path1.read_bytes() == fixture_content

        # Second call for the same date must not hit the network again.
        path2 = client._ensure_cached(date(2026, 7, 2))
        assert mock_get.call_count == 1
        assert path2 == path1


def test_get_short_vol_series_walks_backward_never_forward(tmp_path: Path):
    """Structural PIT guarantee: every date requested must be <= as_of_date.
    This is independent of (and a redundant safeguard alongside)
    scripts/signal_uplift.py's metric_as_of > candidate_date exclusion.
    """
    client = FinraShortVolumeClient(tmp_path)
    fixture_content = _FIXTURE.read_bytes()
    requested_dates = []

    def _fake_get(url, timeout):
        # extract YYYYMMDD from the URL to track what was requested
        date_str = url.rsplit("CNMSshvol", 1)[1].rstrip(".txt")
        requested_dates.append(date_str)
        return MagicMock(status_code=200, content=fixture_content)

    as_of = date(2026, 7, 2)
    with patch("data_layer.finra_shvol_client.requests.get", side_effect=_fake_get):
        series = client.get_short_vol_series("AAPL", as_of_date=as_of, lookback_days=5)

    assert len(series) == 5
    for requested in requested_dates:
        requested_date = date(int(requested[:4]), int(requested[4:6]), int(requested[6:8]))
        assert requested_date <= as_of, f"requested {requested_date} which is AFTER as_of_date {as_of} — lookahead"


def test_get_short_vol_series_gives_up_after_max_search_window(tmp_path: Path):
    client = FinraShortVolumeClient(tmp_path)
    with patch("data_layer.finra_shvol_client.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=403)  # nothing ever available
        series = client.get_short_vol_series("AAPL", as_of_date=date(2026, 7, 2), lookback_days=25)
    assert series == []
