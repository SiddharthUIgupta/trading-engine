from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from analyst_layer.short_interest_provider import ShortInterestSignalProvider
from data_layer.exceptions import ProviderFetchError
from data_layer.models import ShortableStatus, ShortInterestSnapshot


def _snapshot(**overrides) -> ShortInterestSnapshot:
    defaults = dict(
        symbol="GME", as_of=date(2026, 6, 15), shares_short=57_054_439,
        short_percent_of_float=0.1394, days_to_cover=9.62, shares_short_prior_month=57_937_281,
    )
    defaults.update(overrides)
    return ShortInterestSnapshot(**defaults)


def _shortable(shortable=True, easy_to_borrow=True) -> ShortableStatus:
    return ShortableStatus(symbol="GME", as_of=datetime(2026, 7, 5), shortable=shortable, easy_to_borrow=easy_to_borrow)


def test_compute_returns_all_five_metrics_when_everything_available():
    openbb = MagicMock()
    openbb.get_short_interest_snapshot.return_value = _snapshot()
    alpaca = MagicMock()
    alpaca.get_shortable_status.return_value = _shortable(True, False)

    provider = ShortInterestSignalProvider(openbb, alpaca)
    result = provider.compute("GME", pit_snapshot=None)

    assert result["short_percent_of_float"] == pytest.approx(0.1394)
    assert result["days_to_cover"] == pytest.approx(9.62)
    assert result["shortable"] == 1.0
    assert result["easy_to_borrow"] == 0.0
    # mom_change = (57054439 - 57937281) / 57937281
    assert result["short_interest_mom_change"] == pytest.approx((57_054_439 - 57_937_281) / 57_937_281)


def test_compute_handles_missing_prior_month_gracefully():
    openbb = MagicMock()
    openbb.get_short_interest_snapshot.return_value = _snapshot(shares_short_prior_month=None)
    alpaca = MagicMock()
    alpaca.get_shortable_status.return_value = _shortable()

    provider = ShortInterestSignalProvider(openbb, alpaca)
    result = provider.compute("GME", pit_snapshot=None)

    assert result["short_interest_mom_change"] is None
    assert result["short_percent_of_float"] is not None


def test_compute_handles_alpaca_failure_gracefully_still_returns_openbb_metrics():
    openbb = MagicMock()
    openbb.get_short_interest_snapshot.return_value = _snapshot()
    alpaca = MagicMock()
    alpaca.get_shortable_status.side_effect = ProviderFetchError("alpaca down")

    provider = ShortInterestSignalProvider(openbb, alpaca)
    result = provider.compute("GME", pit_snapshot=None)

    assert result is not None
    assert result["short_percent_of_float"] is not None
    assert result["shortable"] is None
    assert result["easy_to_borrow"] is None


def test_compute_returns_none_when_no_short_interest_data_available():
    """Empty, not Failed — some tickers (illiquid, new IPOs) legitimately
    have no short-interest data at all.
    """
    openbb = MagicMock()
    openbb.get_short_interest_snapshot.side_effect = ProviderFetchError("no data")
    alpaca = MagicMock()

    provider = ShortInterestSignalProvider(openbb, alpaca)
    result = provider.compute("NEWIPO", pit_snapshot=None)

    assert result is None


def test_get_metric_as_of_returns_the_actual_settlement_date_not_candidate_date():
    """Regression test for the PIT-honesty design: metric_as_of must be
    short_interest.as_of (the real settlement date), never candidate_date —
    otherwise scripts/signal_uplift.py's staleness reporting would silently
    read 0 for a signal that is, in reality, ~20 days stale.

    Returns a per-metric dict, not one scalar — the OpenBB-sourced metrics
    (short_percent_of_float, days_to_cover, short_interest_mom_change) share
    the settlement date, but the Alpaca-sourced flags (shortable,
    easy_to_borrow) are fetched live and must carry their own fetch-time
    as-of, never the unrelated settlement date.
    """
    openbb = MagicMock()
    openbb.get_short_interest_snapshot.return_value = _snapshot(as_of=date(2026, 6, 15))
    alpaca = MagicMock()
    alpaca.get_shortable_status.return_value = _shortable()

    provider = ShortInterestSignalProvider(openbb, alpaca)
    result = provider.compute("GME", pit_snapshot=None)
    as_of = provider.get_metric_as_of("GME", "2026-07-05", result)

    assert as_of["short_percent_of_float"] == "2026-06-15"
    assert as_of["days_to_cover"] == "2026-06-15"
    assert as_of["short_interest_mom_change"] == "2026-06-15"
    assert as_of["short_percent_of_float"] != "2026-07-05", "must not silently fall back to candidate_date"

    # Alpaca flags must NOT inherit the OpenBB settlement date — they have
    # their own live-fetch-time as-of, which must be materially different.
    assert as_of["shortable"] != as_of["short_percent_of_float"]
    assert as_of["easy_to_borrow"] != as_of["short_percent_of_float"]
    # Fetched "now" during this test run — check dynamically, not a hardcoded
    # literal date (which breaks the moment a day passes, as this did).
    assert as_of["shortable"].startswith(datetime.utcnow().strftime("%Y-%m-%d")), \
        "Alpaca flags should be fetched 'now', during this test run"
