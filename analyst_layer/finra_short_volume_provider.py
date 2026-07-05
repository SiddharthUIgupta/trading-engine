"""FINRA daily short-sale volume shadow signal.

Unlike short_interest_provider.py, this one CAN be made properly point-in-
time clean: FINRA publishes real historical daily files (not just a
"current" snapshot), so "the short-volume ratio as of candidate_date" is a
well-defined, fetchable historical fact. compute() reads the anchor date
from pit_snapshot.bars[-1].timestamp — the batch job builds a real price-
history snapshot ending at candidate_date (same convention Kronos's job
already uses), which is how this provider learns what date to query without
changing the SignalProvider protocol's fixed (ticker, pit_snapshot) shape.

get_metric_as_of returns the actual trade date of the most recent file used.
In the typical case (batch job scoring a candidate a few days old, by which
point T+1 data is already published) this equals candidate_date exactly —
genuinely PIT-clean, unlike short interest's inherent ~20-day settlement
lag. If the job ever scores a same-day candidate before T+1 data exists,
this correctly reports the prior available trading day instead — still
<= candidate_date, honest staleness, not a leak.
"""
from __future__ import annotations

import statistics

from data_layer.finra_shvol_client import FinraShortVolumeClient
from data_layer.models import PriceSeries

_LOOKBACK_DAYS = 25  # covers the 20-day z-score plus buffer for missing/thin days


class FinraShortVolumeSignalProvider:
    name = "finra_short_volume"
    # Bump on any change to which metrics are emitted or how they're computed —
    # signal_values rows are keyed on (candidate_id, signal_name, signal_version,
    # metric_name), so a version bump keeps old and new results from mixing.
    version = "finra-short-volume-v1"

    def __init__(self, client: FinraShortVolumeClient) -> None:
        self._client = client
        self._last_as_of: str | None = None

    def compute(self, ticker: str, pit_snapshot: PriceSeries) -> dict[str, float] | None:
        as_of_date = pit_snapshot.bars[-1].timestamp.date()
        series = self._client.get_short_vol_series(ticker, as_of_date, lookback_days=_LOOKBACK_DAYS)

        if len(series) < 20:
            # Not enough history for a real 20-day z-score — Empty, not Failed.
            # Common for illiquid/newly-listed tickers, not an error condition.
            return None

        # series is ordered most-recent-first (walked backward from as_of_date).
        dates_and_ratios = series
        ratios = [r for _, r in dates_and_ratios]

        # Cached for get_metric_as_of() — safe because the harness calls
        # compute() then, only on success, immediately calls
        # get_metric_as_of() for the same candidate before moving on.
        self._last_as_of = dates_and_ratios[0][0].isoformat()

        short_vol_ratio = ratios[0]
        short_vol_ratio_5d_avg = statistics.mean(ratios[:5])
        window_20d = ratios[:20]
        mean_20d = statistics.mean(window_20d)
        stdev_20d = statistics.stdev(window_20d) if len(set(window_20d)) > 1 else 0.0
        zscore_20d = (short_vol_ratio - mean_20d) / stdev_20d if stdev_20d > 0 else 0.0

        return {
            "short_vol_ratio": short_vol_ratio,
            "short_vol_ratio_5d_avg": short_vol_ratio_5d_avg,
            "short_vol_ratio_zscore_20d": zscore_20d,
        }

    def get_metric_as_of(self, ticker: str, candidate_date: str, result: dict[str, float]) -> str | None:
        return self._last_as_of
