"""Ignition detector shadow signal — pure function over price/volume data
already available, no new data source.

100% point-in-time clean by construction: every metric is computed purely
from pit_snapshot.bars, which already ends exactly at candidate_date. No
get_metric_as_of needed — the harness's existing default (candidate_date)
is correct, identical to Kronos.

No existing ATR/true-range/consecutive-days helper found anywhere in
analyst_layer/ (confirmed by search before writing this) — these are new,
small, standard formulas. gap_pct reuses the same
(today_open - prior_close) / prior_close formula already used in
orb_scanner.py/gap_scanner.py, for consistency.
"""
from __future__ import annotations

import statistics

from data_layer.models import PriceBar, PriceSeries

_LOOKBACK_BARS = 21  # 20 days of history + today


class IgnitionSignalProvider:
    name = "ignition"
    # Bump on any change to which metrics are emitted or how they're computed.
    version = "ignition-v1"

    def compute(self, ticker: str, pit_snapshot: PriceSeries) -> dict[str, float] | None:
        bars = pit_snapshot.bars[-_LOOKBACK_BARS:]
        if len(bars) < _LOOKBACK_BARS:
            return None  # Empty — not enough history for a 20-day window

        today = bars[-1]
        prior = bars[-2]

        volumes = [b.volume for b in bars[:-1]]  # trailing 20 days, excluding today
        vol_mean = statistics.mean(volumes)
        vol_stdev = statistics.stdev(volumes) if len(set(volumes)) > 1 else 0.0
        volume_zscore_20d = (today.volume - vol_mean) / vol_stdev if vol_stdev > 0 else 0.0

        gap_pct = (today.open - prior.close) / prior.close if prior.close > 0 else 0.0

        true_ranges = [_true_range(bars[i], bars[i - 1]) for i in range(1, len(bars))]
        atr_20d = statistics.mean(true_ranges[:-1])  # trailing 20 days, excluding today's own TR
        today_tr = true_ranges[-1]
        range_expansion = today_tr / atr_20d if atr_20d > 0 else 0.0

        consec_up_days = 0
        for i in range(len(bars) - 1, 0, -1):
            if bars[i].close > bars[i - 1].close:
                consec_up_days += 1
            else:
                break

        return {
            "volume_zscore_20d": volume_zscore_20d,
            "gap_pct": gap_pct,
            "range_expansion": range_expansion,
            "consec_up_days": float(consec_up_days),
        }


def _true_range(bar: PriceBar, prev_bar: PriceBar) -> float:
    return max(
        bar.high - bar.low,
        abs(bar.high - prev_bar.close),
        abs(bar.low - prev_bar.close),
    )
