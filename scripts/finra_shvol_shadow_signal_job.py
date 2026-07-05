#!/usr/bin/env python3
"""FINRA daily short-sale volume shadow signal batch job.

Manually invoked, not scheduled — same measurement-only philosophy as
scripts/kronos_shadow_signal_job.py.

Usage:
    source .venv/bin/activate
    python scripts/finra_shvol_shadow_signal_job.py
    python scripts/finra_shvol_shadow_signal_job.py --lookback-days 14

Reads candidates via state_store.get_candidates_needing_signal — only scores
candidates that don't already have a finra_short_volume/<version> row.
build_pit_snapshot fetches real price history ending at candidate_date (same
convention as kronos_shadow_signal_job.py) so the provider has a real,
point-in-time anchor to query FINRA's historical daily files against — see
analyst_layer/finra_short_volume_provider.py for why this signal, unlike
short interest, can be made genuinely PIT-clean.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from config.settings import get_settings
from data_layer.exceptions import DataLayerError
from data_layer.finra_shvol_client import FinraShortVolumeClient
from data_layer.models import PriceSeries
from data_layer.openbb_client import OpenBBDataClient
from analyst_layer.finra_short_volume_provider import FinraShortVolumeSignalProvider
from analyst_layer.shadow_signals import run_provider_on_candidates
from execution_layer.state_store import StateStore

_EXPECTED_METRICS = ["short_vol_ratio", "short_vol_ratio_5d_avg", "short_vol_ratio_zscore_20d"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=None, help="how far back to look for un-enriched candidates")
    args = parser.parse_args()

    settings = get_settings()
    lookback_days = args.lookback_days if args.lookback_days is not None else settings.finra_shvol_job_lookback_days

    state_store = StateStore(settings.state_db_path)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)
    finra_client = FinraShortVolumeClient(settings.finra_shvol_cache_dir)
    provider = FinraShortVolumeSignalProvider(finra_client)

    candidates = state_store.get_candidates_needing_signal(
        provider.name, provider.version, lookback_days=lookback_days
    )
    logger.info("%d candidate(s) need %s/%s (lookback=%dd)", len(candidates), provider.name, provider.version, lookback_days)
    if not candidates:
        return

    def build_pit_snapshot(ticker: str, candidate_date_str: str) -> PriceSeries | None:
        candidate_date = date.fromisoformat(candidate_date_str)
        # Enough buffer beyond finra_shvol_lookback_days for weekends/holidays.
        start = candidate_date - timedelta(days=int(settings.finra_shvol_lookback_days * 1.6) + 10)
        try:
            series = data_client.get_price_history(ticker, start_date=start, end_date=candidate_date)
        except DataLayerError as exc:
            logger.debug("%s: price history fetch failed (%s) — Empty", ticker, exc)
            return None
        if not series.bars:
            return None
        return series

    t0 = time.time()
    counts = run_provider_on_candidates(
        provider=provider,
        candidates=candidates,
        state_store=state_store,
        build_pit_snapshot=build_pit_snapshot,
        expected_metric_names=_EXPECTED_METRICS,
        timeout_s=60.0,
    )
    elapsed = time.time() - t0
    logger.info(
        "Done: %d ok, %d empty, %d failed — %.1fs total",
        counts["ok"], counts["empty"], counts["failed"], elapsed,
    )


if __name__ == "__main__":
    sys.exit(main())
