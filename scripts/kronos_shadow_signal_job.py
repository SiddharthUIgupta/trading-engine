#!/usr/bin/env python3
"""Kronos-small shadow signal batch job — scores already-logged candidates.

Manually invoked, not scheduled — this is a measurement job, not automation,
and it deliberately does not run inside the long-lived main_alpha.py daemon:
torch's memory footprint (measured ~1-2GB resident) is released when this
short-lived process exits, which matters on a Pi with tight RAM headroom.

Usage:
    source .venv/bin/activate
    python scripts/kronos_shadow_signal_job.py
    python scripts/kronos_shadow_signal_job.py --lookback-days 60

Reads candidates via state_store.get_candidates_needing_signal (same
NOT-EXISTS pattern as the existing forward-return backfill) — only scores
candidates that don't already have a kronos_small/<version> row, so re-running
this job is always safe and incremental.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from config.settings import get_settings
from data_layer.exceptions import DataLayerError
from data_layer.models import PriceSeries
from data_layer.openbb_client import OpenBBDataClient
from analyst_layer.kronos_provider import KronosSignalProvider
from analyst_layer.shadow_signals import run_provider_on_candidates
from execution_layer.state_store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=30, help="how far back to look for un-enriched candidates")
    args = parser.parse_args()

    settings = get_settings()
    state_store = StateStore(settings.state_db_path)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

    logger.info("Loading Kronos-small (one-time cost per process run)...")
    t0 = time.time()
    provider = KronosSignalProvider(settings)
    logger.info("Model load: %.1fs", time.time() - t0)

    candidates = state_store.get_candidates_needing_signal(
        provider.name, provider.version, lookback_days=args.lookback_days
    )
    logger.info("%d candidate(s) need %s/%s", len(candidates), provider.name, provider.version)
    if not candidates:
        return

    def build_pit_snapshot(ticker: str, candidate_date_str: str) -> PriceSeries | None:
        candidate_date = date.fromisoformat(candidate_date_str)
        # Extra buffer beyond kronos_lookback_bars so weekends/holidays don't
        # starve the tail end of the window.
        start = candidate_date - timedelta(days=int(settings.kronos_lookback_bars * 1.6) + 10)
        try:
            series = data_client.get_price_history(ticker, start_date=start, end_date=candidate_date)
        except DataLayerError as exc:
            logger.debug("%s: price history fetch failed (%s) — Empty", ticker, exc)
            return None
        if len(series.bars) < settings.kronos_lookback_bars:
            return None
        return series

    t0 = time.time()
    counts = run_provider_on_candidates(
        provider=provider,
        candidates=candidates,
        state_store=state_store,
        build_pit_snapshot=build_pit_snapshot,
        expected_metric_names=["p_touch_win", "med_ret_21d", "path_dispersion"],
        timeout_s=settings.kronos_inference_timeout_s,
    )
    elapsed = time.time() - t0
    logger.info(
        "Done: %d ok, %d empty, %d failed — %.1fs total (%.1fs/candidate avg)",
        counts["ok"], counts["empty"], counts["failed"],
        elapsed, elapsed / len(candidates) if candidates else 0.0,
    )


if __name__ == "__main__":
    sys.exit(main())
