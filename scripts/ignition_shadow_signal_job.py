#!/usr/bin/env python3
"""Ignition detector shadow signal batch job.

Manually invoked, not scheduled — same measurement-only philosophy as
scripts/kronos_shadow_signal_job.py. Pure price/volume function — no new
data source, no external API, no caching needed.

Usage:
    source .venv/bin/activate
    python scripts/ignition_shadow_signal_job.py
    python scripts/ignition_shadow_signal_job.py --lookback-days 14
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from config.settings import get_settings
from data_layer.exceptions import DataLayerError
from data_layer.models import PriceSeries
from data_layer.openbb_client import OpenBBDataClient
from analyst_layer.ignition_provider import IgnitionSignalProvider
from analyst_layer.shadow_signals import run_provider_on_candidates
from execution_layer.state_store import StateStore

_EXPECTED_METRICS = ["volume_zscore_20d", "gap_pct", "range_expansion", "consec_up_days"]
_LOOKBACK_CALENDAR_DAYS = 45  # enough calendar days to reliably yield 21 trading bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=30, help="how far back to look for un-enriched candidates")
    args = parser.parse_args()

    settings = get_settings()
    state_store = StateStore(settings.state_db_path)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)
    provider = IgnitionSignalProvider()

    candidates = state_store.get_candidates_needing_signal(
        provider.name, provider.version, lookback_days=args.lookback_days
    )
    logger.info("%d candidate(s) need %s/%s", len(candidates), provider.name, provider.version)
    if not candidates:
        return

    def build_pit_snapshot(ticker: str, candidate_date_str: str) -> PriceSeries | None:
        candidate_date = date.fromisoformat(candidate_date_str)
        start = candidate_date - timedelta(days=_LOOKBACK_CALENDAR_DAYS)
        try:
            series = data_client.get_price_history(ticker, start_date=start, end_date=candidate_date)
        except DataLayerError as exc:
            logger.debug("%s: price history fetch failed (%s) — Empty", ticker, exc)
            return None
        if not series.bars:
            return None
        return series

    counts = run_provider_on_candidates(
        provider=provider,
        candidates=candidates,
        state_store=state_store,
        build_pit_snapshot=build_pit_snapshot,
        expected_metric_names=_EXPECTED_METRICS,
        timeout_s=10.0,
    )
    logger.info("Done: %d ok, %d empty, %d failed", counts["ok"], counts["empty"], counts["failed"])


if __name__ == "__main__":
    sys.exit(main())
