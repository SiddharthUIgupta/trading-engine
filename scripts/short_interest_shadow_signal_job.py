#!/usr/bin/env python3
"""Short-interest / squeeze-potential shadow signal batch job.

Manually invoked, not scheduled — same measurement-only, off-the-hot-path
philosophy as scripts/kronos_shadow_signal_job.py.

Usage:
    source .venv/bin/activate
    python scripts/short_interest_shadow_signal_job.py
    python scripts/short_interest_shadow_signal_job.py --lookback-days 14

Reads candidates via state_store.get_candidates_needing_signal — only scores
candidates that don't already have a short_interest/<version> row, so
re-running this job is always safe and incremental. Defaults to a 7-day
lookback (not Kronos's 30) — short interest data has no free point-in-time
history, only a ~20-day-stale "current" snapshot, so this job is only
meaningful for reasonably fresh candidates. See config/settings.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from config.settings import get_settings
from data_layer.alpaca_reference_client import AlpacaAssetReferenceClient
from data_layer.models import PriceBar, PriceSeries
from data_layer.openbb_client import OpenBBDataClient
from analyst_layer.short_interest_provider import ShortInterestSignalProvider
from analyst_layer.shadow_signals import run_provider_on_candidates
from execution_layer.state_store import StateStore

_EXPECTED_METRICS = ["short_percent_of_float", "days_to_cover", "short_interest_mom_change", "shortable", "easy_to_borrow"]

# This provider doesn't use pit_snapshot at all (see short_interest_provider.py
# docstring) — a trivial single-bar placeholder satisfies the SignalProvider
# harness's non-None check without a real price-history fetch.
_PLACEHOLDER_SNAPSHOT = PriceSeries(
    symbol="_", interval="1d",
    bars=[PriceBar(symbol="_", timestamp=datetime(2000, 1, 1), open=1.0, high=1.0, low=1.0, close=1.0, volume=0)],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=None, help="how far back to look for un-enriched candidates")
    args = parser.parse_args()

    settings = get_settings()
    lookback_days = args.lookback_days if args.lookback_days is not None else settings.short_interest_job_lookback_days

    state_store = StateStore(settings.state_db_path)
    openbb_client = OpenBBDataClient(pat=settings.openbb_pat or None)
    alpaca_client = AlpacaAssetReferenceClient(settings)
    provider = ShortInterestSignalProvider(openbb_client, alpaca_client)

    candidates = state_store.get_candidates_needing_signal(
        provider.name, provider.version, lookback_days=lookback_days
    )
    logger.info("%d candidate(s) need %s/%s (lookback=%dd)", len(candidates), provider.name, provider.version, lookback_days)
    if not candidates:
        return

    t0 = time.time()
    counts = run_provider_on_candidates(
        provider=provider,
        candidates=candidates,
        state_store=state_store,
        build_pit_snapshot=lambda ticker, candidate_date: _PLACEHOLDER_SNAPSHOT,
        expected_metric_names=_EXPECTED_METRICS,
        timeout_s=30.0,
    )
    elapsed = time.time() - t0
    logger.info(
        "Done: %d ok, %d empty, %d failed — %.1fs total",
        counts["ok"], counts["empty"], counts["failed"], elapsed,
    )

    _log_easy_to_borrow_transitions(state_store, provider, candidates)


def _log_easy_to_borrow_transitions(
    state_store: StateStore, provider: ShortInterestSignalProvider, candidates: list[dict]
) -> None:
    """The flip is the signal, not the level — a ticker sitting at
    easy_to_borrow=False for months is old news by the time this job would
    surface it; the moment it CHANGES is what's actionable. Tracked via
    signal_ticker_state (per-ticker, not per-candidate — see
    state_store.py's docstring on why signal_values can't answer "the last
    value for this ticker" unambiguously: the same ticker+date can have
    multiple candidate_id rows across strategies).
    """
    with state_store._connect() as conn:
        for candidate in candidates:
            row = conn.execute(
                "SELECT value FROM signal_values WHERE candidate_id=? AND signal_name=? AND signal_version=? "
                "AND metric_name='easy_to_borrow' AND status='ok'",
                (candidate["id"], provider.name, provider.version),
            ).fetchone()
            if row is None or row[0] is None:
                continue

            ticker = candidate["ticker"]
            new_value = "true" if row[0] == 1.0 else "false"
            old_value = state_store.get_ticker_signal_state(ticker, provider.name, "easy_to_borrow")

            if old_value is not None and old_value != new_value:
                state_store.record_event(
                    event_type="easy_to_borrow_flip",
                    detail=f"{ticker}: {old_value} -> {new_value}",
                )
                logger.info("%s: easy_to_borrow flipped %s -> %s", ticker, old_value, new_value)

            state_store.set_ticker_signal_state(ticker, provider.name, "easy_to_borrow", new_value)


if __name__ == "__main__":
    sys.exit(main())
