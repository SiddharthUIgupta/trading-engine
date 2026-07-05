"""Shadow-signal measurement pipeline — Alpha Plane, but never on the decision path.

Every provider here scores already-committed candidate ledger rows (the same
candidates thesis/recovery/gap/swing/news already logged via
state_store.log_candidate, after that day's trade decision is done). Nothing
in this module or its providers can influence a trade: it only ever reads
candidates that were logged before this code even runs, and it only ever
writes to signal_values — a table nothing in the decision path reads back.

A signal is shadow, promoted, or deleted — see CLAUDE.md "Signal lifecycle".
Promotion (using a signal in the risk gate) is a separate, explicit, future
task; this module builds no gating hooks.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Protocol

from data_layer.models import PriceSeries
from execution_layer.state_store import StateStore

logger = logging.getLogger(__name__)


class SignalProvider(Protocol):
    """A shadow signal source. compute() returning None means "nothing to
    say" (Empty) — that's not an error and must not be logged as Failed.
    Raising, or exceeding the harness's timeout, is Failed. Either way the
    harness — not the provider — is responsible for making sure one
    candidate's problem never blocks or delays the next.
    """

    name: str
    version: str

    def compute(self, ticker: str, pit_snapshot: PriceSeries) -> dict[str, float] | None: ...


def run_provider_on_candidates(
    provider: SignalProvider,
    candidates: list[dict],
    state_store: StateStore,
    build_pit_snapshot,
    expected_metric_names: list[str],
    timeout_s: float = 120.0,
) -> dict[str, int]:
    """Runs `provider` against every candidate row, writing results to
    signal_values. Never raises — a provider that raises or times out is
    logged as Failed (NULL values written for every expected metric) and
    processing moves on to the next candidate.

    `build_pit_snapshot(ticker, candidate_date_str) -> PriceSeries | None` is
    injected so this harness has no data_layer/network dependency of its own;
    if it returns None (e.g. no price history available), that candidate is
    Empty, not Failed.

    Returns counts: {"ok": n, "empty": n, "failed": n}.
    """
    counts = {"ok": 0, "empty": 0, "failed": 0}

    for candidate in candidates:
        ticker = candidate["ticker"]
        candidate_id = candidate["id"]
        candidate_date = candidate["candidate_date"]

        pit_snapshot = build_pit_snapshot(ticker, candidate_date)
        if pit_snapshot is None:
            counts["empty"] += 1
            state_store.record_signal_values(
                candidate_id, provider.name, provider.version,
                dict.fromkeys(expected_metric_names, None), status="empty",
            )
            continue

        # A fresh, un-managed executor per candidate — deliberately not a
        # `with` block. ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
        # which blocks until the submitted work item actually finishes even
        # after future.result(timeout=...) has already raised — i.e. it would
        # silently turn a 1s timeout into the full hang duration. shutdown()
        # here is wait=False: a hung thread is abandoned, not waited on.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(provider.compute, ticker, pit_snapshot)
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            logger.warning("%s: %s provider timed out after %.0fs — marking Failed", ticker, provider.name, timeout_s)
            counts["failed"] += 1
            state_store.record_signal_values(
                candidate_id, provider.name, provider.version,
                dict.fromkeys(expected_metric_names, None), status="failed",
            )
            executor.shutdown(wait=False)
            continue
        except Exception as exc:  # noqa: BLE001 — this is the harness boundary, not a decision path; a
            # provider must never be able to take down the batch job by raising.
            logger.warning("%s: %s provider raised — marking Failed: %s", ticker, provider.name, exc)
            counts["failed"] += 1
            state_store.record_signal_values(
                candidate_id, provider.name, provider.version,
                dict.fromkeys(expected_metric_names, None), status="failed",
            )
            executor.shutdown(wait=False)
            continue
        else:
            executor.shutdown(wait=False)

        if result is None:
            counts["empty"] += 1
            state_store.record_signal_values(
                candidate_id, provider.name, provider.version,
                dict.fromkeys(expected_metric_names, None), status="empty",
            )
            continue

        counts["ok"] += 1
        state_store.record_signal_values(candidate_id, provider.name, provider.version, result, status="ok")

    return counts
