from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from execution_layer.state_store import StateStore
from scripts.short_interest_shadow_signal_job import _log_easy_to_borrow_transitions


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "transitions_test.sqlite3")


def _seed_candidate_with_etb(store: StateStore, ticker: str, etb_value: float, provider) -> dict:
    store.log_candidate(
        candidate_date=date.today(), strategy="thesis", ticker=ticker,
        llm_verdict="BUY", gate_result="APPROVED", traded=True,
    )
    cid = store.get_candidate_id(date.today(), "thesis", ticker)
    store.record_signal_values(
        cid, provider.name, provider.version, {"easy_to_borrow": etb_value}, status="ok",
    )
    return {"id": cid, "ticker": ticker, "candidate_date": date.today().isoformat()}


def _provider():
    p = MagicMock()
    p.name = "short_interest"
    p.version = "short-interest-v1"
    return p


def test_no_event_on_first_observation(store: StateStore):
    """The very first time a ticker is seen, there's no 'prior' value to
    compare against — must not fire a spurious transition event.
    """
    provider = _provider()
    candidate = _seed_candidate_with_etb(store, "GME", 1.0, provider)

    _log_easy_to_borrow_transitions(store, provider, [candidate])

    events = store.get_events(event_type_like="easy_to_borrow_flip", limit=10)
    assert len(events) == 0
    assert store.get_ticker_signal_state("GME", "short_interest", "easy_to_borrow") == "true"


def test_no_event_when_value_unchanged(store: StateStore):
    provider = _provider()
    store.set_ticker_signal_state("GME", "short_interest", "easy_to_borrow", "true")
    candidate = _seed_candidate_with_etb(store, "GME", 1.0, provider)

    _log_easy_to_borrow_transitions(store, provider, [candidate])

    events = store.get_events(event_type_like="easy_to_borrow_flip", limit=10)
    assert len(events) == 0


def test_event_fires_on_true_to_false_flip(store: StateStore):
    provider = _provider()
    store.set_ticker_signal_state("GME", "short_interest", "easy_to_borrow", "true")
    candidate = _seed_candidate_with_etb(store, "GME", 0.0, provider)

    _log_easy_to_borrow_transitions(store, provider, [candidate])

    events = store.get_events(event_type_like="easy_to_borrow_flip", limit=10)
    assert len(events) == 1
    assert "GME" in events[0]["detail"]
    assert "true -> false" in events[0]["detail"]
    assert store.get_ticker_signal_state("GME", "short_interest", "easy_to_borrow") == "false"


def test_event_fires_on_false_to_true_flip(store: StateStore):
    provider = _provider()
    store.set_ticker_signal_state("GME", "short_interest", "easy_to_borrow", "false")
    candidate = _seed_candidate_with_etb(store, "GME", 1.0, provider)

    _log_easy_to_borrow_transitions(store, provider, [candidate])

    events = store.get_events(event_type_like="easy_to_borrow_flip", limit=10)
    assert len(events) == 1
    assert "false -> true" in events[0]["detail"]


def test_missing_easy_to_borrow_value_is_skipped_gracefully(store: StateStore):
    """A candidate whose easy_to_borrow metric wasn't recorded (e.g. Alpaca
    lookup failed for that ticker) must not crash the transition check.
    """
    provider = _provider()
    store.log_candidate(
        candidate_date=date.today(), strategy="thesis", ticker="NOETB",
        llm_verdict="BUY", gate_result="APPROVED", traded=True,
    )
    cid = store.get_candidate_id(date.today(), "thesis", "NOETB")
    candidate = {"id": cid, "ticker": "NOETB", "candidate_date": date.today().isoformat()}

    _log_easy_to_borrow_transitions(store, provider, [candidate])  # must not raise

    events = store.get_events(event_type_like="easy_to_borrow_flip", limit=10)
    assert len(events) == 0
