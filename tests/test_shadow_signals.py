from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import pytest

from analyst_layer.shadow_signals import run_provider_on_candidates
from data_layer.models import PriceBar, PriceSeries
from execution_layer.state_store import StateStore


def _snapshot() -> PriceSeries:
    return PriceSeries(
        symbol="AAPL", interval="1d",
        bars=[PriceBar(symbol="AAPL", timestamp=datetime(2026, 1, 2), open=100, high=101, low=99, close=100, volume=1000)],
    )


def _seed_candidate(store: StateStore, ticker: str = "AAPL") -> int:
    store.log_candidate(
        candidate_date=date.today(), strategy="thesis", ticker=ticker,
        llm_verdict="BUY", gate_result="APPROVED", traded=True,
    )
    cid = store.get_candidate_id(date.today(), "thesis", ticker)
    assert cid is not None
    return cid


class _OkProvider:
    name = "stub_ok"
    version = "v1"

    def compute(self, ticker, pit_snapshot):
        return {"metric_a": 0.5, "metric_b": -0.2}


class _EmptyProvider:
    name = "stub_empty"
    version = "v1"

    def compute(self, ticker, pit_snapshot):
        return None


class _RaisingProvider:
    name = "stub_raising"
    version = "v1"

    def compute(self, ticker, pit_snapshot):
        raise RuntimeError("boom")


class _HangingProvider:
    name = "stub_hanging"
    version = "v1"

    def compute(self, ticker, pit_snapshot):
        time.sleep(5)
        return {"metric_a": 1.0}


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "shadow_test.sqlite3")


def _get_rows(store: StateStore, candidate_id: int, signal_name: str) -> list[dict]:
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT metric_name, value, status, metric_as_of FROM signal_values WHERE candidate_id=? AND signal_name=?",
            (candidate_id, signal_name),
        ).fetchall()
    return [{"metric_name": r[0], "value": r[1], "status": r[2], "metric_as_of": r[3]} for r in rows]


def test_ok_provider_writes_correct_rows(store: StateStore):
    cid = _seed_candidate(store)
    counts = run_provider_on_candidates(
        _OkProvider(), [{"id": cid, "ticker": "AAPL", "candidate_date": date.today().isoformat()}],
        store, build_pit_snapshot=lambda t, d: _snapshot(),
        expected_metric_names=["metric_a", "metric_b"], timeout_s=5.0,
    )
    assert counts == {"ok": 1, "empty": 0, "failed": 0}
    rows = {r["metric_name"]: r for r in _get_rows(store, cid, "stub_ok")}
    assert rows["metric_a"]["value"] == 0.5
    assert rows["metric_a"]["status"] == "ok"
    assert rows["metric_b"]["value"] == -0.2


def test_provider_without_get_metric_as_of_defaults_to_candidate_date(store: StateStore):
    """Kronos (and any price-history-only provider) has no get_metric_as_of
    — candidate_date is the correct as-of by construction, zero staleness.
    """
    cid = _seed_candidate(store)
    today_str = date.today().isoformat()
    run_provider_on_candidates(
        _OkProvider(), [{"id": cid, "ticker": "AAPL", "candidate_date": today_str}],
        store, build_pit_snapshot=lambda t, d: _snapshot(),
        expected_metric_names=["metric_a", "metric_b"], timeout_s=5.0,
    )
    rows = {r["metric_name"]: r for r in _get_rows(store, cid, "stub_ok")}
    assert rows["metric_a"]["metric_as_of"] == today_str


class _CustomAsOfProvider:
    name = "stub_custom_as_of"
    version = "v1"

    def compute(self, ticker, pit_snapshot):
        return {"metric_a": 1.0}

    def get_metric_as_of(self, ticker, candidate_date, result):
        return "2020-01-01"  # deliberately different from candidate_date


def test_provider_with_get_metric_as_of_uses_custom_value(store: StateStore):
    """A provider like short_interest, whose data reflects a settlement
    date earlier than candidate_date, must have that staleness recorded —
    not silently defaulted to candidate_date.
    """
    cid = _seed_candidate(store)
    run_provider_on_candidates(
        _CustomAsOfProvider(), [{"id": cid, "ticker": "AAPL", "candidate_date": date.today().isoformat()}],
        store, build_pit_snapshot=lambda t, d: _snapshot(),
        expected_metric_names=["metric_a"], timeout_s=5.0,
    )
    rows = _get_rows(store, cid, "stub_custom_as_of")
    assert rows[0]["metric_as_of"] == "2020-01-01"


def test_provider_returning_none_is_empty_not_failed(store: StateStore):
    cid = _seed_candidate(store)
    counts = run_provider_on_candidates(
        _EmptyProvider(), [{"id": cid, "ticker": "AAPL", "candidate_date": date.today().isoformat()}],
        store, build_pit_snapshot=lambda t, d: _snapshot(),
        expected_metric_names=["metric_a"], timeout_s=5.0,
    )
    assert counts == {"ok": 0, "empty": 1, "failed": 0}
    rows = _get_rows(store, cid, "stub_empty")
    assert rows[0]["status"] == "empty"
    assert rows[0]["value"] is None


def test_provider_raising_is_failed_with_null_and_does_not_raise(store: StateStore):
    cid = _seed_candidate(store)
    counts = run_provider_on_candidates(
        _RaisingProvider(), [{"id": cid, "ticker": "AAPL", "candidate_date": date.today().isoformat()}],
        store, build_pit_snapshot=lambda t, d: _snapshot(),
        expected_metric_names=["metric_a"], timeout_s=5.0,
    )
    assert counts == {"ok": 0, "empty": 0, "failed": 1}
    rows = _get_rows(store, cid, "stub_raising")
    assert rows[0]["status"] == "failed"
    assert rows[0]["value"] is None


def test_no_pit_snapshot_is_empty(store: StateStore):
    cid = _seed_candidate(store)
    counts = run_provider_on_candidates(
        _OkProvider(), [{"id": cid, "ticker": "AAPL", "candidate_date": date.today().isoformat()}],
        store, build_pit_snapshot=lambda t, d: None,
        expected_metric_names=["metric_a", "metric_b"], timeout_s=5.0,
    )
    assert counts == {"ok": 0, "empty": 1, "failed": 0}


def test_hanging_provider_times_out_and_does_not_block_next_candidate(store: StateStore):
    """Regression test: a hung provider must be marked Failed within timeout_s
    and must NOT prevent the next candidate from being processed.
    """
    cid1 = _seed_candidate(store, "AAPL")
    cid2 = _seed_candidate(store, "MSFT")

    t0 = time.monotonic()
    counts = run_provider_on_candidates(
        _HangingProvider(),
        [
            {"id": cid1, "ticker": "AAPL", "candidate_date": date.today().isoformat()},
            {"id": cid2, "ticker": "MSFT", "candidate_date": date.today().isoformat()},
        ],
        store, build_pit_snapshot=lambda t, d: _snapshot(),
        expected_metric_names=["metric_a"], timeout_s=1.0,
    )
    elapsed = time.monotonic() - t0

    assert counts == {"ok": 0, "empty": 0, "failed": 2}
    assert elapsed < 5.0, "both candidates should time out at ~1s each, not wait for the full 5s sleep"
    for cid in (cid1, cid2):
        rows = _get_rows(store, cid, "stub_hanging")
        assert rows[0]["status"] == "failed"
        assert rows[0]["value"] is None


def test_lesson_store_and_prompt_construction_never_reference_signal_values():
    """Structural guard: shadow signals must never quietly become a live
    input without an explicit promotion decision. If a future edit joins
    signal_values into lesson_store or any prompt-construction code, this
    test should catch it.
    """
    import ast

    repo_root = Path(__file__).resolve().parent.parent
    suspects = [
        repo_root / "analyst_layer" / "lesson_store.py",
        repo_root / "analyst_layer" / "graph.py",
        repo_root / "analyst_layer" / "vol_graph.py",
    ]
    for path in suspects:
        if not path.exists():
            continue
        source = path.read_text()
        assert "signal_values" not in source, f"{path} references signal_values — shadow signal leaking into a live path"


def test_shadow_signal_modules_never_import_execution_layer_protection():
    """Guard against scope creep: shadow_signals.py / kronos_provider.py must
    never import protection_plane or touch order_intents/breaker_state.

    Uses ast for the import check, not string matching — a plain substring
    search on "protection_plane" would false-positive the moment a docstring
    explains this exact constraint in prose (this bit an equivalent check in
    test_alpaca_reference_client.py — see that test's docstring).
    """
    import ast

    repo_root = Path(__file__).resolve().parent.parent
    for filename in ("shadow_signals.py", "kronos_provider.py", "short_interest_provider.py"):
        path = repo_root / "analyst_layer" / filename
        source = path.read_text()
        tree = ast.parse(source)

        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        # state_store IS expected (that's how a signal persists its measurement) —
        # only protection_plane specifically is the forbidden import here.
        assert not any("protection_plane" in m for m in imported_modules), f"{filename} imports protection_plane"
        # order_intents/breaker_state are table names, not importable modules —
        # a real string check still makes sense for these (no AST equivalent).
        assert "order_intents" not in source, f"{filename} references order_intents"
        assert "breaker_state" not in source, f"{filename} references breaker_state"
