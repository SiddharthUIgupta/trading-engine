from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from execution_layer.state_store import StateStore, CURRENT_OPTIONS_STRATEGY_VERSION


def test_new_option_sale_defaults_to_current_strategy_version(tmp_path: Path):
    store = StateStore(tmp_path / "sv_test.sqlite3")
    store.record_realized_option_sale(
        contract_symbol="AAPL260731C00200000", underlying_symbol="AAPL",
        sale_date=date.today(), contracts=1, sale_price=5.0, cost_basis=3.0,
    )
    rows = store.get_all_realized_option_sales()
    assert rows[0]["strategy_version"] == CURRENT_OPTIONS_STRATEGY_VERSION


def test_explicit_strategy_version_overrides_default(tmp_path: Path):
    store = StateStore(tmp_path / "sv_test2.sqlite3")
    store.record_realized_option_sale(
        contract_symbol="AAPL260731C00200000", underlying_symbol="AAPL",
        sale_date=date.today(), contracts=1, sale_price=5.0, cost_basis=3.0,
        strategy_version="orb_options_v1",
    )
    rows = store.get_all_realized_option_sales()
    assert rows[0]["strategy_version"] == "orb_options_v1"


def test_pnl_by_strategy_version_splits_correctly(tmp_path: Path):
    store = StateStore(tmp_path / "sv_test3.sqlite3")
    store.record_realized_option_sale(
        contract_symbol="A", underlying_symbol="A", sale_date=date.today(),
        contracts=1, sale_price=2.0, cost_basis=1.0, strategy_version="orb_options_v1",
    )
    store.record_realized_option_sale(
        contract_symbol="B", underlying_symbol="B", sale_date=date.today(),
        contracts=1, sale_price=1.0, cost_basis=2.0, strategy_version="orb_options_v1",
    )
    store.record_realized_option_sale(
        contract_symbol="C", underlying_symbol="C", sale_date=date.today(),
        contracts=1, sale_price=5.0, cost_basis=1.0,  # defaults to current version
    )

    results = {r["strategy_version"]: r for r in store.get_realized_option_pnl_by_strategy_version()}

    assert results["orb_options_v1"]["trade_count"] == 2
    assert results["orb_options_v1"]["total_realized_pnl"] == (2.0 - 1.0) * 100 + (1.0 - 2.0) * 100
    assert results["orb_options_v1"]["is_current"] is False

    assert results[CURRENT_OPTIONS_STRATEGY_VERSION]["trade_count"] == 1
    assert results[CURRENT_OPTIONS_STRATEGY_VERSION]["is_current"] is True


def test_backfill_splits_on_exact_commit_timestamp_not_guessing(tmp_path: Path):
    """Regression test for the backfill script: created_at (full timestamp)
    must be used for the split, not sale_date (date-only) — same-day trades
    before/after the fix commit must land on opposite sides.
    """
    from scripts.backfill_strategy_version import backfill, _SPLIT_UTC, _PRE_FIX_VERSION, _POST_FIX_VERSION

    store = StateStore(tmp_path / "sv_test4.sqlite3")
    with store._connect() as conn:
        # Two rows on the exact same sale_date, straddling the split instant.
        before_ts = _SPLIT_UTC.replace(hour=_SPLIT_UTC.hour - 1).isoformat().replace("+00:00", "")
        after_ts = _SPLIT_UTC.replace(hour=_SPLIT_UTC.hour + 1).isoformat().replace("+00:00", "")
        conn.execute(
            "INSERT INTO realized_option_sales (contract_symbol, underlying_symbol, sale_date, contracts, "
            "sale_price, cost_basis, realized_pnl, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("BEFORE", "X", _SPLIT_UTC.date().isoformat(), 1, 1.0, 1.0, 0.0, before_ts),
        )
        conn.execute(
            "INSERT INTO realized_option_sales (contract_symbol, underlying_symbol, sale_date, contracts, "
            "sale_price, cost_basis, realized_pnl, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("AFTER", "X", _SPLIT_UTC.date().isoformat(), 1, 1.0, 1.0, 0.0, after_ts),
        )
        conn.commit()

    counts = backfill(store)
    assert counts == {"pre": 1, "post": 1}

    rows = {r["contract_symbol"]: r for r in store.get_all_realized_option_sales()}
    assert rows["BEFORE"]["strategy_version"] == _PRE_FIX_VERSION
    assert rows["AFTER"]["strategy_version"] == _POST_FIX_VERSION


def test_backfill_dry_run_writes_nothing(tmp_path: Path):
    from scripts.backfill_strategy_version import backfill, _SPLIT_UTC

    store = StateStore(tmp_path / "sv_test4b.sqlite3")
    with store._connect() as conn:
        ts = _SPLIT_UTC.replace(hour=_SPLIT_UTC.hour - 1).isoformat().replace("+00:00", "")
        conn.execute(
            "INSERT INTO realized_option_sales (contract_symbol, underlying_symbol, sale_date, contracts, "
            "sale_price, cost_basis, realized_pnl, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("X", "X", _SPLIT_UTC.date().isoformat(), 1, 1.0, 1.0, 0.0, ts),
        )
        conn.commit()

    counts = backfill(store, dry_run=True)
    assert counts == {"pre": 1, "post": 0}
    rows = store.get_all_realized_option_sales()
    assert rows[0]["strategy_version"] is None, "dry_run must not write anything"


def test_backfill_is_idempotent_skips_already_versioned_rows(tmp_path: Path):
    from scripts.backfill_strategy_version import backfill

    store = StateStore(tmp_path / "sv_test5.sqlite3")
    store.record_realized_option_sale(
        contract_symbol="A", underlying_symbol="A", sale_date=date.today(),
        contracts=1, sale_price=1.0, cost_basis=1.0, strategy_version="orb_options_v1",
    )
    counts = backfill(store)
    assert counts == {"pre": 0, "post": 0}, "already-versioned row must not be touched"
