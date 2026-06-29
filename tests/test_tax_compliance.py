from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from execution_layer.state_store import StateStore
from execution_layer.tax_compliance import WashSaleGuard


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "wash_sale_test.sqlite3")


def test_check_before_buy_allows_when_no_history(tmp_path: Path):
    guard = WashSaleGuard(_store(tmp_path), lookback_days=30)
    assert guard.check_before_buy("AAPL", date(2026, 6, 19)) is None


def test_check_before_buy_blocks_recent_loss_sale(tmp_path: Path):
    store = _store(tmp_path)
    store.record_realized_sale("AAPL", sale_date=date(2026, 6, 1), quantity=10, sale_price=90.0, cost_basis=100.0)
    guard = WashSaleGuard(store, lookback_days=30)

    violation = guard.check_before_buy("AAPL", today=date(2026, 6, 19))

    assert violation is not None
    assert violation.ticker == "AAPL"
    assert "wash sale" in violation.reason


def test_check_before_buy_ignores_loss_sale_outside_window(tmp_path: Path):
    store = _store(tmp_path)
    store.record_realized_sale("AAPL", sale_date=date(2026, 4, 1), quantity=10, sale_price=90.0, cost_basis=100.0)
    guard = WashSaleGuard(store, lookback_days=30)

    assert guard.check_before_buy("AAPL", today=date(2026, 6, 19)) is None


def test_check_before_buy_ignores_gain_sale(tmp_path: Path):
    store = _store(tmp_path)
    store.record_realized_sale("AAPL", sale_date=date(2026, 6, 10), quantity=10, sale_price=120.0, cost_basis=100.0)
    guard = WashSaleGuard(store, lookback_days=30)

    assert guard.check_before_buy("AAPL", today=date(2026, 6, 19)) is None


def test_warn_before_sell_returns_none_with_no_position(tmp_path: Path):
    guard = WashSaleGuard(_store(tmp_path), lookback_days=30)
    assert guard.warn_before_sell("AAPL", proposed_sale_price=90.0, today=date(2026, 6, 19)) is None


def test_warn_before_sell_returns_none_for_gain(tmp_path: Path):
    store = _store(tmp_path)
    store.upsert_position("AAPL", quantity=10, avg_entry_price=100.0, last_buy_at=date(2026, 6, 10).isoformat())
    guard = WashSaleGuard(store, lookback_days=30)

    assert guard.warn_before_sell("AAPL", proposed_sale_price=120.0, today=date(2026, 6, 19)) is None


def test_warn_before_sell_warns_on_loss_within_window(tmp_path: Path):
    store = _store(tmp_path)
    store.upsert_position("AAPL", quantity=10, avg_entry_price=100.0, last_buy_at=date(2026, 6, 10).isoformat())
    guard = WashSaleGuard(store, lookback_days=30)

    warning = guard.warn_before_sell("AAPL", proposed_sale_price=90.0, today=date(2026, 6, 19))

    assert warning is not None
    assert "AAPL" in warning


def test_warn_before_sell_silent_when_buy_outside_window(tmp_path: Path):
    store = _store(tmp_path)
    store.upsert_position("AAPL", quantity=10, avg_entry_price=100.0, last_buy_at=date(2026, 1, 1).isoformat())
    guard = WashSaleGuard(store, lookback_days=30)

    assert guard.warn_before_sell("AAPL", proposed_sale_price=90.0, today=date(2026, 6, 19)) is None


def test_record_realized_sale_computes_pnl(tmp_path: Path):
    store = _store(tmp_path)
    pnl = store.record_realized_sale("AAPL", sale_date=date(2026, 6, 19), quantity=10, sale_price=90.0, cost_basis=100.0)
    assert pnl == -100.0


def test_get_recent_loss_sales_excludes_gains(tmp_path: Path):
    store = _store(tmp_path)
    store.record_realized_sale("AAPL", sale_date=date(2026, 6, 19), quantity=10, sale_price=120.0, cost_basis=100.0)
    losses = store.get_recent_loss_sales("AAPL", since=date(2026, 6, 1))
    assert losses == []


def test_upsert_position_preserves_last_buy_at_on_sell_update(tmp_path: Path):
    store = _store(tmp_path)
    store.upsert_position("AAPL", quantity=10, avg_entry_price=100.0, last_buy_at="2026-06-10")
    store.upsert_position("AAPL", quantity=5, avg_entry_price=90.0)  # sell-driven update, no last_buy_at passed

    position = store.get_position("AAPL")
    assert position["last_buy_at"] == "2026-06-10"
    assert position["quantity"] == 5
