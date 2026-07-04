from __future__ import annotations

from pathlib import Path

from execution_layer.state_store import StateStore


def test_connect_uses_wal_and_raised_busy_timeout(tmp_path: Path):
    """Alpha and Protection are separate processes writing the same SQLite
    file. Python's sqlite3.connect() defaults to a 5-second busy timeout and
    the rollback ("delete") journal mode — under real write-write contention
    between the two daemons (e.g. Protection consuming an intent while Alpha
    is mid-write), a collision that outlasts 5 seconds raises
    `database is locked`. Most call sites (e.g. mark_order_intent_processed
    inside consume_order_intents) don't catch OperationalError, so a single
    collision could kill an entire intraday_monitoring tick, exit checks
    included.

    Asserts the actual connection configuration directly via PRAGMA rather
    than timing a real lock — deterministic and fast, and precisely targets
    what changed: busy_timeout raised from the 5000ms default to 30000ms,
    and journal_mode switched from the default 'delete' to 'wal' (which also
    lets Alpha's reads proceed without blocking on Protection's writes).
    """
    store = StateStore(tmp_path / "concurrency_test.sqlite3")
    conn = store._connect()
    try:
        busy_timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    assert busy_timeout_ms == 30_000, f"expected 30s busy timeout, got {busy_timeout_ms}ms"
    assert journal_mode == "wal", f"expected WAL journal mode, got {journal_mode!r}"
