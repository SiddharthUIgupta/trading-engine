#!/usr/bin/env python3
"""One-time backfill: tag existing realized_option_sales rows with
strategy_version, split on the actual options-position-cap fix commit —
not guessed.

Split point: commit fb9f6d1 "Add options position cap + floating refresh
button in dashboard", 2026-06-29 10:50:15 -0700 = 2026-06-29T17:50:15 UTC.
Before this commit, the ORB options track had no position cap and
accumulated 30 simultaneous positions (see ARCHITECTURE.md "Realized
Performance" — this is the documented root cause of the options bucket's
-$8,647 realized loss). This is the only documented strategy-relevant risk
change to the options bucket so far.

created_at (full timestamp) is used for the split, not sale_date (date-only)
— 21 of the 42 existing rows share the exact fix date (2026-06-29), so
date-only granularity can't tell pre-fix from post-fix trades that same day.

Safe to re-run: only backfills rows where strategy_version IS NULL.

Usage:
    source .venv/bin/activate
    python scripts/backfill_strategy_version.py
    python scripts/backfill_strategy_version.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution_layer.state_store import StateStore

_SPLIT_UTC = datetime(2026, 6, 29, 17, 50, 15, tzinfo=timezone.utc)
_PRE_FIX_VERSION = "orb_options_v1"
_POST_FIX_VERSION = "orb_options_v2"


def backfill(store: StateStore, dry_run: bool = False) -> dict:
    """Splits un-versioned realized_option_sales rows on _SPLIT_UTC using
    each row's full created_at timestamp. Returns {"pre": n, "post": n}.
    Safe to call repeatedly — only touches strategy_version IS NULL rows.
    """
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at FROM realized_option_sales WHERE strategy_version IS NULL"
        ).fetchall()

        pre_ids, post_ids = [], []
        for row_id, created_at in rows:
            ts = datetime.fromisoformat(created_at).replace(tzinfo=timezone.utc)
            (pre_ids if ts < _SPLIT_UTC else post_ids).append(row_id)

        if not dry_run:
            if pre_ids:
                conn.executemany(
                    "UPDATE realized_option_sales SET strategy_version=? WHERE id=?",
                    [(_PRE_FIX_VERSION, i) for i in pre_ids],
                )
            if post_ids:
                conn.executemany(
                    "UPDATE realized_option_sales SET strategy_version=? WHERE id=?",
                    [(_POST_FIX_VERSION, i) for i in post_ids],
                )
            conn.commit()

    return {"pre": len(pre_ids), "post": len(post_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()

    from config.settings import get_settings
    settings = get_settings()
    store = StateStore(settings.state_db_path)

    counts = backfill(store, dry_run=args.dry_run)
    if counts["pre"] + counts["post"] == 0:
        print("No un-versioned rows — nothing to backfill.")
        return
    print(f"{counts['pre'] + counts['post']} un-versioned rows: {counts['pre']} -> {_PRE_FIX_VERSION}, {counts['post']} -> {_POST_FIX_VERSION}")
    if args.dry_run:
        print("--dry-run: no changes written.")
    else:
        print("Backfill complete.")


if __name__ == "__main__":
    sys.exit(main())
