"""File-based IPC between the dashboard (writer) and the engine (reader).

The dashboard writes a JSON trigger file; the engine's polling job reads it,
dispatches to the right runtime method, and deletes it. One pending trigger
at a time — a new write overwrites the previous one.

Trigger history is also appended to a daily JSONL file so the dashboard
can show accumulated output across page refreshes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TRIGGER_FILE = Path("state/manual_trigger.json")
_HISTORY_DIR = Path("state/trigger_history")

VALID_SCANS = ("thesis", "swing", "momentum", "options", "gap")

SCAN_LABELS = {
    "thesis": "Thesis Scan",
    "gap": "Gap Scan",
    "swing": "Swing Scan",
    "momentum": "Momentum Scan",
    "options": "Options Scan",
}


def _history_file(day: date | None = None) -> Path:
    d = day or date.today()
    return _HISTORY_DIR / f"{d.isoformat()}.jsonl"


def write_trigger(scan: str) -> str:
    """Write trigger for engine pickup and append to history. Returns ISO timestamp."""
    if scan not in VALID_SCANS:
        raise ValueError(f"Unknown scan: {scan!r}. Must be one of {VALID_SCANS}")
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    _TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _TRIGGER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"scan": scan, "requested_at": ts}))
    tmp.replace(_TRIGGER_FILE)
    # Append to daily history
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(_history_file(), "a") as f:
        f.write(json.dumps({"scan": scan, "label": SCAN_LABELS[scan], "fired_at": ts}) + "\n")
    logger.info("Manual trigger written: %s", scan)
    return ts


def read_trigger_history(day: date | None = None) -> list[dict]:
    """Return all trigger entries for the given day, oldest first."""
    path = _history_file(day)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def read_and_clear_trigger() -> str | None:
    if not _TRIGGER_FILE.exists():
        return None
    try:
        data = json.loads(_TRIGGER_FILE.read_text())
        _TRIGGER_FILE.unlink(missing_ok=True)
        scan = data.get("scan")
        if scan not in VALID_SCANS:
            logger.warning("Ignoring unknown trigger scan value: %r", scan)
            return None
        return scan
    except Exception as exc:
        logger.warning("Failed to read trigger file: %s", exc)
        _TRIGGER_FILE.unlink(missing_ok=True)
        return None
