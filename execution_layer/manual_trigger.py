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
import re
from datetime import datetime, date, timezone
from pathlib import Path

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

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


def write_trigger(scan: str, tickers: list[str] | None = None) -> str:
    """Write trigger for engine pickup and append to history. Returns ISO timestamp.

    When `tickers` is provided the engine bypasses the full OpenBB screen and
    runs consensus directly on those tickers under the given scan strategy.
    """
    if scan not in VALID_SCANS:
        raise ValueError(f"Unknown scan: {scan!r}. Must be one of {VALID_SCANS}")
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    _TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"scan": scan, "requested_at": ts}
    if tickers:
        clean = [t.upper().strip() for t in tickers if t.strip()]
        invalid = [t for t in clean if not _TICKER_RE.match(t)]
        if invalid:
            raise ValueError(f"Invalid ticker format (expected 1-5 uppercase letters): {invalid}")
        payload["tickers"] = clean
    tmp = _TRIGGER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(_TRIGGER_FILE)
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    label = f"{SCAN_LABELS[scan]}: {', '.join(payload.get('tickers', []))}" if tickers else SCAN_LABELS[scan]
    with open(_history_file(), "a") as f:
        f.write(json.dumps({"scan": scan, "tickers": tickers or [], "label": label, "fired_at": ts}) + "\n")
    logger.info("Manual trigger written: scan=%s tickers=%s", scan, tickers)
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


def read_and_clear_trigger() -> dict | None:
    """Return {"scan": str, "tickers": list[str]} or None. Deletes trigger file."""
    if not _TRIGGER_FILE.exists():
        return None
    try:
        data = json.loads(_TRIGGER_FILE.read_text())
        _TRIGGER_FILE.unlink(missing_ok=True)
        scan = data.get("scan")
        if scan not in VALID_SCANS:
            logger.warning("Ignoring unknown trigger scan value: %r", scan)
            return None
        return {"scan": scan, "tickers": data.get("tickers") or []}
    except Exception as exc:
        logger.warning("Failed to read trigger file: %s", exc)
        _TRIGGER_FILE.unlink(missing_ok=True)
        return None
