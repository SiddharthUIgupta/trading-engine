"""File-based IPC between the dashboard (writer) and the engine (reader).

The dashboard writes a JSON trigger file; the engine's polling job reads it,
dispatches to the right runtime method, and deletes it. One pending trigger
at a time — a new write overwrites the previous one.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TRIGGER_FILE = Path("state/manual_trigger.json")

VALID_SCANS = ("thesis", "swing", "momentum", "options", "gap")


def write_trigger(scan: str) -> None:
    if scan not in VALID_SCANS:
        raise ValueError(f"Unknown scan: {scan!r}. Must be one of {VALID_SCANS}")
    _TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _TRIGGER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"scan": scan, "requested_at": datetime.now(timezone.utc).isoformat()}))
    tmp.replace(_TRIGGER_FILE)
    logger.info("Manual trigger written: %s", scan)


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
