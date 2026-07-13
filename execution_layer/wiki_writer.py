"""Write structured notes to the claude-obsidian vault.

Two callers:
  1. Trading agents (protection_plane._run_reflection) — write pattern/rule notes
     when the reflection agent surfaces non-noise lessons from closed trades.
  2. Claude Code sessions (scripts/wiki_note.py) — persist session discoveries
     so they survive across sessions and appear in the Risk Officer's context.

Notes go into wiki/{category}/ so they're crawled by contextual-prefix.py
and BM25-indexed. The retrieval pipeline then surfaces them in the consensus
prompt via _fetch_trade_memory() in alpha_plane.py.

Categories:
  strategy  — regime/track patterns, entry/exit rules, sizing observations
  sessions  — architectural discoveries, import paths, bug root causes
  patterns  — agent-detected recurring loss/win patterns across multiple trades
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_VAULT_ROOT = Path.home() / "Projects" / "claude-obsidian"
_SCRIPTS_DIR = _VAULT_ROOT / "scripts"

_VALID_CATEGORIES = {"strategy", "sessions", "patterns"}


def write_wiki_note(
    title: str,
    body: str,
    category: str = "strategy",
    tags: list[str] | None = None,
) -> Path | None:
    """Write a markdown note to wiki/{category}/ and trigger incremental re-index.

    Returns the path written, or None if the vault is not provisioned.
    """
    if not _VAULT_ROOT.exists():
        return None

    if category not in _VALID_CATEGORIES:
        logger.warning("wiki_writer: unknown category %r — using 'strategy'", category)
        category = "strategy"

    target_dir = _VAULT_ROOT / "wiki" / category
    target_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(title)
    path = target_dir / f"{date_str}-{slug}.md"

    tag_line = ""
    if tags:
        tag_line = "\n\ntags: " + ", ".join(tags)

    content = f"# {title}\n\nDate: {date_str}{tag_line}\n\n{body.strip()}\n"
    path.write_text(content, encoding="utf-8")
    logger.info("wiki note written: %s", path)

    _reindex(path)
    return path


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _reindex(note_path: Path) -> None:
    prefix_script = _SCRIPTS_DIR / "contextual-prefix.py"
    bm25_script = _SCRIPTS_DIR / "bm25-index.py"
    if not prefix_script.exists() or not bm25_script.exists():
        return

    relative = note_path.relative_to(_VAULT_ROOT)
    try:
        subprocess.run(
            ["python3", str(prefix_script), str(relative)],
            cwd=str(_VAULT_ROOT), capture_output=True, timeout=30,
        )
    except Exception as exc:
        logger.debug("wiki note chunk generation failed: %s", exc)
        return

    try:
        subprocess.run(
            ["python3", str(bm25_script), "build"],
            cwd=str(_VAULT_ROOT), capture_output=True, timeout=30,
        )
    except Exception as exc:
        logger.debug("wiki note BM25 rebuild failed: %s", exc)
