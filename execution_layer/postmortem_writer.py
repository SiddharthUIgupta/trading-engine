"""Write a markdown post-mortem to the claude-obsidian vault after each closed trade.

The file lands in wiki/postmortems/ so it's picked up by the contextual-prefix
pipeline and becomes searchable via scripts/retrieve.py.  Two best-effort
subprocess calls update the index immediately so the next Risk Officer call
already sees the new memory:
  1. contextual-prefix.py <file>  — generates BM25-indexable chunks (tier-3
                                    synthetic, no external calls, fast)
  2. bm25-index.py build          — rebuilds the inverted index

Both calls are fire-and-forget inside the reflection daemon thread.  If the
vault is not set up yet (path doesn't exist), the whole module is a no-op.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_VAULT_ROOT = Path.home() / "Projects" / "claude-obsidian"
_POSTMORTEM_DIR = _VAULT_ROOT / "wiki" / "postmortems"
_SCRIPTS_DIR = _VAULT_ROOT / "scripts"


def write_postmortem(
    ticker: str,
    strategy: str,
    regime: str,
    entry_price: float | None,
    pnl: float,
    agent_signals: list[dict],
    what_happened: str,
    root_cause: str,
    lessons: list[str],
) -> None:
    """Write a structured post-mortem markdown file and re-index the vault."""
    if not _VAULT_ROOT.exists():
        return

    _POSTMORTEM_DIR.mkdir(parents=True, exist_ok=True)

    date_str = date.today().isoformat()
    outcome = "WIN" if pnl > 0 else "LOSS"
    filename = f"{ticker}-{date_str}.md"
    path = _POSTMORTEM_DIR / filename

    lines = [
        f"# {ticker} {date_str} {outcome}",
        "",
        f"Strategy: {strategy}  Regime: {regime}",
    ]
    if entry_price is not None:
        lines.append(f"Entry: {entry_price:.2f}  PnL: {pnl:+.2f}")
    else:
        lines.append(f"PnL: {pnl:+.2f}")

    if agent_signals:
        lines += ["", "## Agent stances"]
        for s in agent_signals:
            name = s.get("agent_name", "unknown")
            stance = s.get("stance", "?")
            conf = s.get("confidence", "?")
            rationale = s.get("rationale", "")
            lines.append(f"- {name}: {stance} ({conf}) — {rationale}")

    if what_happened:
        lines += ["", "## What happened", what_happened]

    if root_cause:
        lines += ["", "## Root cause", root_cause]

    if lessons:
        lines += ["", "## Lessons"]
        for lesson in lessons:
            lines.append(f"- {lesson}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("post-mortem written: %s", path)

    _reindex(path)


def _reindex(postmortem_path: Path) -> None:
    prefix_script = _SCRIPTS_DIR / "contextual-prefix.py"
    bm25_script = _SCRIPTS_DIR / "bm25-index.py"
    if not prefix_script.exists() or not bm25_script.exists():
        return

    relative = postmortem_path.relative_to(_VAULT_ROOT)
    try:
        subprocess.run(
            ["python3", str(prefix_script), str(relative)],
            cwd=str(_VAULT_ROOT), capture_output=True, timeout=30,
        )
    except Exception as exc:
        logger.debug("postmortem chunk generation failed: %s", exc)
        return

    try:
        subprocess.run(
            ["python3", str(bm25_script), "build"],
            cwd=str(_VAULT_ROOT), capture_output=True, timeout=30,
        )
    except Exception as exc:
        logger.debug("postmortem BM25 rebuild failed: %s", exc)
