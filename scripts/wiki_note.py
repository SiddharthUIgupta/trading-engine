#!/usr/bin/env python3
"""Write a note to the claude-obsidian vault from a Claude Code session.

Use this mid-session whenever something non-obvious is discovered:
  - a strategy pattern or regime observation
  - an architectural decision or root cause
  - a recurring agent behaviour worth remembering

The note goes into wiki/{category}/ and is immediately BM25-indexed so
future sessions and the Risk Officer prompt can retrieve it.

Usage:
    python3 scripts/wiki_note.py "Title" "Body text" [--category strategy]
    python3 scripts/wiki_note.py "Title" --file /path/to/body.txt [--category sessions]

Categories: strategy (default), sessions, patterns
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution_layer.wiki_writer import write_wiki_note


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a note to the claude-obsidian vault.")
    parser.add_argument("title", help="Note title (becomes the H1 heading and filename slug)")
    parser.add_argument("body", nargs="?", default=None, help="Note body (markdown)")
    parser.add_argument("--file", help="Read body from this file instead of the positional arg")
    parser.add_argument(
        "--category", default="strategy",
        choices=["strategy", "sessions", "patterns"],
        help="Wiki subdirectory: strategy (default), sessions, patterns",
    )
    parser.add_argument("--tags", help="Comma-separated tags", default=None)
    args = parser.parse_args()

    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    elif args.body:
        body = args.body
    else:
        print("Reading body from stdin (Ctrl-D to finish)…", file=sys.stderr)
        body = sys.stdin.read()

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    path = write_wiki_note(title=args.title, body=body, category=args.category, tags=tags)

    if path is None:
        print("ERROR: claude-obsidian vault not found at ~/Projects/claude-obsidian", file=sys.stderr)
        sys.exit(1)

    print(f"Written: {path}")


if __name__ == "__main__":
    main()
