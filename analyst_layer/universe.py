"""Builds the day's scan universe from OpenBB's discovery screens instead
of a fixed watchlist. Pure function, no I/O — runtime.py does the fetching.
"""
from __future__ import annotations

from data_layer.models import MarketMover


def prerank_movers(movers: list[MarketMover], limit: int) -> list[str]:
    """Sorts the raw discovery pool by |percent_change| and takes the top
    `limit` — this is what keeps the heavier per-ticker fetches (intraday
    bars + float lookup) bounded regardless of how many names the discovery
    screens return.
    """
    deduped = list({m.symbol: m for m in movers}.values())
    ranked = sorted(deduped, key=lambda m: abs(m.percent_change), reverse=True)
    return [m.symbol for m in ranked[:limit]]
