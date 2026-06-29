"""Free, key-free headline fetching via Google News' public RSS search
endpoint. Used as the headline source for sentiment scoring instead of
OpenBB's `news.company` — the free yfinance provider behind that call
returns a much smaller (~10), less financially-targeted headline set, with
no sentiment field of its own (see sentiment_lexicon.py for why that
mattered). Google News' RSS feed is a public, unauthenticated endpoint
Google itself serves for exactly this purpose — not page scraping.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

from data_layer.exceptions import ProviderFetchError

_TIMEOUT_SECONDS = 10
_USER_AGENT = "Mozilla/5.0 (compatible; trading-engine/1.0)"


def fetch_headlines(symbol: str, limit: int = 50) -> list[str]:
    """Returns recent headline titles for `symbol`, most recent first
    (the feed's own ordering). Querying "<symbol> stock" rather than the
    bare symbol biases results toward financial coverage over unrelated
    matches.
    """
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=_TIMEOUT_SECONDS, headers={"User-Agent": _USER_AGENT})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:  # noqa: BLE001
        raise ProviderFetchError(f"Google News fetch failed for {symbol}: {exc}") from exc

    items = root.findall(".//item")[:limit]
    return [item.findtext("title", default="") for item in items]
