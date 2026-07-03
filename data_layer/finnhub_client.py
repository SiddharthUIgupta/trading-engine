"""Finnhub financial news client.

Finance-specific news replacing Google News RSS (per-ticker sentiment) and
yfinance Search (macro news). Finnhub free tier: 60 requests/minute.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from data_layer.exceptions import ProviderFetchError

logger = logging.getLogger(__name__)

_DEFAULT_DAYS_BACK = 7
_MAX_HEADLINES = 50


def _get_client(api_key: str):
    try:
        import finnhub
    except ImportError as exc:
        raise ImportError("finnhub-python not installed: pip install finnhub-python") from exc
    return finnhub.Client(api_key=api_key)


def fetch_company_headlines(symbol: str, api_key: str, days_back: int = _DEFAULT_DAYS_BACK) -> list[str]:
    """Return recent Finnhub news headlines for a ticker, newest first."""
    to_dt = date.today()
    from_dt = to_dt - timedelta(days=days_back)
    try:
        client = _get_client(api_key)
        articles = client.company_news(symbol, _from=from_dt.isoformat(), to=to_dt.isoformat())
        return [a["headline"] for a in articles if a.get("headline")][:_MAX_HEADLINES]
    except Exception as exc:
        raise ProviderFetchError(f"Finnhub company news failed for {symbol}: {exc}") from exc


def fetch_market_news(api_key: str, categories: tuple[str, ...] = ("general", "merger")) -> list[tuple[str, str]]:
    """Return (headline, summary) tuples from Finnhub's finance news feed.

    Fetches multiple categories and deduplicates by headline.
    """
    client = _get_client(api_key)
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    for category in categories:
        try:
            articles = client.general_news(category, min_id=0)
            for a in articles:
                headline = a.get("headline", "").strip()
                if headline and headline not in seen:
                    seen.add(headline)
                    results.append((headline, a.get("summary", "").strip()))
        except Exception as exc:
            logger.warning("Finnhub general news failed for category '%s': %s", category, exc)
    if not results:
        raise ProviderFetchError("Finnhub market news returned no articles across all categories")
    return results
