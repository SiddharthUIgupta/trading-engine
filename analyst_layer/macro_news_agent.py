"""Pre-market macro news agent.

Two-LLM-call design (both use cheap Haiku):
  1. Query generation: given today's date, LLM writes 5-8 yfinance Search
     queries — no hardcoded topics. Queries adapt to what's actually happening
     now: Fed meeting week, earnings season, tariff negotiations, etc.
  2. Analysis: LLM reads all fetched news items and returns:
       - market_sentiment + confidence → feeds into VIX adjustment in regime
       - news_ticker_signals → individual stocks with clear catalysts today

news_ticker_signals are passed back to the runtime. Bullish tickers go
through the full LLM consensus in thesis_scan_and_trade (bypassing the
technical pull-back screen — the catalyst IS the entry signal). Bearish
tickers are surfaced as potential put candidates on red-SPY days.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from anthropic import Anthropic

logger = logging.getLogger(__name__)

_QUERY_GEN_SYSTEM = (
    "You are a financial analyst generating search queries to discover the most "
    "important US stock market news today. Queries should surface both macro "
    "trends and individual stock catalysts."
)

_QUERY_GEN_USER = """\
Today is {today}.

Generate exactly 6 search queries for yfinance news search that would surface:
- The biggest macro/economic stories affecting US equity markets today
- Any specific sector themes dominating news this week
- Individual stock catalysts (earnings surprises, M&A, product launches, FDA approvals, short squeezes)

Make queries specific to what's likely happening NOW — include the month/year, \
reference current themes if you know them (e.g. AI stocks, rate decisions, earnings season).

Return ONLY a JSON array of 6 strings, nothing else."""

_ANALYSIS_SYSTEM = (
    "You are a senior equity analyst reading financial news. "
    "You identify both macro market direction AND specific stock opportunities from news."
)

_ANALYSIS_USER = """\
Today is {today}. Below are recent financial news items fetched from yfinance.

Your tasks:
1. Assess the OVERALL US equity market sentiment right now
2. Identify individual stocks with CLEAR, specific catalysts in the news today

News items:
{news_text}

Return ONLY a JSON object with these exact fields:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": <float 0.0-1.0>,
  "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "reasoning": "<one concise sentence>",
  "news_tickers": [
    {{
      "ticker": "ASTS",
      "catalyst": "<brief description of the specific catalyst>",
      "direction": "bullish" | "bearish"
    }}
  ]
}}

Rules for market sentiment:
- "bullish": multiple items converge on positive catalysts (rate cuts, strong GDP, trade deals)
- "bearish": multiple items converge on negative catalysts (rate hikes, recession signals, escalation)
- "neutral": mixed or no dominant theme
- confidence > 0.7 only when evidence is clear and convergent

Rules for news_tickers:
- Include ONLY stocks with a specific, actionable catalyst (earnings beat/miss, acquisition, approval, partnership)
- Do NOT include stocks just mentioned in passing or in market roundups
- Include 0-8 tickers maximum; quality over quantity
- "bearish" tickers = negative catalyst, useful for put consideration
- Return empty list [] if no individual stock has a clear enough catalyst"""


@dataclass(frozen=True)
class NewsTickerSignal:
    ticker: str
    catalyst: str
    direction: str  # "bullish" | "bearish"


@dataclass(frozen=True)
class MacroSentiment:
    sentiment: str                                          # "bullish" | "bearish" | "neutral"
    confidence: float                                       # 0.0–1.0
    key_themes: list[str] = field(default_factory=list)
    reasoning: str = ""
    headlines_read: int = 0
    queries_used: list[str] = field(default_factory=list)
    news_tickers: list[NewsTickerSignal] = field(default_factory=list)


def _fetch_finnhub_news(api_key: str) -> tuple[list[str], int]:
    """Fetch finance-focused news from Finnhub (general + merger categories).

    Returns (formatted_text_lines, unique_item_count).
    """
    from data_layer.finnhub_client import fetch_market_news
    from data_layer.exceptions import ProviderFetchError
    try:
        articles = fetch_market_news(api_key, categories=("general", "merger"))
        lines = [
            f"- {headline}" + (f"\n  {summary}" if summary else "")
            for headline, summary in articles
        ]
        return lines, len(lines)
    except ProviderFetchError as exc:
        logger.warning("Finnhub market news failed: %s", exc)
        return [], 0


def _fetch_yfinance_news(queries: list[str], count_per_query: int = 10) -> tuple[list[str], int]:
    """Fetch news via yfinance Search for each query (fallback when no Finnhub key).

    Returns (formatted_text_lines, unique_item_count).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available — macro news skipped")
        return [], 0

    seen_titles: set[str] = set()
    lines: list[str] = []

    for query in queries:
        try:
            results = yf.Search(query, news_count=count_per_query).news or []
            for item in results:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or {}
                title = (content.get("title") or item.get("title") or "").strip()
                summary = (content.get("summary") or "").strip()

                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                lines.append(f"- {title}" + (f"\n  {summary}" if summary else ""))
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance Search failed for '%s': %s", query, exc)

    return lines, len(seen_titles)


def assess_macro_sentiment(
    client: Anthropic,
    model: str,
    today: date | None = None,
    finnhub_api_key: str = "",
) -> MacroSentiment:
    """Run the macro news assessment.

    With a Finnhub key: fetches finance-specific news directly (general +
    merger categories) — no query generation step needed.
    Without a key: falls back to LLM-generated queries → yfinance Search.

    Returns neutral, zero-confidence result on any failure — the regime
    treats that as 'no adjustment', which is the safe default.
    """
    if today is None:
        today = date.today()
    today_str = today.strftime("%B %d, %Y")

    # ── Fetch news ────────────────────────────────────────────────────────────
    queries: list[str] = []
    if finnhub_api_key:
        news_lines, item_count = _fetch_finnhub_news(finnhub_api_key)
        logger.info("Macro news: %d items from Finnhub for %s", item_count, today_str)
    else:
        # Fallback: LLM generates queries → yfinance Search
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                system=_QUERY_GEN_SYSTEM,
                messages=[{"role": "user", "content": _QUERY_GEN_USER.format(today=today_str)}],
            )
            parsed = json.loads(resp.content[0].text.strip())
            if not isinstance(parsed, list):
                raise ValueError(f"expected list, got {type(parsed)}")
            queries = [str(q) for q in parsed[:8] if q]
            logger.info("Macro news: LLM generated %d queries for %s", len(queries), today_str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Macro news: query generation failed (%s) — using date-anchored fallback", exc)
            month_year = today.strftime("%B %Y")
            queries = [
                f"stock market news {month_year}",
                f"Federal Reserve {month_year}",
                f"earnings results {month_year}",
                "market movers stocks today",
            ]
        news_lines, item_count = _fetch_yfinance_news(queries, count_per_query=10)
    if not news_lines:
        logger.warning("Macro news: no news fetched — returning neutral")
        return MacroSentiment(
            sentiment="neutral", confidence=0.0,
            reasoning="no news fetched",
            queries_used=queries,
        )
    logger.info("Macro news: %d unique news items fetched across %d queries", item_count, len(queries))

    # ── Step 3: LLM analyzes sentiment + extracts tickers ────────────────────
    news_text = "\n".join(news_lines[:60])
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": _ANALYSIS_USER.format(
                today=today_str, news_text=news_text
            )}],
        )
        data = json.loads(resp.content[0].text.strip())

        sentiment = str(data.get("sentiment", "neutral")).lower()
        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        key_themes = [str(t) for t in (data.get("key_themes") or [])][:5]
        reasoning = str(data.get("reasoning", ""))

        news_tickers: list[NewsTickerSignal] = []
        for entry in (data.get("news_tickers") or [])[:8]:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper().strip()
            catalyst = str(entry.get("catalyst", "")).strip()
            direction = str(entry.get("direction", "bullish")).lower()
            if ticker and catalyst and direction in ("bullish", "bearish"):
                news_tickers.append(NewsTickerSignal(ticker=ticker, catalyst=catalyst, direction=direction))

        result = MacroSentiment(
            sentiment=sentiment,
            confidence=confidence,
            key_themes=key_themes,
            reasoning=reasoning,
            headlines_read=item_count,
            queries_used=queries,
            news_tickers=news_tickers,
        )
        bullish = [s.ticker for s in news_tickers if s.direction == "bullish"]
        bearish = [s.ticker for s in news_tickers if s.direction == "bearish"]
        logger.info(
            "Macro news: %s conf=%.2f | %s | bullish=%s bearish=%s",
            sentiment.upper(), confidence, reasoning, bullish, bearish,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("Macro news: analysis failed (%s) — returning neutral", exc)
        return MacroSentiment(
            sentiment="neutral", confidence=0.0,
            reasoning=f"analysis failed: {exc}",
            headlines_read=item_count,
            queries_used=queries,
        )
