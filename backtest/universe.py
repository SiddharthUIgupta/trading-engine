"""Historical backtest universes.

Neither track's live universe is reconstructable historically: OpenBB's
discovery screens (active/gainers/losers, aggressive_small_caps) are live
snapshots with no historical query — there's no way to ask "who was in
today's gainers screen on 2024-03-15." So backtesting needs its own,
honestly-disclosed universe choice instead of pretending to replay the
live mechanism bar-for-bar.

Thesis track (PIT): point-in-time S&P 500 membership sourced from
fja05680/sp500 on GitHub (sp500_ticker_start_end.csv). For each bar date
during a multi-year backtest we check which tickers were actually IN the
S&P 500 on that date. This eliminates look-ahead survivorship bias —
previously we used the CURRENT Wikipedia constituent list, which is
maximally flattering for a buy-the-drawdown strategy because every name in
it survived whatever drawdown you're backtesting a purchase of.

Fallback: if the PIT CSV download fails, falls back to the Wikipedia
current-constituent list with a prominent warning so the bias is visible.

Momentum track: today's live discovery-screen movers (active/gainers/
losers), the same function the live system already uses. This is NOT a
true historical reconstruction (today's volatile names weren't
necessarily volatile two months ago) — it's the best honestly-available
proxy for "a population of names that behave the way this screen is
looking for," used because there is no free alternative.
"""
from __future__ import annotations

import logging
import urllib.request
from datetime import date
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_PIT_CSV_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
_USER_AGENT = "Mozilla/5.0 (compatible; trading-engine-backtest/1.0)"

# Module-level cache — one download per backtest run.
_pit_df: pd.DataFrame | None = None


def _load_pit_df() -> pd.DataFrame | None:
    global _pit_df
    if _pit_df is not None:
        return _pit_df
    try:
        req = urllib.request.Request(_PIT_CSV_URL, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as r:
            csv_text = r.read().decode()
        df = pd.read_csv(StringIO(csv_text))
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        # Normalize tickers — yfinance uses '-' for share classes
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        _pit_df = df
        logger.info("PIT S&P 500 constituent table loaded: %d rows", len(df))
        return _pit_df
    except Exception as exc:
        logger.warning("PIT constituent CSV download failed: %s", exc)
        return None


def get_sp500_universe_pit(as_of: date | None = None) -> list[str]:
    """Return the S&P 500 constituent list as of `as_of` (defaults to today).

    Uses the fja05680/sp500 point-in-time membership CSV so backtest runs
    see the index as it existed on each bar date rather than today's list.
    Falls back to Wikipedia current list if the download fails (logged as WARNING
    so the survivorship-bias caveat is always visible in the output).
    """
    target = pd.Timestamp(as_of or date.today())
    df = _load_pit_df()
    if df is None:
        logger.warning(
            "PIT constituent data unavailable — falling back to current Wikipedia S&P 500. "
            "SURVIVORSHIP BIAS WARNING: results will be optimistic for buy-the-drawdown strategies."
        )
        return get_sp500_universe()

    mask = (df["start_date"] <= target) & (df["end_date"].isna() | (df["end_date"] >= target))
    tickers = df.loc[mask, "ticker"].dropna().unique().tolist()
    logger.debug("PIT universe as of %s: %d tickers", target.date(), len(tickers))
    return tickers


def get_sp500_universe() -> list[str]:
    """Current S&P 500 from Wikipedia. DO NOT use for backtesting buy-the-drawdown
    strategies — this list excludes every company that was ever removed (delisted,
    acquired, went bankrupt), which is maximally flattering survivorship bias.
    Use get_sp500_universe_pit() for backtest runs.
    """
    response = requests.get(_WIKIPEDIA_SP500_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    # Yahoo/yfinance uses '-' where Wikipedia uses '.' for share classes (e.g. BRK.B -> BRK-B).
    return [str(s).replace(".", "-") for s in table["Symbol"].tolist()]


def get_pit_membership(
    window_start: date,
    window_end: date,
) -> dict[str, list[tuple[date, date | None]]]:
    """Return a dict mapping ticker → list of (start, end) membership periods
    that overlap the given window. `end=None` means the ticker is still in the index.

    Used by the backtest to skip signal generation on bars where the ticker
    wasn't yet in (or had been removed from) the S&P 500.
    """
    df = _load_pit_df()
    if df is None:
        logger.warning(
            "PIT data unavailable — returning empty membership dict. "
            "Caller should fall back to get_sp500_universe() with bias warning."
        )
        return {}

    ws = pd.Timestamp(window_start)
    we = pd.Timestamp(window_end)
    # Keep rows where the membership period overlaps the backtest window
    in_window = (df["start_date"] <= we) & (df["end_date"].isna() | (df["end_date"] >= ws))
    subset = df[in_window]

    membership: dict[str, list[tuple[date, date | None]]] = {}
    for _, row in subset.iterrows():
        ticker = row["ticker"]
        start = row["start_date"].date() if pd.notna(row["start_date"]) else window_start
        end = row["end_date"].date() if pd.notna(row["end_date"]) else None
        membership.setdefault(ticker, []).append((start, end))
    return membership


def get_momentum_backtest_universe(data_client, limit: int = 150) -> list[str]:
    movers = data_client.get_market_movers()
    seen: list[str] = []
    for mover in movers:
        if mover.symbol not in seen:
            seen.append(mover.symbol)
        if len(seen) >= limit:
            break
    return seen
