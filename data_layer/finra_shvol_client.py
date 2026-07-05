"""FINRA daily short-sale volume (Reg SHO "CNMS" files) — free, no auth.

Verified live (2026-07-05, not from memory): the URL pattern is
https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt, pipe-
delimited (Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market),
ONE row per ticker per date (Market is a comma-separated list of venues
that traded it that day, not a per-venue breakdown to sum). ShortVolume/
TotalVolume are high-precision floats, not integers. No rate-limiting or
User-Agent requirement observed across 5 consecutive fetches. Real lag
confirmed empirically: on Sunday 2026-07-05, the most recent available file
was Thursday 2026-07-02 (Friday's returned 403) — T+1 trading-day lag,
weekend/holiday dates simply don't exist yet and must be skipped by walking
backward, never forward (the forward direction would be a lookahead leak).

One file covers the ENTIRE market (~12,241 tickers/day) — caching the raw
file means a whole batch run needs only as many downloads as the lookback
window requires, regardless of how many candidates get scored.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
_TIMEOUT_SECONDS = 15
_MAX_BACKWARD_SEARCH_DAYS = 40  # generous — covers holiday clusters, thin/halted tickers


class FinraShortVolumeClient:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, trade_date: date) -> Path:
        return self._cache_dir / f"CNMSshvol{trade_date.strftime('%Y%m%d')}.txt"

    def _ensure_cached(self, trade_date: date) -> Path | None:
        """Downloads and caches the raw file for trade_date if not already
        present. Returns None (not an error) if FINRA has no file for this
        date — expected and routine for weekends/holidays/not-yet-published
        days, never raised as an exception.
        """
        path = self._cache_path(trade_date)
        if path.exists():
            return path

        url = _URL_TEMPLATE.format(date_str=trade_date.strftime("%Y%m%d"))
        try:
            response = requests.get(url, timeout=_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            logger.debug("FINRA fetch failed for %s: %s", trade_date, exc)
            return None

        if response.status_code != 200:
            logger.debug("FINRA file not available for %s (status=%d)", trade_date, response.status_code)
            return None

        path.write_bytes(response.content)
        return path

    def _parse_ticker_ratio(self, path: Path, ticker: str) -> float | None:
        with open(path) as f:
            next(f)  # header
            for line in f:
                fields = line.rstrip("\n").split("|")
                if len(fields) < 5:
                    continue
                if fields[1] == ticker:
                    short_volume = float(fields[2])
                    total_volume = float(fields[4])
                    if total_volume <= 0:
                        return None
                    return short_volume / total_volume
        return None

    def get_short_vol_series(
        self, ticker: str, as_of_date: date, lookback_days: int = 25
    ) -> list[tuple[date, float]]:
        """Walks backward from as_of_date — never forward, this is the
        structural point-in-time guarantee, independent of the lookahead
        exclusion in scripts/signal_uplift.py. Skips dates with no file
        (weekends/holidays). Stops once lookback_days values are collected
        or the backward search window is exhausted (thin/halted ticker or
        a data gap too large to be worth searching further).
        """
        results: list[tuple[date, float]] = []
        current = as_of_date
        searched = 0
        while len(results) < lookback_days and searched < _MAX_BACKWARD_SEARCH_DAYS:
            path = self._ensure_cached(current)
            if path is not None:
                ratio = self._parse_ticker_ratio(path, ticker)
                if ratio is not None:
                    results.append((current, ratio))
            current -= timedelta(days=1)
            searched += 1
        return results
