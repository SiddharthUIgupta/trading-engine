"""Pre-market gap scanner.

Runs at 9:05 AM ET — after pre-market has had 35 minutes to develop conviction
but before the 9:30 open. Finds stocks gapping ≥ MIN_GAP_PCT from prior close
and returns them ranked by magnitude so the runtime can run expedited consensus
and queue orders for the opening print.

Data sources (in order of preference):
1. akshare stock_us_famous_spot_em() — one call, ~100 liquid US names, already
   includes % change from prior close. Fast (~2s).
2. yfinance fast_info — supplementary check for watchlist tickers not in akshare.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIN_AVG_DOLLAR_VOL = 5_000_000   # $5M/day minimum — same threshold as pre-filter


@dataclass
class GapCandidate:
    symbol: str
    prev_close: float
    premarket_price: float
    gap_pct: float          # signed: positive = gap up, negative = gap down
    avg_dollar_vol: float   # estimated from prior volume × price

    @property
    def direction(self) -> str:
        return "up" if self.gap_pct > 0 else "down"

    def __repr__(self) -> str:
        return (
            f"GapCandidate({self.symbol} {self.gap_pct:+.1%} "
            f"prev={self.prev_close:.2f} → pre={self.premarket_price:.2f})"
        )


def _akshare_movers(min_gap_pct: float) -> dict[str, GapCandidate]:
    """Pull pre-market movers from akshare famous US stocks feed."""
    try:
        from data_layer.akshare_client import get_us_movers
        movers = get_us_movers(min_change_pct=min_gap_pct * 100)
        result: dict[str, GapCandidate] = {}
        for m in movers:
            if m.price <= 0 or m.change_pct == 0:
                continue
            prev = m.price / (1 + m.change_pct / 100)
            # Estimate dollar vol from market cap (rough proxy) — refined by yfinance below
            avg_dv = m.market_cap * 0.005 if m.market_cap else 0.0
            result[m.symbol] = GapCandidate(
                symbol=m.symbol,
                prev_close=round(prev, 4),
                premarket_price=m.price,
                gap_pct=m.change_pct / 100,
                avg_dollar_vol=avg_dv,
            )
        return result
    except Exception as exc:
        logger.debug("akshare mover fetch failed: %s", exc)
        return {}


def _yf_fast_info(symbol: str) -> GapCandidate | None:
    """Check a single ticker's pre-market gap via yfinance fast_info."""
    try:
        import yfinance as yf
        fi = yf.Ticker(symbol).fast_info
        prev = getattr(fi, "previous_close", None)
        last = getattr(fi, "last_price", None)
        if not prev or not last or prev <= 0:
            return None
        gap_pct = (last - prev) / prev
        # Estimate avg dollar vol from 3-month average volume × price
        avg_vol = getattr(fi, "three_month_average_volume", None) or 0
        avg_dv = avg_vol * last
        return GapCandidate(
            symbol=symbol,
            prev_close=round(prev, 4),
            premarket_price=round(last, 4),
            gap_pct=round(gap_pct, 6),
            avg_dollar_vol=avg_dv,
        )
    except Exception as exc:
        logger.debug("yfinance fast_info failed for %s: %s", symbol, exc)
        return None


def scan_premarket_gaps(
    watchlist: list[str],
    min_gap_pct: float = 0.05,
    gap_up_only: bool = True,
    max_candidates: int = 5,
) -> list[GapCandidate]:
    """Find stocks with a meaningful pre-market gap.

    Returns up to max_candidates, sorted by abs(gap_pct) descending.

    Parameters
    ----------
    watchlist:
        System watchlist tickers — always checked via yfinance even if not
        in the akshare famous-stocks universe.
    min_gap_pct:
        Minimum absolute gap to qualify (0.05 = 5%).
    gap_up_only:
        When True, only surfaces gap-ups (we're long-only, no shorting on Alpaca).
    max_candidates:
        Cap — consensus runs in parallel but 5 is enough to finish before open.
    """
    candidates: dict[str, GapCandidate] = {}

    # Source 1: akshare — fast broad scan of liquid US names
    ak_movers = _akshare_movers(min_gap_pct)
    candidates.update(ak_movers)
    logger.info("Gap scanner: akshare returned %d movers ≥%.0f%%", len(ak_movers), min_gap_pct * 100)

    # Source 2: yfinance — supplement with watchlist tickers not already covered
    extra_syms = [s for s in watchlist if s not in candidates and s.isalpha()]
    if extra_syms:
        with ThreadPoolExecutor(max_workers=min(8, len(extra_syms))) as pool:
            futures = {pool.submit(_yf_fast_info, sym): sym for sym in extra_syms}
            for fut in as_completed(futures):
                result = fut.result()
                if result and abs(result.gap_pct) >= min_gap_pct:
                    candidates[result.symbol] = result

    # Refine dollar volume via yfinance for any candidate that only has akshare proxy
    to_refine = [c for c in candidates.values() if c.avg_dollar_vol == 0]
    if to_refine:
        with ThreadPoolExecutor(max_workers=min(8, len(to_refine))) as pool:
            futures = {pool.submit(_yf_fast_info, c.symbol): c.symbol for c in to_refine}
            for fut in as_completed(futures):
                result = fut.result()
                if result and result.symbol in candidates:
                    candidates[result.symbol].avg_dollar_vol = result.avg_dollar_vol

    # Apply filters
    filtered = [
        c for c in candidates.values()
        if abs(c.gap_pct) >= min_gap_pct
        and c.avg_dollar_vol >= _MIN_AVG_DOLLAR_VOL
        and (not gap_up_only or c.gap_pct > 0)
        and c.premarket_price > 1.0        # no sub-dollar stocks
    ]

    # Sort by magnitude descending
    filtered.sort(key=lambda c: abs(c.gap_pct), reverse=True)
    result = filtered[:max_candidates]

    logger.info(
        "Gap scanner: %d candidate(s) after filters (min_gap=%.0f%%, liquid, %s): %s",
        len(result), min_gap_pct * 100,
        "up only" if gap_up_only else "both directions",
        [(c.symbol, f"{c.gap_pct:+.1%}") for c in result],
    )
    return result
