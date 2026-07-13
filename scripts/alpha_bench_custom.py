#!/usr/bin/env python3
"""Alpha bench on our actual traded universe (tickers from realized_sales).

run_alpha_bench() in Vibe-Trading only supports csi300/sp500/btc-usdt — fixed
indices that don't match our actual traded tickers.  Our universe is whatever
the OpenBB screens surface on any given day (gainers, active, losers,
undervalued_growth, aggressive_small_caps) — any market cap, not just small
caps.  Testing factor IC on sp500 or csi300 tells us nothing specific about
the stocks we actually trade.  This script runs the same IC calculation
(Spearman rank correlation of factor values vs next-day returns) directly on
our realized-sale tickers via yfinance + Vibe-Trading's Registry and
compute_ic_series utilities.

IC threshold for promotion: ic_mean > 0.03 at ic_count >= 300  (CLAUDE.md)

Usage:
    ~/trading-engine/.venv/bin/python scripts/alpha_bench_custom.py
    ~/trading-engine/.venv/bin/python scripts/alpha_bench_custom.py --zoo alpha101
    ~/trading-engine/.venv/bin/python scripts/alpha_bench_custom.py --top 30

Output: prints factor_id, ic_mean, ic_count for all factors above threshold.
Also writes a CSV to state/alpha_bench_results.csv for the full run.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DB = Path(__file__).resolve().parent.parent / "state" / "trading_engine.sqlite3"
_VIBE = Path.home() / "Projects" / "Vibe-Trading" / "agent"
_IC_THRESHOLD = 0.03
_N_THRESHOLD = 300
_OUT_CSV = Path(__file__).resolve().parent.parent / "state" / "alpha_bench_results.csv"


def get_thesis_tickers() -> list[str]:
    conn = sqlite3.connect(_DB)
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM realized_sales ORDER BY ticker"
    ).fetchall()
    conn.close()
    tickers = [r[0] for r in rows]
    logger.info("thesis universe: %d tickers from realized_sales", len(tickers))
    return tickers


def build_panel(tickers: list[str], period: str = "2y") -> dict:
    """Download OHLCV for each ticker and assemble into the panel dict format
    that Registry.compute() and compute_ic_series() expect.

    panel = {
        "open":   DataFrame(index=dates, columns=tickers),
        "high":   ...,
        "low":    ...,
        "close":  ...,
        "volume": ...,
    }
    """
    import pandas as pd
    import yfinance as yf

    frames: dict[str, dict[str, pd.Series]] = {
        "open": {}, "high": {}, "low": {}, "close": {}, "volume": {}
    }
    skipped = 0
    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                skipped += 1
                continue
            # yfinance returns MultiIndex columns when group_by='ticker'; flatten
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            for field in ("open", "high", "low", "close", "volume"):
                if field in raw.columns:
                    frames[field][ticker] = raw[field]
        except Exception as exc:
            logger.debug("yfinance failed for %s: %s", ticker, exc)
            skipped += 1

    if skipped:
        logger.warning("skipped %d tickers (no data)", skipped)

    panel = {}
    for field, series_dict in frames.items():
        if series_dict:
            panel[field] = pd.DataFrame(series_dict)
    return panel


def run_bench(zoo: str | None = None, top: int = 20, period: str = "2y") -> list[dict]:
    if not _VIBE.exists():
        logger.error("Vibe-Trading not found at %s — run: pip install vibe-trading-ai", _VIBE)
        sys.exit(1)

    sys.path.insert(0, str(_VIBE))

    try:
        from src.factors.registry import Registry, get_default_registry
        from src.factors.factor_analysis_core import compute_ic_series, compute_forward_returns
    except ImportError as exc:
        logger.error("Vibe-Trading import failed: %s", exc)
        sys.exit(1)

    tickers = get_thesis_tickers()
    if not tickers:
        logger.error("No tickers in realized_sales — run trades first")
        sys.exit(1)

    logger.info("building panel (downloading %d tickers via yfinance, period=%s)…", len(tickers), period)
    panel = build_panel(tickers, period=period)

    close_df = panel.get("close")
    if close_df is None or close_df.empty:
        logger.error("panel is empty — all yfinance downloads failed")
        sys.exit(1)

    logger.info("panel shape: %s × %s (dates × tickers)", *close_df.shape)

    try:
        return_df = compute_forward_returns(panel)
    except Exception as exc:
        logger.error("forward returns failed: %s", exc)
        sys.exit(1)

    registry = get_default_registry()
    alpha_ids = registry.list(zoo=zoo) if zoo else registry.list()
    logger.info("testing %d factors (zoo=%s)…", len(alpha_ids), zoo or "all")

    results: list[dict] = []
    failures = 0
    for i, aid in enumerate(alpha_ids):
        if i % 50 == 0:
            logger.info("  %d/%d tested…", i, len(alpha_ids))
        try:
            factor_df = registry.compute(aid, panel)
            ic_series = compute_ic_series(factor_df, return_df)
            if ic_series.empty:
                continue
            ic_mean = float(ic_series.mean())
            ic_count = int(len(ic_series))
            ic_std = float(ic_series.std())
            ir = ic_mean / ic_std if ic_std > 0 else 0.0
            results.append({
                "factor_id": aid,
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "ir": round(ir, 4),
                "ic_count": ic_count,
            })
        except Exception:
            failures += 1

    logger.info("tested %d factors, %d failures", len(results), failures)

    # Write full CSV
    results.sort(key=lambda r: abs(r["ic_mean"]), reverse=True)
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["factor_id", "ic_mean", "ic_std", "ir", "ic_count"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("full results written to %s", _OUT_CSV)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha bench on thesis universe")
    parser.add_argument("--zoo", default=None, help="Restrict to one zoo (e.g. alpha101)")
    parser.add_argument("--top", type=int, default=20, help="How many to print (default 20)")
    parser.add_argument("--period", default="2y", help="yfinance period string, e.g. 2y, 3y (default 2y)")
    args = parser.parse_args()

    results = run_bench(zoo=args.zoo, top=args.top, period=args.period)

    candidates = [
        r for r in results
        if r["ic_mean"] > _IC_THRESHOLD and r["ic_count"] >= _N_THRESHOLD
    ]
    candidates.sort(key=lambda r: r["ic_mean"], reverse=True)

    print(f"\n{'='*60}")
    print(f"PROMOTE CANDIDATES (ic_mean > {_IC_THRESHOLD}, ic_count >= {_N_THRESHOLD})")
    print(f"{'='*60}")
    if not candidates:
        print("None found — insufficient sample or no edge at this threshold.")
        print(f"Tip: run again with more trade history or lower --zoo to test a subset.")
    else:
        print(f"{'factor_id':<30} {'ic_mean':>8} {'ic_count':>9}")
        print("-" * 52)
        for r in candidates[:args.top]:
            print(f"{r['factor_id']:<30} {r['ic_mean']:>8.4f} {r['ic_count']:>9}")
        print(f"\nSet in .env: PROMOTED_VW_FACTORS={','.join(r['factor_id'] for r in candidates[:args.top])}")

    print(f"\nFull results: {_OUT_CSV}")


if __name__ == "__main__":
    main()
