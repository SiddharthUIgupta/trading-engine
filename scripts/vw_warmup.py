#!/usr/bin/env python3
"""Bootstrap the VW bandit from 2 years of historical price data.

Simulates which tickers would have triggered the swing and momentum scanners
on each historical trading day, computes the actual forward return as the
outcome label, and feeds everything into the VW model as training examples.

No LLM calls — scanner logic only. Because agent stance/confidence data
isn't available for historical trades, VW learns track × regime patterns
and scanner quality (did the setup actually lead to a win?). This is a
solid head-start until live closed trades accumulate.

Usage:
    source .venv/bin/activate
    python scripts/vw_warmup.py

Safe to re-run: will append to an existing model, not overwrite it.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

from analyst_layer import prefilter, swing_scanner
from analyst_layer.vw_bandit import VWSignalBandit
from config.settings import get_settings
from data_layer.models import PriceBar, PriceSeries

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = [
    # Core watchlist
    "AAPL", "MSFT", "NVDA", "SPY", "QQQ", "AMZN", "META", "TSLA",
    # Extended liquid universe — more training examples
    "GOOGL", "NFLX", "AMD", "INTC", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "UNH", "V", "MA",
    "DIS", "MCD", "NKE", "HD", "WMT", "COST",
    "CRM", "ADBE", "NOW", "SNOW", "UBER", "ABNB",
    "LLY", "AVGO", "ORCL", "QCOM", "MU", "ARM",
]

LOOKBACK_YEARS = 2
MOMENTUM_HOLD_DAYS = 5   # exit after 5 trading days
SWING_HOLD_DAYS = 15     # exit after 15 trading days (within 21-day max)
MIN_HISTORY_BARS = 65    # need 60 days for SMA50 + a few extra

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_series(ticker: str, closes: list[float], volumes: list[int]) -> PriceSeries:
    """Wrap raw closes/volumes into a PriceSeries the scanner can consume."""
    bars = [
        PriceBar(
            symbol=ticker,
            timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),  # placeholder date
            open=c, high=c, low=c, close=c, volume=int(v),
        )
        for c, v in zip(closes, volumes)
    ]
    return PriceSeries(symbol=ticker, interval="1d", bars=bars)


def _pct_return(entry: float, exit_price: float) -> float:
    return (exit_price - entry) / entry * 100.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()
    vw_model_path = settings.state_db_path.parent / "vw_bandit.model"

    print(f"VW model path: {vw_model_path}")
    if vw_model_path.exists():
        print("Existing model found — will append new examples (not overwrite).")
    else:
        print("No existing model — creating fresh.")

    bandit = VWSignalBandit(model_path=vw_model_path)
    existing_count = bandit.example_count
    print(f"Starting example count: {existing_count}")

    end_date = date.today()
    start_date = end_date - timedelta(days=int(LOOKBACK_YEARS * 365) + 90)  # extra buffer for SMA warmup

    print(f"\nDownloading {len(TICKERS)} tickers from {start_date} to {end_date}...")
    raw = yf.download(
        TICKERS,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if raw.empty:
        print("ERROR: no data downloaded — check network / yfinance")
        return

    closes_df = raw["Close"]
    volumes_df = raw["Volume"]

    total_examples = 0
    swing_examples = 0
    momentum_examples = 0
    skipped = 0

    print("\nSimulating entries...")
    for ticker in TICKERS:
        if ticker not in closes_df.columns:
            print(f"  {ticker}: no data, skipping")
            continue

        closes_series = closes_df[ticker].dropna()
        volumes_series = volumes_df[ticker].dropna() if ticker in volumes_df.columns else None

        closes_arr = closes_series.values.tolist()
        volumes_arr = (
            [int(v) for v in volumes_series.values.tolist()]
            if volumes_series is not None
            else [0] * len(closes_arr)
        )

        if len(closes_arr) < MIN_HISTORY_BARS + SWING_HOLD_DAYS:
            print(f"  {ticker}: too short ({len(closes_arr)} bars), skipping")
            continue

        ticker_examples = 0

        # Slide through each candidate entry date
        for i in range(MIN_HISTORY_BARS, len(closes_arr) - SWING_HOLD_DAYS):
            window_closes = closes_arr[i - MIN_HISTORY_BARS : i + 1]
            window_volumes = volumes_arr[i - MIN_HISTORY_BARS : i + 1]
            entry_price = closes_arr[i]

            # Regime using momentum SMA windows (10/30)
            regime = prefilter.compute_regime(
                window_closes,
                settings.filter_sma_short_window,
                settings.filter_sma_long_window,
            )

            # ── Swing track ────────────────────────────────────────────────
            price_series = _make_price_series(ticker, window_closes, window_volumes)
            sig = swing_scanner.evaluate_swing_candidate(price_series)
            if sig.passed:
                exit_price = closes_arr[i + SWING_HOLD_DAYS]
                pnl = _pct_return(entry_price, exit_price)
                bandit.learn(
                    track="swing",
                    regime=regime,
                    signals=[],  # no agent signals available in backtest
                    pnl=pnl,
                )
                swing_examples += 1
                ticker_examples += 1

            # ── Momentum track ─────────────────────────────────────────────
            # Momentum entry criterion: price > SMA10 > SMA30 AND today's
            # close higher than yesterday's (price momentum confirmation).
            if len(window_closes) >= 31:
                sma10 = sum(window_closes[-10:]) / 10
                sma30 = sum(window_closes[-30:]) / 30
                prev_close = window_closes[-2]
                if (
                    entry_price > sma10 > sma30
                    and entry_price > prev_close  # upward day
                ):
                    exit_price = closes_arr[min(i + MOMENTUM_HOLD_DAYS, len(closes_arr) - 1)]
                    pnl = _pct_return(entry_price, exit_price)
                    bandit.learn(
                        track="momentum",
                        regime=regime,
                        signals=[],
                        pnl=pnl,
                    )
                    momentum_examples += 1
                    ticker_examples += 1

        total_examples += ticker_examples
        if ticker_examples > 0:
            print(f"  {ticker}: {ticker_examples} examples")
        else:
            skipped += 1

    print(f"\n{'─' * 50}")
    print(f"Done.")
    print(f"  Swing examples:    {swing_examples}")
    print(f"  Momentum examples: {momentum_examples}")
    print(f"  Total new:         {total_examples}")
    print(f"  Tickers skipped:   {skipped}")
    print(f"  Model total:       {bandit.example_count} examples")
    print(f"  Model saved to:    {vw_model_path}")

    bandit.close()


if __name__ == "__main__":
    main()
