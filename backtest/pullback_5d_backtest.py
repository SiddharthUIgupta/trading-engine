"""Backtests a short-horizon pullback setup ("pullback_5d") — research only,
no live wiring. See CLAUDE.md "Signal lifecycle" / invariant #6: this is
exactly the kind of PIT backtest artifact required before any strategy could
even be considered for arming, and this task produces only that artifact.

Deliberately NOT in analyst_layer/: anything placed there is one import away
from being reachable by a live scanner. Keeping the signal function here is
a structural guarantee of "no live wiring," not just a convention.

Deliberately a NEW walk-forward loop, not a reuse of thesis_backtest.py's
_walk_forward: that function is tightly coupled to the thesis signal
(252-day rolling high/low, evaluate_thesis_candidate, stop-loss/trailing-
stop-only exits) and has no hook for a 20-session window, an SMA/liquidity
pre-filter, or a fixed-session time exit — none of which this strategy can
do without. Making _walk_forward generic enough for both would mean
modifying the shared function thesis depends on. What IS reused: PIT
membership (backtest.universe.get_pit_membership / _in_membership from
thesis_backtest.py), the Trade dataclass, and the same slippage convention.
thesis_backtest.py itself has zero diff from this file's existence.

Methodology (disclosed, not hidden):
- Entry: pullback 5-10% (or swept range) below the 20-session high, close
  above the 200-day SMA (uptrend intact), 20-day avg dollar volume >= floor.
  Entry fills at the NEXT session's open, matching thesis's convention.
- Exit: stop-loss checked against intraday low (same "fill at the
  theoretical stop price, not the bar's worst tick" choice as thesis), OR a
  fixed session-count time exit at that session's close — whichever
  triggers first. No trailing-stop/profit-target for v1.
- One position per ticker at a time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from backtest.metrics import Trade
from backtest.thesis_backtest import _in_membership
from backtest.universe import get_pit_membership, get_sp500_universe
from data_layer.exceptions import DataLayerError

logger = logging.getLogger(__name__)

_HIGH_WINDOW = 20
_SMA_WINDOW = 200
_VOLUME_WINDOW = 20
# 200-day SMA needs 200 bars of history before it's valid; add buffer for
# weekends/holidays already baked into trading-day counts (this is bar count,
# not calendar days, so no extra buffer needed here).
_MIN_BARS_REQUIRED = _SMA_WINDOW + 10


@dataclass(frozen=True)
class Pullback5dSignal:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_pullback_5d(
    close: float,
    high_20d: float,
    sma_200: float,
    avg_dollar_vol_20d: float,
    pullback_min_pct: float,
    pullback_max_pct: float,
    min_dollar_vol: float,
) -> Pullback5dSignal:
    """Three conjunctive conditions — all must hold:
    1. pullback_min_pct <= (high_20d - close)/high_20d <= pullback_max_pct
    2. close > sma_200 (uptrend intact, not a falling knife)
    3. avg_dollar_vol_20d >= min_dollar_vol (tradable liquidity)
    """
    if high_20d <= 0 or sma_200 <= 0:
        return Pullback5dSignal(passed=False, reasons=["invalid high/SMA input"])

    pullback_pct = (high_20d - close) / high_20d
    in_band = pullback_min_pct <= pullback_pct <= pullback_max_pct
    above_sma = close > sma_200
    liquid_enough = avg_dollar_vol_20d >= min_dollar_vol

    reasons = [
        f"pullback {pullback_pct:.1%} vs band [{pullback_min_pct:.1%}, {pullback_max_pct:.1%}]: {'OK' if in_band else 'FAIL'}",
        f"close {close:.2f} vs 200d SMA {sma_200:.2f}: {'OK' if above_sma else 'FAIL'}",
        f"avg $vol {avg_dollar_vol_20d:,.0f} vs floor {min_dollar_vol:,.0f}: {'OK' if liquid_enough else 'FAIL'}",
    ]
    return Pullback5dSignal(passed=in_band and above_sma and liquid_enough, reasons=reasons)


@dataclass(frozen=True)
class PreparedTickerData:
    ticker: str
    bars: list
    rolling_high_20d: pd.Series
    sma_200: pd.Series
    avg_dollar_vol_20d: pd.Series
    membership_periods: list[tuple[date, date | None]] | None


@dataclass(frozen=True)
class PreparedUniverseData:
    tickers_data: list[PreparedTickerData]
    total_tickers: int
    no_data_count: int
    insufficient_history_count: int


def fetch_and_prepare_universe(
    data_client,
    universe: list[str] | None = None,
    years_back: int = 3,
    use_pit_universe: bool = True,
) -> PreparedUniverseData:
    """Fetches price history and computes the rolling series ONCE per
    ticker — a param sweep (e.g. the 18-cell grid) should call this exactly
    once, then run the walk-forward loop many times over the cached result,
    not re-fetch the network for every parameter combination.
    """
    end = date.today()
    start = end - timedelta(days=years_back * 365 + 60)

    pit_membership: dict[str, list[tuple[date, date | None]]] = {}
    if use_pit_universe and universe is None:
        pit_membership = get_pit_membership(start, end)
        if pit_membership:
            tickers = list(pit_membership.keys())
            logger.info("pullback_5d: PIT universe %d tickers", len(tickers))
        else:
            logger.warning("pullback_5d: PIT data unavailable — falling back to Wikipedia (survivorship-biased).")
            tickers = get_sp500_universe()
    elif universe is not None:
        tickers = universe
    else:
        logger.warning("pullback_5d: use_pit_universe=False — Wikipedia current constituents (survivorship-biased).")
        tickers = get_sp500_universe()

    tickers_data: list[PreparedTickerData] = []
    no_data_count = 0
    insufficient_history_count = 0
    for n, ticker in enumerate(tickers):
        membership_periods = pit_membership.get(ticker) if pit_membership else None

        try:
            series = data_client.get_price_history(ticker, start_date=start, end_date=end)
        except DataLayerError as exc:
            logger.debug("%s: skipped (%s)", ticker, exc)
            no_data_count += 1
            continue

        bars = series.bars
        if len(bars) < _MIN_BARS_REQUIRED:
            insufficient_history_count += 1
            continue

        closes = pd.Series([b.close for b in bars])
        dollar_vol = pd.Series([b.close * b.volume for b in bars])
        tickers_data.append(PreparedTickerData(
            ticker=ticker, bars=bars,
            rolling_high_20d=closes.rolling(_HIGH_WINDOW).max(),
            sma_200=closes.rolling(_SMA_WINDOW).mean(),
            avg_dollar_vol_20d=dollar_vol.rolling(_VOLUME_WINDOW).mean(),
            membership_periods=membership_periods,
        ))
        if (n + 1) % 50 == 0:
            logger.info("pullback_5d: fetched %d/%d tickers", n + 1, len(tickers))

    skipped = no_data_count + insufficient_history_count
    if skipped:
        logger.warning(
            "pullback_5d: %d of %d tickers skipped (%d no price data, %d insufficient history)",
            skipped, len(tickers), no_data_count, insufficient_history_count,
        )

    return PreparedUniverseData(
        tickers_data=tickers_data, total_tickers=len(tickers),
        no_data_count=no_data_count, insufficient_history_count=insufficient_history_count,
    )


def run_pullback_5d_on_prepared(
    prepared: PreparedUniverseData,
    pullback_min_pct: float = 0.05,
    pullback_max_pct: float = 0.10,
    stop_loss_pct: float = 0.08,
    time_exit_sessions: int = 5,
    min_dollar_vol: float = 20_000_000.0,
    entry_slippage_pct: float = 0.0,
    exit_slippage_pct: float = 0.0,
) -> list[Trade]:
    """Runs the walk-forward loop over already-fetched data — no network
    calls. This is what a parameter sweep should call for every grid cell.
    """
    all_trades: list[Trade] = []
    for td in prepared.tickers_data:
        trades = _walk_forward_pullback_5d(
            td.ticker, td.bars, td.rolling_high_20d, td.sma_200, td.avg_dollar_vol_20d,
            pullback_min_pct, pullback_max_pct, stop_loss_pct, time_exit_sessions, min_dollar_vol,
            entry_slippage_pct, exit_slippage_pct,
            membership_periods=td.membership_periods,
        )
        all_trades.extend(trades)
    return all_trades


def run_pullback_5d_backtest(
    data_client,
    universe: list[str] | None = None,
    years_back: int = 3,
    pullback_min_pct: float = 0.05,
    pullback_max_pct: float = 0.10,
    stop_loss_pct: float = 0.08,
    time_exit_sessions: int = 5,
    min_dollar_vol: float = 20_000_000.0,
    entry_slippage_pct: float = 0.0,
    exit_slippage_pct: float = 0.0,
    use_pit_universe: bool = True,
) -> list[Trade]:
    """Single-call convenience wrapper (fetch + run in one call) — for a
    param sweep, call fetch_and_prepare_universe() once and
    run_pullback_5d_on_prepared() per cell instead.
    """
    prepared = fetch_and_prepare_universe(data_client, universe, years_back, use_pit_universe)
    return run_pullback_5d_on_prepared(
        prepared, pullback_min_pct, pullback_max_pct, stop_loss_pct, time_exit_sessions,
        min_dollar_vol, entry_slippage_pct, exit_slippage_pct,
    )


def _walk_forward_pullback_5d(
    ticker, bars, rolling_high_20d, sma_200, avg_dollar_vol_20d,
    pullback_min_pct, pullback_max_pct, stop_loss_pct, time_exit_sessions, min_dollar_vol,
    entry_slippage_pct: float = 0.0,
    exit_slippage_pct: float = 0.0,
    membership_periods: list[tuple[date, date | None]] | None = None,
) -> list[Trade]:
    trades: list[Trade] = []
    in_position = False
    entry_price = entry_date = None
    sessions_held = 0

    i = _SMA_WINDOW
    last_index = len(bars) - 1
    while i <= last_index:
        bar = bars[i]
        bar_date = bar.timestamp.date()

        if not in_position:
            if not _in_membership(bar_date, membership_periods):
                i += 1
                continue

            high_20d, sma200, dvol20d = rolling_high_20d[i], sma_200[i], avg_dollar_vol_20d[i]
            if pd.isna(high_20d) or pd.isna(sma200) or pd.isna(dvol20d):
                i += 1
                continue

            signal = evaluate_pullback_5d(
                bar.close, float(high_20d), float(sma200), float(dvol20d),
                pullback_min_pct, pullback_max_pct, min_dollar_vol,
            )
            if signal.passed and i < last_index:
                entry_price = bars[i + 1].open * (1 + entry_slippage_pct)
                entry_date = bars[i + 1].timestamp.date()
                in_position = True
                sessions_held = 0
                i += 1
                continue
        else:
            sessions_held += 1
            stop_price = entry_price * (1 - stop_loss_pct)
            if bar.low <= stop_price:
                fill = stop_price * (1 - exit_slippage_pct)
                trades.append(Trade(ticker, entry_date, entry_price, bar_date, fill, "stop-loss"))
                in_position = False
                i += 1
                continue

            if sessions_held >= time_exit_sessions:
                fill = bar.close * (1 - exit_slippage_pct)
                trades.append(Trade(ticker, entry_date, entry_price, bar_date, fill, "time-exit"))
                in_position = False
                i += 1
                continue
        i += 1

    if in_position:
        trades.append(Trade(ticker, entry_date, entry_price))  # still open at backtest end

    return trades
