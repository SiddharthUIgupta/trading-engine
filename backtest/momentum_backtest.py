"""Backtests the momentum track's actual entry screen (analyst_layer.
momentum_scanner.evaluate_low_float_momentum, unmodified) and exit rule
(execution_layer.exit_rules.evaluate_exit's logic, replicated bar-by-bar
below) against real 5-minute intraday bars.

Hard, disclosed data constraint: free yfinance intraday history caps out
at roughly 60 calendar days regardless of how far back it's asked for —
verified live, not assumed. That bounds this backtest to ~40 trading
days no matter what. That is NOT a large enough sample to draw a real
conclusion from; this exists to sanity-check the screen's behavior, not
to certify it the way the thesis backtest's 3-year window can attempt to.

Universe constraint, also disclosed: there's no free way to query "who
was in today's gainers/losers screen" on a past date — discovery screens
are live-only. This uses *today's* movers as a population proxy (same
function the live system already calls), which is not a true historical
reconstruction.

Float is also a single current snapshot reused across the whole window
(no free historical float series exists) — another disclosed
simplification, not a hidden one.

Methodology:
- Entry: the deterministic screen evaluated using each day's full
  completed intraday session (no lookahead — only that day's bars).
  On a pass, entry fills at the next trading day's first 5-min bar open.
- Exit: walked forward bar-by-bar through subsequent 5-min bars (the
  live bracket is tight enough that daily granularity would hide
  same-day stop/target hits). Stop-loss and take-profit checked against
  each bar's low/high respectively; if a single bar's range could have
  triggered both, stop-loss takes priority — a conservative assumption,
  not an optimistic one, so it doesn't inflate the win rate.
- One position per ticker at a time: the scan resumes only after the
  open trade resolves (or data runs out) — no overlapping/averaging-in
  entries while a position is open, same discipline as the thesis
  backtest, so each trade is a clean, comparable unit.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

from analyst_layer.momentum_scanner import evaluate_low_float_momentum
from backtest.metrics import Trade
from backtest.universe import get_momentum_backtest_universe
from data_layer.exceptions import DataLayerError
from data_layer.models import PriceSeries

logger = logging.getLogger(__name__)

_MAX_FREE_INTRADAY_DAYS = 59  # verified live: yfinance 5m history caps here regardless of how far back asked


def run_momentum_backtest(
    data_client,
    universe: list[str] | None = None,
    volume_lookback_days: int = 10,
    max_float_shares: int = 20_000_000,
    ema_short_period: int = 9,
    ema_long_period: int = 20,
    min_daily_gain_pct: float = 0.10,
    clean_body_dominance_threshold: float = 0.55,
    clean_lookback_bars: int = 12,
    min_relative_volume: float = 5.0,
    price_min: float = 1.0,
    price_max: float = 20.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.03,
    trailing_stop_pct: float = 0.015,
) -> list[Trade]:
    if universe is None:
        universe = get_momentum_backtest_universe(data_client)

    end = date.today()
    intraday_start = end - timedelta(days=_MAX_FREE_INTRADAY_DAYS)
    daily_start = intraday_start - timedelta(days=volume_lookback_days * 3)  # buffer for the trailing-volume average

    all_trades: list[Trade] = []
    global_fail_counts: dict[str, int] = {}  # aggregated across all tickers/days
    tickers_skipped = 0
    tickers_processed = 0

    for n, ticker in enumerate(universe):
        try:
            intraday = data_client.get_price_history(ticker, start_date=intraday_start, end_date=end, interval="5m")
            daily = data_client.get_price_history(ticker, start_date=daily_start, end_date=end)
            shares_float = data_client.get_shares_float(ticker)
        except DataLayerError as exc:
            logger.debug("%s: skipped (%s)", ticker, exc)
            tickers_skipped += 1
            continue

        bars_by_day: dict[date, list] = defaultdict(list)
        for bar in intraday.bars:
            bars_by_day[bar.timestamp.date()].append(bar)
        trading_days = sorted(bars_by_day.keys())
        daily_by_date = {bar.timestamp.date(): bar for bar in daily.bars}
        daily_dates_sorted = sorted(daily_by_date.keys())

        trades, fail_counts = _walk_forward(
            ticker, trading_days, bars_by_day, daily_by_date, daily_dates_sorted, shares_float,
            volume_lookback_days, max_float_shares, ema_short_period, ema_long_period, min_daily_gain_pct,
            clean_body_dominance_threshold, clean_lookback_bars, min_relative_volume, price_min, price_max,
            stop_loss_pct, take_profit_pct, trailing_stop_pct,
        )
        all_trades.extend(trades)
        tickers_processed += 1
        for reason, count in fail_counts.items():
            global_fail_counts[reason] = global_fail_counts.get(reason, 0) + count
        if (n + 1) % 25 == 0:
            logger.info("Momentum backtest: processed %d/%d tickers, %d trades so far", n + 1, len(universe), len(all_trades))

    # Emit a filter breakdown so we know which criteria are blocking signals
    if global_fail_counts:
        total_rejected = sum(global_fail_counts.values())
        logger.info(
            "Signal filter breakdown (%d tickers processed, %d skipped, %d total day-signals rejected):",
            tickers_processed, tickers_skipped, total_rejected,
        )
        for reason, count in sorted(global_fail_counts.items(), key=lambda x: -x[1]):
            logger.info("  [%5d rejections] %s", count, reason)
    else:
        logger.info("No signals were evaluated (universe empty or all tickers skipped due to data errors).")

    return all_trades


def _walk_forward(
    ticker, trading_days, bars_by_day, daily_by_date, daily_dates_sorted, shares_float,
    volume_lookback_days, max_float_shares, ema_short_period, ema_long_period, min_daily_gain_pct,
    clean_body_dominance_threshold, clean_lookback_bars, min_relative_volume, price_min, price_max,
    stop_loss_pct, take_profit_pct, trailing_stop_pct,
) -> tuple[list[Trade], dict[str, int]]:
    trades: list[Trade] = []
    fail_counts: dict[str, int] = {}  # reason keyword → number of days it was the blocking criterion
    n_days = len(trading_days)
    day_idx = 0

    while day_idx < n_days:
        day = trading_days[day_idx]
        if day not in daily_by_date:
            day_idx += 1
            continue
        day_pos = daily_dates_sorted.index(day)
        if day_pos < volume_lookback_days:
            day_idx += 1
            continue

        prior_days = daily_dates_sorted[day_pos - volume_lookback_days:day_pos]
        average_daily_volume = sum(daily_by_date[d].volume for d in prior_days) / len(prior_days)
        prior_close = daily_by_date[daily_dates_sorted[day_pos - 1]].close
        today_bar = daily_by_date[day]
        today_percent_change = (today_bar.close - prior_close) / prior_close if prior_close > 0 else 0.0

        series = PriceSeries(symbol=ticker, interval="5m", bars=bars_by_day[day])
        signal = evaluate_low_float_momentum(
            intraday_series=series, shares_float=shares_float, today_percent_change=today_percent_change,
            today_volume=today_bar.volume, average_daily_volume=average_daily_volume,
            max_float_shares=max_float_shares, ema_short_period=ema_short_period, ema_long_period=ema_long_period,
            min_daily_gain_pct=min_daily_gain_pct, clean_body_dominance_threshold=clean_body_dominance_threshold,
            clean_lookback_bars=clean_lookback_bars, min_relative_volume=min_relative_volume,
            price_min=price_min, price_max=price_max,
        )
        if not signal.passed:
            # Attribute the rejection to each failing criterion (a signal can fail multiple checks)
            for reason in signal.reasons:
                if "NOT" in reason or "<" in reason or "outside" in reason or ">" in reason and ">=" not in reason:
                    fail_counts[reason] = fail_counts.get(reason, 0) + 1
            logger.debug("%s %s: signal REJECTED — %s", ticker, day, "; ".join(signal.reasons))
            day_idx += 1
            continue
        if day_idx >= n_days - 1:
            day_idx += 1
            continue

        next_day_idx = day_idx + 1
        next_day_bars = bars_by_day[trading_days[next_day_idx]]
        entry_price = next_day_bars[0].open
        entry_date = next_day_bars[0].timestamp.date()
        stop_price = entry_price * (1 - stop_loss_pct)
        target_price = entry_price * (1 + take_profit_pct)
        high_water_mark = entry_price

        exited = False
        scan_idx = next_day_idx
        bars_to_check = next_day_bars[1:]  # the entry bar itself isn't also an exit check
        while True:
            for bar in bars_to_check:
                if bar.low <= stop_price:
                    trades.append(Trade(ticker, entry_date, entry_price, bar.timestamp.date(), stop_price, "stop-loss"))
                    exited = True
                    break
                if bar.high >= target_price:
                    trades.append(Trade(ticker, entry_date, entry_price, bar.timestamp.date(), target_price, "take-profit"))
                    exited = True
                    break
                high_water_mark = max(high_water_mark, bar.high)
                trailing_stop_price = high_water_mark * (1 - trailing_stop_pct)
                if high_water_mark > entry_price and bar.low <= trailing_stop_price:
                    trades.append(Trade(ticker, entry_date, entry_price, bar.timestamp.date(), trailing_stop_price, "trailing-stop"))
                    exited = True
                    break
            if exited:
                break
            scan_idx += 1
            if scan_idx >= n_days:
                break  # ran out of data before any exit fired
            bars_to_check = bars_by_day[trading_days[scan_idx]]

        if not exited:
            trades.append(Trade(ticker, entry_date, entry_price))  # still open when data ran out
            day_idx = n_days  # nothing more to scan — data exhausted
        else:
            day_idx = scan_idx + 1  # resume scanning only after this trade resolved — no overlapping entries

    return trades, fail_counts
