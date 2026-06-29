"""Backtests Opening Range Breakout (analyst_layer.orb_scanner.evaluate_orb,
unmodified) against real 5-minute intraday bars.

Same disclosed data constraint as the momentum backtest: free yfinance
intraday history caps at ~60 calendar days regardless of how far back
it's asked for — this bounds the sample to ~40 trading days, same caveat
as before. Universe constraint is also the same (today's active/
gainers/losers as a population proxy, not a true historical reconstruction)
— but ORB doesn't depend on float at all, so it isn't vulnerable to the
universe/float mismatch that broke the old momentum scanner.

Methodology, day-trading discipline throughout:
- Entry: on a confirmed breakout (close beyond the opening range), enter
  at the NEXT bar's open — not the breakout bar's own close, to avoid
  lookahead.
- Exit: classic ORB risk framing. Stop = the opposite side of the opening
  range (the level that invalidates the breakout thesis). Target = a
  fixed R-multiple of that initial risk. Force-closed at the last bar of
  the SAME session regardless of P&L — no overnight risk, consistent
  with "day trade," not "swing trade."
- Each trading day is an independent opportunity per ticker (no
  multi-day lookback needed, unlike momentum's volume averaging) — a
  trade always resolves (hits stop, target, or EOD close) within the
  same session it opened.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

from analyst_layer.orb_scanner import evaluate_orb
from backtest.metrics import Trade
from backtest.universe import get_momentum_backtest_universe
from data_layer.exceptions import DataLayerError
from data_layer.models import PriceSeries

logger = logging.getLogger(__name__)

_MAX_FREE_INTRADAY_DAYS = 59


def run_orb_backtest(
    data_client,
    universe: list[str] | None = None,
    opening_range_minutes: int = 15,
    risk_reward_multiple: float = 2.0,
    volume_confirmation_multiple: float | None = None,
) -> list[Trade]:
    if universe is None:
        universe = get_momentum_backtest_universe(data_client)

    end = date.today()
    start = end - timedelta(days=_MAX_FREE_INTRADAY_DAYS)

    all_trades: list[Trade] = []
    for n, ticker in enumerate(universe):
        try:
            intraday = data_client.get_price_history(ticker, start_date=start, end_date=end, interval="5m")
        except DataLayerError as exc:
            logger.debug("%s: skipped (%s)", ticker, exc)
            continue

        bars_by_day = defaultdict(list)
        for bar in intraday.bars:
            bars_by_day[bar.timestamp.date()].append(bar)

        for day, day_bars in bars_by_day.items():
            trade = _evaluate_one_session(ticker, day_bars, opening_range_minutes, risk_reward_multiple, volume_confirmation_multiple)
            if trade is not None:
                all_trades.append(trade)

        if (n + 1) % 25 == 0:
            logger.info("ORB backtest: processed %d/%d tickers, %d trades so far", n + 1, len(universe), len(all_trades))

    return all_trades


def _evaluate_one_session(ticker, day_bars, opening_range_minutes, risk_reward_multiple, volume_confirmation_multiple=None) -> Trade | None:
    series = PriceSeries(symbol=ticker, interval="5m", bars=day_bars)
    signal = evaluate_orb(series, opening_range_minutes=opening_range_minutes, volume_confirmation_multiple=volume_confirmation_multiple)
    if signal.direction == "none" or signal.breakout_bar_index is None:
        return None

    entry_idx = signal.breakout_bar_index + 1
    if entry_idx >= len(day_bars):
        return None  # breakout was the last bar of the day -- nothing left to enter on

    entry_bar = day_bars[entry_idx]
    entry_price = entry_bar.open
    entry_date = entry_bar.timestamp.date()

    if signal.direction == "long":
        stop_price = signal.opening_range_low
        risk = entry_price - stop_price
        target_price = entry_price + risk_reward_multiple * risk
    else:
        stop_price = signal.opening_range_high
        risk = stop_price - entry_price
        target_price = entry_price - risk_reward_multiple * risk

    if risk <= 0:
        return None  # degenerate range (e.g. entry already past the stop level) -- not a tradeable setup

    for bar in day_bars[entry_idx:]:
        if signal.direction == "long":
            if bar.low <= stop_price:
                return Trade(ticker, entry_date, entry_price, bar.timestamp.date(), stop_price, "stop-loss", direction="long")
            if bar.high >= target_price:
                return Trade(ticker, entry_date, entry_price, bar.timestamp.date(), target_price, "target", direction="long")
        else:
            if bar.high >= stop_price:
                return Trade(ticker, entry_date, entry_price, bar.timestamp.date(), stop_price, "stop-loss", direction="short")
            if bar.low <= target_price:
                return Trade(ticker, entry_date, entry_price, bar.timestamp.date(), target_price, "target", direction="short")

    # Neither hit -- forced flat at the close of the session (day-trading discipline).
    last_bar = day_bars[-1]
    return Trade(ticker, entry_date, entry_price, last_bar.timestamp.date(), last_bar.close, "eod-close", direction=signal.direction)
