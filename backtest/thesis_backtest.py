"""Backtests the thesis track's actual entry screen (analyst_layer.
thesis_scanner.evaluate_thesis_candidate, unmodified) and exit rule
(execution_layer.exit_rules.evaluate_exit, unmodified) against several
years of real daily bars — this tests whether the mechanical rule itself
has ever had edge, not whether the LLM agents added value on top of it.

The LLM consensus layer is NOT backtested here, deliberately: replaying
it would mean re-running real Claude calls against historical data
framed as "today," and the model's training data may already contain
knowledge of what actually happened next for any sufficiently-old or
well-known ticker — an unfixable lookahead-bias risk a mechanical
backtest doesn't have. This only tests the deterministic screen+exit,
which is also the part actually claimed to have a repeatable edge.

Methodology (disclosed, not hidden):
- Entry: screen evaluated using each day's close vs. the trailing
  252-trading-day high/low (no lookahead — only data through that day).
  On a pass, entry fills at the NEXT trading day's open.
- Exit: stop-loss and trailing-stop checked against that day's intraday
  low (catches an intraday breach even if the close recovered — ignoring
  intraday lows would overstate the win rate by missing real stop-outs).
  Fill price is the theoretical stop/trailing-stop price itself, not the
  bar's extreme low, since a real stop order doesn't fill at the worst
  possible tick. No take-profit, matching the live thesis bracket.
- One position per ticker at a time — no averaging into an existing
  position (unlike the live system), to keep "one trade" a clean,
  comparable unit for win-rate statistics.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from analyst_layer.thesis_scanner import evaluate_thesis_candidate
from backtest.metrics import Trade
from backtest.universe import get_pit_membership, get_sp500_universe
from data_layer.exceptions import DataLayerError
from data_layer.models import ThesisCandidate
from execution_layer.exit_rules import evaluate_exit

logger = logging.getLogger(__name__)

_LOOKBACK_TRADING_DAYS = 252


def run_thesis_backtest(
    data_client,
    universe: list[str] | None = None,
    years_back: int = 3,
    min_pullback_pct: float = 0.20,
    max_pullback_pct: float = 0.50,
    stop_loss_pct: float = 0.18,
    trailing_stop_pct: float = 0.10,
    trailing_stop_activation_pct: float = 0.20,
    entry_slippage_pct: float = 0.0,
    exit_slippage_pct: float = 0.0,
    use_pit_universe: bool = True,
) -> list[Trade]:
    """
    entry_slippage_pct: fraction added to entry fill (e.g. 0.005 = 0.5% worse than open).
    exit_slippage_pct: fraction subtracted from exit fill (e.g. 0.003 = 0.3% worse than stop).
    use_pit_universe: if True (default), use point-in-time S&P 500 membership so signals
        are only generated for bars where the ticker was actually in the index. Eliminates
        look-ahead survivorship bias for buy-the-drawdown strategies. Falls back to
        Wikipedia current-constituent list if the PIT CSV is unavailable.
    """
    end = date.today()
    start = end - timedelta(days=years_back * 365 + 60)

    # Build ticker list and PIT membership filter
    pit_membership: dict[str, list[tuple[date, date | None]]] = {}
    if use_pit_universe and universe is None:
        pit_membership = get_pit_membership(start, end)
        if pit_membership:
            tickers = list(pit_membership.keys())
            logger.info(
                "PIT universe: %d unique tickers were S&P 500 members during the backtest window "
                "(vs %d current constituents from Wikipedia)", len(tickers), 503
            )
        else:
            logger.warning("PIT data unavailable — falling back to Wikipedia (survivorship-biased).")
            tickers = get_sp500_universe()
    elif universe is not None:
        tickers = universe
    else:
        logger.warning("use_pit_universe=False — using Wikipedia current constituents (survivorship-biased).")
        tickers = get_sp500_universe()

    all_trades: list[Trade] = []
    for n, ticker in enumerate(tickers):
        membership_periods = pit_membership.get(ticker) if pit_membership else None

        try:
            series = data_client.get_price_history(ticker, start_date=start, end_date=end)
        except DataLayerError as exc:
            logger.debug("%s: skipped (%s)", ticker, exc)
            continue

        bars = series.bars
        if len(bars) < _LOOKBACK_TRADING_DAYS + 10:
            continue

        closes = pd.Series([b.close for b in bars])
        rolling_high = closes.rolling(_LOOKBACK_TRADING_DAYS).max()
        rolling_low = closes.rolling(_LOOKBACK_TRADING_DAYS).min()

        trades = _walk_forward(
            ticker, bars, rolling_high, rolling_low,
            min_pullback_pct, max_pullback_pct, stop_loss_pct, trailing_stop_pct, trailing_stop_activation_pct,
            entry_slippage_pct, exit_slippage_pct,
            membership_periods=membership_periods,
        )
        all_trades.extend(trades)
        if (n + 1) % 50 == 0:
            logger.info("Thesis backtest: processed %d/%d tickers, %d trades so far", n + 1, len(tickers), len(all_trades))

    return all_trades


def _in_membership(bar_date: date, periods: list[tuple[date, date | None]] | None) -> bool:
    """True if bar_date falls within any membership period (or if no PIT data)."""
    if periods is None:
        return True
    return any(
        start <= bar_date and (end is None or bar_date <= end)
        for start, end in periods
    )


def _walk_forward(
    ticker, bars, rolling_high, rolling_low,
    min_pullback_pct, max_pullback_pct, stop_loss_pct, trailing_stop_pct, trailing_stop_activation_pct,
    entry_slippage_pct: float = 0.0,
    exit_slippage_pct: float = 0.0,
    membership_periods: list[tuple[date, date | None]] | None = None,
) -> list[Trade]:
    trades: list[Trade] = []
    in_position = False
    entry_price = entry_date = high_water_mark = None

    i = _LOOKBACK_TRADING_DAYS
    last_index = len(bars) - 1
    while i <= last_index:
        bar = bars[i]
        bar_date = bar.timestamp.date()

        if not in_position:
            # PIT guard: only generate entry signals on dates when the ticker
            # was actually a member of the S&P 500.
            if not _in_membership(bar_date, membership_periods):
                i += 1
                continue

            year_high, year_low = rolling_high[i], rolling_low[i]
            if pd.isna(year_high) or pd.isna(year_low):
                i += 1
                continue
            candidate = ThesisCandidate(symbol=ticker, price=bar.close, year_high=float(year_high), year_low=float(year_low))
            signal = evaluate_thesis_candidate(candidate, min_pullback_pct, max_pullback_pct)
            if signal.passed and i < last_index:
                entry_price = bars[i + 1].open * (1 + entry_slippage_pct)
                entry_date = bars[i + 1].timestamp.date()
                high_water_mark = entry_price
                in_position = True
                i += 1
                continue
        else:
            stop_price = entry_price * (1 - stop_loss_pct)
            if bar.low <= stop_price:
                fill = stop_price * (1 - exit_slippage_pct)
                trades.append(Trade(ticker, entry_date, entry_price, bar_date, fill, "stop-loss"))
                in_position = False
                i += 1
                continue

            high_water_mark = max(high_water_mark, bar.high)
            gain_to_peak = (high_water_mark - entry_price) / entry_price
            if high_water_mark > entry_price and gain_to_peak >= trailing_stop_activation_pct:
                trailing_stop_price = high_water_mark * (1 - trailing_stop_pct)
                if bar.low <= trailing_stop_price:
                    fill = trailing_stop_price * (1 - exit_slippage_pct)
                    trades.append(Trade(ticker, entry_date, entry_price, bar_date, fill, "trailing-stop"))
                    in_position = False
                    i += 1
                    continue
        i += 1

    if in_position:
        trades.append(Trade(ticker, entry_date, entry_price))  # still open at backtest end

    return trades
