#!/usr/bin/env python3
"""Walk-forward backtest using backtrader.

Implements the same entry/exit logic as the live strategies so backtested
metrics directly predict live performance — no strategy drift between
backtesting and execution.

Usage:
    source .venv/bin/activate
    python scripts/backtest.py --strategy swing   --years 2
    python scripts/backtest.py --strategy momentum --tickers AAPL,MSFT,NVDA --years 1
    python scripts/backtest.py --strategy orb     --tickers SPY,QQQ --years 1

Outputs:
    Per-run Sharpe ratio, max drawdown, total return, win rate, trade count.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_COMMISSION = 0.0   # Alpaca is commission-free
_SLIPPAGE   = 0.001  # 0.1% round-trip slippage estimate


def _yf_feed(ticker: str, start: datetime, end: datetime) -> bt.feeds.PandasData | None:
    """Download OHLCV from yfinance and wrap it in a backtrader PandasData feed."""
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df is None or df.empty or len(df) < 30:
        print(f"  [warn] insufficient data for {ticker}")
        return None
    df.index = pd.DatetimeIndex(df.index)
    # backtrader expects lowercase column names
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    # Rename 'adj close' -> 'close' if needed (auto_adjust means adj close IS close)
    return bt.feeds.PandasData(dataname=df, openinterest=None)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Momentum (mirrors live momentum / ORB equity)
# ─────────────────────────────────────────────────────────────────────────────

class MomentumStrategy(bt.Strategy):
    """Volume spike + price > SMA20 momentum entry.

    Entry: close > SMA20 AND today's volume >= 2× 10-day avg volume AND RSI < 70.
    Exit:  5% hard stop, 3% trailing stop, or 7 calendar days max hold.
    """
    params = dict(
        sma_period=20,
        vol_avg_period=10,
        vol_spike_mult=2.0,
        rsi_period=14,
        rsi_overbought=70.0,
        stop_loss=0.05,
        trail_pct=0.03,
        max_hold_bars=7,
        risk_pct=0.02,   # risk 2% of equity per trade
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.vol_sma = bt.indicators.SMA(self.data.volume, period=self.p.vol_avg_period)
        self.entry_price = None
        self.high_since_entry = None
        self.bars_held = 0
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            price = self.data.close[0]
            vol_spike = self.data.volume[0] >= self.p.vol_spike_mult * self.vol_sma[0]
            trend_ok  = price > self.sma[0]
            rsi_ok    = self.rsi[0] < self.p.rsi_overbought

            if vol_spike and trend_ok and rsi_ok:
                equity = self.broker.get_value()
                risk_amt = equity * self.p.risk_pct
                stop_dist = price * self.p.stop_loss
                size = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = price
                    self.high_since_entry = price
                    self.bars_held = 0
        else:
            price = self.data.close[0]
            self.bars_held += 1
            if price > self.high_since_entry:
                self.high_since_entry = price

            trail_stop = self.high_since_entry * (1 - self.p.trail_pct)
            hard_stop  = self.entry_price * (1 - self.p.stop_loss)

            if (price <= hard_stop
                    or price <= trail_stop
                    or self.bars_held >= self.p.max_hold_bars):
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Margin):
            self.order = None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Swing (mirrors live swing_scanner.py)
# ─────────────────────────────────────────────────────────────────────────────

class SwingStrategy(bt.Strategy):
    """SMA20 > SMA50 + RSI [35,65] + entry on pullback to SMA20.

    Entry: SMA20 > SMA50 (uptrend) AND RSI between 35–65 (not extended) AND
           price pulls back within 2% of SMA20 (buy the dip into the trend).
    Exit:  8% hard stop / 5% trailing stop / 21-day max hold.
    """
    params = dict(
        sma_short=20,
        sma_long=50,
        rsi_period=14,
        rsi_low=35.0,
        rsi_high=65.0,
        pullback_pct=0.02,   # entry within 2% of SMA20
        stop_loss=0.08,
        trail_pct=0.05,
        max_hold_bars=21,
        risk_pct=0.02,
    )

    def __init__(self):
        self.sma_s = bt.indicators.SMA(self.data.close, period=self.p.sma_short)
        self.sma_l = bt.indicators.SMA(self.data.close, period=self.p.sma_long)
        self.rsi   = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.entry_price = None
        self.high_since_entry = None
        self.bars_held = 0
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            price = self.data.close[0]
            uptrend = self.sma_s[0] > self.sma_l[0]
            rsi_ok  = self.p.rsi_low <= self.rsi[0] <= self.p.rsi_high
            near_s  = price <= self.sma_s[0] * (1 + self.p.pullback_pct)
            above_l = price > self.sma_l[0]

            if uptrend and rsi_ok and near_s and above_l:
                equity   = self.broker.get_value()
                risk_amt = equity * self.p.risk_pct
                stop_dist = price * self.p.stop_loss
                size = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = price
                    self.high_since_entry = price
                    self.bars_held = 0
        else:
            price = self.data.close[0]
            self.bars_held += 1
            if price > self.high_since_entry:
                self.high_since_entry = price

            trail_stop = self.high_since_entry * (1 - self.p.trail_pct)
            hard_stop  = self.entry_price * (1 - self.p.stop_loss)

            if (price <= hard_stop
                    or price <= trail_stop
                    or self.bars_held >= self.p.max_hold_bars):
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Margin):
            self.order = None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: ORB approximation (Opening Range Breakout on daily bars)
# ─────────────────────────────────────────────────────────────────────────────

class ORBStrategy(bt.Strategy):
    """Opening range breakout approximated on daily bars.

    On daily bars the "opening range" is approximated as the first day's high/low
    of a rolling 5-day window. True ORB needs intraday data; this backtest gives
    directional bias and hold-period P&L, not precise intraday fill prices.

    Entry: today's close > 5-day rolling high AND volume > avg (trending breakout).
    Exit:  2% stop / 2% trailing stop / 3 days max (ORB is intraday/very short term).
    """
    params = dict(
        lookback=5,
        vol_period=10,
        stop_loss=0.02,
        trail_pct=0.02,
        max_hold_bars=3,
        risk_pct=0.015,
    )

    def __init__(self):
        self.high_n = bt.indicators.Highest(self.data.high, period=self.p.lookback)
        self.vol_sma = bt.indicators.SMA(self.data.volume, period=self.p.vol_period)
        self.entry_price = None
        self.high_since_entry = None
        self.bars_held = 0
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            price = self.data.close[0]
            breakout = price > self.high_n[-1]   # today's close broke yesterday's N-day high
            vol_ok   = self.data.volume[0] > self.vol_sma[0]

            if breakout and vol_ok:
                equity   = self.broker.get_value()
                risk_amt = equity * self.p.risk_pct
                stop_dist = price * self.p.stop_loss
                size = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = price
                    self.high_since_entry = price
                    self.bars_held = 0
        else:
            price = self.data.close[0]
            self.bars_held += 1
            if price > self.high_since_entry:
                self.high_since_entry = price

            trail_stop = self.high_since_entry * (1 - self.p.trail_pct)
            hard_stop  = self.entry_price * (1 - self.p.stop_loss)

            if (price <= hard_stop
                    or price <= trail_stop
                    or self.bars_held >= self.p.max_hold_bars):
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Margin):
            self.order = None


# ─────────────────────────────────────────────────────────────────────────────
# Analyzers + runner
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "swing": SwingStrategy,
    "orb": ORBStrategy,
}

_DEFAULT_TICKERS = {
    "momentum": ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "AMD", "TSLA"],
    "swing":    ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "JPM", "UNH", "V"],
    "orb":      ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"],
}


def run_backtest(
    strategy_cls: type,
    ticker: str,
    start: datetime,
    end: datetime,
    starting_cash: float = 100_000,
) -> dict | None:
    feed = _yf_feed(ticker, start, end)
    if feed is None:
        return None

    cerebro = bt.Cerebro()
    cerebro.adddata(feed, name=ticker)
    cerebro.addstrategy(strategy_cls)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=_COMMISSION)
    cerebro.broker.set_slippage_perc(_SLIPPAGE / 2, slip_open=True, slip_limit=True, slip_match=True)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.05, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,     _name="returns", timeframe=bt.TimeFrame.Days)

    results = cerebro.run()
    strat = results[0]

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe     = round(float(sharpe_raw), 3) if sharpe_raw is not None else None
    dd         = strat.analyzers.drawdown.get_analysis()
    ta         = strat.analyzers.trades.get_analysis()

    total_closed = int(ta.get("total", {}).get("closed", 0))
    won          = int(ta.get("won", {}).get("total", 0))
    win_rate     = won / total_closed if total_closed > 0 else 0.0
    final_val    = cerebro.broker.getvalue()
    total_ret    = (final_val - starting_cash) / starting_cash

    return {
        "ticker":     ticker,
        "sharpe":     sharpe,
        "max_dd_pct": round(dd.get("max", {}).get("drawdown", 0.0), 2),
        "total_ret":  round(total_ret * 100, 2),
        "trades":     total_closed,
        "win_rate":   round(win_rate * 100, 1),
        "final_val":  round(final_val, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="backtrader walk-forward backtest")
    parser.add_argument("--strategy", choices=list(_STRATEGY_MAP), default="swing",
                        help="Which strategy to backtest")
    parser.add_argument("--tickers", default="",
                        help="Comma-separated tickers (default: strategy-specific list)")
    parser.add_argument("--years", type=float, default=2.0,
                        help="How many years of history to use (default: 2)")
    parser.add_argument("--cash", type=float, default=100_000,
                        help="Starting cash (default: 100000)")
    args = parser.parse_args()

    strategy_cls = _STRATEGY_MAP[args.strategy]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or _DEFAULT_TICKERS[args.strategy]
    end   = datetime.now()
    start = end - timedelta(days=int(args.years * 365))

    print(f"\n{'='*60}")
    print(f"Strategy: {args.strategy.upper()}  |  {start.date()} → {end.date()}")
    print(f"Starting cash: ${args.cash:,.0f}  |  Tickers: {tickers}")
    print(f"{'='*60}")

    results = []
    for ticker in tickers:
        print(f"  Running {ticker}...", end=" ", flush=True)
        r = run_backtest(strategy_cls, ticker, start, end, args.cash)
        if r:
            results.append(r)
            print(f"return={r['total_ret']:+.1f}%  sharpe={r['sharpe']}  dd={r['max_dd_pct']:.1f}%  trades={r['trades']}  win={r['win_rate']:.0f}%")
        else:
            print("skipped")

    if not results:
        print("\nNo results.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print("AGGREGATE SUMMARY")
    avg_ret    = sum(r["total_ret"] for r in results) / len(results)
    avg_sharpe = [r["sharpe"] for r in results if r["sharpe"] is not None]
    avg_dd     = sum(r["max_dd_pct"] for r in results) / len(results)
    total_tr   = sum(r["trades"] for r in results)
    avg_win    = sum(r["win_rate"] for r in results) / len(results)
    print(f"  Tickers tested  : {len(results)}")
    print(f"  Avg return      : {avg_ret:+.1f}%")
    print(f"  Avg Sharpe      : {sum(avg_sharpe)/len(avg_sharpe):.3f}" if avg_sharpe else "  Avg Sharpe      : n/a")
    print(f"  Avg max DD      : {avg_dd:.1f}%")
    print(f"  Total trades    : {total_tr}")
    print(f"  Avg win rate    : {avg_win:.0f}%")
    best  = max(results, key=lambda r: r["total_ret"])
    worst = min(results, key=lambda r: r["total_ret"])
    print(f"\n  Best:  {best['ticker']} {best['total_ret']:+.1f}%")
    print(f"  Worst: {worst['ticker']} {worst['total_ret']:+.1f}%")


if __name__ == "__main__":
    main()
