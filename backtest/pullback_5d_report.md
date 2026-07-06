# pullback_5d Backtest Report

Generated: 2026-07-06T14:55:24.119347Z

## Methodology

- Universe: point-in-time S&P 500 membership (fja05680/sp500), same as the thesis backtest.
- Prepared 538/567 tickers (26 no price data, 3 insufficient history).
- Slippage: 0.5% entry / 0.3% exit — same convention as the thesis backtest.
- Entry: 5-10% pullback below 20-session high, close > 200-day SMA, 20-day avg dollar volume >= $20M. Entry fills at next session's open.
- Exit: -8% stop (intrabar low) OR flat time exit at session 5 close, whichever comes first. No trailing stop/profit target.

## Best cell (by profit factor)

**pullback=7%-12%, stop=-10%, time_exit=8 sessions**

- n trades: 4355 (+ 51 still open at backtest end)
- Win rate: 45.8%
- Avg win: 4.81%
- Avg loss: -4.68%
- Profit factor: 0.869
- Mean return/trade (after slippage): -0.33%
- Max drawdown (equal-weighted equity curve): 21.14%
- Avg concurrent positions: 54.19 (max 168)
- Confidence: 4355 closed trades — a reasonable sample, though still not a substitute for live validation.

**Stability verdict: STABLE**
(best cell PF=0.869 vs. neighbor avg PF=0.814, n=3 neighbors)

## Year split (best cell)

| Year | n | Win rate | Profit factor | Mean return |
|---|---|---|---|---|
| 2024 | 1545 | 46.6% | 0.867 | -0.30% |
| 2025 | 1630 | 45.8% | 0.819 | -0.45% |
| 2026 | 1180 | 44.8% | 0.929 | -0.21% |

## Full 18-cell grid

| Pullback band | Stop | Time exit | n | Win rate | Profit factor | Mean return |
|---|---|---|---|---|---|---|
| 5%-10% | -6% | 3 | 11820 | 39.6% | 0.561 | -0.71% |
| 5%-10% | -6% | 5 | 9344 | 43.0% | 0.731 | -0.52% |
| 5%-10% | -6% | 8 | 7628 | 43.6% | 0.814 | -0.44% |
| 5%-10% | -8% | 3 | 11746 | 39.8% | 0.558 | -0.72% |
| 5%-10% | -8% | 5 | 9162 | 43.5% | 0.727 | -0.54% |
| 5%-10% | -8% | 8 | 7320 | 45.2% | 0.823 | -0.42% |
| 5%-10% | -10% | 3 | 11733 | 39.9% | 0.562 | -0.72% |
| 5%-10% | -10% | 5 | 9114 | 43.6% | 0.725 | -0.54% |
| 5%-10% | -10% | 8 | 7213 | 46.1% | 0.841 | -0.38% |
| 7%-12% | -6% | 3 | 6943 | 41.6% | 0.634 | -0.61% |
| 7%-12% | -6% | 5 | 5595 | 43.9% | 0.768 | -0.47% |
| 7%-12% | -6% | 8 | 4635 | 43.2% | 0.850 | -0.37% |
| 7%-12% | -8% | 3 | 6874 | 41.9% | 0.626 | -0.63% |
| 7%-12% | -8% | 5 | 5483 | 44.4% | 0.750 | -0.53% |
| 7%-12% | -8% | 8 | 4429 | 44.9% | 0.852 | -0.37% |
| 7%-12% | -10% | 3 | 6858 | 42.0% | 0.625 | -0.63% |
| 7%-12% | -10% | 5 | 5441 | 44.8% | 0.748 | -0.53% |
| 7%-12% | -10% | 8 | 4355 | 45.8% | 0.869 | -0.33% **<- BEST** |
