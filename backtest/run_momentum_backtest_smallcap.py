"""Re-runs the momentum backtest against the small-cap discovery screen
universe (aggressive_small_caps/undervalued_growth -- the same one the
thesis track already uses) instead of active/gainers/losers, since that
default universe was just confirmed to have ~0% float-cap pass rate.
This isolates whether the entry LOGIC has edge, separate from the
universe-sourcing problem.
"""
from __future__ import annotations

import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from config.settings import get_settings  # noqa: E402
from data_layer.openbb_client import OpenBBDataClient  # noqa: E402
from backtest.momentum_backtest import run_momentum_backtest  # noqa: E402
from backtest.metrics import summarize  # noqa: E402

settings = get_settings()
data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

thesis_universe = data_client.get_thesis_universe()
universe = [c.symbol for c in thesis_universe]
print(f"Using {len(universe)} small-cap-screen tickers instead of active/gainers/losers")

trades = run_momentum_backtest(data_client, universe=universe)
report = summarize(trades)

print("\n=== MOMENTUM BACKTEST REPORT (small-cap universe) ===")
print(json.dumps(
    {
        "total_closed_trades": report.total_trades,
        "still_open_at_backtest_end": report.still_open_at_backtest_end,
        "win_rate": report.win_rate,
        "avg_win_pct": report.avg_win_pct,
        "avg_loss_pct": report.avg_loss_pct,
        "profit_factor": report.profit_factor,
        "best_trade_pct": report.best_trade_pct,
        "worst_trade_pct": report.worst_trade_pct,
        "mean_return_pct": report.mean_return_pct,
        "median_return_pct": report.median_return_pct,
        "confidence_note": report.confidence_note,
    },
    indent=2,
))

with open("backtest_results_momentum_smallcap.json", "w") as f:
    json.dump([
        {
            "ticker": t.ticker, "entry_date": t.entry_date.isoformat(), "entry_price": t.entry_price,
            "exit_date": t.exit_date.isoformat() if t.exit_date else None, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "return_pct": t.return_pct,
        }
        for t in trades
    ], f, indent=2)
print("\nFull trade log written to backtest_results_momentum_smallcap.json")
