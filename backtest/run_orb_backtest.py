"""Standalone runner — fetches real intraday data and runs the ORB
backtest, then prints a report.
"""
from __future__ import annotations

import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from config.settings import get_settings  # noqa: E402
from data_layer.openbb_client import OpenBBDataClient  # noqa: E402
from backtest.orb_backtest import run_orb_backtest  # noqa: E402
from backtest.universe import get_momentum_backtest_universe  # noqa: E402
from backtest.metrics import summarize  # noqa: E402

settings = get_settings()
data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

universe = get_momentum_backtest_universe(data_client, limit=300)
print(f"Running ORB backtest (volume-confirmed) against {len(universe)} tickers")

trades = run_orb_backtest(data_client, universe=universe, volume_confirmation_multiple=1.5)
report = summarize(trades)

print("\n=== ORB BACKTEST REPORT (volume confirmation >= 1.5x) ===")
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

with open("backtest_results_orb_volconfirm.json", "w") as f:
    json.dump([
        {
            "ticker": t.ticker, "direction": t.direction, "entry_date": t.entry_date.isoformat(),
            "entry_price": t.entry_price, "exit_date": t.exit_date.isoformat() if t.exit_date else None,
            "exit_price": t.exit_price, "exit_reason": t.exit_reason, "return_pct": t.return_pct,
        }
        for t in trades
    ], f, indent=2)
print("\nFull trade log written to backtest_results_orb_volconfirm.json")
