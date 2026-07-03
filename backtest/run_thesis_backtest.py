"""Standalone runner — fetches real historical data and runs the thesis
backtest, then prints a report. Takes a while (full S&P 500 x 3 years);
run in the background and check logs/thesis_backtest.log.
"""
from __future__ import annotations

import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from config.settings import get_settings  # noqa: E402
from data_layer.openbb_client import OpenBBDataClient  # noqa: E402
from backtest.thesis_backtest import run_thesis_backtest  # noqa: E402
from backtest.metrics import summarize  # noqa: E402

settings = get_settings()
data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

universe = None  # defaults to S&P 500

def _fmt(report) -> dict:
    return {
        "total_closed_trades": report.total_trades,
        "still_open_at_backtest_end": report.still_open_at_backtest_end,
        "win_rate": f"{report.win_rate:.1%}" if report.win_rate is not None else None,
        "avg_win_pct": f"{report.avg_win_pct:.2%}" if report.avg_win_pct is not None else None,
        "avg_loss_pct": f"{report.avg_loss_pct:.2%}" if report.avg_loss_pct is not None else None,
        "profit_factor": round(report.profit_factor, 3) if report.profit_factor is not None else None,
        "mean_return_pct": f"{report.mean_return_pct:.2%}" if report.mean_return_pct is not None else None,
        "median_return_pct": f"{report.median_return_pct:.2%}" if report.median_return_pct is not None else None,
        "best_trade_pct": f"{report.best_trade_pct:.2%}" if report.best_trade_pct is not None else None,
        "worst_trade_pct": f"{report.worst_trade_pct:.2%}" if report.worst_trade_pct is not None else None,
        "confidence_note": report.confidence_note,
    }

# ── Scenario 1: zero slippage (original backtest) ────────────────────────────
logging.getLogger("thesis_backtest").info("Running scenario 1: zero slippage")
trades_clean = run_thesis_backtest(data_client, universe=universe, years_back=3)
report_clean = summarize(trades_clean)

# ── Scenario 2: realistic slippage (0.5% entry, 0.3% exit) ──────────────────
logging.getLogger("thesis_backtest").info("Running scenario 2: 0.5%% entry / 0.3%% exit slippage")
trades_slip = run_thesis_backtest(
    data_client, universe=universe, years_back=3,
    entry_slippage_pct=0.005, exit_slippage_pct=0.003,
)
report_slip = summarize(trades_slip)

result = {
    "zero_slippage": _fmt(report_clean),
    "with_slippage_0.5pct_entry_0.3pct_exit": _fmt(report_slip),
}

print("\n=== THESIS BACKTEST: SLIPPAGE IMPACT ===")
print(json.dumps(result, indent=2))

with open("backtest_results_thesis.json", "w") as f:
    json.dump([
        {
            "ticker": t.ticker, "entry_date": t.entry_date.isoformat(), "entry_price": t.entry_price,
            "exit_date": t.exit_date.isoformat() if t.exit_date else None, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "return_pct": t.return_pct,
            "slippage": "0.5%_entry_0.3%_exit",
        }
        for t in trades_slip
    ], f, indent=2)
print("\nFull trade log (with slippage) written to backtest_results_thesis.json")
