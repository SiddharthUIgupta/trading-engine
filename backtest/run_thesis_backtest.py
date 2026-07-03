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

import sys
use_pit = "--no-pit" not in sys.argv  # pass --no-pit to skip PIT and use Wikipedia (faster but biased)

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

pit_label = "pit_universe" if use_pit else "wikipedia_current_biased"
slippage_note = "0.5%_entry_0.3%_exit"

# ── Scenario 1: PIT universe, zero slippage ───────────────────────────────────
logging.getLogger("thesis_backtest").info("Running scenario 1: %s, zero slippage", pit_label)
trades_clean = run_thesis_backtest(data_client, years_back=3, use_pit_universe=use_pit)
report_clean = summarize(trades_clean)

# ── Scenario 2: PIT universe + realistic slippage ────────────────────────────
logging.getLogger("thesis_backtest").info("Running scenario 2: %s, 0.5%%/0.3%% slippage", pit_label)
trades_slip = run_thesis_backtest(
    data_client, years_back=3,
    entry_slippage_pct=0.005, exit_slippage_pct=0.003,
    use_pit_universe=use_pit,
)
report_slip = summarize(trades_slip)

result = {
    f"{pit_label}_zero_slippage": _fmt(report_clean),
    f"{pit_label}_with_slippage": _fmt(report_slip),
    "_note": (
        "PIT universe uses fja05680/sp500 point-in-time membership — signals only "
        "generated when ticker was actually in the S&P 500 on that bar date. "
        "This eliminates look-ahead survivorship bias for buy-the-drawdown strategies."
        if use_pit else
        "WARNING: Wikipedia current-constituent list — SURVIVORSHIP BIAS. "
        "Results are optimistic for buy-the-drawdown strategies."
    ),
}

print("\n=== THESIS BACKTEST: SLIPPAGE IMPACT ===")
print(json.dumps(result, indent=2))

with open("backtest_results_thesis.json", "w") as f:
    json.dump([
        {
            "ticker": t.ticker, "entry_date": t.entry_date.isoformat(), "entry_price": t.entry_price,
            "exit_date": t.exit_date.isoformat() if t.exit_date else None, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "return_pct": t.return_pct,
            "universe": pit_label, "slippage": slippage_note,
        }
        for t in trades_slip
    ], f, indent=2)
print(f"\nFull trade log ({pit_label}, with slippage) written to backtest_results_thesis.json")
