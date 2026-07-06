"""Standalone runner — fetches real historical data ONCE, then sweeps the
full pullback_5d parameter grid against the cached data (no re-fetching per
cell). Takes a while on the first fetch (full PIT universe x 3 years); run
in the background.

Usage:
    source .venv/bin/activate
    python backtest/run_pullback_5d_backtest.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from config.settings import get_settings  # noqa: E402
from data_layer.openbb_client import OpenBBDataClient  # noqa: E402
from backtest.metrics import summarize  # noqa: E402
from backtest.portfolio_metrics import compute_exposure_and_drawdown  # noqa: E402
from backtest.pullback_5d_backtest import fetch_and_prepare_universe, run_pullback_5d_on_prepared  # noqa: E402

_ENTRY_SLIPPAGE = 0.005
_EXIT_SLIPPAGE = 0.003
_PULLBACK_BANDS = [(0.05, 0.10), (0.07, 0.12)]
_STOPS = [0.06, 0.08, 0.10]
_TIME_EXITS = [3, 5, 8]
_DEFAULT_MIN_DOLLAR_VOL = 20_000_000.0


def _cell_stats(trades) -> dict:
    report = summarize(trades)
    return {
        "n": report.total_trades,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "mean_return_pct": report.mean_return_pct,
    }


def _neighbors(i: int, j: int, k: int, shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    result = []
    for dim, size in enumerate(shape):
        idx = [i, j, k]
        if idx[dim] - 1 >= 0:
            n = list(idx); n[dim] -= 1
            result.append(tuple(n))
        if idx[dim] + 1 < size:
            n = list(idx); n[dim] += 1
            result.append(tuple(n))
    return result


def main() -> None:
    settings = get_settings()
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

    logger.info("Fetching + preparing PIT universe (once for the whole sweep)...")
    prepared = fetch_and_prepare_universe(data_client, years_back=3, use_pit_universe=True)
    logger.info(
        "Prepared %d tickers (%d skipped: %d no data, %d insufficient history)",
        len(prepared.tickers_data), prepared.total_tickers - len(prepared.tickers_data),
        prepared.no_data_count, prepared.insufficient_history_count,
    )

    grid_results = {}
    all_cells_trades = {}
    for (pb_min, pb_max), stop, time_exit in product(_PULLBACK_BANDS, _STOPS, _TIME_EXITS):
        key = (f"{pb_min:.0%}-{pb_max:.0%}", stop, time_exit)
        logger.info("Running cell: pullback=%s stop=%.0f%% time_exit=%d", key[0], stop * 100, time_exit)
        trades = run_pullback_5d_on_prepared(
            prepared, pullback_min_pct=pb_min, pullback_max_pct=pb_max,
            stop_loss_pct=stop, time_exit_sessions=time_exit,
            min_dollar_vol=_DEFAULT_MIN_DOLLAR_VOL,
            entry_slippage_pct=_ENTRY_SLIPPAGE, exit_slippage_pct=_EXIT_SLIPPAGE,
        )
        grid_results[key] = _cell_stats(trades)
        all_cells_trades[key] = trades

    # Best cell by profit factor (None/inf handled: treat None as -inf, inf as a large finite number for sorting)
    def _pf_sort_key(item):
        pf = item[1]["profit_factor"]
        if pf is None:
            return float("-inf")
        return pf

    ranked = sorted(grid_results.items(), key=_pf_sort_key, reverse=True)
    best_key, best_stats = ranked[0]

    keys_list = list(grid_results.keys())
    band_order = [f"{a:.0%}-{b:.0%}" for a, b in _PULLBACK_BANDS]
    best_i = band_order.index(best_key[0])
    best_j = _STOPS.index(best_key[1])
    best_k = _TIME_EXITS.index(best_key[2])
    neighbor_indices = _neighbors(best_i, best_j, best_k, (len(band_order), len(_STOPS), len(_TIME_EXITS)))
    neighbor_keys = [(band_order[i], _STOPS[j], _TIME_EXITS[k]) for i, j, k in neighbor_indices]
    neighbor_pfs = [grid_results[nk]["profit_factor"] for nk in neighbor_keys if grid_results[nk]["profit_factor"] is not None]

    best_pf = best_stats["profit_factor"] or 0.0
    if neighbor_pfs:
        neighbor_avg = sum(neighbor_pfs) / len(neighbor_pfs)
        # STABLE if the best cell isn't wildly detached from its neighbors' average.
        is_stable = neighbor_avg > 0 and (best_pf / neighbor_avg) < 1.5
        stability_verdict = "STABLE" if is_stable else "OVERFIT — best cell is a spike relative to its neighbors"
    else:
        stability_verdict = "UNKNOWN — no valid neighbor cells to compare against"

    # Full report for the best cell: year split + exposure/drawdown.
    best_trades = all_cells_trades[best_key]
    closed_best = [t for t in best_trades if t.is_closed]
    years = sorted({t.exit_date.year for t in closed_best}) if closed_best else []
    year_split = {}
    for y in years:
        year_trades = [t for t in closed_best if t.exit_date.year == y]
        year_split[y] = _cell_stats(year_trades)

    exposure = compute_exposure_and_drawdown(best_trades)
    overall = summarize(best_trades)

    # ── Write JSON trade log for the best cell ──────────────────────────────
    with open("backtest/pullback_5d_results.json", "w") as f:
        json.dump([
            {
                "ticker": t.ticker, "entry_date": t.entry_date.isoformat(), "entry_price": t.entry_price,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None, "exit_price": t.exit_price,
                "exit_reason": t.exit_reason, "return_pct": t.return_pct,
            }
            for t in best_trades
        ], f, indent=2)

    # ── Write markdown report ───────────────────────────────────────────────
    lines = []
    lines.append("# pullback_5d Backtest Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Universe: point-in-time S&P 500 membership (fja05680/sp500), same as the thesis backtest.")
    lines.append(f"- Prepared {len(prepared.tickers_data)}/{prepared.total_tickers} tickers "
                 f"({prepared.no_data_count} no price data, {prepared.insufficient_history_count} insufficient history).")
    lines.append(f"- Slippage: {_ENTRY_SLIPPAGE:.1%} entry / {_EXIT_SLIPPAGE:.1%} exit — same convention as the thesis backtest.")
    lines.append("- Entry: 5-10% pullback below 20-session high, close > 200-day SMA, 20-day avg dollar volume >= $20M. "
                 "Entry fills at next session's open.")
    lines.append("- Exit: -8% stop (intrabar low) OR flat time exit at session 5 close, whichever comes first. No trailing stop/profit target.")
    lines.append("")
    lines.append("## Best cell (by profit factor)")
    lines.append("")
    lines.append(f"**pullback={best_key[0]}, stop=-{best_key[1]:.0%}, time_exit={best_key[2]} sessions**")
    lines.append("")
    lines.append(f"- n trades: {overall.total_trades} (+ {overall.still_open_at_backtest_end} still open at backtest end)")
    lines.append(f"- Win rate: {overall.win_rate:.1%}" if overall.win_rate is not None else "- Win rate: n/a")
    lines.append(f"- Avg win: {overall.avg_win_pct:.2%}" if overall.avg_win_pct is not None else "- Avg win: n/a")
    lines.append(f"- Avg loss: {overall.avg_loss_pct:.2%}" if overall.avg_loss_pct is not None else "- Avg loss: n/a")
    lines.append(f"- Profit factor: {overall.profit_factor:.3f}" if overall.profit_factor not in (None, float("inf")) else f"- Profit factor: {overall.profit_factor}")
    lines.append(f"- Mean return/trade (after slippage): {overall.mean_return_pct:.2%}" if overall.mean_return_pct is not None else "- Mean return/trade: n/a")
    lines.append(f"- Max drawdown (equal-weighted equity curve): {exposure.max_drawdown_pct:.2%}" if exposure.max_drawdown_pct is not None else "- Max drawdown: n/a")
    lines.append(f"- Avg concurrent positions: {exposure.avg_concurrent_positions:.2f} (max {exposure.max_concurrent_positions})")
    lines.append(f"- Confidence: {overall.confidence_note}")
    lines.append("")
    lines.append(f"**Stability verdict: {stability_verdict}**")
    if neighbor_pfs:
        lines.append(f"(best cell PF={best_pf:.3f} vs. neighbor avg PF={sum(neighbor_pfs)/len(neighbor_pfs):.3f}, n={len(neighbor_pfs)} neighbors)")
    lines.append("")
    lines.append("## Year split (best cell)")
    lines.append("")
    lines.append("| Year | n | Win rate | Profit factor | Mean return |")
    lines.append("|---|---|---|---|---|")
    for y in years:
        s = year_split[y]
        pf_str = f"{s['profit_factor']:.3f}" if s["profit_factor"] not in (None, float("inf")) else str(s["profit_factor"])
        wr_str = f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "n/a"
        mr_str = f"{s['mean_return_pct']:.2%}" if s["mean_return_pct"] is not None else "n/a"
        lines.append(f"| {y} | {s['n']} | {wr_str} | {pf_str} | {mr_str} |")
    lines.append("")
    lines.append("## Full 18-cell grid")
    lines.append("")
    lines.append("| Pullback band | Stop | Time exit | n | Win rate | Profit factor | Mean return |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in keys_list:
        s = grid_results[key]
        pf_str = f"{s['profit_factor']:.3f}" if s["profit_factor"] not in (None, float("inf")) else str(s["profit_factor"])
        wr_str = f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "n/a"
        mr_str = f"{s['mean_return_pct']:.2%}" if s["mean_return_pct"] is not None else "n/a"
        marker = " **<- BEST**" if key == best_key else ""
        lines.append(f"| {key[0]} | -{key[1]:.0%} | {key[2]} | {s['n']} | {wr_str} | {pf_str} | {mr_str}{marker} |")
    lines.append("")

    report_text = "\n".join(lines)
    with open("backtest/pullback_5d_report.md", "w") as f:
        f.write(report_text)

    print(report_text)
    print("\nWritten to backtest/pullback_5d_report.md and backtest/pullback_5d_results.json")


if __name__ == "__main__":
    sys.exit(main())
