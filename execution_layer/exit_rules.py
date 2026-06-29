"""Deterministic, zero-LLM exit rules — the default path for closing a held
position intraday. A 15-minute LLM call on every position, every day, is
the easiest way to quietly turn a paper-trading experiment into a live
decision-making system; plain thresholds are the default, LLM review is an
opt-in escalation (see runtime.py::_check_intraday_exits).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str


def evaluate_exit(
    avg_entry_price: float,
    current_price: float,
    high_water_mark: float,
    stop_loss_pct: float,
    take_profit_pct: float | None,
    trailing_stop_pct: float,
    trailing_stop_activation_pct: float = 0.0,
) -> ExitDecision:
    """`take_profit_pct=None` disables the fixed target entirely — the
    thesis track uses this to let a winner run instead of capping it at a
    few percent. `trailing_stop_activation_pct` gates the trailing stop so
    it only engages once the position is up that much from entry — the
    momentum track leaves this at 0 (trail from the first tick of profit);
    the thesis track sets it higher (e.g. only trail after +20%) so normal
    short-term volatility on a long-horizon pick doesn't stop it out early.
    """
    pnl_pct = (current_price - avg_entry_price) / avg_entry_price

    if pnl_pct <= -stop_loss_pct:
        return ExitDecision(True, f"stop-loss hit: {pnl_pct:+.2%} from entry {avg_entry_price:.2f}")

    if take_profit_pct is not None and pnl_pct >= take_profit_pct:
        return ExitDecision(True, f"target hit: {pnl_pct:+.2%} from entry {avg_entry_price:.2f}")

    gain_to_peak_pct = (high_water_mark - avg_entry_price) / avg_entry_price
    if high_water_mark > avg_entry_price and gain_to_peak_pct >= trailing_stop_activation_pct:
        drawdown_from_peak = (current_price - high_water_mark) / high_water_mark
        if drawdown_from_peak <= -trailing_stop_pct:
            return ExitDecision(
                True, f"trailing stop hit: {drawdown_from_peak:+.2%} from peak {high_water_mark:.2f}"
            )

    return ExitDecision(False, "")
