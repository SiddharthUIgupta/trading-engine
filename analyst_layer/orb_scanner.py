"""Opening Range Breakout (ORB) — deterministic, zero-LLM, decades-old
day-trading pattern: define a range from the first few minutes of the
session, then trade a confirmed breakout above or below it. Single-stock
based — no discovery-screen universe dependency, unlike the low-float
momentum scanner this replaces (that one turned out to almost never fire,
on any universe — see backtest/momentum_backtest.py results).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_layer.models import PriceSeries


@dataclass(frozen=True)
class OrbSignal:
    direction: str  # "long", "short", or "none"
    reasons: list[str] = field(default_factory=list)
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    breakout_bar_index: int | None = None


def evaluate_orb(
    intraday_series: PriceSeries,
    opening_range_minutes: int = 15,
    bar_minutes: int = 5,
    volume_confirmation_multiple: float | None = None,
) -> OrbSignal:
    """Opening range = the high/low of the first `opening_range_minutes` of
    the session. A breakout is a bar CLOSING beyond that range, not just
    touching it intrabar — confirmation that avoids whipsaw on a single
    wick poking through. Returns the first confirmed breakout scanning
    forward in time within this one session.

    `volume_confirmation_multiple`, when set, additionally requires the
    breakout bar's volume to be at least that multiple of the opening
    range's own average volume — standard breakout-trading practice
    (a break on weak volume is a common false signal) rather than
    something invented to flatter a backtest.
    """
    bars = intraday_series.bars
    n_range_bars = max(1, opening_range_minutes // bar_minutes)
    if len(bars) <= n_range_bars:
        return OrbSignal(direction="none", reasons=["not enough bars to establish an opening range"])

    range_bars = bars[:n_range_bars]
    opening_range_high = max(b.high for b in range_bars)
    opening_range_low = min(b.low for b in range_bars)
    range_avg_volume = sum(b.volume for b in range_bars) / len(range_bars)

    def _volume_confirms(bar) -> bool:
        if volume_confirmation_multiple is None:
            return True
        return range_avg_volume > 0 and bar.volume >= volume_confirmation_multiple * range_avg_volume

    for i in range(n_range_bars, len(bars)):
        bar = bars[i]
        if bar.close > opening_range_high and _volume_confirms(bar):
            return OrbSignal(
                direction="long",
                reasons=[f"bar {i} closed {bar.close:.2f} above opening range high {opening_range_high:.2f}"],
                opening_range_high=opening_range_high, opening_range_low=opening_range_low, breakout_bar_index=i,
            )
        if bar.close < opening_range_low and _volume_confirms(bar):
            return OrbSignal(
                direction="short",
                reasons=[f"bar {i} closed {bar.close:.2f} below opening range low {opening_range_low:.2f}"],
                opening_range_high=opening_range_high, opening_range_low=opening_range_low, breakout_bar_index=i,
            )

    return OrbSignal(
        direction="none", reasons=["no confirmed breakout this session"],
        opening_range_high=opening_range_high, opening_range_low=opening_range_low,
    )
