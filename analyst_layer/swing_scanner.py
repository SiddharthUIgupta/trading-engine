"""Swing trade scanner: screens for multi-week holding candidates (3–6 weeks).

Entry criteria — all conjunctive (every one must pass):
  1. Intact uptrend: price > SMA20 > SMA50.
  2. RSI in the healthy zone [35, 65] — not chasing overbought, not catching a knife.
  3. Minimum average daily value $5M — liquidity for clean fills and exits.

Score ranks passing candidates so the position cap cuts from the bottom.
Exit rules are enforced by runtime._check_swing_exits:
  - Per-position stop at entry * (1 − swing_stop_loss_pct)  [default 8%]
  - Trailing stop after a configured gain  [default: 12% activation, 7% trail]
  - Max hold in calendar days  [default 21]
  - Adverse news catalyst for the specific ticker
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_layer.models import PriceSeries


@dataclass(frozen=True)
class SwingSignal:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def evaluate_swing_candidate(
    price_series: PriceSeries,
    sma_short: int = 20,
    sma_long: int = 50,
    rsi_period: int = 14,
    rsi_max: float = 65.0,
    rsi_min: float = 35.0,
    min_adv_usd: float = 5_000_000,
) -> SwingSignal:
    """Returns a SwingSignal indicating whether this ticker is a valid swing entry.

    All three checks are conjunctive — a fail on any one returns immediately.
    """
    bars = price_series.bars
    if len(bars) < sma_long + 1:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"insufficient history: {len(bars)} bars, need {sma_long + 1}"],
        )

    closes = [b.close for b in bars]
    sma_short_val = sum(closes[-sma_short:]) / sma_short
    sma_long_val = sum(closes[-sma_long:]) / sma_long
    current_price = closes[-1]

    # 1. Uptrend: price > SMA20 > SMA50
    if current_price <= sma_short_val:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"price {current_price:.2f} <= SMA{sma_short} {sma_short_val:.2f} — no uptrend"],
        )
    if current_price <= sma_long_val:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"price {current_price:.2f} <= SMA{sma_long} {sma_long_val:.2f} — below long trend"],
        )
    if sma_short_val <= sma_long_val:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"SMA{sma_short} {sma_short_val:.2f} <= SMA{sma_long} {sma_long_val:.2f} — trend not intact"],
        )

    # 2. RSI in healthy range
    rsi_val = _compute_rsi(closes, rsi_period)
    if rsi_val > rsi_max:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"RSI {rsi_val:.1f} > {rsi_max:.0f} — overbought, avoid chasing"],
        )
    if rsi_val < rsi_min:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"RSI {rsi_val:.1f} < {rsi_min:.0f} — oversold, not a swing entry"],
        )

    # 3. Liquidity: average daily value over last 20 bars
    lookback = min(20, len(bars))
    adv = sum(b.close * b.volume for b in bars[-lookback:]) / lookback
    if adv < min_adv_usd:
        return SwingSignal(
            passed=False, score=0.0,
            reasons=[f"ADV ${adv:,.0f} < minimum ${min_adv_usd:,.0f}"],
        )

    reasons = [
        f"uptrend: {current_price:.2f} > SMA{sma_short} {sma_short_val:.2f} > SMA{sma_long} {sma_long_val:.2f}",
        f"RSI {rsi_val:.1f} in [{rsi_min:.0f}, {rsi_max:.0f}]",
        f"ADV ${adv / 1_000_000:.1f}M",
    ]

    # Score: reward steep trend separation and RSI close to healthy midpoint (50)
    sma_separation = (sma_short_val - sma_long_val) / sma_long_val
    rsi_score = max(1.0 - abs(rsi_val - 50.0) / 15.0, 0.0)
    score = round(sma_separation * 10 + rsi_score, 4)

    return SwingSignal(passed=True, score=score, reasons=reasons)


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """Wilder's smoothed RSI. Returns 50 (neutral) if data is insufficient."""
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
