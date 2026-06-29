"""Low-float intraday momentum scanner — the user's 7-criteria spec (the
standard "Warrior Trading" style scanner): price above VWAP, 9 EMA above
20 EMA, float under a cap, "clean" price action, a minimum daily gain,
relative volume, and a price band. All seven are conjunctive — a candidate
must clear every one, not just one of them, unlike
analyst_layer/prefilter.py's "any reason trips it" logic. Deliberately
zero-LLM, same reasoning as prefilter.py.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from data_layer.models import PriceBar, PriceSeries


@dataclass(frozen=True)
class MomentumSignal:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0  # today's % gain, for ranking among passed candidates only


def _ema_series(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    alpha = 2 / (period + 1)
    series = [sum(values[:period]) / period]  # seed with SMA
    for v in values[period:]:
        series.append(alpha * v + (1 - alpha) * series[-1])
    return series


def _latest_ema(values: list[float], period: int) -> float | None:
    series = _ema_series(values, period)
    return series[-1] if series else None


def _vwap(bars: list[PriceBar]) -> float | None:
    cum_pv = 0.0
    cum_v = 0
    for bar in bars:
        typical_price = (bar.high + bar.low + bar.close) / 3
        cum_pv += typical_price * bar.volume
        cum_v += bar.volume
    return cum_pv / cum_v if cum_v > 0 else None


def _avg_body_dominance(bars: list[PriceBar], lookback: int) -> float | None:
    """Proxy for "clean price action": how much of each candle is decisive
    body vs. indecisive wick, averaged over the most recent bars. Closer to
    1.0 = decisive moves; closer to 0.0 = choppy, wick-heavy indecision.
    This is an approximation of a visual judgment call, not the judgment
    itself — tune `clean_body_dominance_threshold` if it doesn't match intent.
    """
    recent = bars[-lookback:]
    ratios = [
        abs(bar.close - bar.open) / (bar.high - bar.low)
        for bar in recent
        if (bar.high - bar.low) > 0
    ]
    return statistics.mean(ratios) if ratios else None


def evaluate_low_float_momentum(
    intraday_series: PriceSeries,
    shares_float: int,
    today_percent_change: float,
    today_volume: int,
    average_daily_volume: float,
    max_float_shares: int,
    ema_short_period: int,
    ema_long_period: int,
    min_daily_gain_pct: float,
    clean_body_dominance_threshold: float,
    clean_lookback_bars: int,
    min_relative_volume: float,
    price_min: float,
    price_max: float,
) -> MomentumSignal:
    bars = intraday_series.bars
    closes = [bar.close for bar in bars]
    current_price = closes[-1] if closes else None
    reasons: list[str] = []
    checks_passed = 0

    vwap = _vwap(bars)
    above_vwap = vwap is not None and current_price is not None and current_price > vwap
    checks_passed += above_vwap
    reasons.append(
        f"price {current_price:.2f} {'above' if above_vwap else 'NOT above'} VWAP {vwap:.2f}"
        if vwap is not None and current_price is not None
        else "VWAP unavailable"
    )

    ema_short = _latest_ema(closes, ema_short_period)
    ema_long = _latest_ema(closes, ema_long_period)
    ema_bullish = ema_short is not None and ema_long is not None and ema_short > ema_long
    checks_passed += ema_bullish
    reasons.append(
        f"EMA{ema_short_period} {ema_short:.2f} {'>' if ema_bullish else '<='} EMA{ema_long_period} {ema_long:.2f}"
        if ema_short is not None and ema_long is not None
        else f"insufficient bars for EMA{ema_long_period}"
    )

    low_float = shares_float <= max_float_shares
    checks_passed += low_float
    reasons.append(
        f"float {shares_float:,} {'<=' if low_float else '>'} cap {max_float_shares:,}"
    )

    body_dominance = _avg_body_dominance(bars, clean_lookback_bars)
    is_clean = body_dominance is not None and body_dominance >= clean_body_dominance_threshold
    checks_passed += is_clean
    reasons.append(
        f"body dominance {body_dominance:.2f} {'>=' if is_clean else '<'} {clean_body_dominance_threshold:.2f}"
        if body_dominance is not None
        else "insufficient bars for price-action check"
    )

    strong_gain = today_percent_change >= min_daily_gain_pct
    checks_passed += strong_gain
    reasons.append(
        f"{today_percent_change:+.1%} today {'>=' if strong_gain else '<'} {min_daily_gain_pct:+.1%} minimum"
    )

    relative_volume = today_volume / average_daily_volume if average_daily_volume > 0 else 0.0
    high_rvol = relative_volume >= min_relative_volume
    checks_passed += high_rvol
    reasons.append(
        f"RVOL {relative_volume:.1f}x {'>=' if high_rvol else '<'} {min_relative_volume:.1f}x minimum"
    )

    # price_max == 0 means no upper cap (any price stock allowed)
    above_min = current_price is not None and current_price >= price_min
    below_max = price_max <= 0 or (current_price is not None and current_price <= price_max)
    in_price_band = above_min and below_max
    checks_passed += in_price_band
    cap_str = f"{price_max:.2f}" if price_max > 0 else "∞"
    reasons.append(
        f"price {current_price:.2f} {'within' if in_price_band else 'outside'} [{price_min:.2f}, {cap_str}]"
        if current_price is not None
        else "price unavailable"
    )

    passed = checks_passed == 7
    return MomentumSignal(passed=passed, reasons=reasons, score=today_percent_change if passed else 0.0)
