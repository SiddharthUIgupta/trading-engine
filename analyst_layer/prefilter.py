"""Deterministic, zero-LLM pre-filter — gates whether a ticker is even worth
the expensive 4-agent consensus today. No model in the loop here: pure
arithmetic over OpenBB data, so cost is flat regardless of watchlist size,
and there's no hallucination risk in deciding "is anything happening here."
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from data_layer.models import FilingSummary, FilingType, PriceBar, PriceSeries, SentimentSnapshot


@dataclass(frozen=True)
class FilterSignal:
    passed: bool
    regime: str  # "bullish_crossover" | "bearish_crossover" | "neutral"
    reasons: list[str] = field(default_factory=list)


def _sma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _rsi(closes: list[float], period: int) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(len(closes) - period, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0  # zero price movement at all — undefined (0/0), not "all gains"
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _rsquared(closes: list[float], window: int) -> tuple[float, float] | None:
    """R² and slope sign of a linear fit over the last `window` closes.

    From qlib's Alpha158 factor library. R² measures how cleanly prices have
    trended — a high R² with positive slope means the move is consistent and
    linear, not choppy. More informative than SMA crossover alone because a
    crossover can fire on a single noisy candle; R² ≥ 0.80 requires sustained
    directional consistency over the whole window.

    Returns (r_squared, slope) or None if insufficient data.
    """
    if len(closes) < window:
        return None
    y = closes[-window:]
    n = len(y)
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    ss_xy = sum((i - x_mean) * (y[i] - y_mean) for i in range(n))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((v - y_mean) ** 2 for v in y)
    if ss_xx == 0 or ss_yy == 0:
        return None
    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
    return r_squared, slope


def _range_position(bars: list[PriceBar], window: int) -> float | None:
    """Where today's close sits within the N-day high/low range (0=low, 1=high).

    Williams %R / Stochastic variant from qlib's Alpha158. More context-aware
    than RSI because it's anchored to actual price extremes over the window,
    not momentum of closes. ≤ 0.15 = at the low end of the range (potential
    support/oversold positioning); ≥ 0.85 = at the high end (extended).
    """
    if len(bars) < window:
        return None
    recent = bars[-window:]
    high_n = max(b.high for b in recent)
    low_n = min(b.low for b in recent)
    if high_n == low_n:
        return None
    return (bars[-1].close - low_n) / (high_n - low_n)


def _kbar(bars: list[PriceBar]) -> tuple[float, float] | None:
    """Candlestick body metrics from qlib Alpha158.

    KBAR = (close - open) / (high - low):  body size as fraction of full range.
      Near ±1 = strong directional day (close at extreme of range).
      Near  0  = indecisive candle (doji / spinning top).
    KSFT = (2*close - high - low) / (high - low):  close position in the range.
      Positive = closed in upper half (bullish bias); negative = lower half.

    Returns (kbar, ksft) of the most recent bar, or None if the bar has no range.
    """
    if not bars:
        return None
    b = bars[-1]
    rng = b.high - b.low
    if rng < 1e-9:
        return None
    kbar = (b.close - b.open) / rng
    ksft = (2 * b.close - b.high - b.low) / rng
    return kbar, ksft


def _volume_pressure(bars: list[PriceBar], window: int) -> tuple[float, float] | None:
    """Up/down volume pressure ratio from qlib Alpha158 (VSUMP / VSUMN).

    VSUMP = fraction of window volume traded on up-close days.
    VSUMN = fraction of window volume traded on down-close days.

    VSUMP > 0.65 → buyers dominating volume (bullish).
    VSUMN > 0.65 → sellers dominating volume (bearish distribution).
    """
    if len(bars) < window + 1:
        return None
    recent = bars[-(window + 1):]
    vol_up = vol_down = 0.0
    total_vol = 0.0
    for i in range(1, len(recent)):
        v = recent[i].volume
        total_vol += v
        if recent[i].close > recent[i - 1].close:
            vol_up += v
        elif recent[i].close < recent[i - 1].close:
            vol_down += v
    if total_vol == 0:
        return None
    return vol_up / total_vol, vol_down / total_vol


def _amihud_illiquidity(bars: list[PriceBar], window: int) -> float | None:
    """Amihud (2002) illiquidity ratio — from qlib Alpha158.

    ILLIQ = mean(|daily_return| / dollar_volume) × 10^6

    High ILLIQ = price moves a lot per dollar traded = illiquid.
    Liquid stocks (low ILLIQ) have tight spreads and good execution quality.
    We flag stocks above a threshold to warn agents of execution risk.
    """
    if len(bars) < window + 1:
        return None
    recent = bars[-(window + 1):]
    ratios = []
    for i in range(1, len(recent)):
        b = recent[i]
        dollar_vol = b.close * b.volume
        if dollar_vol == 0:
            continue
        ret = abs(b.close / recent[i - 1].close - 1)
        ratios.append(ret / dollar_vol)
    if not ratios:
        return None
    return (sum(ratios) / len(ratios)) * 1_000_000


def _return_volume_corr(bars: list[PriceBar], window: int) -> float | None:
    """Rolling correlation of daily returns vs log-volume changes over `window` days.

    From qlib's Alpha158: Corr(close/Ref(close,1), Log(volume/Ref(volume,1)+1), N).
    Positive = volume rises on up days (healthy trend confirmation).
    Negative = volume rises on down days (distribution / selling into strength).
    Threshold of |0.30| separates noise from a meaningful directional bias.
    """
    if len(bars) < window + 1:
        return None
    recent = bars[-(window + 1):]
    returns, vol_changes = [], []
    for i in range(1, len(recent)):
        returns.append(recent[i].close / recent[i - 1].close - 1)
        v_prev = recent[i - 1].volume
        v_curr = recent[i].volume
        vol_changes.append(math.log(v_curr / v_prev + 1) if v_prev > 0 else 0.0)

    n = len(returns)
    if n < 2:
        return None
    r_mean = sum(returns) / n
    v_mean = sum(vol_changes) / n
    num = sum((returns[i] - r_mean) * (vol_changes[i] - v_mean) for i in range(n))
    r_std = math.sqrt(sum((r - r_mean) ** 2 for r in returns) / n)
    v_std = math.sqrt(sum((v - v_mean) ** 2 for v in vol_changes) / n)
    if r_std == 0 or v_std == 0:
        return None
    return num / (n * r_std * v_std)


def compute_regime(closes: list[float], short_window: int, long_window: int) -> str:
    """Public so callers (e.g. runtime.py's intraday reversal check) can
    recompute "what regime is this ticker in right now" without duplicating
    the SMA-crossover logic.
    """
    sma_short = _sma(closes, short_window)
    sma_long = _sma(closes, long_window)
    if sma_short is None or sma_long is None:
        return "neutral"
    return "bullish_crossover" if sma_short > sma_long else "bearish_crossover"


def evaluate_ticker(
    price_series: PriceSeries,
    sentiment: SentimentSnapshot,
    filings: list[FilingSummary],
    today: date,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
    sma_short_window: int,
    sma_long_window: int,
    volume_spike_multiplier: float,
    sentiment_abs_threshold: float,
    recent_filing_days: int,
    shares_float: int | None = None,
) -> FilterSignal:
    bars = price_series.bars
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    reasons: list[str] = []

    regime = compute_regime(closes, sma_short_window, sma_long_window)
    if regime != "neutral" and len(closes) > sma_long_window:
        prior_regime = compute_regime(closes[:-1], sma_short_window, sma_long_window)
        if prior_regime != "neutral" and prior_regime != regime:
            reasons.append(f"moving-average crossover flipped to {regime}")

    rsi = _rsi(closes, rsi_period)
    if rsi is not None and (rsi <= rsi_oversold or rsi >= rsi_overbought):
        reasons.append(f"RSI {rsi:.1f} outside [{rsi_oversold:.0f}, {rsi_overbought:.0f}] band")

    if len(volumes) >= 11:
        baseline = statistics.mean(volumes[-11:-1])
        if baseline > 0 and volumes[-1] >= volume_spike_multiplier * baseline:
            reasons.append(f"volume spike: {volumes[-1]} vs {volume_spike_multiplier:.1f}x average ({baseline:.0f})")

    if abs(sentiment.score) >= sentiment_abs_threshold:
        reasons.append(f"sentiment score {sentiment.score:+.2f} beyond {sentiment_abs_threshold:.2f} threshold")

    # Turnover rate: volume ÷ float. More manipulation-resistant than headline
    # sentiment; > 3% means a meaningful fraction of the float changed hands today,
    # which is a structurally different signal from a volume spike vs. average
    # (RVOL can spike on a thin day; turnover rate can't).
    if shares_float is not None and shares_float > 0 and volumes:
        turnover_rate = volumes[-1] / shares_float
        if turnover_rate >= 0.03:
            reasons.append(
                f"turnover rate {turnover_rate:.1%} of float ({shares_float:,} shares) — elevated float activity"
            )

    # Closest proxy available to "earnings surprise" without an analyst EPS
    # estimate to compare against: a recently filed 8-K usually means a real
    # material event just happened, worth a closer look regardless of what
    # the price/volume/sentiment numbers say yet.
    cutoff = today - timedelta(days=recent_filing_days)
    recent_8k = [f for f in filings if f.filing_type == FilingType.EIGHT_K and f.filed_on >= cutoff]
    if recent_8k:
        reasons.append(f"recent 8-K filed {recent_8k[0].filed_on.isoformat()}")

    # ── qlib Alpha158 factors (OHLCV-only, no extra data fetch) ──────────────

    # R² trend quality: linear fit over 20 days. High R² + consistent slope means
    # the move is linear and sustained, not noise. Stronger confirmation than a
    # single SMA crossover which can fire on one spiky candle.
    trend_result = _rsquared(closes, window=20)
    if trend_result is not None:
        r_sq, slope = trend_result
        if r_sq >= 0.80:
            direction = "up" if slope > 0 else "down"
            reasons.append(f"clean {direction}trend: R²={r_sq:.2f} over 20 days")

    # N-day range position: where today's close sits in the 30-day high/low band.
    # More context-aware than RSI — anchored to actual price extremes, not
    # momentum of closes.
    range_pos = _range_position(bars, window=30)
    if range_pos is not None:
        if range_pos <= 0.15:
            reasons.append(
                f"at low end of 30-day range ({range_pos:.0%}) — potential support / oversold positioning"
            )
        elif range_pos >= 0.85:
            reasons.append(
                f"at high end of 30-day range ({range_pos:.0%}) — extended / potential resistance"
            )

    # Return-volume correlation over 20 days. Positive = volume rises on up days
    # (trend confirmation); negative = volume rises on down days (distribution).
    rv_corr = _return_volume_corr(bars, window=20)
    if rv_corr is not None and abs(rv_corr) >= 0.30:
        label = "volume confirming direction" if rv_corr > 0 else "volume rising on down days (distribution signal)"
        reasons.append(f"return-volume correlation {rv_corr:+.2f} over 20 days — {label}")

    # ── Alpha158: candlestick body (KBAR / KSFT) ─────────────────────────────
    kbar_result = _kbar(bars)
    if kbar_result is not None:
        kbar, ksft = kbar_result
        if abs(kbar) >= 0.70:
            direction = "bullish" if kbar > 0 else "bearish"
            reasons.append(
                f"strong {direction} candle body: KBAR={kbar:+.2f} (close vs open spans {abs(kbar):.0%} of day's range)"
            )
        if abs(ksft) >= 0.60:
            half = "upper" if ksft > 0 else "lower"
            reasons.append(
                f"close in {half} half of day's range: KSFT={ksft:+.2f} (indicates {half}-half momentum)"
            )

    # ── Alpha158: volume directional pressure (VSUMP / VSUMN) ────────────────
    pressure = _volume_pressure(bars, window=15)
    if pressure is not None:
        vsump, vsumn = pressure
        if vsump >= 0.65:
            reasons.append(
                f"bullish volume pressure: {vsump:.0%} of 15-day volume traded on up-close days (VSUMP)"
            )
        elif vsumn >= 0.65:
            reasons.append(
                f"bearish volume pressure: {vsumn:.0%} of 15-day volume traded on down-close days (VSUMN — distribution)"
            )

    # ── Alpha158: Amihud illiquidity ─────────────────────────────────────────
    illiq = _amihud_illiquidity(bars, window=20)
    if illiq is not None and illiq > 0.5:
        reasons.append(
            f"elevated illiquidity (ILLIQ={illiq:.3f}): price moves sharply per dollar traded — watch execution slippage"
        )

    return FilterSignal(passed=bool(reasons), regime=regime, reasons=reasons or ["no threshold crossed"])
