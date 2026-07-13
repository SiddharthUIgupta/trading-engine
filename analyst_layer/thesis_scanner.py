"""Thesis-track screen — deterministic, zero-LLM. The screen's only job
is "is this worth a closer look," not "is this a good business." That
judgment belongs to the Fundamental/SEC and Macro/Sentiment agents
downstream. The min-pullback floor is 0.0 by default (configurable via
THESIS_MIN_PULLBACK_PCT), meaning breakout names near their 52-week highs
pass through and are evaluated by the LLM — the screen does not hard-block
momentum or high-performing stocks. The only structural gate is the
max-pullback ceiling (default 50%): beyond that, a drawdown is more likely
to be genuine impairment than dislocation, and the recovery track handles
those with different parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_layer.models import PriceSeries, ThesisCandidate


@dataclass(frozen=True)
class ThesisSignal:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0  # range extremity: 0=middle of 52w range (least notable), 1=at either extreme


def evaluate_thesis_candidate(
    candidate: ThesisCandidate, min_pullback_pct: float, max_pullback_pct: float
) -> ThesisSignal:
    """With min_pullback_pct=0.0 (default): always passes the floor check,
    so breakout names near their 52-week highs reach the LLM alongside
    classic pullback plays. Score = range extremity: how close the stock is
    to either end of its 52-week range (1.0 = at high or low, 0.0 = dead
    middle). This ranks both breakouts AND deep dislocations above boring
    mid-range drifters, without bias toward either direction.
    """
    pullback_pct = (candidate.year_high - candidate.price) / candidate.year_high
    passed = min_pullback_pct <= pullback_pct <= max_pullback_pct
    if pullback_pct > max_pullback_pct:
        reason = f"pulled back {pullback_pct:.1%} from 52w high {candidate.year_high:.2f} — beyond {max_pullback_pct:.1%} ceiling (likely distress, not dislocation)"
    else:
        reason = (
            f"pulled back {pullback_pct:.1%} from 52w high {candidate.year_high:.2f} "
            f"(range: {candidate.year_low:.2f}–{candidate.year_high:.2f})"
        )

    if not passed:
        return ThesisSignal(passed=False, reasons=[reason], score=0.0)

    range_width = candidate.year_high - candidate.year_low
    if range_width > 0:
        range_pos = (candidate.price - candidate.year_low) / range_width  # 0=at 52w low, 1=at 52w high
        score = abs(range_pos - 0.5) * 2  # 0=middle of range, 1=at either extreme
    else:
        score = 0.5

    return ThesisSignal(passed=True, reasons=[reason], score=round(score, 4))


def evaluate_shrink_volume_pullback(price_series: PriceSeries) -> ThesisSignal:
    """Shrink-volume retest signal from ZhuLinsen/daily_stock_analysis.

    Three conjunctive conditions for a high-quality uptrend continuation entry:
    1. MA5 > MA10 > MA20 — uptrend confirmed across multiple timeframes
    2. Price within 2% of MA5 or MA10 — retesting support, not breaking down
    3. Today's volume < 70% of prior 5-day average — sellers absent on the dip

    Used as a ranking boost in the thesis track: candidates that pass all
    three score higher and surface above plain 52-week-high pullbacks that
    lack the structural confirmation. Non-passing candidates are NOT blocked —
    a genuine dislocation thesis doesn't require an intact uptrend.
    """
    bars = price_series.bars
    if len(bars) < 21:
        return ThesisSignal(
            passed=False,
            reasons=["insufficient price history for shrink-volume check (need 21 bars)"],
        )

    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]

    ma5 = sum(closes[-5:]) / 5
    ma10 = sum(closes[-10:]) / 10
    ma20 = sum(closes[-20:]) / 20
    current_price = closes[-1]

    if not (ma5 > ma10 > ma20):
        return ThesisSignal(
            passed=False,
            reasons=[f"no confirmed uptrend: MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f}"],
        )

    near_ma5 = abs(current_price - ma5) / ma5 <= 0.02
    near_ma10 = abs(current_price - ma10) / ma10 <= 0.02
    if not (near_ma5 or near_ma10):
        return ThesisSignal(
            passed=False,
            reasons=[
                f"price {current_price:.2f} not retesting MA5 ({ma5:.2f}) or MA10 ({ma10:.2f})"
                f" — drift too wide for a quality retest"
            ],
        )
    retest_label = "MA5" if near_ma5 else "MA10"
    retest_level = ma5 if near_ma5 else ma10

    avg_vol_5d = sum(volumes[-6:-1]) / 5 if len(volumes) >= 6 else 0.0
    today_vol = volumes[-1]
    vol_ratio = today_vol / avg_vol_5d if avg_vol_5d > 0 else 1.0
    if vol_ratio >= 0.70:
        return ThesisSignal(
            passed=False,
            reasons=[f"volume not shrinking: {vol_ratio:.0%} of 5d avg — sellers still active on pullback"],
        )

    return ThesisSignal(
        passed=True,
        reasons=[
            f"uptrend: MA5={ma5:.2f} > MA10={ma10:.2f} > MA20={ma20:.2f}",
            f"retesting {retest_label} ({retest_level:.2f}) within 2%",
            f"volume at {vol_ratio:.0%} of 5d avg — sellers absent",
        ],
        score=1.0 - vol_ratio,  # quieter = higher score
    )
