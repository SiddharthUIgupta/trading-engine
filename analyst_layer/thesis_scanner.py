"""Thesis-track screen — deterministic, zero-LLM, the opposite shape of
the momentum scanner's net. Where momentum requires a stock already moving
up fast, this looks for quality-pool names having a quiet pullback: down
some % from their 52-week high. It's deliberately permissive (no hard
profitability bar) — the screen's only job is "is this worth a closer
look," not "is this a good business." That judgment belongs to the
Fundamental/SEC and Macro/Sentiment agents downstream, the same way the
momentum scanner only screens, it doesn't decide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_layer.models import PriceSeries, ThesisCandidate


@dataclass(frozen=True)
class ThesisSignal:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0  # pullback magnitude, for ranking among passed candidates only


def evaluate_thesis_candidate(
    candidate: ThesisCandidate, min_pullback_pct: float, max_pullback_pct: float
) -> ThesisSignal:
    """Pullback must fall within [min_pullback_pct, max_pullback_pct] — a
    floor to filter out noise, and a ceiling because a pullback beyond
    ~50% is statistically much more likely to be a genuinely impaired
    business (failed trial, accounting problem, secular decline) than a
    quality name temporarily out of favor. The screen stays permissive
    inside that band; it's not trying to call which businesses are sound,
    just which ones are worth the Fundamental/SEC agent's attention.
    """
    pullback_pct = (candidate.year_high - candidate.price) / candidate.year_high
    passed = min_pullback_pct <= pullback_pct <= max_pullback_pct
    if pullback_pct > max_pullback_pct:
        reason = f"pulled back {pullback_pct:.1%} from 52w high {candidate.year_high:.2f} — beyond {max_pullback_pct:.1%} ceiling (likely distress, not dislocation)"
    else:
        reason = (
            f"pulled back {pullback_pct:.1%} from 52w high {candidate.year_high:.2f} "
            f"{'>=' if pullback_pct >= min_pullback_pct else '<'} {min_pullback_pct:.1%} minimum"
        )
    return ThesisSignal(passed=passed, reasons=[reason], score=pullback_pct if passed else 0.0)


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
