"""Market recovery scanner — deterministic, zero-LLM.

Finds stocks recovering from a recent correction: pulled back 15–40% from
their 60-day high but now showing positive 5-day momentum, price back above
MA20, and volume picking up vs. the 20-day average.

Different shape from thesis_scanner.py: thesis looks at 52-week-high
dislocations (individual fundamental story); recovery looks for names
bouncing off a broad market correction with confirmed buying volume
(macro tailwind story — "market is recovering, which stocks are leading").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_layer.models import PriceSeries


@dataclass(frozen=True)
class RecoverySignal:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0  # composite for ranking; higher = stronger bounce conviction


def evaluate_recovery_candidate(
    price_series: PriceSeries,
    min_pullback_pct: float = 0.15,
    max_pullback_pct: float = 0.40,
    momentum_days: int = 5,
    volume_pickup_ratio: float = 1.20,
) -> RecoverySignal:
    """Four conjunctive conditions — all must pass:

    1. Pulled back 15–40% from 60-day high: in correction territory but not broken.
       Below 15% = noise, not a real correction. Above 40% = likely distress,
       not a recoverable dip.
    2. 5-day momentum positive: price higher than it was 5 trading days ago —
       the turn has already started, we're not trying to catch a falling knife.
    3. Price above MA20: short-term trend confirmed, not just a dead-cat bounce.
    4. 3-day avg volume ≥ 1.2× 20-day avg: buyers returning with conviction.
       Without volume, the bounce is low-quality and likely to fail.
    """
    bars = price_series.bars
    if len(bars) < 22:
        return RecoverySignal(
            passed=False,
            reasons=["insufficient price history (need ≥22 bars)"],
        )

    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    current = closes[-1]

    # 1. Pullback from 60-day high
    high_60d = max(closes)
    pullback = (high_60d - current) / high_60d if high_60d > 0 else 0.0
    if pullback < min_pullback_pct:
        return RecoverySignal(
            passed=False,
            reasons=[
                f"pullback {pullback:.1%} < {min_pullback_pct:.0%} floor "
                "— not enough correction to be a recovery play"
            ],
        )
    if pullback > max_pullback_pct:
        return RecoverySignal(
            passed=False,
            reasons=[
                f"pullback {pullback:.1%} > {max_pullback_pct:.0%} ceiling "
                "— likely distress, not a cyclical correction"
            ],
        )

    # 2. 5-day momentum positive
    lookback = min(momentum_days, len(closes) - 1)
    prior = closes[-(lookback + 1)]
    momentum_pct = (current - prior) / prior if prior > 0 else 0.0
    if momentum_pct <= 0:
        return RecoverySignal(
            passed=False,
            reasons=[f"5d momentum {momentum_pct:+.1%} — price still declining, no turn yet"],
        )

    # 3. Price above MA20
    ma20 = sum(closes[-20:]) / 20
    if current < ma20:
        return RecoverySignal(
            passed=False,
            reasons=[
                f"price {current:.2f} below MA20 {ma20:.2f} "
                "— recovery not yet confirmed by moving average"
            ],
        )

    # 4. Volume pickup
    vol_3d = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else 0.0
    vol_20d = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0.0
    vol_ratio = vol_3d / vol_20d if vol_20d > 0 else 1.0
    if vol_ratio < volume_pickup_ratio:
        return RecoverySignal(
            passed=False,
            reasons=[
                f"volume {vol_ratio:.2f}x 20d avg < {volume_pickup_ratio:.2f}x threshold "
                "— buyers not yet committed"
            ],
        )

    # Score: blend of momentum magnitude and volume conviction
    score = momentum_pct * 0.6 + (vol_ratio - 1.0) * 0.4
    return RecoverySignal(
        passed=True,
        reasons=[
            f"down {pullback:.1%} from 60d high — corrected but not broken",
            f"5d momentum +{momentum_pct:.1%} — turn already underway",
            f"price {current:.2f} above MA20 {ma20:.2f}",
            f"volume {vol_ratio:.1f}x 20d avg — buyers returning",
        ],
        score=score,
    )
