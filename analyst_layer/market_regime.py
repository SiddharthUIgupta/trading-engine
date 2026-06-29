"""Pre-market market regime assessment — zero LLM, pure arithmetic.

Runs once daily before any agent call. Determines which trading tracks
to arm for the day based on VIX level/trend and broad market direction.

The goal: the system decides which strategies make sense today, so the
user only ever sets capability flags once (does the broker account support
options writing?) and never touches strategy toggles again.

Decision rules
--------------
Vol / premium-selling:
  ARM  when VIX ≥ 18 (enough premium) AND VIX ≤ 40 (not extreme fear)
       AND VIX is NOT spiking (>15% rise in a week while already elevated).
  DISARM otherwise — bad risk/reward for sellers in low-vol or panic environments.

ORB equity (long only):
  ARM  unless VIX > 30 AND market is bearish — long breakouts don't follow
       through when the broad tape is selling off into fear.
  DISARM in that specific combination only; self-gating otherwise (no ORB
       signal = no trade).

ORB options (calls + puts, defined risk):
  ARM  when market has directional conviction (SPY SMA10 meaningfully above
       or below SMA30). Options premium paid on ORB calls/puts is wasted in
       choppy, range-bound markets where breakouts reverse.
  DISARM when SPY SMA10 ≈ SMA30 (neutral regime).

Thesis / pullback:
  ARM  when VIX ≤ 30. Self-gating (no pullback candidates = no trades).
  DISARM in extreme fear where "pullbacks" statistically become breakdowns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from data_layer.models import PriceBar


@dataclass
class DailyRegime:
    vix_current: float
    vix_trend: str        # "rising" | "falling" | "stable"
    market_trend: str     # "bullish" | "bearish" | "neutral"

    arm_orb_equity: bool
    arm_orb_options: bool
    arm_thesis: bool
    arm_vol: bool

    reasons: dict[str, list[str]] = field(default_factory=dict)
    assessed_at: datetime = field(default_factory=datetime.utcnow)

    def log_summary(self) -> str:
        def _flag(armed: bool) -> str:
            return "ARMED   " if armed else "DISARMED"

        lines = [
            f"=== DAILY REGIME: VIX={self.vix_current:.1f} ({self.vix_trend}), market={self.market_trend} ===",
            f"  ORB equity   [{_flag(self.arm_orb_equity)}]  {'; '.join(self.reasons.get('orb_equity', []))}",
            f"  ORB options  [{_flag(self.arm_orb_options)}]  {'; '.join(self.reasons.get('orb_options', []))}",
            f"  Thesis       [{_flag(self.arm_thesis)}]  {'; '.join(self.reasons.get('thesis', []))}",
            f"  Vol/premium  [{_flag(self.arm_vol)}]  {'; '.join(self.reasons.get('vol', []))}",
        ]
        return "\n".join(lines)


def assess_daily_regime(
    spy_closes: list[float],
    vix_bars: list[PriceBar],
) -> DailyRegime:
    """Assess which tracks to arm for the day.

    Parameters
    ----------
    spy_closes:
        Daily SPY close prices, most recent last. Needs ≥ 30 bars for a
        meaningful SMA10/SMA30 comparison; fewer bars → neutral.
    vix_bars:
        Daily VIX bars, most recent last. Needs ≥ 5 bars for trend; fewer → stable.
    """
    # ── VIX ──────────────────────────────────────────────────────────────────
    vix_current: float = vix_bars[-1].close if vix_bars else 18.0
    vix_1w: float = vix_bars[-5].close if len(vix_bars) >= 5 else vix_current
    vix_week_chg = (vix_current - vix_1w) / vix_1w if vix_1w > 0 else 0.0

    vix_trend = (
        "rising"  if vix_week_chg >  0.05
        else "falling" if vix_week_chg < -0.05
        else "stable"
    )
    # "spiking": VIX rose >15% in a week AND is already elevated — selling
    # premium here means selling into expanding IV, the worst time to be short vol.
    vix_spiking = vix_week_chg > 0.15 and vix_current > 25

    # ── SPY trend (SMA10 vs SMA30) ────────────────────────────────────────────
    market_trend = "neutral"
    if len(spy_closes) >= 30:
        sma10 = sum(spy_closes[-10:]) / 10
        sma30 = sum(spy_closes[-30:]) / 30
        sep = (sma10 - sma30) / sma30 if sma30 > 0 else 0.0
        if sep > 0.005:
            market_trend = "bullish"
        elif sep < -0.005:
            market_trend = "bearish"

    reasons: dict[str, list[str]] = {}

    # ── ORB equity (long only) ────────────────────────────────────────────────
    if vix_current > 30 and market_trend == "bearish":
        arm_orb_equity = False
        reasons["orb_equity"] = [
            f"VIX={vix_current:.1f} + bearish market: long ORB breakouts unreliable in fear-driven downtrend"
        ]
    else:
        arm_orb_equity = True
        reasons["orb_equity"] = [
            f"VIX={vix_current:.1f}, market={market_trend}: conditions support ORB equity longs"
        ]

    # ── ORB options (both directions, defined risk) ───────────────────────────
    if market_trend == "neutral":
        arm_orb_options = False
        reasons["orb_options"] = [
            "SPY SMA10 ≈ SMA30 (neutral/choppy): ORB breakouts unreliable — premium paid likely wasted"
        ]
    else:
        arm_orb_options = True
        reasons["orb_options"] = [
            f"market is {market_trend}: directional conviction supports ORB options"
        ]

    # ── Thesis pullback ───────────────────────────────────────────────────────
    if vix_current > 30:
        arm_thesis = False
        reasons["thesis"] = [
            f"VIX={vix_current:.1f} > 30: extreme fear — pullbacks statistically likely to be breakdowns"
        ]
    else:
        arm_thesis = True
        reasons["thesis"] = [
            f"VIX={vix_current:.1f}: environment supports thesis pullback entries"
        ]

    # ── Vol / premium selling ─────────────────────────────────────────────────
    if vix_current < 18:
        arm_vol = False
        reasons["vol"] = [
            f"VIX={vix_current:.1f} < 18: too little premium — risk/reward unfavorable for sellers"
        ]
    elif vix_spiking:
        arm_vol = False
        reasons["vol"] = [
            f"VIX={vix_current:.1f} spiking +{vix_week_chg:.0%} from {vix_1w:.1f} (1w ago): "
            "selling into expanding IV — wait for vol to stabilize"
        ]
    elif vix_current > 40:
        arm_vol = False
        reasons["vol"] = [
            f"VIX={vix_current:.1f} > 40: extreme fear, short premium tail risk too high"
        ]
    else:
        arm_vol = True
        reasons["vol"] = [
            f"VIX={vix_current:.1f} ({vix_trend}), market={market_trend}: premium-selling conditions favorable"
        ]

    return DailyRegime(
        vix_current=vix_current,
        vix_trend=vix_trend,
        market_trend=market_trend,
        arm_orb_equity=arm_orb_equity,
        arm_orb_options=arm_orb_options,
        arm_thesis=arm_thesis,
        arm_vol=arm_vol,
        reasons=reasons,
    )
