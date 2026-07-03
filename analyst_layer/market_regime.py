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

    # Optional macro news sentiment (from macro_news_agent). None when disabled or unavailable.
    macro_sentiment: str | None = None   # "bullish" | "bearish" | "neutral"
    macro_confidence: float = 0.0
    macro_themes: list[str] = field(default_factory=list)
    vix_effective: float | None = None   # vix_smooth after macro adjustment
    # Individual stock signals extracted from news. Each: {"ticker", "catalyst", "direction"}
    news_ticker_signals: list[dict] = field(default_factory=list)

    def log_summary(self) -> str:
        def _flag(armed: bool) -> str:
            return "ARMED   " if armed else "DISARMED"

        macro_line = ""
        if self.macro_sentiment is not None:
            themes = ", ".join(self.macro_themes) if self.macro_themes else "no themes"
            vix_adj = f" → effective VIX={self.vix_effective:.1f}" if self.vix_effective is not None else ""
            macro_line = f"\n  Macro news   [{self.macro_sentiment.upper():8s}] conf={self.macro_confidence:.2f}{vix_adj} | {themes}"
            if self.news_ticker_signals:
                bullish = [s["ticker"] for s in self.news_ticker_signals if s.get("direction") == "bullish"]
                bearish = [s["ticker"] for s in self.news_ticker_signals if s.get("direction") == "bearish"]
                if bullish:
                    macro_line += f"\n  News tickers [BULLISH ] {bullish}"
                if bearish:
                    macro_line += f"\n  News tickers [BEARISH ] {bearish}"

        lines = [
            f"=== DAILY REGIME: VIX={self.vix_current:.1f} ({self.vix_trend}), market={self.market_trend} ==={macro_line}",
            f"  ORB equity   [{_flag(self.arm_orb_equity)}]  {'; '.join(self.reasons.get('orb_equity', []))}",
            f"  ORB options  [{_flag(self.arm_orb_options)}]  {'; '.join(self.reasons.get('orb_options', []))}",
            f"  Thesis       [{_flag(self.arm_thesis)}]  {'; '.join(self.reasons.get('thesis', []))}",
            f"  Vol/premium  [{_flag(self.arm_vol)}]  {'; '.join(self.reasons.get('vol', []))}",
        ]
        return "\n".join(lines)


def assess_daily_regime(
    spy_closes: list[float],
    vix_bars: list[PriceBar],
    macro_sentiment: str | None = None,
    macro_confidence: float = 0.0,
    macro_themes: list[str] | None = None,
    macro_vix_adjustment: float = 3.0,
    macro_min_confidence: float = 0.6,
    news_ticker_signals: list[dict] | None = None,
) -> DailyRegime:
    """Assess which tracks to arm for the day.

    Parameters
    ----------
    spy_closes:
        Daily SPY close prices, most recent last. Needs ≥ 30 bars for a
        meaningful SMA10/SMA30 comparison; fewer bars → neutral.
    vix_bars:
        Daily VIX bars, most recent last. Needs ≥ 5 bars for trend; fewer → stable.
    macro_sentiment:
        Optional macro news sentiment ("bullish"/"bearish"/"neutral") from the
        macro_news_agent. MONOTONIC: only "bearish" at sufficient confidence
        has any effect, RAISING vix_smooth by up to `macro_vix_adjustment`
        points (can disarm tracks near a threshold). Bullish/neutral sentiment
        never lowers effective VIX — a risk guard that headline sentiment can
        relax is not a guard.
    macro_confidence:
        Confidence of the macro_sentiment assessment (0–1). Only applied when
        >= macro_min_confidence.
    macro_themes:
        Key themes from the news scoring, stored in DailyRegime for logging.
    macro_vix_adjustment:
        Maximum upward VIX adjustment from bearish news (in VIX points).
        Scaled linearly by confidence.
    macro_min_confidence:
        Minimum confidence to apply any macro adjustment. Below this threshold
        the news is too uncertain to override pure technical data.
    """
    # ── VIX ──────────────────────────────────────────────────────────────────
    vix_current: float = vix_bars[-1].close if vix_bars else 18.0
    vix_1w: float = vix_bars[-5].close if len(vix_bars) >= 5 else vix_current
    vix_week_chg = (vix_current - vix_1w) / vix_1w if vix_1w > 0 else 0.0

    # 5-day smoothed VIX for threshold decisions — prevents daily whipsawing
    # when spot VIX sits at 29.8 vs 30.2 and flips strategy arm states each day.
    _smooth_window = min(5, len(vix_bars))
    vix_smooth: float = sum(b.close for b in vix_bars[-_smooth_window:]) / _smooth_window if vix_bars else 18.0

    # ── Macro news adjustment to effective VIX (MONOTONIC) ──────────────────
    # News sentiment may only TIGHTEN the regime (raise effective VIX), never
    # loosen it. Bullish sentiment must never arm a track that raw volatility
    # says should be disarmed — headline sentiment is at its most confidently
    # bullish immediately before regime breaks, so a guard that bullish news
    # can relax is not a guard.
    vix_effective = vix_smooth
    if macro_sentiment == "bearish" and macro_confidence >= macro_min_confidence:
        vix_effective = vix_smooth + macro_vix_adjustment * macro_confidence
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Macro news (bearish, conf=%.2f) raised effective VIX: %.1f → %.1f",
            macro_confidence, vix_smooth, vix_effective,
        )
    vix_smooth = vix_effective

    vix_trend = (
        "rising"  if vix_week_chg >  0.05
        else "falling" if vix_week_chg < -0.05
        else "stable"
    )
    # "spiking": VIX rose >15% in a week AND is already elevated — selling
    # premium here means selling into expanding IV, the worst time to be short vol.
    vix_spiking = vix_week_chg > 0.15 and vix_smooth > 25

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
    if vix_smooth > 30 and market_trend == "bearish":
        arm_orb_equity = False
        reasons["orb_equity"] = [
            f"VIX(5d avg)={vix_smooth:.1f} + bearish market: long ORB breakouts unreliable in fear-driven downtrend"
        ]
    else:
        arm_orb_equity = True
        reasons["orb_equity"] = [
            f"VIX(5d avg)={vix_smooth:.1f}, market={market_trend}: conditions support ORB equity longs"
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
    if vix_smooth > 30:
        arm_thesis = False
        reasons["thesis"] = [
            f"VIX(5d avg)={vix_smooth:.1f} > 30: elevated fear — pullbacks statistically likely to be breakdowns"
        ]
    else:
        arm_thesis = True
        reasons["thesis"] = [
            f"VIX(5d avg)={vix_smooth:.1f}: environment supports thesis pullback entries"
        ]

    # ── Vol / premium selling ─────────────────────────────────────────────────
    if vix_smooth < 18:
        arm_vol = False
        reasons["vol"] = [
            f"VIX(5d avg)={vix_smooth:.1f} < 18: too little premium — risk/reward unfavorable for sellers"
        ]
    elif vix_spiking:
        arm_vol = False
        reasons["vol"] = [
            f"VIX={vix_current:.1f} spiking +{vix_week_chg:.0%} from {vix_1w:.1f} (1w ago): "
            "selling into expanding IV — wait for vol to stabilize"
        ]
    elif vix_smooth > 40:
        arm_vol = False
        reasons["vol"] = [
            f"VIX(5d avg)={vix_smooth:.1f} > 40: extreme fear, short premium tail risk too high"
        ]
    else:
        arm_vol = True
        reasons["vol"] = [
            f"VIX={vix_current:.1f} (5d avg={vix_smooth:.1f}, {vix_trend}), market={market_trend}: "
            "premium-selling conditions favorable"
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
        macro_sentiment=macro_sentiment,
        macro_confidence=macro_confidence,
        macro_themes=list(macro_themes) if macro_themes else [],
        vix_effective=vix_effective if (macro_sentiment and macro_confidence >= macro_min_confidence) else None,
        news_ticker_signals=list(news_ticker_signals) if news_ticker_signals else [],
    )
