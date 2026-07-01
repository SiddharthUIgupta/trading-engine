"""Tests for the autonomous daily regime assessment."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from analyst_layer.market_regime import DailyRegime, assess_daily_regime
from data_layer.models import PriceBar


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vix_bars(current: float, week_ago: float | None = None, n: int = 30) -> list[PriceBar]:
    bars = []
    now = datetime.now()
    for i in range(n):
        days_ago = n - i
        close = week_ago if (week_ago is not None and days_ago == 5) else current
        bars.append(PriceBar(
            symbol="^VIX", timestamp=now - timedelta(days=days_ago),
            open=close, high=close * 1.02, low=close * 0.98, close=close, volume=0,
        ))
    return bars


def _spy_closes(trend: str, n: int = 40) -> list[float]:
    """Generate SPY close series producing the requested SMA10/SMA30 regime."""
    price = 500.0
    closes = []
    for _ in range(n):
        if trend == "bullish":
            price *= 1.003
        elif trend == "bearish":
            price *= 0.997
        # neutral: no movement → SMA10 == SMA30
        closes.append(price)
    return closes


# ── Vol track arming ──────────────────────────────────────────────────────────

def test_vol_armed_when_vix_in_sweet_spot():
    regime = assess_daily_regime(_spy_closes("bullish"), _vix_bars(22.0))
    assert regime.arm_vol is True


def test_vol_disarmed_when_vix_too_low():
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(13.0))
    assert regime.arm_vol is False
    assert "too little premium" in regime.reasons["vol"][0]


def test_vol_disarmed_when_vix_spiking():
    # VIX went from 20 a week ago to 28 now — >15% spike while already elevated
    regime = assess_daily_regime(_spy_closes("bearish"), _vix_bars(current=28.0, week_ago=20.0))
    assert regime.arm_vol is False
    assert "spiking" in regime.reasons["vol"][0]


def test_vol_disarmed_when_vix_extreme():
    regime = assess_daily_regime(_spy_closes("bearish"), _vix_bars(45.0))
    assert regime.arm_vol is False
    assert "extreme fear" in regime.reasons["vol"][0]  # VIX > 40 path still says "extreme fear"


def test_vol_armed_when_vix_elevated_but_not_spiking():
    # VIX at 28 but stable (week ago was also 28)
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(current=28.0, week_ago=27.0))
    assert regime.arm_vol is True


# ── ORB equity arming ─────────────────────────────────────────────────────────

def test_orb_equity_armed_in_normal_bull_market():
    regime = assess_daily_regime(_spy_closes("bullish"), _vix_bars(15.0))
    assert regime.arm_orb_equity is True


def test_orb_equity_disarmed_in_fear_plus_downtrend():
    # VIX > 30 AND bearish market → long ORB unreliable
    regime = assess_daily_regime(_spy_closes("bearish"), _vix_bars(32.0))
    assert regime.arm_orb_equity is False
    assert "fear-driven downtrend" in regime.reasons["orb_equity"][0]


def test_orb_equity_armed_when_vix_high_but_market_neutral():
    # High VIX alone isn't enough to disarm — needs BOTH fear AND downtrend
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(32.0))
    assert regime.arm_orb_equity is True


def test_orb_equity_armed_when_market_bearish_but_vix_normal():
    # Bearish market with normal VIX (orderly selloff) → still arm
    regime = assess_daily_regime(_spy_closes("bearish"), _vix_bars(18.0))
    assert regime.arm_orb_equity is True


# ── ORB options arming ────────────────────────────────────────────────────────

def test_orb_options_armed_in_trending_market():
    regime = assess_daily_regime(_spy_closes("bullish"), _vix_bars(18.0))
    assert regime.arm_orb_options is True


def test_orb_options_armed_in_bearish_trending_market():
    # Bearish trend → put options make sense
    regime = assess_daily_regime(_spy_closes("bearish"), _vix_bars(22.0))
    assert regime.arm_orb_options is True


def test_orb_options_disarmed_in_neutral_market():
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(15.0))
    assert regime.arm_orb_options is False
    assert "neutral" in regime.reasons["orb_options"][0].lower()


# ── Thesis arming ─────────────────────────────────────────────────────────────

def test_thesis_armed_in_normal_environment():
    regime = assess_daily_regime(_spy_closes("bullish"), _vix_bars(18.0))
    assert regime.arm_thesis is True


def test_thesis_disarmed_in_extreme_fear():
    regime = assess_daily_regime(_spy_closes("bearish"), _vix_bars(35.0))
    assert regime.arm_thesis is False
    assert "elevated fear" in regime.reasons["thesis"][0]


def test_thesis_disarmed_at_vix_boundary():
    # Exactly at threshold (VIX=30 should still be armed, 31 should not)
    assert assess_daily_regime(_spy_closes("neutral"), _vix_bars(30.0)).arm_thesis is True
    assert assess_daily_regime(_spy_closes("neutral"), _vix_bars(31.0)).arm_thesis is False


# ── VIX trend classification ──────────────────────────────────────────────────

def test_vix_trend_rising():
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(current=22.0, week_ago=19.0))
    assert regime.vix_trend == "rising"


def test_vix_trend_falling():
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(current=15.0, week_ago=19.0))
    assert regime.vix_trend == "falling"


def test_vix_trend_stable():
    regime = assess_daily_regime(_spy_closes("neutral"), _vix_bars(current=18.0, week_ago=17.5))
    assert regime.vix_trend == "stable"


# ── Market trend classification ───────────────────────────────────────────────

def test_market_trend_bullish_for_uptrend():
    regime = assess_daily_regime(_spy_closes("bullish", n=40), _vix_bars(18.0))
    assert regime.market_trend == "bullish"


def test_market_trend_bearish_for_downtrend():
    regime = assess_daily_regime(_spy_closes("bearish", n=40), _vix_bars(18.0))
    assert regime.market_trend == "bearish"


def test_market_trend_neutral_for_flat():
    regime = assess_daily_regime(_spy_closes("neutral", n=40), _vix_bars(18.0))
    assert regime.market_trend == "neutral"


# ── Graceful degradation ──────────────────────────────────────────────────────

def test_empty_vix_bars_falls_back_to_default():
    # No VIX data → defaults to 18.0 (should arm vol)
    regime = assess_daily_regime(_spy_closes("bullish"), [])
    assert regime.vix_current == 18.0
    assert regime.arm_vol is True


def test_short_spy_series_produces_neutral_trend():
    # < 30 bars → can't compute SMA30 → neutral
    regime = assess_daily_regime([500.0] * 10, _vix_bars(18.0))
    assert regime.market_trend == "neutral"


# ── log_summary ───────────────────────────────────────────────────────────────

def test_log_summary_contains_all_tracks():
    regime = assess_daily_regime(_spy_closes("bullish"), _vix_bars(22.0))
    summary = regime.log_summary()
    assert "ORB equity" in summary
    assert "ORB options" in summary
    assert "Thesis" in summary
    assert "Vol/premium" in summary
    assert "ARMED" in summary
