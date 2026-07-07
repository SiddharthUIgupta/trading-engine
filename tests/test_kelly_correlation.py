"""Tests for Kelly sizing and correlation guard."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from analyst_layer.kelly import compute_kelly_fraction, kelly_fraction_from_pnl_history, MIN_TRADES_FOR_KELLY
from analyst_layer.correlation import (
    pairwise_correlation,
    check_portfolio_correlation,
    apply_correlation_adjustment,
    HARD_BLOCK_THRESHOLD,
    SOFT_REDUCE_THRESHOLD,
    CORR_PENALTY,
)


# ── compute_kelly_fraction ────────────────────────────────────────────────────

def test_kelly_positive_edge():
    # win_rate=0.6, W/L=1.5 → f = 0.6 - 0.4/1.5 = 0.6 - 0.267 = 0.333 → half = 0.167
    f = compute_kelly_fraction(0.6, 1.5)
    assert abs(f - 0.1667) < 0.001


def test_kelly_no_edge_returns_zero():
    # win_rate=0.4, W/L=1.0 → 0.4 - 0.6/1.0 = -0.2 → 0
    assert compute_kelly_fraction(0.4, 1.0) == 0.0


def test_kelly_zero_win_rate_returns_zero():
    assert compute_kelly_fraction(0.0, 2.0) == 0.0


def test_kelly_zero_win_loss_ratio_returns_zero():
    assert compute_kelly_fraction(0.6, 0.0) == 0.0


def test_kelly_full_vs_half():
    full = compute_kelly_fraction(0.6, 2.0, half_kelly=False)
    half = compute_kelly_fraction(0.6, 2.0, half_kelly=True)
    assert abs(full - half * 2) < 1e-9


# ── kelly_fraction_from_pnl_history ──────────────────────────────────────────

def _pnls(wins: int, win_amt: float, losses: int, loss_amt: float) -> list[float]:
    return [win_amt] * wins + [-loss_amt] * losses


def test_kelly_from_history_below_min_trades_uses_fallback():
    frac, reason = kelly_fraction_from_pnl_history([50.0] * 5, max_position_size_pct=0.05)
    assert frac == pytest.approx(0.025)  # 0.5 * 0.05
    assert "bootstrapping" in reason


def test_kelly_from_history_sufficient_trades():
    pnls = _pnls(wins=10, win_amt=100.0, losses=5, loss_amt=50.0)
    frac, reason = kelly_fraction_from_pnl_history(pnls, max_position_size_pct=0.05)
    assert 0 < frac <= 0.05
    assert "win_rate" in reason


def test_kelly_from_history_capped_at_max():
    # Very high edge — Kelly would suggest >5% but must be capped
    pnls = _pnls(wins=20, win_amt=500.0, losses=5, loss_amt=10.0)
    frac, _ = kelly_fraction_from_pnl_history(pnls, max_position_size_pct=0.05)
    assert frac <= 0.05


def test_kelly_from_history_no_winning_trades():
    pnls = [-100.0] * MIN_TRADES_FOR_KELLY
    frac, reason = kelly_fraction_from_pnl_history(pnls, max_position_size_pct=0.05)
    assert frac == 0.0
    assert "no winning" in reason


def test_kelly_from_history_all_wins():
    pnls = [100.0] * MIN_TRADES_FOR_KELLY
    frac, reason = kelly_fraction_from_pnl_history(pnls, max_position_size_pct=0.05)
    assert frac == pytest.approx(0.05)
    assert "conservatively" in reason


# ── pairwise_correlation ──────────────────────────────────────────────────────

def _trending_series(n: int = 60, slope: float = 1.002, noise: float = 0.005) -> list[float]:
    """Trending price series with small random noise so returns have non-zero variance."""
    import random
    rng = random.Random(99)
    price = 100.0
    out = []
    for _ in range(n):
        out.append(price)
        price *= slope * (1 + rng.gauss(0, noise))
    return out


def test_correlation_identical_series_is_one():
    s = _trending_series(60)
    r = pairwise_correlation(s, s)
    assert r is not None
    assert abs(r - 1.0) < 1e-6


def test_correlation_opposite_series_is_negative():
    # Build prices from a shared random return sequence r_i:
    #   up:   each bar multiplied by exp(+r_i)  → log return = +r_i
    #   down: each bar multiplied by exp(-r_i)  → log return = -r_i
    # pairwise_correlation then computes Pearson(r_i, -r_i) = -1.0 exactly.
    import math, random
    rng = random.Random(42)
    returns = [rng.gauss(0, 0.02) for _ in range(60)]
    up: list[float] = [100.0]
    down: list[float] = [100.0]
    for ret in returns:
        up.append(up[-1] * math.exp(ret))
        down.append(down[-1] * math.exp(-ret))
    r = pairwise_correlation(up, down)
    assert r is not None
    assert r < -0.99


def test_correlation_short_series_returns_none():
    s = _trending_series(10)
    assert pairwise_correlation(s, s) is None


# ── check_portfolio_correlation ───────────────────────────────────────────────

def test_no_held_positions_returns_zero():
    max_r, desc = check_portfolio_correlation(_trending_series(), {})
    assert max_r == 0.0
    assert "no existing" in desc


def test_high_correlation_detected():
    s = _trending_series(60)
    max_r, desc = check_portfolio_correlation(s, {"HELD": s})
    assert max_r > 0.9
    assert "HELD" in desc


def test_uncorrelated_returns_low_value():
    import random
    rng = random.Random(42)
    a = [100.0 + rng.gauss(0, 1) for _ in range(60)]
    b = [100.0 + rng.gauss(0, 1) for _ in range(60)]
    # Two independent random series shouldn't have r > 0.5 reliably
    max_r, _ = check_portfolio_correlation(a, {"OTHER": b})
    assert max_r < 0.7  # not guaranteed but extremely likely with random data


# ── Regression: a strong INVERSE correlation must never be treated the same
# as a duplicate-exposure (positive) correlation — this guard exists to catch
# "buying QQQ when already long SPY", not to flag hedges. Previously used
# abs(max_r), so a strongly negative (hedging) correlation was hard-blocked/
# size-reduced exactly like a strongly positive one.

def _inverse_pair(n: int = 60) -> tuple[list[float], list[float]]:
    import math
    import random
    rng = random.Random(7)
    returns = [rng.gauss(0, 0.02) for _ in range(n)]
    up: list[float] = [100.0]
    down: list[float] = [100.0]
    for ret in returns:
        up.append(up[-1] * math.exp(ret))
        down.append(down[-1] * math.exp(-ret))
    return up, down


def test_strong_inverse_correlation_returns_zero_not_absolute_value():
    proposed, held = _inverse_pair()
    max_r, desc = check_portfolio_correlation(proposed, {"HEDGE": held})
    assert max_r == 0.0  # not ~1.0, which abs(-0.99...) would have produced
    assert "HEDGE" in desc  # still reported for visibility, just not penalized


def test_strong_inverse_correlation_does_not_hard_block_via_full_pipeline():
    proposed, held = _inverse_pair()
    max_r, desc = check_portfolio_correlation(proposed, {"HEDGE": held})
    fraction, reason, blocked = apply_correlation_adjustment(
        kelly_fraction=0.04, max_correlation=max_r, correlation_description=desc,
    )
    assert blocked is False
    assert fraction == 0.04  # unpenalized — a hedge is not overlapping exposure


def test_positive_correlation_still_picked_over_inverse_when_both_held():
    """When one held position is a near-duplicate and another is an inverse
    hedge, the duplicate (positive) correlation must still be the one that
    drives the block/reduce decision, not get averaged away or masked."""
    proposed = _trending_series(60)
    _, inverse_held = _inverse_pair()
    max_r, desc = check_portfolio_correlation(proposed, {"DUPLICATE": proposed, "HEDGE": inverse_held})
    assert max_r > 0.9
    assert "DUPLICATE" in desc


# ── apply_correlation_adjustment ─────────────────────────────────────────────

def test_hard_block_above_threshold():
    frac, reason, blocked = apply_correlation_adjustment(
        kelly_fraction=0.04,
        max_correlation=HARD_BLOCK_THRESHOLD + 0.01,
        correlation_description="SPY=+0.88",
    )
    assert blocked is True
    assert frac == 0.0
    assert "HARD BLOCK" in reason


def test_soft_reduce_between_thresholds():
    base = 0.04
    frac, reason, blocked = apply_correlation_adjustment(
        kelly_fraction=base,
        max_correlation=(SOFT_REDUCE_THRESHOLD + HARD_BLOCK_THRESHOLD) / 2,
        correlation_description="QQQ=+0.78",
    )
    assert blocked is False
    assert abs(frac - base * (1.0 - CORR_PENALTY)) < 1e-6
    assert "reduction" in reason


def test_no_adjustment_below_soft_threshold():
    base = 0.04
    frac, reason, blocked = apply_correlation_adjustment(
        kelly_fraction=base,
        max_correlation=SOFT_REDUCE_THRESHOLD - 0.1,
        correlation_description="AMZN=+0.55",
    )
    assert blocked is False
    assert frac == base
    assert "acceptable" in reason


# ── StateStore integration ────────────────────────────────────────────────────

def test_state_store_get_pnl_history_empty(tmp_path: Path):
    from execution_layer.state_store import StateStore
    store = StateStore(tmp_path / "test.sqlite3")
    assert store.get_pnl_history() == []


def test_state_store_get_pnl_history_returns_values(tmp_path: Path):
    from datetime import date
    from execution_layer.state_store import StateStore
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_realized_sale("AAPL", date.today(), 10, 155.0, 150.0)  # +$50
    store.record_realized_sale("MSFT", date.today(), 5, 290.0, 300.0)   # -$50
    pnls = store.get_pnl_history()
    assert len(pnls) == 2
    assert any(p > 0 for p in pnls)
    assert any(p < 0 for p in pnls)
