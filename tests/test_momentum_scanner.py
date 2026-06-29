from __future__ import annotations

from datetime import datetime, timedelta

from analyst_layer.momentum_scanner import evaluate_low_float_momentum
from data_layer.models import PriceBar, PriceSeries

# Wide price band by default so most tests exercise only the criterion
# they're named for — the dedicated price-band tests below use the real
# default band ($1-$20) instead.
_KWARGS = dict(
    max_float_shares=20_000_000,
    ema_short_period=9,
    ema_long_period=20,
    min_daily_gain_pct=0.10,
    clean_body_dominance_threshold=0.55,
    clean_lookback_bars=12,
    today_volume=600_000,
    average_daily_volume=100_000,  # RVOL = 6x
    min_relative_volume=5.0,
    price_min=1.0,
    price_max=1000.0,
)


def _clean_uptrend_series(n: int = 25) -> PriceSeries:
    """Decisive green candles, steadily rising — clean body dominance,
    bullish EMA crossover, and price ends above VWAP almost by construction.
    """
    bars = [
        PriceBar(
            symbol="TEST", timestamp=datetime(2026, 6, 22, 9, 30) + timedelta(minutes=5 * i),
            open=100.0 + i, high=101.0 + i, low=100.0 + i, close=101.0 + i, volume=10_000,
        )
        for i in range(n)
    ]
    return PriceSeries(symbol="TEST", interval="5m", bars=bars)


def _choppy_flat_series(n: int = 25) -> PriceSeries:
    """Big wicks, tiny bodies, no net direction — fails the "clean" check
    and the EMA/VWAP checks (no real trend to be above/below decisively).
    """
    bars = [
        PriceBar(
            symbol="TEST", timestamp=datetime(2026, 6, 22, 9, 30) + timedelta(minutes=5 * i),
            open=100.0, high=105.0, low=95.0, close=100.0 + (0.1 if i % 2 == 0 else -0.1), volume=10_000,
        )
        for i in range(n)
    ]
    return PriceSeries(symbol="TEST", interval="5m", bars=bars)


def test_all_criteria_pass_on_clean_low_float_mover():
    signal = evaluate_low_float_momentum(
        intraday_series=_clean_uptrend_series(),
        shares_float=5_000_000,
        today_percent_change=0.15,
        **_KWARGS,
    )
    assert signal.passed is True
    assert signal.score == 0.15


def test_fails_on_float_too_large_even_if_everything_else_passes():
    signal = evaluate_low_float_momentum(
        intraday_series=_clean_uptrend_series(),
        shares_float=200_000_000,
        today_percent_change=0.15,
        **_KWARGS,
    )
    assert signal.passed is False
    assert signal.score == 0.0
    assert any("float" in r and ">" in r for r in signal.reasons)


def test_fails_on_insufficient_daily_gain():
    signal = evaluate_low_float_momentum(
        intraday_series=_clean_uptrend_series(),
        shares_float=5_000_000,
        today_percent_change=0.02,
        **_KWARGS,
    )
    assert signal.passed is False


def test_fails_on_choppy_price_action():
    signal = evaluate_low_float_momentum(
        intraday_series=_choppy_flat_series(),
        shares_float=5_000_000,
        today_percent_change=0.15,
        **_KWARGS,
    )
    assert signal.passed is False
    assert any("body dominance" in r for r in signal.reasons)


def test_fails_with_too_few_bars_for_ema():
    short_series = PriceSeries(
        symbol="TEST",
        interval="5m",
        bars=[
            PriceBar(
                symbol="TEST", timestamp=datetime(2026, 6, 22, 9, 30), open=100.0, high=101.0, low=100.0, close=101.0,
                volume=10_000,
            )
        ],
    )
    signal = evaluate_low_float_momentum(
        intraday_series=short_series, shares_float=5_000_000, today_percent_change=0.15, **_KWARGS
    )
    assert signal.passed is False
    assert any("insufficient bars" in r for r in signal.reasons)


def test_fails_on_low_relative_volume():
    kwargs = {**_KWARGS, "today_volume": 50_000, "average_daily_volume": 100_000}  # RVOL = 0.5x
    signal = evaluate_low_float_momentum(
        intraday_series=_clean_uptrend_series(), shares_float=5_000_000, today_percent_change=0.15, **kwargs
    )
    assert signal.passed is False
    assert any("RVOL" in r and "<" in r for r in signal.reasons)


def test_fails_on_zero_average_volume_does_not_crash():
    kwargs = {**_KWARGS, "today_volume": 50_000, "average_daily_volume": 0.0}
    signal = evaluate_low_float_momentum(
        intraday_series=_clean_uptrend_series(), shares_float=5_000_000, today_percent_change=0.15, **kwargs
    )
    assert signal.passed is False


def test_fails_on_price_outside_band():
    """_clean_uptrend_series ends at $125 — outside the real default $1-$20 band."""
    kwargs = {**_KWARGS, "price_min": 1.0, "price_max": 20.0}
    signal = evaluate_low_float_momentum(
        intraday_series=_clean_uptrend_series(), shares_float=5_000_000, today_percent_change=0.15, **kwargs
    )
    assert signal.passed is False
    assert any("outside" in r for r in signal.reasons)


def test_passes_with_price_inside_band():
    low_priced_series = PriceSeries(
        symbol="TEST",
        interval="5m",
        bars=[
            PriceBar(
                symbol="TEST", timestamp=datetime(2026, 6, 22, 9, 30) + timedelta(minutes=5 * i),
                open=5.0 + i * 0.1, high=5.1 + i * 0.1, low=5.0 + i * 0.1, close=5.1 + i * 0.1, volume=10_000,
            )
            for i in range(25)
        ],
    )
    kwargs = {**_KWARGS, "price_min": 1.0, "price_max": 20.0}
    signal = evaluate_low_float_momentum(
        intraday_series=low_priced_series, shares_float=5_000_000, today_percent_change=0.15, **kwargs
    )
    assert signal.passed is True
