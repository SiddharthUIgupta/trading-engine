from __future__ import annotations

from datetime import date, datetime, timedelta

from analyst_layer.prefilter import compute_regime, evaluate_ticker
from data_layer.models import (
    FilingSummary,
    FilingType,
    PriceBar,
    PriceSeries,
    SentimentPolarity,
    SentimentSnapshot,
)

_FILTER_KWARGS = dict(
    rsi_period=14,
    rsi_oversold=30.0,
    rsi_overbought=70.0,
    sma_short_window=10,
    sma_long_window=30,
    volume_spike_multiplier=2.0,
    sentiment_abs_threshold=0.3,
    recent_filing_days=3,
)


def _flat_price_series(symbol: str = "AAPL", n: int = 35, price: float = 100.0, volume: int = 1_000_000) -> PriceSeries:
    bars = [
        PriceBar(
            symbol=symbol,
            timestamp=datetime(2026, 1, 1) + timedelta(days=i),
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=volume,
        )
        for i in range(n)
    ]
    return PriceSeries(symbol=symbol, interval="1d", bars=bars)


def _neutral_sentiment() -> SentimentSnapshot:
    return SentimentSnapshot(
        symbol="AAPL", as_of=datetime(2026, 1, 1), source="test", score=0.0,
        polarity=SentimentPolarity.NEUTRAL, headline_count=5,
    )


def test_flat_quiet_ticker_is_filtered_out():
    signal = evaluate_ticker(
        price_series=_flat_price_series(),
        sentiment=_neutral_sentiment(),
        filings=[],
        today=date(2026, 2, 5),
        **_FILTER_KWARGS,
    )
    assert signal.passed is False
    assert signal.reasons == ["no threshold crossed"]


def test_strong_sentiment_passes_filter():
    sentiment = SentimentSnapshot(
        symbol="AAPL", as_of=datetime(2026, 1, 1), source="test", score=0.6,
        polarity=SentimentPolarity.BULLISH, headline_count=20,
    )
    signal = evaluate_ticker(
        price_series=_flat_price_series(), sentiment=sentiment, filings=[], today=date(2026, 2, 5), **_FILTER_KWARGS
    )
    assert signal.passed is True
    assert any("sentiment" in r for r in signal.reasons)


def test_volume_spike_passes_filter():
    bars = _flat_price_series(n=35).bars
    spiked = list(bars[:-1]) + [bars[-1].model_copy(update={"volume": bars[-1].volume * 5})]
    series = PriceSeries(symbol="AAPL", interval="1d", bars=spiked)

    signal = evaluate_ticker(
        price_series=series, sentiment=_neutral_sentiment(), filings=[], today=date(2026, 2, 5), **_FILTER_KWARGS
    )
    assert signal.passed is True
    assert any("volume spike" in r for r in signal.reasons)


def test_recent_8k_filing_passes_filter():
    filing = FilingSummary(
        symbol="AAPL", filing_type=FilingType.EIGHT_K, filed_on=date(2026, 2, 4), summary="material event", url=None
    )
    signal = evaluate_ticker(
        price_series=_flat_price_series(),
        sentiment=_neutral_sentiment(),
        filings=[filing],
        today=date(2026, 2, 5),
        **_FILTER_KWARGS,
    )
    assert signal.passed is True
    assert any("8-K" in r for r in signal.reasons)


def test_old_8k_filing_outside_window_does_not_pass_filter():
    filing = FilingSummary(
        symbol="AAPL", filing_type=FilingType.EIGHT_K, filed_on=date(2026, 1, 1), summary="material event", url=None
    )
    signal = evaluate_ticker(
        price_series=_flat_price_series(),
        sentiment=_neutral_sentiment(),
        filings=[filing],
        today=date(2026, 2, 5),
        **_FILTER_KWARGS,
    )
    assert signal.passed is False


def test_rsi_oversold_passes_filter():
    # monotonically declining closes drive RSI toward 0 (oversold)
    bars = [
        PriceBar(
            symbol="AAPL", timestamp=datetime(2026, 1, 1) + timedelta(days=i),
            open=100 - i, high=100 - i + 0.5, low=100 - i - 0.5, close=100 - i, volume=1_000_000,
        )
        for i in range(35)
    ]
    series = PriceSeries(symbol="AAPL", interval="1d", bars=bars)

    signal = evaluate_ticker(
        price_series=series, sentiment=_neutral_sentiment(), filings=[], today=date(2026, 2, 5), **_FILTER_KWARGS
    )
    assert signal.passed is True
    assert any("RSI" in r for r in signal.reasons)


def test_compute_regime_bullish_vs_bearish():
    rising = [100.0 + i for i in range(35)]
    falling = [135.0 - i for i in range(35)]
    assert compute_regime(rising, 10, 30) == "bullish_crossover"
    assert compute_regime(falling, 10, 30) == "bearish_crossover"


def test_compute_regime_neutral_with_insufficient_data():
    assert compute_regime([100.0, 101.0], 10, 30) == "neutral"
