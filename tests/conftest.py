from __future__ import annotations

from datetime import date, datetime

import pytest

from data_layer.models import (
    FilingSummary,
    FilingType,
    FundamentalsSnapshot,
    PriceBar,
    PriceSeries,
    SentimentPolarity,
    SentimentSnapshot,
)


@pytest.fixture
def sample_price_series() -> PriceSeries:
    bars = [
        PriceBar(
            symbol="AAPL",
            timestamp=datetime(2026, 6, day, 16, 0),
            open=190.0 + day,
            high=192.0 + day,
            low=189.0 + day,
            close=191.0 + day,
            volume=1_000_000,
        )
        for day in range(1, 16)
    ]
    return PriceSeries(symbol="AAPL", interval="1d", bars=bars)


@pytest.fixture
def sample_sentiment() -> SentimentSnapshot:
    return SentimentSnapshot(
        symbol="AAPL",
        as_of=datetime(2026, 6, 17, 9, 0),
        source="benzinga",
        score=0.42,
        polarity=SentimentPolarity.BULLISH,
        headline_count=23,
    )


@pytest.fixture
def sample_fundamentals() -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        symbol="AAPL",
        as_of=date(2026, 6, 17),
        eps=6.1,
        revenue=391_000_000_000,
        pe_ratio=31.2,
        revisions=[],
    )


@pytest.fixture
def sample_filings() -> list[FilingSummary]:
    return [
        FilingSummary(
            symbol="AAPL",
            filing_type=FilingType.TEN_Q,
            filed_on=date(2026, 5, 1),
            period_end=date(2026, 3, 31),
            summary="Q2 results filed.",
            url="https://example.com/filing",
        )
    ]
