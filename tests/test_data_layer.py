from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_layer.exceptions import DataValidationError, ProviderFetchError
from data_layer.models import OrderBookSnapshot, PriceSeries
from data_layer.openbb_client import OpenBBDataClient


def _client_with_mocked_obb(mock_obb: MagicMock) -> OpenBBDataClient:
    client = OpenBBDataClient()
    client._obb = mock_obb  # bypass the real `from openbb import obb` lazy import
    return client


def test_get_price_history_validates_and_returns_price_series():
    mock_obb = MagicMock()
    df = pd.DataFrame(
        [
            {"date": datetime(2026, 6, 1), "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000},
            {"date": datetime(2026, 6, 2), "open": 103.0, "high": 108.0, "low": 102.0, "close": 107.0, "volume": 1200},
        ]
    )
    mock_obb.equity.price.historical.return_value.to_df.return_value = df

    client = _client_with_mocked_obb(mock_obb)
    series = client.get_price_history("AAPL", date(2026, 6, 1), date(2026, 6, 2))

    assert isinstance(series, PriceSeries)
    assert len(series.bars) == 2
    assert series.bars[0].close == 103.0


def test_get_price_history_raises_data_validation_error_on_malformed_row():
    mock_obb = MagicMock()
    bad_df = pd.DataFrame([{"date": datetime(2026, 6, 1), "open": 100.0, "high": 50.0, "low": 99.0, "close": 103.0, "volume": 1000}])
    mock_obb.equity.price.historical.return_value.to_df.return_value = bad_df

    client = _client_with_mocked_obb(mock_obb)
    with pytest.raises(DataValidationError):
        client.get_price_history("AAPL", date(2026, 6, 1), date(2026, 6, 2))


def test_get_price_history_wraps_provider_exception():
    mock_obb = MagicMock()
    mock_obb.equity.price.historical.side_effect = RuntimeError("provider timeout")

    client = _client_with_mocked_obb(mock_obb)
    with pytest.raises(ProviderFetchError):
        client.get_price_history("AAPL", date(2026, 6, 1), date(2026, 6, 2))


def test_get_order_book_rejects_crossed_book():
    mock_obb = MagicMock()
    df = pd.DataFrame([{"date": datetime(2026, 6, 1), "bid": 105.0, "bid_size": 10, "ask": 100.0, "ask_size": 10}])
    mock_obb.equity.price.quote.return_value.to_df.return_value = df

    client = _client_with_mocked_obb(mock_obb)
    with pytest.raises(DataValidationError):
        client.get_order_book("AAPL")


def test_get_order_book_valid_snapshot():
    mock_obb = MagicMock()
    df = pd.DataFrame([{"date": datetime(2026, 6, 1), "bid": 99.5, "bid_size": 10, "ask": 100.5, "ask_size": 8}])
    mock_obb.equity.price.quote.return_value.to_df.return_value = df

    client = _client_with_mocked_obb(mock_obb)
    snapshot = client.get_order_book("AAPL")
    assert isinstance(snapshot, OrderBookSnapshot)
    assert snapshot.bids[0].price < snapshot.asks[0].price
