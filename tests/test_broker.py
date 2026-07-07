from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from alpaca.common.exceptions import APIError

from execution_layer.broker import AlpacaBroker


def _api_error(status_code: int, message: str = "error") -> APIError:
    error = json.dumps({"code": 40410000 if status_code == 404 else 50000000, "message": message})
    http_error = MagicMock()
    http_error.response.status_code = status_code
    return APIError(error, http_error)


@pytest.fixture
def broker() -> AlpacaBroker:
    return AlpacaBroker(trading_client=MagicMock(), is_live=False)


# ── Regression: get_position_detail/get_position_shares previously caught a
# bare Exception and treated ANY failure (network error, 5xx, rate limit) as
# "no open position" — silently returning None/0.0. Reconciliation reads that
# as "position closed" and deletes the local record, permanently stripping
# stop-loss protection from a still-open position over a transient API blip.

def test_get_position_detail_returns_none_on_genuine_404(broker: AlpacaBroker):
    broker._client.get_open_position.side_effect = _api_error(404, "position does not exist")
    assert broker.get_position_detail("AAPL") is None


def test_get_position_detail_raises_on_non_404_api_error(broker: AlpacaBroker):
    broker._client.get_open_position.side_effect = _api_error(500, "internal server error")
    with pytest.raises(APIError):
        broker.get_position_detail("AAPL")


def test_get_position_shares_returns_zero_on_genuine_404(broker: AlpacaBroker):
    broker._client.get_open_position.side_effect = _api_error(404, "position does not exist")
    assert broker.get_position_shares("AAPL") == 0.0


def test_get_position_shares_raises_on_non_404_api_error(broker: AlpacaBroker):
    broker._client.get_open_position.side_effect = _api_error(503, "service unavailable")
    with pytest.raises(APIError):
        broker.get_position_shares("AAPL")
