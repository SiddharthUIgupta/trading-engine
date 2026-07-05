"""Read-only Alpaca reference data — asset metadata, not trading actions.

Deliberately NOT execution_layer.AlpacaBroker: this repo's architecture is
one-directional (data_layer -> analyst_layer -> execution_layer), and
AlpacaBroker lives in execution_layer. A shadow-signal provider in
analyst_layer needs Alpaca's shortable/easy_to_borrow flags but must never
import execution_layer to get them — this client builds its own read-only
TradingClient directly from settings, the same way OpenBBDataClient takes
settings.openbb_pat directly rather than going through another layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient

from config.settings import Settings
from data_layer.exceptions import ProviderFetchError
from data_layer.models import ShortableStatus

logger = logging.getLogger(__name__)


class AlpacaAssetReferenceClient:
    def __init__(self, settings: Settings) -> None:
        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=not settings.is_live,
        )

    def get_shortable_status(self, symbol: str) -> ShortableStatus | None:
        """Returns None (not an error) if the symbol isn't a recognized
        tradable asset on Alpaca — a real, non-error case for illiquid or
        newly-listed tickers, distinct from an actual API failure.
        """
        try:
            asset = self._client.get_asset(symbol)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "not found" in message or "404" in message:
                return None
            raise ProviderFetchError(f"get_asset failed for {symbol}: {exc}") from exc

        return ShortableStatus(
            symbol=symbol,
            as_of=datetime.now(timezone.utc),
            shortable=bool(asset.shortable),
            easy_to_borrow=bool(asset.easy_to_borrow),
        )
