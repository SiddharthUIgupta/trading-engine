from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from config.settings import Settings
from data_layer.alpaca_reference_client import AlpacaAssetReferenceClient


def test_get_shortable_status_returns_real_fields():
    with patch("data_layer.alpaca_reference_client.TradingClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_asset = MagicMock(shortable=True, easy_to_borrow=False)
        mock_client.get_asset.return_value = mock_asset
        mock_client_cls.return_value = mock_client

        settings = Settings(_env_file=None)
        client = AlpacaAssetReferenceClient(settings)
        status = client.get_shortable_status("GME")

    assert status is not None
    assert status.symbol == "GME"
    assert status.shortable is True
    assert status.easy_to_borrow is False


def test_get_shortable_status_returns_none_for_unrecognized_symbol():
    with patch("data_layer.alpaca_reference_client.TradingClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get_asset.side_effect = Exception("asset not found: 404")
        mock_client_cls.return_value = mock_client

        settings = Settings(_env_file=None)
        client = AlpacaAssetReferenceClient(settings)
        status = client.get_shortable_status("NOTATICKER")

    assert status is None


def test_get_shortable_status_raises_on_genuine_api_failure():
    from data_layer.exceptions import ProviderFetchError

    with patch("data_layer.alpaca_reference_client.TradingClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get_asset.side_effect = Exception("connection timeout")
        mock_client_cls.return_value = mock_client

        settings = Settings(_env_file=None)
        client = AlpacaAssetReferenceClient(settings)

        try:
            client.get_shortable_status("GME")
            assert False, "expected ProviderFetchError"
        except ProviderFetchError:
            pass


def test_alpaca_reference_client_never_imports_execution_layer():
    """Guard against an architecture violation: this file lives in
    data_layer and must never import execution_layer (AlpacaBroker lives
    there) — see the module docstring's explanation of why. Uses ast, not
    string matching — the docstring itself mentions "execution_layer" in
    prose, which a naive text search would misfire on.
    """
    import ast

    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / "data_layer" / "alpaca_reference_client.py").read_text()
    tree = ast.parse(source)

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(m == "execution_layer" or m.startswith("execution_layer.") for m in imported_modules)
