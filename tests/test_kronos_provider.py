from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from data_layer.models import PriceBar, PriceSeries


def _synthetic_snapshot(n_bars: int = 64) -> PriceSeries:
    # Flat price series so the provider's entry_price (bars[-1].close) is
    # deterministic and matches the test's assumed entry=100.0 baseline.
    bars = [
        PriceBar(
            symbol="TEST", timestamp=datetime(2026, 1, 1) + timedelta(days=i),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1_000_000,
        )
        for i in range(n_bars)
    ]
    return PriceSeries(symbol="TEST", interval="1d", bars=bars)


def test_p_touch_win_reads_thresholds_from_settings_not_hardcoded():
    """The touch-probability calc must use settings.thesis_trailing_stop_activation_pct
    (win) and settings.thesis_stop_loss_pct (loss) — not hardcoded 0.20/0.18 —
    since those are this repo's real thesis exit rules, not arbitrary numbers.
    Mocks the model/predictor so this test has no torch dependency and runs fast.
    """
    from unittest.mock import MagicMock, patch

    settings = Settings(_env_file=None, THESIS_TRAILING_STOP_ACTIVATION_PCT="0.35", THESIS_STOP_LOSS_PCT="0.11")
    assert settings.thesis_trailing_stop_activation_pct == 0.35
    assert settings.thesis_stop_loss_pct == 0.11

    with patch("analyst_layer.kronos_provider.Kronos") as mock_kronos, \
         patch("analyst_layer.kronos_provider.KronosTokenizer") as mock_tokenizer, \
         patch("analyst_layer.kronos_provider.KronosPredictor") as mock_predictor_cls:
        mock_predictor = MagicMock()
        mock_predictor.device = "cpu"
        mock_predictor_cls.return_value = mock_predictor

        # Two paths: one that touches +35% before -11% (a "win"), one that touches
        # -11% first (a "loss") — if the provider used the old hardcoded 0.20/0.18,
        # this exact path pair would misclassify.
        entry = 100.0
        idx = pd.bdate_range(start="2026-03-01", periods=21)
        win_path = pd.DataFrame({
            "open": [entry] * 21, "high": [entry] * 21, "low": [entry] * 21,
            "close": [entry * 1.36] * 10 + [entry * 1.40] * 11,  # jumps straight to +36-40%, past 0.35 not 0.20
            "volume": [0] * 21, "amount": [0] * 21,
        }, index=idx)
        loss_path = pd.DataFrame({
            "open": [entry] * 21, "high": [entry] * 21, "low": [entry] * 21,
            "close": [entry * 0.87] * 21,  # -13%: past 0.11 (new threshold) but NOT past old hardcoded 0.18
            "volume": [0] * 21, "amount": [0] * 21,
        }, index=idx)
        mock_predictor.predict_paths.return_value = [win_path, loss_path]

        from analyst_layer.kronos_provider import KronosSignalProvider
        provider = KronosSignalProvider(settings)
        result = provider.compute("TEST", _synthetic_snapshot())

    assert result is not None
    # With settings-driven thresholds (0.35/0.11): win_path touches +36% (>=0.35) -> win;
    # loss_path touches -13% (<=-0.11) -> loss. p_touch_win should be 0.5 (1 of 2).
    assert result["p_touch_win"] == pytest.approx(0.5)


@pytest.mark.skipif(
    os.environ.get("RUN_KRONOS_INTEGRATION_TEST") != "1",
    reason="Real Kronos inference — slow (~1min+) and needs network access to HuggingFace. "
           "Set RUN_KRONOS_INTEGRATION_TEST=1 to run.",
)
def test_real_kronos_inference_end_to_end():
    """Opt-in, slow, real integration test — actually loads the model and runs
    inference. Skipped by default so the main suite stays fast and doesn't
    depend on network/HF availability.
    """
    from config.settings import get_settings
    from data_layer.openbb_client import OpenBBDataClient
    from analyst_layer.kronos_provider import KronosSignalProvider

    settings = get_settings()
    dc = OpenBBDataClient(pat=settings.openbb_pat or None)
    series = dc.get_price_history("AAPL", start_date=date.today() - timedelta(days=200), end_date=date.today())

    provider = KronosSignalProvider(settings)
    result = provider.compute("AAPL", series)

    assert result is not None
    assert set(result.keys()) == {"p_touch_win", "med_ret_21d", "path_dispersion"}
    assert 0.0 <= result["p_touch_win"] <= 1.0
    assert np.isfinite(result["med_ret_21d"])
    assert result["path_dispersion"] >= 0.0
