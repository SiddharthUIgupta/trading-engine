"""Kronos-small shadow signal provider.

Vendored model: analyst_layer/vendor/kronos (MIT license, github.com/shiyu-coder/Kronos).
Loads once per process — this provider is meant to be used from a short-lived
batch script (scripts/kronos_shadow_signal_job.py), never imported into the
long-running Alpha/Protection daemons, so torch's memory footprint is fully
released when the batch job exits.

lookback_bars defaults to 64, not the model's max_context=512 — measured on
this Pi's CPU: 30 paths at 512 bars took 666s/ticker (~7x over budget), 64
bars/30 paths took ~58s/ticker with real margin. See config/settings.py
kronos_lookback_bars docstring.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analyst_layer.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer
from data_layer.models import PriceSeries
from config.settings import Settings

logger = logging.getLogger(__name__)

_TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
_MODEL_REPO = "NeoQuasar/Kronos-small"


class KronosSignalProvider:
    name = "kronos_small"
    # Bump this if lookback/paths/horizon change — signal_values rows are keyed
    # on (candidate_id, signal_name, signal_version, metric_name), so a version
    # bump means old and new results never collide, and the uplift report
    # never silently mixes results computed under different settings.
    version = "kronos-small-v1"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        logger.info("Loading %s / %s (this happens once per process)...", _MODEL_REPO, _TOKENIZER_REPO)
        tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_REPO)
        model = Kronos.from_pretrained(_MODEL_REPO)
        self._predictor = KronosPredictor(model, tokenizer, max_context=settings.kronos_max_context)
        logger.info("Kronos-small loaded on device=%s", self._predictor.device)

    def compute(self, ticker: str, pit_snapshot: PriceSeries) -> dict[str, float] | None:
        bars = pit_snapshot.bars[-self._settings.kronos_lookback_bars:]
        if len(bars) < self._settings.kronos_lookback_bars:
            logger.debug("%s: only %d bars available, need %d — Empty", ticker, len(bars), self._settings.kronos_lookback_bars)
            return None

        df = pd.DataFrame({
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        })
        x_timestamp = pd.Series([pd.Timestamp(b.timestamp) for b in bars])
        pred_len = self._settings.kronos_horizon_sessions
        y_timestamp = pd.Series(pd.bdate_range(start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), periods=pred_len))

        paths = self._predictor.predict_paths(
            df=df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=1.0, top_p=0.9,
            sample_count=self._settings.kronos_mc_paths, verbose=False,
        )

        entry_price = bars[-1].close
        path_returns_21d = np.array([(path["close"].iloc[-1] - entry_price) / entry_price for path in paths])

        win_pct = self._settings.thesis_trailing_stop_activation_pct
        loss_pct = self._settings.thesis_stop_loss_pct
        touches_win = 0
        for path in paths:
            closes = path["close"].values
            cum_return = (closes - entry_price) / entry_price
            win_idx = np.argmax(cum_return >= win_pct) if np.any(cum_return >= win_pct) else None
            loss_idx = np.argmax(cum_return <= -loss_pct) if np.any(cum_return <= -loss_pct) else None
            if win_idx is not None and (loss_idx is None or win_idx <= loss_idx):
                touches_win += 1

        return {
            "p_touch_win": touches_win / len(paths),
            "med_ret_21d": float(np.median(path_returns_21d)),
            "path_dispersion": float(np.std(path_returns_21d)),
        }
