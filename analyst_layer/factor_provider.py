"""Bridge to Vibe-Trading's alpha factor zoo (analyst_layer never imports
execution_layer, so this stays a pure feature-lookup — no sizing decisions).

Shadow only: values computed here feed the VW bandit's |factors namespace
(see vw_bandit.py) but do not influence Kelly sizing until settings.
promoted_vw_factors is populated AND scripts/signal_uplift.py shows edge
at n>=300 closed trades (CLAUDE.md Phase 2, Step 6 — not yet approved).

Import path confirmed against the installed package layout: pyproject.toml
maps `agent/` as the setuptools package-dir root, so `agent/src/factors/
registry.py` installs as `src.factors.registry` — not `src.alpha.
factor_registry` as an earlier draft of CLAUDE.md assumed.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def compute_factor_features(ticker: str, factor_ids: list[str]) -> dict[str, float]:
    """Returns {factor_id: latest_value} for each promoted factor.

    Empty dict when factor_ids is empty (the default — this function is a
    no-op until PROMOTED_VW_FACTORS is explicitly set) or when Vibe-Trading
    isn't importable on this machine.
    """
    if not factor_ids:
        return {}

    try:
        from src.factors.registry import Registry
    except ImportError as exc:
        logger.warning("factor_provider: Vibe-Trading not importable — %s", exc)
        return {}

    try:
        registry = Registry()
    except Exception as exc:
        logger.warning("factor_provider: Registry() init failed: %s", exc)
        return {}

    raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        logger.warning("factor_provider: no OHLCV data for %s", ticker)
        return {}
    panel = {ticker: raw}

    out: dict[str, float] = {}
    for factor_id in factor_ids:
        try:
            result: pd.DataFrame = registry.compute(factor_id, panel)
            series = result[ticker].dropna()
            if series.empty:
                logger.debug("factor_provider: %s produced no values for %s", factor_id, ticker)
                continue
            out[factor_id] = float(series.iloc[-1])
        except Exception as exc:
            logger.debug("factor_provider: %s failed for %s: %s", factor_id, ticker, exc)
    return out
