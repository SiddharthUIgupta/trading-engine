"""One-shot runner: vol options scan for AAPL using real APIs.

Uses real Anthropic, OpenBB/yfinance, and Alpaca paper-trading credentials
from .env. Runs the full scan → consensus → (conditional) order flow and
prints what happened. Safe to run any time — paper account only.
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from anthropic import Anthropic

from config.settings import Settings
from data_layer.openbb_client import OpenBBDataClient
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.state_store import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_vol_aapl")


def main() -> None:
    settings = Settings()

    if not settings.vol_options_track_enabled:
        logger.error("VOL_OPTIONS_TRACK_ENABLED is not set — check .env")
        sys.exit(1)

    logger.info("Settings: model=%s subagent=%s vol_track=%s",
                settings.anthropic_model, settings.anthropic_subagent_model,
                settings.vol_options_track_enabled)
    logger.info("Vol params: target_dte=%d  min/max_dte=%d/%d  profit_target=%.0f%%  loss_limit=%.0fx  roll_dte=%dd",
                settings.vol_options_target_dte, settings.vol_options_min_dte,
                settings.vol_options_max_dte, settings.vol_options_profit_target_pct * 100,
                settings.vol_options_loss_limit_multiplier, settings.vol_options_roll_dte)

    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)
    broker = AlpacaBroker.from_settings(settings)

    equity = broker.get_equity()
    logger.info("Alpaca paper account equity: $%.2f", equity)

    breaker = CircuitBreaker(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        daily_profit_target_usd=settings.daily_profit_target_usd,
    )
    breaker.start_trading_day(equity=equity, today=date.today())

    state_store = StateStore(settings.state_db_path)

    runtime = TradingRuntime(
        settings=settings,
        data_client=data_client,
        broker=broker,
        circuit_breaker=breaker,
        state_store=state_store,
        anthropic_client=anthropic_client,
        watchlist=["AAPL"],
    )

    logger.info("=" * 60)
    logger.info("Running vol options scan for AAPL ...")
    logger.info("=" * 60)
    runtime.vol_options_scan_and_trade()

    logger.info("=" * 60)
    logger.info("DONE — state summary:")
    for pos in state_store.get_option_positions():
        if pos["strategy"] == "vol_short":
            logger.info("  vol_short  %s  qty=%d  avg_price=%.2f  exp=%s",
                        pos["contract_symbol"], pos["quantity"],
                        pos["avg_entry_price"], pos["expiration"])
    events = state_store.get_events(limit=20)
    for e in events:
        if "vol" in e["event_type"]:
            logger.info("  event: %s — %s", e["event_type"], e["detail"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
