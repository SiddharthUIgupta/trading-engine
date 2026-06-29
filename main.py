"""Entrypoint: wires the three layers together and starts the scheduler.

Run with: python main.py
Stop with Ctrl+C — the scheduler shuts down cleanly and the process exits.
"""
from __future__ import annotations

import logging
import signal
import time

from anthropic import Anthropic

from config.settings import get_settings
from data_layer.openbb_client import OpenBBDataClient
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.scheduler import build_scheduler
from execution_layer.state_store import StateStore

WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "AMZN", "META", "TSLA"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("main")

    settings = get_settings()
    if settings.is_live:
        logger.warning("LIVE TRADING MODE — real capital is at risk. Confirm this is intentional.")
    else:
        logger.info("Paper trading mode (default).")

    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)
    broker = AlpacaBroker.from_settings(settings)
    circuit_breaker = CircuitBreaker(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        daily_profit_target_usd=settings.daily_profit_target_usd,
    )
    state_store = StateStore(settings.state_db_path)

    runtime = TradingRuntime(
        settings=settings,
        data_client=data_client,
        broker=broker,
        circuit_breaker=circuit_breaker,
        state_store=state_store,
        anthropic_client=anthropic_client,
        watchlist=WATCHLIST,
    )
    scheduler = build_scheduler(runtime)
    runtime._halt_callback = scheduler.pause  # wire the breaker's halt action to actually pause the loop

    scheduler.start()
    logger.info("Scheduler started. Jobs: %s", [job.id for job in scheduler.get_jobs()])

    stop = {"requested": False}

    def _handle_sigint(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        while not stop["requested"]:
            time.sleep(1)
    finally:
        logger.info("Shutting down scheduler.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
