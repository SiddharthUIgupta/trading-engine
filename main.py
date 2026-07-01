"""Entrypoint: wires the three layers together and starts the scheduler.

Run with: python main.py
Stop with Ctrl+C — the scheduler shuts down cleanly and the process exits.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
import traceback
from datetime import date, timedelta

from anthropic import Anthropic

from config.settings import get_settings
from data_layer.openbb_client import OpenBBDataClient
from execution_layer import alerting
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import GlobalRiskState, RobustCircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.scheduler import build_scheduler
from execution_layer.state_store import StateStore

WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "AMZN", "META", "TSLA"]


def startup_health_check(
    broker: AlpacaBroker,
    data_client: OpenBBDataClient,
    state_store: StateStore,
    env: str,
) -> float:
    """Verify broker, data, and DB are reachable before the scheduler starts.

    Returns account equity so the caller can include it in the startup alert.
    Raises on any failure — the process should exit rather than run broken.
    """
    logger = logging.getLogger("startup")

    logger.info("Health check: verifying Alpaca connectivity...")
    equity = broker.get_equity()
    logger.info("Health check: Alpaca OK — equity $%.2f", equity)

    logger.info("Health check: verifying data layer...")
    today = date.today()
    data_client.get_price_history("SPY", start_date=today - timedelta(days=5), end_date=today)
    logger.info("Health check: data layer OK")

    logger.info("Health check: verifying SQLite...")
    state_store.get_positions()
    logger.info("Health check: SQLite OK")

    logger.info("Health check: all systems go (equity $%.2f, env=%s)", equity, env)
    return equity


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("main")

    settings = get_settings()
    env = "LIVE" if settings.is_live else "PAPER"
    if settings.is_live:
        logger.warning("LIVE TRADING MODE — real capital is at risk. Confirm this is intentional.")
    else:
        logger.info("Paper trading mode (default).")

    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)
    broker = AlpacaBroker.from_settings(settings)
    global_risk = GlobalRiskState(
        max_weekly_drawdown_pct=settings.max_weekly_drawdown_pct,
        max_trailing_drawdown_pct=settings.max_trailing_drawdown_pct,
    )
    intraday_breaker = RobustCircuitBreaker(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        capital_limit_pct=settings.intraday_capital_pct,
        daily_profit_target_usd=settings.daily_profit_target_usd,
        name="intraday",
        consecutive_loss_limit=settings.consecutive_loss_limit,
        global_state=global_risk,
    )
    options_breaker = RobustCircuitBreaker(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        capital_limit_pct=settings.options_capital_pct,
        daily_profit_target_usd=settings.daily_profit_target_usd,
        name="options",
        consecutive_loss_limit=settings.consecutive_loss_limit,
        global_state=global_risk,
    )
    thesis_breaker = RobustCircuitBreaker(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        capital_limit_pct=settings.thesis_capital_pct,
        daily_profit_target_usd=settings.daily_profit_target_usd,
        name="thesis",
        consecutive_loss_limit=settings.consecutive_loss_limit,
        global_state=global_risk,
    )
    swing_breaker = RobustCircuitBreaker(
        max_position_size_pct=settings.max_position_size_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        capital_limit_pct=settings.swing_capital_pct,
        daily_profit_target_usd=settings.daily_profit_target_usd,
        name="swing",
        consecutive_loss_limit=settings.consecutive_loss_limit,
        global_state=global_risk,
    )
    state_store = StateStore(settings.state_db_path)

    try:
        equity = startup_health_check(broker, data_client, state_store, env)
    except Exception as exc:
        logger.critical("Startup health check FAILED — aborting: %s", exc)
        alerting.alert_crash(f"Startup health check failed: {exc}")
        sys.exit(1)

    runtime = TradingRuntime(
        settings=settings,
        data_client=data_client,
        broker=broker,
        intraday_breaker=intraday_breaker,
        options_breaker=options_breaker,
        thesis_breaker=thesis_breaker,
        swing_breaker=swing_breaker,
        state_store=state_store,
        anthropic_client=anthropic_client,
        watchlist=WATCHLIST,
        global_risk_state=global_risk,
    )
    scheduler = build_scheduler(runtime)
    runtime._halt_callback = scheduler.pause

    scheduler.start()
    logger.info("Scheduler started. Jobs: %s", [job.id for job in scheduler.get_jobs()])
    alerting.alert_startup(equity=equity, env=env)

    stop = {"requested": False}

    def _handle_sigint(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        while not stop["requested"]:
            time.sleep(1)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.critical("Unhandled exception in main loop: %s\n%s", exc, tb)
        alerting.alert_crash(tb)
        raise
    finally:
        logger.info("Shutting down scheduler.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logging.getLogger("main").critical("Fatal crash: %s", exc)
        alerting.alert_crash(tb)
        sys.exit(1)
