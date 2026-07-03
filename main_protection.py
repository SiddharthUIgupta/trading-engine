"""Protection Plane entry point — exits, reconciliation, order execution.

Runs as trading-engine-protection systemd unit. If the Alpha Plane crashes
(LLM timeout, scraper failure), this process keeps running and all existing
positions remain protected by stop-losses and exit rules.

Schedule (ET):
  Every 15 min 9:30–16:00 — intraday_monitoring (reconcile → consume intents → exits)
  15:30 ET                 — pre_close_orb_exit
"""
from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_engine_protection.log"),
    ],
)
logger = logging.getLogger(__name__)

from anthropic import Anthropic
from config.settings import get_settings
from data_layer.openbb_client import OpenBBDataClient
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import CircuitBreaker
from execution_layer.state_store import StateStore
from execution_layer.protection_plane import ProtectionRuntime
from execution_layer import alerting

ET = "America/New_York"

settings = get_settings()
state_store = StateStore(settings.state_db_path)
broker = AlpacaBroker(
    api_key=settings.alpaca_api_key,
    secret_key=settings.alpaca_secret_key,
    paper=settings.alpaca_paper,
)
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

intraday_breaker = CircuitBreaker(
    name="intraday",
    max_daily_drawdown_pct=settings.circuit_breaker_max_daily_drawdown_pct,
)
options_breaker = CircuitBreaker(
    name="options",
    max_daily_drawdown_pct=settings.options_circuit_breaker_max_daily_drawdown_pct,
)
thesis_breaker = CircuitBreaker(
    name="thesis",
    max_daily_drawdown_pct=settings.thesis_circuit_breaker_max_daily_drawdown_pct,
)
swing_breaker = CircuitBreaker(
    name="swing",
    max_daily_drawdown_pct=settings.swing_circuit_breaker_max_daily_drawdown_pct,
)

runtime = ProtectionRuntime(
    settings=settings,
    broker=broker,
    state_store=state_store,
    anthropic_client=anthropic_client,
    data_client=data_client,
    intraday_breaker=intraday_breaker,
    options_breaker=options_breaker,
    thesis_breaker=thesis_breaker,
    swing_breaker=swing_breaker,
)

scheduler = BlockingScheduler(timezone=ET)

# Every 15 minutes during market hours
scheduler.add_job(
    runtime.intraday_monitoring,
    CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone=ET),
    id="protection_intraday_monitoring",
    max_instances=1,
    misfire_grace_time=60,
)

# Pre-close ORB exit at 3:30pm ET
scheduler.add_job(
    runtime.pre_close_orb_exit,
    CronTrigger(day_of_week="mon-fri", hour=15, minute=30, timezone=ET),
    id="protection_pre_close_orb_exit",
    max_instances=1,
    misfire_grace_time=120,
)

def _shutdown(signum, frame):  # noqa: ANN001
    logger.info("Protection Plane shutting down (signal %d)", signum)
    scheduler.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

logger.info("=== TRADING ENGINE: PROTECTION PLANE STARTED ===")
logger.info("Intraday monitoring every 15 min (9:30-16:00 ET)")
logger.info("Pre-close ORB exit at 15:30 ET")

try:
    scheduler.start()
except Exception as exc:
    logger.critical("Protection Plane scheduler crashed: %s", exc, exc_info=True)
    sys.exit(1)
