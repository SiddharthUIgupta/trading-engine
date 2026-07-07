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
from execution_layer.guardrails import GlobalRiskState, RobustCircuitBreaker
from execution_layer.state_store import StateStore
from execution_layer.protection_plane import ProtectionRuntime
from execution_layer import alerting

ET = "America/New_York"

settings = get_settings()
state_store = StateStore(settings.state_db_path)
broker = AlpacaBroker.from_settings(settings)
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

global_risk_state = GlobalRiskState(
    max_weekly_drawdown_pct=settings.max_weekly_drawdown_pct,
    max_trailing_drawdown_pct=settings.max_trailing_drawdown_pct,
)

intraday_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.intraday_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="intraday",
    global_state=global_risk_state,
)
options_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.options_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="options",
    global_state=global_risk_state,
)
thesis_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.thesis_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="thesis",
    global_state=global_risk_state,
)
swing_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.swing_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="swing",
    global_state=global_risk_state,
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
    global_risk_state=global_risk_state,
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
    alerting.alert_startup(equity=broker.get_equity(), env=settings.trading_env)
except Exception as exc:  # noqa: BLE001 — a broken startup alert must never block the process from starting
    logger.warning("Startup alert failed: %s", exc)

try:
    scheduler.start()
except Exception as exc:
    logger.critical("Protection Plane scheduler crashed: %s", exc, exc_info=True)
    try:
        alerting.alert_crash(str(exc))
    except Exception as alert_exc:  # noqa: BLE001 — the crash alert itself must not mask the real crash
        logger.warning("Crash alert failed: %s", alert_exc)
    sys.exit(1)
