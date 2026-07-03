"""Alpha Plane entry point — scanning, LLM consensus, candidate sizing.

Runs as trading-engine-alpha systemd unit. Writes approved BUY intents to
the order_intents DB table. Protection Plane reads and executes them.

If this process crashes (LLM timeout, OpenBB outage, scan exception), the
Protection Plane continues running and existing positions remain protected.

Schedule (ET):
  8:15 AM   — thesis_scan_and_trade (pre-market universe + LLM consensus)
  9:05 AM   — gap_scan_and_queue (pre-market gap up candidates)
  9:30 AM   — market_open_execution (queue any pending intents for 9:30 open)
  9:35 AM   — swing_scan_and_trade
  Every 15m 9:30–16:00 — momentum_scan_and_trade (ORB equity), options_scan_and_trade
  10:00 AM, 1:00 PM — vol_options_scan_and_trade
  16:30 PM  — post_market_logging
  Every 15s — check_manual_trigger
"""
from __future__ import annotations

import logging
import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_engine_alpha.log"),
    ],
)
logger = logging.getLogger(__name__)

from anthropic import Anthropic
from config.settings import get_settings
from data_layer.openbb_client import OpenBBDataClient
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import RobustCircuitBreaker
from execution_layer.state_store import StateStore
from execution_layer.alpha_plane import AlphaRuntime
from execution_layer import alerting

ET = "America/New_York"

WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "AMZN", "META", "TSLA"]

settings = get_settings()
state_store = StateStore(settings.state_db_path)
broker = AlpacaBroker.from_settings(settings)
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

intraday_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.intraday_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="intraday",
)
options_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.options_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="options",
)
thesis_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.thesis_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="thesis",
)
swing_breaker = RobustCircuitBreaker(
    max_position_size_pct=settings.max_position_size_pct,
    max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
    capital_limit_pct=settings.swing_capital_pct,
    daily_profit_target_usd=settings.daily_profit_target_usd,
    name="swing",
)

watchlist = WATCHLIST

runtime = AlphaRuntime(
    settings=settings,
    data_client=data_client,
    broker=broker,
    intraday_breaker=intraday_breaker,
    options_breaker=options_breaker,
    thesis_breaker=thesis_breaker,
    swing_breaker=swing_breaker,
    state_store=state_store,
    anthropic_client=anthropic_client,
    watchlist=watchlist,
)

from execution_layer import alerting as _alerting

def _heartbeat(fn, job_id):  # noqa: ANN001
    def _wrapped():
        try:
            fn()
        finally:
            _alerting.ping_heartbeat(job_id)
    return _wrapped

scheduler = BlockingScheduler(timezone=ET)

scheduler.add_job(
    _heartbeat(runtime.pre_market_scan, "pre_market_scan"),
    CronTrigger(day_of_week="mon-fri", hour=8, minute=15, timezone=ET),
    id="alpha_pre_market_scan", max_instances=1, misfire_grace_time=300,
)
scheduler.add_job(
    _heartbeat(runtime.gap_scan_and_queue, "gap_scan_and_queue"),
    CronTrigger(day_of_week="mon-fri", hour=9, minute=5, timezone=ET),
    id="alpha_gap_scan", max_instances=1, misfire_grace_time=120,
)
scheduler.add_job(
    _heartbeat(runtime.market_open_execution, "market_open_execution"),
    CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=ET),
    id="alpha_market_open", max_instances=1, misfire_grace_time=120,
)
scheduler.add_job(
    _heartbeat(runtime.swing_scan_and_trade, "swing_scan_and_trade"),
    CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone=ET),
    id="alpha_swing_scan", max_instances=1, misfire_grace_time=120,
)
scheduler.add_job(
    _heartbeat(runtime.thesis_scan_and_trade, "thesis_scan_and_trade"),
    CronTrigger(day_of_week="mon-fri", hour=10, minute=0, timezone=ET),
    id="alpha_thesis_scan", max_instances=1, misfire_grace_time=300,
)
scheduler.add_job(
    _heartbeat(runtime.momentum_scan_and_trade, "momentum_scan_and_trade"),
    CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone=ET),
    id="alpha_momentum_scan", max_instances=1, misfire_grace_time=60,
)
scheduler.add_job(
    _heartbeat(runtime.options_scan_and_trade, "options_scan_and_trade"),
    CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone=ET),
    id="alpha_options_scan", max_instances=1, misfire_grace_time=60,
)
scheduler.add_job(
    _heartbeat(runtime.vol_options_scan_and_trade, "vol_options_scan_and_trade"),
    CronTrigger(day_of_week="mon-fri", hour="10,13", minute=0, timezone=ET),
    id="alpha_vol_options_scan", max_instances=1, misfire_grace_time=120,
)
scheduler.add_job(
    _heartbeat(runtime.post_market_logging, "post_market_logging"),
    CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=ET),
    id="alpha_post_market_logging", max_instances=1, misfire_grace_time=300,
)
scheduler.add_job(
    runtime.check_manual_trigger,
    "interval",
    seconds=15,
    id="alpha_manual_trigger_watcher",
    max_instances=1,
)

def _shutdown(signum, frame):  # noqa: ANN001
    logger.info("Alpha Plane shutting down (signal %d)", signum)
    scheduler.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

logger.info("=== TRADING ENGINE: ALPHA PLANE STARTED ===")
logger.info("Intents will be written to order_intents table for Protection to execute")

try:
    scheduler.start()
except Exception as exc:
    logger.critical("Alpha Plane scheduler crashed: %s", exc, exc_info=True)
    sys.exit(1)
