"""Chronological workflow scheduling via APScheduler.

Wires the four named phases from the mandate (pre-market scan,
market-open execution, intraday monitoring, post-market logging) to
cron triggers in the exchange timezone. This module only knows about
*when* — all the *what* lives in execution_layer.runtime.TradingRuntime,
which this module calls into.
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from execution_layer import alerting
from execution_layer.runtime import TradingRuntime

logger = logging.getLogger(__name__)


def _heartbeat(fn, job_id: str):
    """Wraps a job so it pings the heartbeat file (and optionally healthchecks.io)
    after every successful completion. A job that hangs or crashes never pings,
    making silence detectable.
    """
    def _wrapped():
        fn()
        alerting.ping_heartbeat(job_id)
    _wrapped.__name__ = fn.__name__
    return _wrapped

EXCHANGE_TZ = "America/New_York"


def build_scheduler(runtime: TradingRuntime) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=EXCHANGE_TZ)

    scheduler.add_job(
        _heartbeat(runtime.pre_market_scan, "pre_market_scan"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=0, timezone=EXCHANGE_TZ),
        id="pre_market_scan",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        _heartbeat(runtime.gap_scan_and_queue, "gap_scan_and_queue"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=5, timezone=EXCHANGE_TZ),
        id="gap_scan_and_queue",
        misfire_grace_time=120,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.market_open_execution, "market_open_execution"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=EXCHANGE_TZ),
        id="market_open_execution",
        misfire_grace_time=120,
    )
    scheduler.add_job(
        _heartbeat(runtime.intraday_monitoring, "intraday_monitoring"),
        trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone=EXCHANGE_TZ),
        id="intraday_monitoring",
        misfire_grace_time=300,  # 5 min: a late tick beats a silent skip during fast markets
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.momentum_scan_and_trade, "momentum_scan_and_trade"),
        trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30", timezone=EXCHANGE_TZ),
        id="momentum_scan_and_trade",
        misfire_grace_time=120,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.options_scan_and_trade, "options_scan_and_trade"),
        trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="15,45", timezone=EXCHANGE_TZ),
        id="options_scan_and_trade",
        misfire_grace_time=120,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.thesis_scan_and_trade, "thesis_scan_and_trade"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=15, timezone=EXCHANGE_TZ),
        id="thesis_scan_and_trade",
        misfire_grace_time=300,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.swing_scan_and_trade, "swing_scan_and_trade"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=45, timezone=EXCHANGE_TZ),
        id="swing_scan_and_trade",
        misfire_grace_time=300,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.vol_options_scan_and_trade, "vol_options_scan_morning"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=10, minute=0, timezone=EXCHANGE_TZ),
        id="vol_options_scan_morning",
        misfire_grace_time=300,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.vol_options_scan_and_trade, "vol_options_scan_afternoon"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=13, minute=0, timezone=EXCHANGE_TZ),
        id="vol_options_scan_afternoon",
        misfire_grace_time=300,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.pre_close_orb_exit, "pre_close_orb_exit"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=30, timezone=EXCHANGE_TZ),
        id="pre_close_orb_exit",
        misfire_grace_time=300,  # raised from 60 — a Pi restart at 3:30pm must not skip this
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _heartbeat(runtime.post_market_logging, "post_market_logging"),
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=EXCHANGE_TZ),
        id="post_market_logging",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        runtime.check_manual_trigger,
        trigger=IntervalTrigger(seconds=15),
        id="manual_trigger_watcher",
        max_instances=1,
        coalesce=True,
    )

    logger.info("Scheduler built with jobs: %s", [job.id for job in scheduler.get_jobs()])
    return scheduler
