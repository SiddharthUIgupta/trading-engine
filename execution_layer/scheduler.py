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

from execution_layer.runtime import TradingRuntime

logger = logging.getLogger(__name__)

EXCHANGE_TZ = "America/New_York"


def build_scheduler(runtime: TradingRuntime) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=EXCHANGE_TZ)

    scheduler.add_job(
        runtime.pre_market_scan,
        trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=0, timezone=EXCHANGE_TZ),
        id="pre_market_scan",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        runtime.market_open_execution,
        trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=EXCHANGE_TZ),
        id="market_open_execution",
        misfire_grace_time=120,
    )
    scheduler.add_job(
        runtime.intraday_monitoring,
        trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone=EXCHANGE_TZ),
        id="intraday_monitoring",
        misfire_grace_time=60,
    )
    scheduler.add_job(
        runtime.momentum_scan_and_trade,
        # 30 min, not 15 — each scan costs ~2 OpenBB calls per prerank
        # candidate; this cadence keeps Yahoo Finance call volume well
        # under rate-limit risk. No-ops automatically in static-watchlist
        # mode (DYNAMIC_UNIVERSE_ENABLED=false).
        trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30", timezone=EXCHANGE_TZ),
        id="momentum_scan_and_trade",
        misfire_grace_time=120,
    )
    scheduler.add_job(
        runtime.options_scan_and_trade,
        # Same momentum screen and cadence as the equity momentum track, but
        # offset 15 min from it (":15,:45" vs ":00,:30") so the two jobs'
        # independent OpenBB re-scans don't fire in the same instant. No-ops
        # automatically unless OPTIONS_TRACK_ENABLED=true (defaults false —
        # this is the highest-variance track in the system; opt-in only).
        trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="15,45", timezone=EXCHANGE_TZ),
        id="options_scan_and_trade",
        misfire_grace_time=120,
    )
    scheduler.add_job(
        runtime.thesis_scan_and_trade,
        # Once daily, not intraday — the pullback-from-52-week-high screen
        # uses fields that don't change minute to minute, so rescanning
        # every 30 min like the momentum track would just burn API calls
        # for no new information. No-ops automatically if THESIS_TRACK_ENABLED=false.
        trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=15, timezone=EXCHANGE_TZ),
        id="thesis_scan_and_trade",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        runtime.vol_options_scan_and_trade,
        # Morning entry window: 30 min after open so opening-hour vol
        # spikes (which can artificially inflate IV) have settled.
        # No-ops automatically unless VOL_OPTIONS_TRACK_ENABLED=true.
        trigger=CronTrigger(day_of_week="mon-fri", hour=10, minute=0, timezone=EXCHANGE_TZ),
        id="vol_options_scan_morning",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        runtime.vol_options_scan_and_trade,
        # Afternoon entry window: catches tickers whose IV spiked intraday
        # (competitor earnings, macro events) that weren't eligible at 10 AM.
        # The double-entry guard in vol_options_scan_and_trade prevents
        # re-entering a ticker already opened in the morning window.
        trigger=CronTrigger(day_of_week="mon-fri", hour=13, minute=0, timezone=EXCHANGE_TZ),
        id="vol_options_scan_afternoon",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        runtime.post_market_logging,
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=EXCHANGE_TZ),
        id="post_market_logging",
        misfire_grace_time=300,
    )

    logger.info("Scheduler built with jobs: %s", [job.id for job in scheduler.get_jobs()])
    return scheduler
