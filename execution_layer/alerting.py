"""Alerting for critical trading engine events.

Sends to Telegram (primary) and Gmail SMTP (optional secondary).
Never raises — a broken alert must never crash the trading process.

Telegram env vars (required for alerts to fire): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Gmail env vars (optional): GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_EMAIL_TO
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

_GMAIL_USER = os.getenv("GMAIL_USER", "")
_GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
_TO = os.getenv("ALERT_EMAIL_TO") or _GMAIL_USER
_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_HEALTHCHECKS_URL = os.getenv("HEALTHCHECKS_IO_URL", "")
_HEARTBEAT_FILE = Path(os.getenv("HEARTBEAT_FILE", "state/heartbeat.json"))


def _send(subject: str, html: str) -> None:
    if not _GMAIL_USER or not _GMAIL_PASSWORD:
        logger.debug("Email alert skipped — GMAIL_USER/GMAIL_APP_PASSWORD not configured")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = _GMAIL_USER
        msg["To"] = _TO
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(_GMAIL_USER, _GMAIL_PASSWORD)
            server.sendmail(_GMAIL_USER, _TO, msg.as_string())
        logger.info("Email alert sent: %s", subject)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Email alert delivery failed (%s): %s", subject, exc)


def _send_telegram(text: str) -> None:
    if not _TELEGRAM_TOKEN or not _TELEGRAM_CHAT_ID:
        logger.debug("Telegram alert skipped — TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured")
        return
    try:
        payload = json.dumps({"chat_id": _TELEGRAM_CHAT_ID, "text": text}).encode()
        url = f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
        logger.info("Telegram alert sent: %s", text[:60])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram alert delivery failed: %s", exc)


def _broadcast(subject: str, html: str, brief: str | None = None) -> None:
    """Send to all configured transports. brief is the Telegram plain-text message;
    defaults to subject when not provided."""
    _send(subject, html)
    _send_telegram(brief if brief is not None else subject)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _table(*rows: tuple[str, str]) -> str:
    cells = "".join(
        f"<tr><td style='padding:6px 12px;border-bottom:1px solid #eee;color:#555'><b>{k}</b></td>"
        f"<td style='padding:6px 12px;border-bottom:1px solid #eee'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table style='border-collapse:collapse;font-family:monospace;font-size:14px'>{cells}</table>"


def alert_buy(
    ticker: str,
    shares: int,
    price: float,
    strategy: str,
    order_id: str = "",
    equity: float = 0.0,
) -> None:
    total = shares * price
    subject = f"🟢 BUY EXECUTED: {ticker} — {shares} shares @ ${price:.2f}"
    html = f"""
    <h2 style='color:#16a34a;margin-bottom:4px'>✅ BUY ORDER EXECUTED</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Ticker", f"<b>{ticker}</b>"),
        ("Strategy", strategy),
        ("Shares", str(shares)),
        ("Price", f"${price:.2f}"),
        ("Total Cost", f"${total:,.2f}"),
        ("Account Equity", f"${equity:,.2f}" if equity else "—"),
        ("Order ID", order_id or "—"),
    )}
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"BUY {ticker}: {shares} shares @ ${price:.2f} (${total:,.0f}) [{strategy}]")


def alert_option_buy(
    contract_symbol: str,
    underlying: str,
    contracts: int,
    premium: float,
    strategy: str,
    equity: float = 0.0,
) -> None:
    total = contracts * premium * 100
    subject = f"🟢 OPTIONS BUY: {contract_symbol} — {contracts}x @ ${premium:.2f}"
    html = f"""
    <h2 style='color:#16a34a;margin-bottom:4px'>✅ OPTIONS BUY EXECUTED</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Contract", f"<b>{contract_symbol}</b>"),
        ("Underlying", underlying),
        ("Strategy", strategy),
        ("Contracts", str(contracts)),
        ("Premium/contract", f"${premium:.2f}"),
        ("Total Cost", f"${total:,.2f}"),
        ("Account Equity", f"${equity:,.2f}" if equity else "—"),
    )}
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"OPTIONS BUY {contract_symbol} ({underlying}): {contracts}x @ ${premium:.2f} (${total:,.0f}) [{strategy}]")


def alert_circuit_breaker(reason: str, equity: float = 0.0) -> None:
    subject = f"🔴 CIRCUIT BREAKER TRIPPED — {reason[:60]}"
    html = f"""
    <h2 style='color:#dc2626;margin-bottom:4px'>🚨 CIRCUIT BREAKER TRIPPED</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Reason", reason),
        ("Equity at trip", f"${equity:,.2f}" if equity else "—"),
    )}
    <p>All new stock trades are halted for the rest of the trading day.</p>
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"CIRCUIT BREAKER: {reason[:120]}")


def alert_profit_locked(equity: float, gain: float) -> None:
    subject = f"🔒 PROFIT LOCKED — +${gain:,.2f} today"
    html = f"""
    <h2 style='color:#2563eb;margin-bottom:4px'>🔒 DAILY PROFIT TARGET HIT</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Today's gain", f"<b>+${gain:,.2f}</b>"),
        ("Current equity", f"${equity:,.2f}"),
    )}
    <p>No new entries for the rest of the day. Existing positions remain open.</p>
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"PROFIT LOCKED: +${gain:,.2f} today | equity ${equity:,.2f}")


def alert_daily_summary(equity: float, realized_pnl: float, open_positions: int) -> None:
    color = "#16a34a" if realized_pnl >= 0 else "#dc2626"
    sign = "+" if realized_pnl >= 0 else ""
    subject = f"📊 Daily Summary — {sign}${realized_pnl:,.2f} | Equity ${equity:,.2f}"
    html = f"""
    <h2 style='color:{color};margin-bottom:4px'>📊 DAILY TRADING SUMMARY</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Realized P&L today", f"<b style='color:{color}'>{sign}${realized_pnl:,.2f}</b>"),
        ("Account equity", f"${equity:,.2f}"),
        ("Open positions", str(open_positions)),
    )}
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"DAILY: {sign}${realized_pnl:,.2f} | equity ${equity:,.2f} | {open_positions} open")


def alert_signal_uplift_summary(lines: list[str]) -> None:
    """Weekly shadow-signal report — piggybacks on the Friday post_market_logging
    tick rather than its own APScheduler job. `lines` is one row per
    (signal_name, signal_version, metric_name) from scripts/signal_uplift.py,
    e.g. "kronos_small/kronos-small-v1/p_touch_win (n=412): IC=+0.041 -> PROMOTE-CANDIDATE".
    """
    subject = f"📈 Weekly Signal Uplift Report — {len(lines)} signal(s)"
    rows_html = "".join(f"<li style='margin-bottom:4px'>{line}</li>" for line in lines) or "<li>No signal data yet.</li>"
    html = f"""
    <h2 style='color:#2563eb;margin-bottom:4px'>📈 WEEKLY SIGNAL UPLIFT REPORT</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    <ul style='font-family:monospace;font-size:13px'>{rows_html}</ul>
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Shadow signals only — not gating any trade decision. Promotion requires a separate explicit task.<br>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    brief_lines = "\n".join(lines[:5]) or "No signal data yet."
    _broadcast(subject, html, f"SIGNAL REPORT ({len(lines)} signals):\n{brief_lines}")


def alert_startup(equity: float, env: str) -> None:
    subject = f"🚀 Engine Started — {env.upper()} | Equity ${equity:,.2f}"
    html = f"""
    <h2 style='color:#7c3aed;margin-bottom:4px'>🚀 TRADING ENGINE STARTED</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Environment", f"<b>{env.upper()}</b>"),
        ("Account equity", f"${equity:,.2f}"),
    )}
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"ENGINE STARTED: {env.upper()} | equity ${equity:,.2f}")


def alert_crash(exc: str) -> None:
    subject = "💥 ENGINE CRASH — unhandled exception"
    html = f"""
    <h2 style='color:#dc2626;margin-bottom:4px'>💥 TRADING ENGINE CRASHED</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(("Exception", f"<pre style='color:red'>{exc[:500]}</pre>"))}
    <p>The process has exited. Restart it manually or check systemd.</p>
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"ENGINE CRASHED: {exc[:200]}")


def alert_zero_buy_streak(strategy: str, streak: int) -> None:
    subject = f"⚠️ SILENT ENGINE — {strategy} has placed 0 BUYs for {streak} sessions"
    html = f"""
    <h2 style='color:#d97706;margin-bottom:4px'>⚠️ ZERO-BUY STREAK DETECTED</h2>
    <p style='color:#666;margin-top:0'>{_now()}</p>
    {_table(
        ("Strategy", strategy),
        ("Consecutive sessions with 0 BUYs", str(streak)),
    )}
    <p>The scan ran but placed no buy orders for {streak} consecutive sessions.
    This may indicate a silent failure, regime disarm, or the screen finding nothing —
    but ≥3 sessions in a row warrants manual inspection.</p>
    <p style='color:#999;font-size:12px;margin-top:16px'>
      Trading Engine · Raspberry Pi · SiddharthUIgupta/trading-engine
    </p>
    """
    _broadcast(subject, html, f"SILENT ENGINE: {strategy} — 0 buys for {streak} sessions")


def ping_heartbeat(job_name: str) -> None:
    """Record that `job_name` completed. Two side effects:
    1. Writes a timestamp entry to the local heartbeat JSON file.
    2. If HEALTHCHECKS_IO_URL is set, pings healthchecks.io (free tier supports
       one check — useful for the most critical job; set the env var to a
       healthchecks.io ping URL and point it at thesis_scan_and_trade).
    Never raises — a broken heartbeat must never crash the trading process.
    """
    try:
        _HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(_HEARTBEAT_FILE.read_text()) if _HEARTBEAT_FILE.exists() else {}
        except Exception:
            existing = {}
        existing[job_name] = datetime.now(timezone.utc).isoformat()
        _HEARTBEAT_FILE.write_text(json.dumps(existing, indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Heartbeat file write failed for %s: %s", job_name, exc)

    if _HEALTHCHECKS_URL:
        try:
            url = _HEALTHCHECKS_URL.rstrip("/") + f"/{job_name}" if "/" not in _HEALTHCHECKS_URL.split("//")[-1].split("?")[0].split("/")[-1] else _HEALTHCHECKS_URL
            urllib.request.urlopen(url, timeout=5)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
            logger.warning("Healthchecks.io ping failed for %s: %s", job_name, exc)
