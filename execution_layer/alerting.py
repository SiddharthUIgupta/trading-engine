"""Email alerting for critical trading engine events.

Sends HTML emails via Gmail SMTP for: trade executions, circuit breaker
trips, profit locks, daily summaries, and crashes. Never raises — a broken
alert must never crash the trading process.

Required env vars: GMAIL_USER, GMAIL_APP_PASSWORD
Optional: ALERT_EMAIL_TO (defaults to GMAIL_USER)
"""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

_GMAIL_USER = os.getenv("GMAIL_USER", "")
_GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
_TO = os.getenv("ALERT_EMAIL_TO") or _GMAIL_USER


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
        logger.info("Alert sent: %s", subject)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alert delivery failed (%s): %s", subject, exc)


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
    _send(subject, html)


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
    _send(subject, html)


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
    _send(subject, html)


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
    _send(subject, html)


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
    _send(subject, html)


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
    _send(subject, html)


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
    _send(subject, html)
