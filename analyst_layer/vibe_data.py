"""Vibe-Trading data enrichment bridge.

Calls Vibe-Trading's SEC EDGAR tools (SecFilingsTool,
FinancialStatementsTool) and our own prefilter Alpha158 functions to produce
richer context strings that are injected into the agent prompts before the
consensus run. Both functions degrade gracefully to "" if Vibe-Trading is not
installed or any fetch fails — the agents still run on whatever OpenBB data
they already have.

Import convention: Vibe-Trading tools are lazy-loaded (added to sys.path only
on first call) so this module can be imported without the Vibe-Trading package
installed on developer machines.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

from analyst_layer.prefilter import (
    _amihud_illiquidity,
    _kbar,
    _range_position,
    _return_volume_corr,
    _rsquared,
    _rsi,
    _volume_pressure,
)
from data_layer.models import PriceSeries

_log = logging.getLogger(__name__)
_VIBE_AGENT = Path.home() / "Projects" / "Vibe-Trading" / "agent"

_sec_tool = None
_fin_tool = None
_vibe_loaded = False


def _load_vibe() -> bool:
    global _sec_tool, _fin_tool, _vibe_loaded
    if _vibe_loaded:
        return _sec_tool is not None
    _vibe_loaded = True
    if not _VIBE_AGENT.exists():
        return False
    if str(_VIBE_AGENT) not in sys.path:
        sys.path.insert(0, str(_VIBE_AGENT))
    try:
        from src.tools.sec_filings_tool import SecFilingsTool
        from src.tools.financial_statements_tool import FinancialStatementsTool
        _sec_tool = SecFilingsTool()
        _fin_tool = FinancialStatementsTool()
        return True
    except Exception as exc:
        _log.debug("Vibe-Trading tools unavailable: %s", exc)
        return False


def fetch_sec_context(ticker: str) -> str:
    """Pull recent SEC filings + annual income statement for `ticker`.

    Returns a markdown-formatted string ready to append to the fundamental
    agent's prompt. Returns "" if Vibe-Trading is unavailable or the fetch
    fails — the agent still runs without it.
    """
    if not _load_vibe():
        return ""
    try:
        parts: list[str] = [f"## SEC EDGAR data for {ticker}"]

        # Recent filings (10-K, 10-Q, 8-K)
        try:
            raw = _sec_tool.execute(ticker=ticker, limit=5)
            filings = raw if isinstance(raw, list) else (raw.get("filings") or raw.get("results") or [])
            if filings:
                parts.append("\n### Recent SEC filings")
                for f in filings[:5]:
                    form = f.get("form") or f.get("filing_type") or "?"
                    filed = f.get("filed") or f.get("filed_on") or f.get("date") or "?"
                    desc = f.get("description") or f.get("summary") or ""
                    parts.append(f"- {form} filed {filed}: {desc[:200]}")
        except Exception as exc:
            _log.debug("SecFilingsTool failed for %s: %s", ticker, exc)

        # Annual income statement (revenue, net income)
        try:
            raw = _fin_tool.execute(code=f"{ticker}.US", statement="income", period="annual")
            rows = raw if isinstance(raw, list) else (raw.get("data") or raw.get("results") or [])
            if rows:
                parts.append("\n### Annual income statement (most recent 3 years)")
                for row in rows[:3]:
                    year = row.get("date") or row.get("period") or row.get("fiscal_date") or "?"
                    rev = row.get("revenue") or row.get("totalRevenue") or None
                    net = row.get("netIncome") or row.get("net_income") or None
                    rev_str = f"${rev/1e9:.2f}B" if rev and abs(rev) >= 1e9 else (f"${rev/1e6:.1f}M" if rev else "N/A")
                    net_str = f"${net/1e9:.2f}B" if net and abs(net) >= 1e9 else (f"${net/1e6:.1f}M" if net else "N/A")
                    parts.append(f"- {year}: Revenue {rev_str}, Net income {net_str}")
        except Exception as exc:
            _log.debug("FinancialStatementsTool failed for %s: %s", ticker, exc)

        if len(parts) == 1:
            return ""
        return "\n".join(parts) + "\n"
    except Exception as exc:
        _log.debug("fetch_sec_context failed for %s: %s", ticker, exc)
        return ""


def compute_technical_signals(price_series: PriceSeries) -> str:
    """Compute Alpha158 signals from the price series and format them for the
    technical agent's prompt. Supplements the agent's existing SMA/vol output
    with trend quality (R²), range position, candlestick body metrics, volume
    pressure, return-volume correlation, and Amihud illiquidity.

    Returns "" if the series is too short for meaningful computation (< 15 bars).
    """
    bars = price_series.bars
    closes = [b.close for b in bars]
    if len(bars) < 15:
        return ""

    parts: list[str] = ["\n## Alpha158 technical signals"]

    # Trend quality: R² over last 20 bars
    rsq = _rsquared(closes, window=min(20, len(closes)))
    if rsq is not None:
        r2, slope = rsq
        direction = "upward" if slope > 0 else "downward"
        parts.append(f"- Trend R² ({direction}): {r2:.3f}  "
                     f"[≥0.80 = clean linear trend; <0.40 = choppy]")

    # Range position: where price sits in the 30-day high/low range
    rp = _range_position(bars, window=min(30, len(bars)))
    if rp is not None:
        label = "at low end (potential support)" if rp <= 0.15 else ("at high end (extended)" if rp >= 0.85 else "mid-range")
        parts.append(f"- 30-day range position: {rp:.0%}  [{label}]")

    # RSI
    rsi_val = _rsi(closes, period=min(14, len(closes) - 1))
    if rsi_val is not None:
        label = "oversold" if rsi_val <= 30 else ("overbought" if rsi_val >= 70 else "neutral")
        parts.append(f"- RSI(14): {rsi_val:.1f}  [{label}]")

    # Candlestick body (most recent bar)
    kb = _kbar(bars)
    if kb is not None:
        kbar, ksft = kb
        body_label = "strong bull body" if kbar > 0.7 else ("strong bear body" if kbar < -0.7 else "indecisive candle")
        close_label = "closed upper half (bullish)" if ksft > 0 else "closed lower half (bearish)"
        parts.append(f"- Last bar KBAR: {kbar:+.3f}  KSFT: {ksft:+.3f}  [{body_label}; {close_label}]")

    # Volume pressure over last 10 bars
    vp = _volume_pressure(bars, window=min(10, len(bars) - 1))
    if vp is not None:
        vsump, vsumn = vp
        vol_label = "buyers dominating" if vsump > 0.65 else ("sellers dominating" if vsumn > 0.65 else "balanced")
        parts.append(f"- Volume pressure: up={vsump:.0%} down={vsumn:.0%}  [{vol_label}]")

    # Return-volume correlation
    rvc = _return_volume_corr(bars, window=min(20, len(bars) - 1))
    if rvc is not None:
        rvc_label = "volume confirms trend" if rvc > 0.30 else ("distribution (selling into strength)" if rvc < -0.30 else "no clear volume bias")
        parts.append(f"- Return-volume corr: {rvc:+.3f}  [{rvc_label}]")

    # Amihud illiquidity
    illiq = _amihud_illiquidity(bars, window=min(20, len(bars) - 1))
    if illiq is not None:
        liq_label = "execution risk: illiquid" if illiq > 1.0 else "liquid"
        parts.append(f"- Amihud illiquidity: {illiq:.4f}  [{liq_label}]")

    if len(parts) == 1:
        return ""
    return "\n".join(parts) + "\n"
