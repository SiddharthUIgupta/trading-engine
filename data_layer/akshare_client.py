"""AkShare supplementary data client.

Provides two things the main OpenBB + yfinance stack doesn't easily cover:
1. Structured US macro indicators (CPI, PMI, NFP, consumer sentiment)
   — injected into agent consensus prompts as hard numbers, not just news sentiment
2. US market movers from Eastmoney source
   — alternative discovery layer complementing OpenBB's discovery screens

All functions degrade gracefully when akshare is unavailable or a fetch fails.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

try:
    import akshare as ak
    _AK_AVAILABLE = True
except ImportError:
    _AK_AVAILABLE = False
    logger.warning("akshare not installed — AkshareClient will be a no-op")


@dataclass
class MacroSnapshot:
    """Hard US macro economic indicators for injection into agent prompts."""
    cpi_current: float | None = None
    cpi_previous: float | None = None
    pmi_mfg_current: float | None = None
    pmi_mfg_previous: float | None = None
    pmi_svc_current: float | None = None
    pmi_svc_previous: float | None = None
    non_farm_current: float | None = None
    non_farm_previous: float | None = None
    consumer_sentiment: float | None = None
    initial_jobless: float | None = None
    as_of: str = field(default_factory=lambda: date.today().isoformat())

    def is_empty(self) -> bool:
        return all(
            v is None for v in [
                self.cpi_current, self.pmi_mfg_current, self.non_farm_current,
                self.consumer_sentiment, self.initial_jobless,
            ]
        )

    def format_for_prompt(self) -> str:
        """Single-block macro context block for injection into consensus prompts."""
        if self.is_empty():
            return ""
        lines = ["[US MACRO INDICATORS]"]
        if self.cpi_current is not None:
            arrow = "↑" if (self.cpi_previous and self.cpi_current > self.cpi_previous) else "↓"
            lines.append(
                f"  CPI (monthly rate): {self.cpi_current:.1f}% {arrow}"
                + (f" (prev {self.cpi_previous:.1f}%)" if self.cpi_previous else "")
            )
        if self.pmi_mfg_current is not None:
            status = "expanding" if self.pmi_mfg_current >= 50 else "contracting"
            lines.append(
                f"  ISM Mfg PMI: {self.pmi_mfg_current:.1f} ({status})"
                + (f" prev {self.pmi_mfg_previous:.1f}" if self.pmi_mfg_previous else "")
            )
        if self.pmi_svc_current is not None:
            status = "expanding" if self.pmi_svc_current >= 50 else "contracting"
            lines.append(
                f"  ISM Svc PMI: {self.pmi_svc_current:.1f} ({status})"
                + (f" prev {self.pmi_svc_previous:.1f}" if self.pmi_svc_previous else "")
            )
        if self.non_farm_current is not None:
            lines.append(
                f"  Non-Farm Payrolls: {self.non_farm_current:+.0f}K"
                + (f" (prev {self.non_farm_previous:+.0f}K)" if self.non_farm_previous else "")
            )
        if self.consumer_sentiment is not None:
            lines.append(f"  Michigan Consumer Sentiment: {self.consumer_sentiment:.1f}")
        if self.initial_jobless is not None:
            lines.append(f"  Initial Jobless Claims: {self.initial_jobless:.0f}K")
        return "\n".join(lines) + "\n"

    def derive_macro_tone(self) -> str | None:
        """Best-effort macro tone for logging — 'bullish', 'bearish', or 'neutral'."""
        bullish_signals = 0
        bearish_signals = 0
        if self.pmi_mfg_current is not None:
            if self.pmi_mfg_current >= 52:
                bullish_signals += 1
            elif self.pmi_mfg_current < 48:
                bearish_signals += 1
        if self.non_farm_current is not None:
            if self.non_farm_current > 150:
                bullish_signals += 1
            elif self.non_farm_current < 50:
                bearish_signals += 1
        if self.consumer_sentiment is not None:
            if self.consumer_sentiment > 80:
                bullish_signals += 1
            elif self.consumer_sentiment < 60:
                bearish_signals += 1
        if bullish_signals > bearish_signals:
            return "bullish"
        if bearish_signals > bullish_signals:
            return "bearish"
        if bullish_signals == 0 and bearish_signals == 0:
            return None
        return "neutral"


@dataclass
class AkMover:
    symbol: str
    price: float
    change_pct: float
    market_cap: float | None = None
    pe_ratio: float | None = None


def get_macro_snapshot() -> MacroSnapshot:
    """Pull latest US macro indicators from akshare Eastmoney/investing.com feeds."""
    if not _AK_AVAILABLE:
        return MacroSnapshot()

    snap = MacroSnapshot()

    def _safe_pull(fn, current_field: str, previous_field: str) -> None:
        try:
            df = fn()
            df = df.dropna(subset=["今值"])
            if not df.empty:
                row = df.iloc[-1]
                setattr(snap, current_field, float(row["今值"]))
                if row["前值"] and str(row["前值"]).strip() not in ("", "nan"):
                    try:
                        setattr(snap, previous_field, float(row["前值"]))
                    except (ValueError, TypeError):
                        pass
        except Exception as exc:
            logger.debug("akshare pull %s failed: %s", fn.__name__, exc)

    _safe_pull(ak.macro_usa_cpi_monthly, "cpi_current", "cpi_previous")
    _safe_pull(ak.macro_usa_ism_pmi, "pmi_mfg_current", "pmi_mfg_previous")
    _safe_pull(ak.macro_usa_ism_non_pmi, "pmi_svc_current", "pmi_svc_previous")
    _safe_pull(ak.macro_usa_non_farm, "non_farm_current", "non_farm_previous")
    _safe_pull(ak.macro_usa_initial_jobless, "initial_jobless", "_ignore1")

    try:
        df = ak.macro_usa_michigan_consumer_sentiment()
        df = df.dropna(subset=["今值"])
        if not df.empty:
            snap.consumer_sentiment = float(df.iloc[-1]["今值"])
    except Exception as exc:
        logger.debug("akshare consumer sentiment failed: %s", exc)

    return snap


def get_us_movers(min_change_pct: float = 3.0) -> list[AkMover]:
    """Pull notable US stock movers from Eastmoney via akshare.

    Returns stocks with abs(change_pct) >= min_change_pct, sorted by magnitude.
    Complements OpenBB's gainers/losers screens as an independent data source.
    """
    if not _AK_AVAILABLE:
        return []

    try:
        df = ak.stock_us_famous_spot_em()
        movers: list[AkMover] = []
        for _, row in df.iterrows():
            try:
                raw_change = row.get("涨跌幅", 0)
                change_pct = float(raw_change) if raw_change and str(raw_change) not in ("", "nan") else 0.0
                if abs(change_pct) < min_change_pct:
                    continue
                code = str(row.get("代码", ""))
                symbol = code.split(".")[-1] if "." in code else code
                if not symbol or not symbol.isalpha() or len(symbol) > 5:
                    continue
                raw_cap = row.get("总市值", 0)
                raw_pe = row.get("市盈率", 0)
                movers.append(AkMover(
                    symbol=symbol,
                    price=float(row.get("最新价", 0) or 0),
                    change_pct=change_pct,
                    market_cap=float(raw_cap) if raw_cap and str(raw_cap) not in ("", "nan") else None,
                    pe_ratio=float(raw_pe) if raw_pe and str(raw_pe) not in ("", "nan", "0") else None,
                ))
            except (ValueError, TypeError):
                continue
        return sorted(movers, key=lambda m: abs(m.change_pct), reverse=True)
    except Exception as exc:
        logger.debug("akshare US movers fetch failed: %s", exc)
        return []
