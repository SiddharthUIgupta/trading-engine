from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from config.settings import get_settings  # noqa: E402
from execution_layer.broker import AlpacaBroker  # noqa: E402
from execution_layer.state_store import StateStore  # noqa: E402
from dashboard.custom_ui import render_custom_ui  # noqa: E402


st.set_page_config(page_title="trading-engine", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }
    iframe {
        height: 100vh !important;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _settings():
    return get_settings()


def _store() -> StateStore:
    return StateStore(_settings().state_db_path)


def _broker() -> AlpacaBroker | None:
    try:
        return AlpacaBroker.from_settings(_settings())
    except Exception as exc:
        st.warning(f"Couldn't reach Alpaca: {exc}")
        return None


def main():
    settings = _settings()
    store = _store()
    broker = _broker()

    equity = 0.0
    if broker is not None:
        try:
            equity = float(broker.get_equity())
        except Exception:
            pass

    today_str = date.today().isoformat()

    # ── Today's realized P&L ──────────────────────────────────────────────────
    try:
        all_sales = store.get_all_realized_sales(limit=500)
        today_equity_pnl = sum(
            float(s["realized_pnl"] or 0) for s in all_sales
            if s.get("sale_date") == today_str
        )
    except Exception:
        today_equity_pnl = 0.0

    try:
        all_opt_sales = store.get_all_realized_option_sales(limit=200)
        today_opt_pnl = sum(
            float(s["realized_pnl"] or 0) for s in all_opt_sales
            if s.get("sale_date") == today_str
        )
    except Exception:
        today_opt_pnl = 0.0

    # Day-start equity recorded by pre_market_scan
    try:
        ds_events = store.get_events(event_type_like="day_start_equity", limit=5)
        ds_today = [e for e in ds_events if e["created_at"][:10] == today_str]
        base_eq = float(ds_today[0]["detail"]) if ds_today else max(equity, 1.0)
    except Exception:
        base_eq = max(equity, 1.0)

    # ── Breakers ─────────────────────────────────────────────────────────────
    cap = settings.max_daily_drawdown_pct
    global_halted = False
    try:
        global_halted = store.is_breaker_halted("global")
    except Exception:
        pass

    def _note(pnl: float, disabled: bool = False) -> str:
        if disabled:
            return "track disabled — no allocation"
        sign = "+" if pnl >= 0 else "−"
        return f"{sign}${abs(pnl):,.0f} today · cap −{cap:.0%}"

    def _dd(pnl: float) -> float:
        return round(pnl / base_eq, 4) if base_eq else 0.0

    # realized_sales has no strategy column, so equity P&L is reported
    # as a single combined bucket (thesis + swing + recovery all share it).
    breakers = [
        {
            "bucket": "Thesis / Recovery",
            "dd": _dd(today_equity_pnl),
            "cap": -cap,
            "note": _note(today_equity_pnl),
            "halted": global_halted,
        },
        {
            "bucket": "Swing",
            "dd": 0.0,
            "cap": -cap,
            "note": _note(0.0),
        },
        {
            "bucket": "Vol / Options",
            "dd": _dd(today_opt_pnl),
            "cap": -cap,
            "note": _note(today_opt_pnl),
        },
        {
            "bucket": "Intraday (ORB)",
            "dd": 0.0,
            "cap": -cap,
            "note": _note(0, disabled=True),
            "disabled": True,
        },
    ]

    # ── Tracks ───────────────────────────────────────────────────────────────
    try:
        regime_events = store.get_events(event_type_like="daily_regime", limit=5)
        regime = next(
            (e["detail"] for e in regime_events if e["created_at"][:10] == today_str),
            "unknown",
        )
    except Exception:
        regime = "unknown"

    def _track_state(enabled: bool, armed: bool = True) -> str:
        if not enabled:
            return "DISABLED"
        return "ARMED" if armed else "DISARMED"

    # Thesis arms in bull/flat; recovery skips pure bear days
    thesis_on = settings.thesis_track_enabled and regime not in ("bear",)
    recovery_on = settings.recovery_track_enabled and regime not in ("bear",)
    vol_on = settings.vol_options_track_enabled
    swing_on = settings.swing_track_enabled

    tracks = [
        {"name": "Thesis Pullback", "state": _track_state(settings.thesis_track_enabled, thesis_on), "on": thesis_on},
        {"name": "Recovery", "state": _track_state(settings.recovery_track_enabled, recovery_on), "on": recovery_on},
        {"name": "Pre-Market Gap", "state": "ARMED", "on": True},
        {"name": "Vol / Premium", "state": _track_state(settings.vol_options_track_enabled, vol_on), "on": vol_on},
        {"name": "Swing", "state": _track_state(swing_on), "on": swing_on},
        {"name": "ORB Equity", "state": "DISABLED", "on": False},
        {"name": "ORB Options", "state": "DISABLED", "on": False},
    ]

    # ── Lessons ──────────────────────────────────────────────────────────────
    lessons = []
    try:
        for lesson in store.get_lessons(limit=10):
            try:
                tags = json.loads(lesson["setup_tags_json"]) if lesson["setup_tags_json"] else []
            except Exception:
                tags = []
            lessons.append({
                "tag": ", ".join(tags) if tags else (lesson["strategy"] or "general"),
                "score": round(float(lesson["score"] or 1.0), 2),
                "delta": "win" if lesson["outcome_was_win"] else "loss",
                "note": lesson["lesson"],
            })
    except Exception:
        pass
    if not lessons:
        lessons = [{"tag": "none yet", "score": 0.0, "delta": "", "note": "No lessons recorded yet."}]

    # ── Agent hit rates (across all scored runs) ──────────────────────────────
    hit_rates = []
    try:
        agent_stats: dict[str, list[int]] = {}  # name -> [wins, total]
        for log in store.get_scored_signal_logs(limit=1000):
            won = (log.get("outcome_pnl") or 0) > 0
            for sig in log.get("signals", []):
                name = sig.get("agent_name", "unknown")
                if name not in agent_stats:
                    agent_stats[name] = [0, 0]
                agent_stats[name][0] += int(won)
                agent_stats[name][1] += 1
        hit_rates = [
            {"role": name, "rate": round(w / max(t, 1), 2)}
            for name, (w, t) in agent_stats.items()
            if t >= 5
        ]
    except Exception:
        pass
    if not hit_rates:
        hit_rates = [{"role": "Awaiting scored trades", "rate": 0.0}]

    # ── Recent scan candidates ────────────────────────────────────────────────
    candidates = []
    try:
        for run in store.get_run_history(limit=5):
            payload = run.get("payload", {})
            proposal = payload.get("proposal", {})
            ticker = payload.get("ticker", "UNKNOWN")
            risk_review = payload.get("risk_review", {})
            verdict = risk_review.get("verdict", "REJECTED").upper()
            subs = [
                {
                    "role": sig.get("agent_name", ""),
                    "model": "Claude",
                    "stance": sig.get("stance", "NEUTRAL"),
                    "conf": 0.5,
                    "note": sig.get("rationale", ""),
                }
                for sig in payload.get("signals", [])
            ]
            candidates.append({
                "ticker": ticker,
                "name": ticker,
                "sector": "Unknown",
                "score": 5.0,
                "dd": 0.0,
                "price": proposal.get("limit_price", 0.0),
                "status": verdict,
                "size": proposal.get("quantity", 0),
                "kelly": 0.0,
                "corr": 0.0,
                "corrVs": "",
                "sub": subs,
                "ro": {"note": " ".join(risk_review.get("reasons", []))},
            })
    except Exception:
        pass
    if not candidates:
        candidates = [
            {
                "ticker": "N/A", "name": "No scans yet today", "sector": "", "score": 0.0,
                "dd": 0.0, "price": 0.0, "status": "PENDING", "size": 0, "kelly": 0.0,
                "corr": 0.0, "corrVs": "", "sub": [],
                "ro": {"note": "Awaiting next scheduled scan."},
            }
        ]

    # ── Open positions ────────────────────────────────────────────────────────
    equity_pos = []
    try:
        for p in [x for x in store.get_positions() if x["quantity"] > 0]:
            detail = broker.get_position_detail(p["ticker"]) if broker else None
            current = detail["current_price"] if detail else p["avg_entry_price"]
            equity_pos.append({
                "ticker": p["ticker"],
                "strat": p.get("strategy", "thesis").capitalize(),
                "qty": p["quantity"],
                "entry": float(p["avg_entry_price"]),
                "last": float(current),
            })
    except Exception:
        pass

    option_pos = []
    try:
        for p in [x for x in store.get_option_positions() if x["quantity"] > 0]:
            detail = broker.get_position_detail(p["contract_symbol"]) if broker else None
            current = detail["current_price"] if detail else p["avg_entry_price"]
            dte_remaining = (date.fromisoformat(p["expiration"]) - date.today()).days
            option_pos.append({
                "contract": p["contract_symbol"],
                "und": p["underlying_symbol"],
                "exp": p["expiration"],
                "dte": dte_remaining,
                "qty": p["quantity"],
                "entry": float(p["avg_entry_price"]),
                "last": float(current),
                "tag": p.get("strategy_version", "LEGACY"),
            })
    except Exception:
        pass

    render_custom_ui(
        tradingMode="Live" if settings.is_live else "Paper",
        autonomyMode="Fully autonomous",
        engineArmed=not global_halted,
        candidates=candidates,
        equityPos=equity_pos,
        optionPos=option_pos,
        breakers=breakers,
        tracks=tracks,
        lessons=lessons,
        hitRates=hit_rates,
    )


if __name__ == "__main__":
    main()
