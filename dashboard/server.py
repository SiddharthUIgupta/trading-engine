"""FastAPI dashboard server — replaces Streamlit.

Serves a standalone HTML/JS dashboard on port 8502. The frontend polls
these JSON endpoints every 30 seconds; no WebSocket, no build step.

Run:
    .venv/bin/uvicorn dashboard.server:app --host 0.0.0.0 --port 8502 --reload
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from execution_layer.broker import AlpacaBroker
from execution_layer.state_store import StateStore

app = FastAPI(title="trading-engine dashboard", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

_settings = get_settings()
_store = StateStore(_settings.state_db_path)
_static = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(_static)), name="static")


def _broker() -> AlpacaBroker | None:
    try:
        return AlpacaBroker.from_settings(_settings)
    except Exception:
        return None


def _today() -> str:
    return date.today().isoformat()


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(_static / "index.html"))


@app.get("/api/status")
def api_status():
    broker = _broker()
    equity = 0.0
    buying_power = 0.0
    if broker:
        try:
            equity = float(broker.get_equity())
            acct = broker._client.get_account()
            buying_power = float(acct.buying_power)
        except Exception:
            pass

    today = _today()
    global_halted = False
    try:
        global_halted = _store.is_breaker_halted("global")
    except Exception:
        pass

    regime = "unknown"
    try:
        evts = _store.get_events(event_type_like="daily_regime", limit=5)
        regime = next((e["detail"] for e in evts if e["created_at"][:10] == today), "unknown")
    except Exception:
        pass

    vw_examples = 0
    try:
        model_path = Path(_settings.state_db_path).parent / "vw_bandit.model"
        if model_path.exists():
            # The model file doesn't expose example count; read it from the log line at startup
            pass
    except Exception:
        pass

    return {
        "mode": "Live" if _settings.is_live else "Paper",
        "equity": equity,
        "buying_power": buying_power,
        "engine_armed": not global_halted,
        "regime": regime,
        "as_of": datetime.utcnow().isoformat(),
    }


@app.get("/api/positions")
def api_positions():
    broker = _broker()
    equity_pos = []
    try:
        for p in [x for x in _store.get_positions() if x["quantity"] > 0]:
            detail = broker.get_position_detail(p["ticker"]) if broker else None
            current = float(detail["current_price"]) if detail else float(p["avg_entry_price"])
            entry = float(p["avg_entry_price"])
            pnl = (current - entry) * int(p["quantity"])
            pnl_pct = (current / entry - 1) * 100 if entry else 0
            equity_pos.append({
                "ticker": p["ticker"],
                "strategy": (p.get("strategy") or "thesis").capitalize(),
                "qty": int(p["quantity"]),
                "entry": entry,
                "current": current,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "stop": p.get("stop_price"),
                "target": p.get("target_price"),
                "regime": p.get("entry_regime", ""),
            })
    except Exception:
        pass

    option_pos = []
    try:
        for p in [x for x in _store.get_option_positions() if x["quantity"] > 0]:
            detail = broker.get_position_detail(p["contract_symbol"]) if broker else None
            current = float(detail["current_price"]) if detail else float(p["avg_entry_price"])
            entry = float(p["avg_entry_price"])
            pnl = (current - entry) * int(p["quantity"]) * 100  # options are per 100
            dte = (date.fromisoformat(p["expiration"]) - date.today()).days
            option_pos.append({
                "contract": p["contract_symbol"],
                "underlying": p["underlying_symbol"],
                "expiration": p["expiration"],
                "dte": dte,
                "qty": int(p["quantity"]),
                "entry": entry,
                "current": current,
                "pnl": round(pnl, 2),
                "strategy_version": p.get("strategy_version", ""),
            })
    except Exception:
        pass

    return {"equity": equity_pos, "options": option_pos}


@app.get("/api/breakers")
def api_breakers():
    today = _today()
    try:
        all_sales = _store.get_all_realized_sales(limit=500)
        today_pnl = sum(float(s["realized_pnl"] or 0) for s in all_sales if s.get("sale_date") == today)
    except Exception:
        today_pnl = 0.0

    try:
        all_opt = _store.get_all_realized_option_sales(limit=200)
        today_opt_pnl = sum(float(s["realized_pnl"] or 0) for s in all_opt if s.get("sale_date") == today)
    except Exception:
        today_opt_pnl = 0.0

    equity = 0.0
    try:
        b = _broker()
        if b:
            equity = float(b.get_equity())
    except Exception:
        pass

    try:
        ds = _store.get_events(event_type_like="day_start_equity", limit=5)
        ds_today = [e for e in ds if e["created_at"][:10] == today]
        base_eq = float(ds_today[0]["detail"]) if ds_today else max(equity, 1.0)
    except Exception:
        base_eq = max(equity, 1.0)

    global_halted = False
    try:
        global_halted = _store.is_breaker_halted("global")
    except Exception:
        pass

    cap = _settings.max_daily_drawdown_pct

    def dd(pnl):
        return round(pnl / base_eq, 4) if base_eq else 0.0

    return {
        "base_equity": base_eq,
        "cap": cap,
        "breakers": [
            {"bucket": "Thesis / Recovery", "pnl": today_pnl, "dd": dd(today_pnl), "cap": -cap, "halted": global_halted},
            {"bucket": "Swing", "pnl": 0.0, "dd": 0.0, "cap": -cap, "halted": False},
            {"bucket": "Vol / Options", "pnl": today_opt_pnl, "dd": dd(today_opt_pnl), "cap": -cap, "halted": False},
            {"bucket": "ORB Equity", "pnl": 0.0, "dd": 0.0, "cap": -cap, "halted": False, "disabled": True},
        ],
    }


@app.get("/api/tracks")
def api_tracks():
    today = _today()
    regime = "unknown"
    try:
        evts = _store.get_events(event_type_like="daily_regime", limit=5)
        regime = next((e["detail"] for e in evts if e["created_at"][:10] == today), "unknown")
    except Exception:
        pass

    s = _settings
    thesis_on = s.thesis_track_enabled and regime not in ("bear",)
    recovery_on = s.recovery_track_enabled and regime not in ("bear",)

    def state(enabled, armed=True):
        if not enabled:
            return "DISABLED"
        return "ARMED" if armed else "DISARMED"

    return {
        "regime": regime,
        "tracks": [
            {"name": "Thesis Pullback", "state": state(s.thesis_track_enabled, thesis_on), "on": thesis_on},
            {"name": "Recovery", "state": state(s.recovery_track_enabled, recovery_on), "on": recovery_on},
            {"name": "Swing", "state": state(s.swing_track_enabled), "on": s.swing_track_enabled},
            {"name": "Vol / Premium", "state": state(s.vol_options_track_enabled), "on": s.vol_options_track_enabled},
            {"name": "ORB Equity", "state": "DISABLED", "on": False},
            {"name": "ORB Options", "state": "DISABLED", "on": False},
        ],
    }


@app.get("/api/candidates")
def api_candidates():
    candidates = []
    try:
        for run in _store.get_run_history(limit=20):
            p = run.get("payload", {})
            proposal = p.get("proposal") or {}
            risk = p.get("risk_review") or {}
            signals = p.get("signals") or []
            ticker = p.get("ticker", "?")
            candidates.append({
                "ticker": ticker,
                "created_at": run.get("created_at", ""),
                "verdict": risk.get("verdict", "unknown").upper(),
                "action": (proposal.get("action") or "HOLD").upper(),
                "quantity": proposal.get("quantity", 0),
                "limit_price": proposal.get("limit_price", 0),
                "reasons": risk.get("reasons") or [],
                "signals": [
                    {
                        "agent": sig.get("agent_name", ""),
                        "stance": sig.get("stance", ""),
                        "rationale": (sig.get("rationale") or "")[:200],
                    }
                    for sig in signals
                ],
                "is_executable": run.get("is_executable", False),
            })
    except Exception:
        pass
    return {"candidates": candidates}


@app.get("/api/performance")
def api_performance():
    today = _today()

    # Recent closed trades
    trades = []
    try:
        for s in _store.get_all_realized_sales(limit=50):
            trades.append({
                "ticker": s["ticker"],
                "date": s["sale_date"],
                "qty": s["quantity"],
                "price": s["sale_price"],
                "pnl": round(float(s["realized_pnl"] or 0), 2),
            })
    except Exception:
        pass

    # Running totals
    total_pnl = sum(t["pnl"] for t in trades)
    today_pnl = sum(t["pnl"] for t in trades if t["date"] == today)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    win_rate = round(len(wins) / max(len(trades), 1), 3)
    avg_win = round(sum(t["pnl"] for t in wins) / max(len(wins), 1), 2)
    avg_loss = round(sum(t["pnl"] for t in losses) / max(len(losses), 1), 2)
    pf = round(abs(sum(t["pnl"] for t in wins) / min(sum(t["pnl"] for t in losses), -0.01)), 2)

    # Agent hit rates
    hit_rates = []
    try:
        agent_stats: dict[str, list[int]] = {}
        for log in _store.get_scored_signal_logs(limit=500):
            won = (log.get("outcome_pnl") or 0) > 0
            for sig in log.get("signals", []):
                name = sig.get("agent_name", "unknown")
                if name not in agent_stats:
                    agent_stats[name] = [0, 0]
                agent_stats[name][0] += int(won)
                agent_stats[name][1] += 1
        hit_rates = [
            {"agent": name, "wins": w, "total": t, "rate": round(w / max(t, 1), 3)}
            for name, (w, t) in agent_stats.items()
            if t >= 5
        ]
    except Exception:
        pass

    # Lessons
    lessons = []
    try:
        for l in _store.get_lessons(limit=10):
            try:
                tags = json.loads(l["setup_tags_json"]) if l["setup_tags_json"] else []
            except Exception:
                tags = []
            lessons.append({
                "tag": ", ".join(tags) or l.get("strategy", "general"),
                "outcome": "win" if l["outcome_was_win"] else "loss",
                "lesson": l["lesson"],
                "score": round(float(l["score"] or 1.0), 2),
            })
    except Exception:
        pass

    return {
        "summary": {
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
        },
        "trades": trades[:30],
        "hit_rates": hit_rates,
        "lessons": lessons,
    }


@app.get("/api/research")
def api_research():
    """Recent nightly swarm research notes from the obsidian vault."""
    vault = Path.home() / "Projects" / "claude-obsidian" / "wiki" / "research"
    notes = []
    if vault.exists():
        for f in sorted(vault.glob("*.md"), reverse=True)[:10]:
            try:
                text = f.read_text()
                lines = text.split("\n")
                title = lines[0].lstrip("# ").strip() if lines else f.stem
                # Extract PM verdict block
                pm_start = next((i for i, l in enumerate(lines) if "## PM Verdict" in l), None)
                verdict_text = ""
                if pm_start is not None:
                    verdict_text = "\n".join(lines[pm_start + 1: pm_start + 6]).strip()
                notes.append({
                    "file": f.stem,
                    "ticker": f.stem.split("-")[0] if "-" in f.stem else f.stem,
                    "date": "-".join(f.stem.split("-")[1:4]) if "-" in f.stem else "",
                    "title": title,
                    "verdict": verdict_text[:300],
                })
            except Exception:
                pass
    return {"notes": notes}
