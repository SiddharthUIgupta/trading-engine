from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from config.settings import get_settings  # noqa: E402
from execution_layer.broker import AlpacaBroker  # noqa: E402
from execution_layer.state_store import StateStore  # noqa: E402

from config.settings import get_settings  # noqa: E402
from execution_layer.broker import AlpacaBroker  # noqa: E402
from execution_layer.state_store import StateStore  # noqa: E402
from dashboard.custom_ui import render_custom_ui
import json


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

    # Extract state logic from the real store


    # Map real run_history to candidates
    candidates = []
    run_history = store.get_run_history(limit=5)
    for run in run_history:
        payload = run.get("payload", {})
        proposal = payload.get("proposal", {})
        ticker = payload.get("ticker", "UNKNOWN")
        risk_review = payload.get("risk_review", {})
        verdict = risk_review.get("verdict", "REJECTED").upper()

        subs = []
        for sig in payload.get("signals", []):
            subs.append({
                "role": sig.get("agent_name", ""),
                "model": "Claude",
                "stance": sig.get("stance", "NEUTRAL"),
                "conf": 0.5,
                "note": sig.get("rationale", "")
            })

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
            "ro": {"note": " ".join(risk_review.get("reasons", []))}
        })

    if not candidates:
        candidates = [
            { "ticker":'N/A', "name":'No scans yet today', "sector":'', "score":0.0, "dd":0.0, "price":0.0, "status":'PENDING', "size":0, "kelly":0.0, "corr":0.0, "corrVs":'',
              "sub":[],
              "ro":{"note":'Awaiting next scheduled scan.'}
            }
        ]

    equity_pos = []

    positions = store.get_positions()
    for p in [x for x in positions if x["quantity"] > 0]:
        detail = broker.get_position_detail(p["ticker"]) if broker else None
        current = detail["current_price"] if detail else p["avg_entry_price"]
        equity_pos.append({
            "ticker": p["ticker"],
            "strat": p.get("strategy", "Thesis").capitalize(),
            "qty": p["quantity"],
            "entry": float(p["avg_entry_price"]),
            "last": float(current)
        })

    option_pos = []
    option_positions = store.get_option_positions()
    for p in [x for x in option_positions if x["quantity"] > 0]:
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
            "tag": p.get("strategy_version", "LEGACY")
        })

    breakers = [
        {"bucket":'Thesis', "dd":-0.008, "cap":-0.05, "note":'−0.8% today · cap −5.0%'},
        {"bucket":'Swing', "dd":0.0, "cap":-0.05, "note":'flat today · cap −5.0%'},
        {"bucket":'Vol / Options', "dd":-0.012, "cap":-0.05, "note":'−1.2% today · cap −5.0%'},
        {"bucket":'Intraday (ORB)', "dd":0, "cap":-0.05, "note":'track disabled — no allocation', "disabled": True}
    ]

    tracks = [
        {"name":'Thesis Pullback', "state":'ARMED', "on":True},
        {"name":'Pre-Market Gap', "state":'ARMED', "on":True},
        {"name":'Vol / Premium', "state":'ARMED', "on":True},
        {"name":'Swing', "state":'ARMED', "on":True},
        {"name":'ORB Equity', "state":'DISABLED', "on":False},
        {"name":'ORB Options', "state":'DISABLED', "on":False}
    ]

    lessons = [
        {"tag":'oversold-pullback + earnings-beat', "score":1.4, "delta":'3W · 1L', "note":'Enter only when RSI < 45 AND fundamental stance >= 0.6. Skips falling-knife setups.'},
    ]

    hit_rates = [
        {"role":'Fundamental agent', "rate":0.68},
        {"role":'Technical agent', "rate":0.55},
        {"role":'Macro sentiment agent', "rate":0.49}
    ]

    render_custom_ui(
        tradingMode="Paper",
        autonomyMode="Fully autonomous",
        engineArmed=True,
        candidates=candidates,
        equityPos=equity_pos,
        optionPos=option_pos,
        breakers=breakers,
        tracks=tracks,
        lessons=lessons,
        hitRates=hit_rates
    )

if __name__ == "__main__":
    main()
