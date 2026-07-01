"""Read-only visualization over the trading engine's state — never writes
anything. Reads directly from StateStore's SQLite file (safe to read while
main.py's scheduler is writing to it concurrently) plus live Alpaca account
state. Run with: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from execution_layer.manual_trigger import VALID_SCANS, write_trigger  # noqa: E402
from analyst_layer.agents.fundamental_agent import FundamentalAgent  # noqa: E402
from analyst_layer.agents.general_analyst_agent import GeneralAnalystAgent  # noqa: E402
from analyst_layer.agents.macro_sentiment_agent import MacroSentimentAgent  # noqa: E402
from analyst_layer.agents.technical_agent import TechnicalAgent  # noqa: E402
from analyst_layer.performance import compute_metrics  # noqa: E402
from analyst_layer.pricing import estimate_cost_usd  # noqa: E402
from analyst_layer.schemas import Action  # noqa: E402
from config.settings import get_settings  # noqa: E402
from data_layer.exceptions import DataLayerError  # noqa: E402
from data_layer.occ_symbol import parse_occ_symbol  # noqa: E402
from data_layer.openbb_client import OpenBBDataClient  # noqa: E402
from execution_layer.broker import AlpacaBroker  # noqa: E402
from execution_layer.state_store import StateStore  # noqa: E402

st.set_page_config(page_title="trading-engine", layout="wide")


@st.cache_resource
def _settings():
    return get_settings()


def _store() -> StateStore:
    return StateStore(_settings().state_db_path)


def _broker() -> AlpacaBroker | None:
    try:
        return AlpacaBroker.from_settings(_settings())
    except Exception as exc:  # noqa: BLE001 — dashboard must degrade, never crash, on a broker hiccup
        st.warning(f"Couldn't reach Alpaca: {exc}")
        return None


def _md(text: str) -> str:
    """Escapes literal `$` before handing free text (agent rationale,
    risk-review reasons — frequently full of dollar amounts) to
    st.markdown/st.caption. Streamlit's renderer treats a pair of `$`
    as inline LaTeX math, which silently swallows every space between
    them — turning "EPS -$0.62, suggesting..." into a wall of run-together
    text up to the next `$` in the paragraph.
    """
    return text.replace("$", "\\$")


_ORDER_STATUS_EXPLANATIONS = {
    "new": "Submitted to Alpaca, not yet routed to an exchange.",
    "accepted": "Accepted and working at the exchange — not yet filled. Common to sit here a while after hours or for less-liquid options.",
    "pending_new": "Still being processed by Alpaca before it's sent to the exchange.",
    "accepted_for_bidding": "Accepted and eligible to be matched against incoming orders — not yet filled.",
    "partially_filled": "Some of the order has filled; the remainder is still working.",
    "filled": "Completely filled.",
    "canceled": "Canceled before it filled — no shares/contracts were bought or sold.",
    "expired": "The order's time-in-force ran out (e.g. a DAY order at market close) without filling.",
    "rejected": "Rejected by the broker or exchange — never filled.",
    "pending_cancel": "A cancel request is in progress.",
    "pending_replace": "A replace (modify) request is in progress.",
}


def _run_adhoc_analysis(ticker: str, settings, store: StateStore):
    """General investment read on a single user-chosen ticker — fundamentals,
    sentiment, and technicals, each independently, then synthesized by a
    4th general-analyst call that weighs the strength and freshness of each
    read against the others (not a vote count — see GeneralAnalystAgent).
    Intentionally NOT the same mechanism the scanners use: no Risk Officer,
    no position sizing, no order ticket. That machinery exists to run the
    autonomous harness toward today's profit target; this is a general
    opinion, the same regardless of equity, drawdown, or what day it is.
    """
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    data_client = OpenBBDataClient(pat=settings.openbb_pat or None)

    today = datetime.utcnow().date()
    sentiment = data_client.get_sentiment(ticker)
    fundamentals = data_client.get_fundamentals(ticker)
    filings = data_client.get_recent_filings(ticker)
    price_series = data_client.get_price_history(ticker, start_date=today - timedelta(days=60), end_date=today)

    usage_log = []

    def _usage_callback(agent_name, model, usage):
        usage_log.append((agent_name, model, usage))

    subagent_model = settings.anthropic_subagent_model
    macro_agent = MacroSentimentAgent(anthropic_client, subagent_model, usage_callback=_usage_callback)
    fundamental_agent = FundamentalAgent(anthropic_client, subagent_model, usage_callback=_usage_callback)
    technical_agent = TechnicalAgent(anthropic_client, subagent_model, usage_callback=_usage_callback)
    # The synthesis itself is the consequential judgment call here — same
    # reasoning as why the Risk Officer gets the more capable model tier.
    general_agent = GeneralAnalystAgent(anthropic_client, settings.anthropic_model, usage_callback=_usage_callback)

    signals = [
        macro_agent.analyze(ticker, sentiment),
        fundamental_agent.analyze(ticker, fundamentals, filings),
        technical_agent.analyze(ticker, price_series),
    ]
    overall = general_agent.synthesize(ticker, signals)

    for agent_name, agent_model, usage in usage_log:
        cost = estimate_cost_usd(agent_model, usage)
        store.record_token_usage(
            agent_name=agent_name, model=agent_model, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0, estimated_cost_usd=cost,
        )
    store.record_event(
        event_type="adhoc_analysis",
        detail=f"{ticker}: overall lean {overall.stance.value} ({overall.confidence.value} confidence)",
    )
    return signals, overall


st.title("trading-engine")

# Floating refresh button — window.location.reload() is the only reliable
# cross-Streamlit-version approach. Equivalent to browser F5: re-runs the
# script and fetches fresh broker + DB data, which is what a dashboard wants.
st.markdown(
    """
    <style>
    #__floating_refresh__ {
        position: fixed;
        bottom: 1.75rem;
        right: 1.75rem;
        z-index: 99999;
        background: #f97316;
        color: white;
        border: none;
        border-radius: 2rem;
        padding: 0.65rem 1.35rem;
        font-size: 0.95rem;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.45);
        transition: background 0.15s ease, transform 0.1s ease;
    }
    #__floating_refresh__:hover {
        background: #ea6c0a;
        transform: scale(1.04);
    }
    </style>
    <button id="__floating_refresh__" onclick="window.location.reload()">↻ Refresh</button>
    """,
    unsafe_allow_html=True,
)

settings = _settings()
store = _store()
broker = _broker()

# ---- Overview ----
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)

equity = None
if broker is not None:
    try:
        equity = broker.get_equity()
        clock = broker._client.get_clock()
        col1.metric("Equity", f"${equity:,.2f}")
        col2.metric("Market", "OPEN" if clock.is_open else "CLOSED")
        col3.metric("Next open" if not clock.is_open else "Closes", str(clock.next_open if not clock.is_open else clock.next_close))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't fetch account/clock: {exc}")

cost_today = store.get_cost_summary(since=datetime.utcnow().date())
col4.metric("Claude spend today", f"${cost_today['total_cost_usd']:.4f}", f"{cost_today['total_calls']} calls")
st.caption("Claude spend is real money billed by Anthropic — entirely separate from the paper-trading equity below.")

# Today's actual starting equity, not the account's all-time opening balance —
# that distinction matters past day one (see runtime.py::pre_market_scan).
day_start_events = store.get_events(event_type_like="day_start_equity", limit=1)
if equity is not None and day_start_events:
    day_start_equity = float(day_start_events[0]["detail"])
    day_pnl = equity - day_start_equity
    progress = max(0.0, min(1.0, day_pnl / settings.daily_profit_target_usd)) if settings.daily_profit_target_usd else 0.0
    st.progress(progress, text=f"Today's P&L: ${day_pnl:+,.2f} / ${settings.daily_profit_target_usd:.2f} target (started today at ${day_start_equity:,.2f})")
elif equity is not None:
    st.caption("Today's P&L unavailable — pre-market scan hasn't recorded a starting equity yet today.")

st.divider()

# ---- Tabs ----
tab_analyze, tab_controls, tab_positions, tab_trades, tab_performance, tab_decisions, tab_scans, tab_cost, tab_events = st.tabs(
    ["Analyze Ticker", "Controls", "Positions", "Trade History", "Performance", "Agent Decisions", "Scanner Activity", "Cost Tracking", "Events Log"]
)

# ---- Analyze Ticker (general read — no position sizing, no order) ----
with tab_analyze:
    st.caption(
        "A general investment read on any ticker you choose — independent fundamentals, sentiment, and "
        "technical signals, synthesized by a 4th general-analyst call that weighs the strength and "
        "freshness of each read, not a vote count. This deliberately skips the Risk Officer/position-sizing "
        "mechanism the scanners use: that machinery exists to size trades for the autonomous harness chasing "
        "today's profit target, not to answer 'is this a good buy' in general. Costs real Claude API money "
        "per click (~$0.01-0.03, 4 calls)."
    )
    ticker_input = st.text_input("Ticker symbol", placeholder="e.g. RDW").strip().upper()
    if st.button("Analyze", disabled=not ticker_input):
        with st.spinner(f"Analyzing {ticker_input}..."):
            try:
                signals, overall = _run_adhoc_analysis(ticker_input, settings, store)
            except DataLayerError as exc:
                st.error(f"Data fetch failed for {ticker_input}: {exc}")
            except Exception as exc:  # noqa: BLE001 — show the user the error, don't crash the page
                st.error(f"Analysis failed: {exc}")
            else:
                badge = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}.get(overall.stance, "⚪")
                st.subheader(f"{badge} Overall: {overall.stance.value} ({overall.confidence.value} confidence)")
                st.caption(_md(overall.rationale))
                st.caption("This is a general opinion, not a sized trade — no position, no order, no equity-relative limits.")
                st.markdown("**Independent specialist reads:**")
                for signal in signals:
                    st.markdown(f"**{signal.agent_name}** ({signal.confidence.value}): {signal.stance.value}")
                    st.caption(_md(signal.rationale))

# ---- Controls ----
with tab_controls:
    st.subheader("Manual Scan Triggers")
    st.caption(
        "Queues a scan on the live engine — it runs within ~15 seconds. "
        "Scans respect all circuit breakers and regime guards exactly as if they "
        "fired on schedule. The engine log will show `=== MANUAL TRIGGER ===`."
    )

    _SCAN_META = {
        "thesis": ("Thesis Scan", "8:15 AM scan — finds stocks pulled back ≥20% from 52w high. Runs full 4-agent consensus on top candidates."),
        "gap": ("Gap Scan", "9:05 AM scan — finds stocks gapping ≥5% pre-market (like META today). Queues approved names for immediate execution."),
        "swing": ("Swing Scan", "9:45 AM scan — finds stocks in uptrend (SMA20>SMA50) at a pullback entry. Multi-week hold."),
        "momentum": ("Momentum Scan", "Runs every 30 min intraday — looks for volume-spike + SMA breakout momentum signals."),
        "options": ("Options Scan", "Runs every 30 min — looks for directional ORB breakouts to trade with defined-risk calls/puts."),
    }

    cols = st.columns(len(_SCAN_META))
    for col, (scan_key, (label, description)) in zip(cols, _SCAN_META.items()):
        with col:
            st.markdown(f"**{label}**")
            st.caption(description)
            if st.button(f"Run {label}", key=f"trigger_{scan_key}"):
                try:
                    write_trigger(scan_key)
                    st.success("Queued — engine will pick it up within 15 seconds.")
                except Exception as exc:
                    st.error(f"Failed to write trigger: {exc}")

    st.divider()
    st.subheader("Live Scan Output")
    st.caption("Today's scan activity from the engine log — candidates found, consensus verdicts, orders placed.")

    _LOG_KEYWORDS = (
        "MANUAL TRIGGER", "THESIS SCAN", "GAP SCAN", "SWING SCAN", "MOMENTUM SCAN",
        "cleared screening", "thesis scan PASSED", "shrink-volume confirmed",
        "BUY", "SELL", "HOLD", "verdict", "approved", "rejected", "amended",
        "kelly", "circuit breaker", "halted", "blocked", "gap_pct", "Gap scanner",
        "consensus", "REGIME", "DAILY REGIME",
    )

    try:
        log_path = Path(__file__).resolve().parent.parent / "logs" / "trading_engine.log"
        today_str = date.today().isoformat()
        matching_lines = []
        with open(log_path) as f:
            for line in f:
                if not line.startswith(today_str):
                    continue
                if any(kw in line for kw in _LOG_KEYWORDS):
                    matching_lines.append(line.rstrip())
        if matching_lines:
            st.code("\n".join(matching_lines[-200:]), language=None)
        else:
            st.info("No scan activity logged today yet.")
    except Exception as exc:
        st.warning(f"Could not read log: {exc}")

# ---- Positions ----
with tab_positions:
    st.subheader("Pending Orders")
    st.caption(
        "Submitted but not yet filled or cancelled — a position only appears below once it actually "
        "fills, so a slow-to-fill order (common with options on the paper account) would otherwise be "
        "invisible anywhere except the raw Events Log."
    )
    pending_orders = broker.get_open_orders() if broker else []
    if not pending_orders:
        st.info("No pending orders.")
    else:
        rows = []
        for o in pending_orders:
            sym = o["symbol"]
            status_note = _ORDER_STATUS_EXPLANATIONS.get(o["status"], "Unrecognized status — check the broker directly.")
            if o.get("legs"):
                # mleg order: expand into one display row per leg
                for leg in o["legs"]:
                    leg_sym = leg["symbol"]
                    parsed = parse_occ_symbol(leg_sym) if leg_sym else None
                    rows.append({
                        "Underlying": parsed.underlying_symbol if parsed else leg_sym,
                        "Type": parsed.option_type if parsed else "option",
                        "Strike": parsed.strike if parsed else None,
                        "Expiration": parsed.expiration.isoformat() if parsed else None,
                        "Side": leg["position_intent"],
                        "Qty": o["qty"],
                        "Limit (net)": o["limit_price"],
                        "Status": o["status"],
                        "What this means": status_note,
                        "Submitted": o["submitted_at"],
                    })
            else:
                parsed = parse_occ_symbol(sym) if sym else None
                rows.append({
                    "Underlying": parsed.underlying_symbol if parsed else (sym or "—"),
                    "Type": parsed.option_type if parsed else "stock",
                    "Strike": parsed.strike if parsed else None,
                    "Expiration": parsed.expiration.isoformat() if parsed else None,
                    "Side": o["side"],
                    "Qty": o["qty"],
                    "Limit (net)": o["limit_price"],
                    "Status": o["status"],
                    "What this means": status_note,
                    "Submitted": o["submitted_at"],
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Equities")
    positions = store.get_positions()
    open_positions = [p for p in positions if p["quantity"] > 0]
    if not open_positions:
        st.info("No open equity positions.")
    else:
        rows = []
        for p in open_positions:
            detail = broker.get_position_detail(p["ticker"]) if broker else None
            rows.append({
                "Ticker": p["ticker"],
                "Strategy": p["strategy"],
                "Qty": p["quantity"],
                "Avg Entry": p["avg_entry_price"],
                "Current Price": detail["current_price"] if detail else None,
                "Unrealized P&L %": f"{detail['unrealized_plpc']:+.2%}" if detail else None,
                "High Water Mark": p["high_water_mark"],
                "Entry Regime": p["entry_regime"],
                "Last Buy": p["last_buy_at"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Options")
    option_positions = store.get_option_positions()
    open_option_positions = [p for p in option_positions if p["quantity"] > 0]
    if not open_option_positions:
        st.info("No open option positions.")
    else:
        rows = []
        for p in open_option_positions:
            detail = broker.get_position_detail(p["contract_symbol"]) if broker else None
            dte_remaining = (date.fromisoformat(p["expiration"]) - date.today()).days
            rows.append({
                "Contract": p["contract_symbol"],
                "Underlying": p["underlying_symbol"],
                "Type": p["option_type"],
                "Strike": p["strike"],
                "Expiration": p["expiration"],
                "DTE": dte_remaining,
                "Contracts": p["quantity"],
                "Avg Premium": p["avg_entry_price"],
                "Current Premium": detail["current_price"] if detail else None,
                "Unrealized P&L %": f"{detail['unrealized_plpc']:+.2%}" if detail else None,
                "Opened": p["opened_at"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("P&L is on premium (per-contract value, already reflecting the 100x multiplier) — not the underlying's share price.")

# ---- Trade History ----
with tab_trades:
    st.subheader("Equities")
    sales = store.get_all_realized_sales(limit=200)
    if not sales:
        st.info("No closed equity trades yet.")
    else:
        df = pd.DataFrame(sales)
        wins = (df["realized_pnl"] > 0).sum()
        total = len(df)
        win_rate = wins / total if total else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total trades", total)
        c2.metric("Win rate", f"{win_rate:.0%}")
        c3.metric("Total realized P&L", f"${df['realized_pnl'].sum():+,.2f}")

        df["cumulative_pnl"] = df["realized_pnl"][::-1].cumsum()[::-1]
        fig = px.line(df[::-1], x="created_at", y="cumulative_pnl", title="Cumulative realized P&L")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("Options")
    option_sales = store.get_all_realized_option_sales(limit=200)
    if not option_sales:
        st.info("No closed option trades yet.")
    else:
        odf = pd.DataFrame(option_sales)
        owins = (odf["realized_pnl"] > 0).sum()
        ototal = len(odf)
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Total trades", ototal)
        oc2.metric("Win rate", f"{owins / ototal:.0%}" if ototal else "0%")
        oc3.metric("Total realized P&L", f"${odf['realized_pnl'].sum():+,.2f}")
        st.dataframe(odf, width="stretch", hide_index=True)

# ---- Performance ----
with tab_performance:
    st.caption(
        "Risk-adjusted performance metrics derived from all closed trades. "
        "Sharpe and Sortino are annualized (×√252) and assume zero risk-free rate. "
        "Max drawdown and Calmar are on the cumulative realized P&L curve, not mark-to-market equity."
    )
    all_equity_sales = store.get_all_realized_sales(limit=1000)
    all_option_sales = store.get_all_realized_option_sales(limit=1000)
    combined_sales = all_equity_sales + all_option_sales

    def _show_metrics(label: str, sales: list[dict]) -> None:
        st.subheader(label)
        if not sales:
            st.info(f"No closed {label.lower()} trades yet.")
            return
        m = compute_metrics(sales)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe ratio", f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "—")
        c2.metric("Sortino ratio", f"{m.sortino_ratio:.2f}" if m.sortino_ratio is not None else "—")
        c3.metric("Calmar ratio", f"{m.calmar_ratio:.2f}" if m.calmar_ratio is not None else "—")
        c4.metric("Max drawdown", f"${m.max_drawdown:,.2f}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Win rate", f"{m.win_rate:.0%}")
        d2.metric("Profit factor", f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞")
        d3.metric("Avg win", f"${m.avg_win:+,.2f}")
        d4.metric("Avg loss", f"${m.avg_loss:+,.2f}")

    _show_metrics("Combined (equity + options)", combined_sales)
    _show_metrics("Equity trades", all_equity_sales)
    _show_metrics("Options trades", all_option_sales)

# ---- Agent Decisions ----
with tab_decisions:
    history = store.get_run_history(limit=100)
    if not history:
        st.info("No consensus runs recorded yet.")
    else:
        for entry in history:
            payload = entry["payload"]
            proposal = payload["proposal"]
            verdict = payload["risk_review"]["verdict"]
            badge = {"approved": "🟢", "amended": "🟡", "rejected": "🔴"}.get(verdict, "⚪")
            with st.expander(
                f"{badge} {entry['created_at']} — {payload['ticker']} — {proposal['action']} "
                f"x{proposal['quantity']} @ {proposal['limit_price']:.2f} ({verdict})"
            ):
                for signal in payload["signals"]:
                    st.markdown(f"**{signal['agent_name']}** ({signal['confidence']}): {signal['stance']}")
                    st.caption(_md(signal["rationale"]))
                st.markdown("**Risk Review reasons:**")
                for reason in payload["risk_review"]["reasons"]:
                    st.caption(_md(f"- {reason}"))

# ---- Scanner Activity ----
with tab_scans:
    for label, prefix in [
        ("Momentum scan", "momentum_scan_summary"),
        ("Thesis scan", "thesis_scan_summary"),
        ("Static prefilter", "prefilter_summary"),
    ]:
        st.subheader(label)
        events = store.get_events(event_type_like=prefix, limit=50)
        if not events:
            st.caption("No scans recorded yet.")
        else:
            st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)

# ---- Cost Tracking ----
with tab_cost:
    overall = store.get_cost_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Claude spend (all time)", f"${overall['total_cost_usd']:.4f}")
    c2.metric("Total calls", overall["total_calls"])
    c3.metric(
        "Cache read tokens",
        f"{overall['total_cache_read_input_tokens']:,}",
        help="Tokens served from the prompt cache at ~0.1x cost",
    )
    if overall["by_agent"]:
        df = pd.DataFrame(overall["by_agent"])
        fig = px.bar(df, x="agent_name", y="cost_usd", title="Spend by agent (all time)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No Claude calls recorded yet.")

# ---- Events Log ----
with tab_events:
    events = store.get_events(limit=300)
    if not events:
        st.info("No events recorded yet.")
    else:
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
