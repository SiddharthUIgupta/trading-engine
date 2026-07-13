#!/usr/bin/env python3
"""Nightly investment committee swarm (Path B — our own Anthropic agents).

Runs overnight on recent thesis candidates. Four agents per ticker:
  Bull analyst   ──┐
                   ├──► CRO ──► PM (final verdict → obsidian vault)
  Bear analyst   ──┘

Results written to wiki/research/TICKER-DATE.md and BM25-indexed so
_fetch_trade_memory() surfaces them to the Risk Officer at market open.

Usage:
    .venv/bin/python scripts/nightly_swarm.py               # top BUY candidates
    .venv/bin/python scripts/nightly_swarm.py --top 10      # more candidates
    .venv/bin/python scripts/nightly_swarm.py --tickers AAPL,NVDA,MSFT
    .venv/bin/python scripts/nightly_swarm.py --dry-run     # print prompts, no LLM calls
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_DB = Path(__file__).resolve().parent.parent / "state" / "trading_engine.sqlite3"
_VAULT = Path.home() / "Projects" / "claude-obsidian" / "wiki" / "research"
_RETRIEVE_SCRIPT = Path.home() / "Projects" / "claude-obsidian" / "scripts" / "retrieve.py"
_VIBE_AGENT = Path.home() / "Projects" / "Vibe-Trading" / "agent"


# ── data helpers ─────────────────────────────────────────────────────────────

def get_recent_candidates(top: int = 10) -> list[str]:
    """Return tickers from the most recent scan date, BUY verdicts first."""
    conn = sqlite3.connect(_DB)
    try:
        latest_date = conn.execute(
            "SELECT MAX(candidate_date) FROM candidates WHERE strategy='thesis'"
        ).fetchone()[0]
        if not latest_date:
            return []
        rows = conn.execute(
            "SELECT ticker, llm_verdict FROM candidates "
            "WHERE strategy='thesis' AND candidate_date=? "
            "ORDER BY CASE llm_verdict WHEN 'BUY' THEN 0 WHEN 'HOLD' THEN 1 ELSE 2 END, ticker",
            (latest_date,),
        ).fetchall()
        tickers = [r[0] for r in rows[:top]]
        logger.info("candidates from %s: %s", latest_date, tickers)
        return tickers
    finally:
        conn.close()


def fetch_price_bars(ticker: str, period: str = "60d") -> list[dict]:
    """Download daily OHLCV bars via yfinance."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return []
        if hasattr(raw.columns, "get_level_values"):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        bars = []
        for ts, row in raw.iterrows():
            bars.append({
                "date": str(ts.date()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
            })
        return bars
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        return []


def build_price_series(ticker: str, bars: list[dict]):
    """Build a PriceSeries from raw bar dicts for vibe_data functions."""
    from data_layer.models import PriceBar, PriceSeries
    price_bars = []
    for b in bars:
        try:
            price_bars.append(PriceBar(
                symbol=ticker,
                timestamp=datetime.fromisoformat(b["date"] + "T00:00:00"),
                open=b["open"], high=b["high"], low=b["low"],
                close=b["close"], volume=int(b["volume"]),
            ))
        except Exception:
            continue
    if not price_bars:
        return None
    return PriceSeries(symbol=ticker, interval="1d", bars=price_bars)


def enrich_ticker(ticker: str) -> dict:
    """Fetch SEC context + technical signals for one ticker."""
    from analyst_layer.vibe_data import compute_technical_signals, fetch_sec_context
    sec = fetch_sec_context(ticker)
    bars = fetch_price_bars(ticker)
    tech = ""
    last_close = None
    if bars:
        series = build_price_series(ticker, bars)
        if series:
            tech = compute_technical_signals(series)
            last_close = bars[-1]["close"]
    return {"sec": sec, "tech": tech, "last_close": last_close}


# ── agents ────────────────────────────────────────────────────────────────────

def _call(client, model: str, system: str, prompt: str, dry_run: bool) -> str:
    if dry_run:
        logger.info("[DRY RUN] would call %s with prompt len=%d", model, len(prompt))
        return f"[DRY RUN — {model}]"
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def run_bull(client, model: str, ticker: str, data: dict, dry_run: bool) -> str:
    system = (
        "You are the Bull analyst on an investment committee. Your job is to build "
        "the strongest possible long case for the given stock. Be specific: cite "
        "actual revenue figures, growth rates, catalysts, and why this beats the market. "
        "End with: UPSIDE TARGET: $X (Y% from current), CONVICTION: 1-5."
    )
    prompt = (
        f"Ticker: {ticker}  Current price: ${data['last_close']}\n"
        f"{data['sec']}\n{data['tech']}\n\n"
        "Build the bull case. Be specific. What makes this a strong BUY right now?"
    )
    return _call(client, model, system, prompt, dry_run)


def run_bear(client, model: str, ticker: str, data: dict, dry_run: bool) -> str:
    system = (
        "You are the Bear analyst on an investment committee. Your job is to identify "
        "every material risk, red flag, and reason NOT to buy this stock. Be specific: "
        "cite actual numbers, competitive threats, balance sheet weaknesses, technicals. "
        "End with: DOWNSIDE TARGET: $X (Y% from current), CONVICTION: 1-5."
    )
    prompt = (
        f"Ticker: {ticker}  Current price: ${data['last_close']}\n"
        f"{data['sec']}\n{data['tech']}\n\n"
        "Build the bear case. What are the real risks here? Why should we pass?"
    )
    return _call(client, model, system, prompt, dry_run)


def run_cro(client, model: str, ticker: str, bull: str, bear: str, dry_run: bool) -> str:
    system = (
        "You are the Chief Risk Officer on an investment committee. Given the bull and "
        "bear arguments, identify the key swing factors — what would make the bull right "
        "vs the bear right. Assess position sizing: what fraction of max position is "
        "appropriate given the uncertainty? End with: SIZING GUIDELINE: X% of max position."
    )
    prompt = (
        f"Ticker: {ticker}\n\n"
        f"BULL CASE:\n{bull}\n\n"
        f"BEAR CASE:\n{bear}\n\n"
        "What are the 2-3 key swing factors? Who has the stronger argument and why? "
        "What sizing is appropriate?"
    )
    return _call(client, model, system, prompt, dry_run)


def run_pm(client, model: str, ticker: str, data: dict, bull: str, bear: str, cro: str, dry_run: bool) -> str:
    system = (
        "You are the Portfolio Manager making the final call. You have heard from the bull, "
        "bear, and risk officer. Give a clear, actionable verdict: BUY, WATCH, or PASS. "
        "For BUY: give entry zone, price target, stop loss, and time horizon. "
        "For WATCH: state the specific trigger that would make you buy. "
        "For PASS: state what would need to change to reconsider. "
        "Be direct. No hedging. One clear decision."
    )
    prompt = (
        f"Ticker: {ticker}  Current price: ${data['last_close']}\n\n"
        f"BULL:\n{bull}\n\n"
        f"BEAR:\n{bear}\n\n"
        f"CRO:\n{cro}\n\n"
        "Final verdict: BUY, WATCH, or PASS? Give entry, target, stop, and thesis in 150 words."
    )
    return _call(client, model, system, prompt, dry_run)


# ── vault write ───────────────────────────────────────────────────────────────

def write_research_note(ticker: str, data: dict, bull: str, bear: str, cro: str, pm: str) -> None:
    if not _VAULT.exists():
        _VAULT.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    verdict_line = "UNKNOWN"
    for word in ("BUY", "WATCH", "PASS"):
        if word in pm[:50]:
            verdict_line = word
            break
    md = "\n".join([
        f"# {ticker} Nightly Research — {today} [{verdict_line}]",
        f"\nPrice at research time: ${data['last_close']}",
        "\n## Bull Case", bull,
        "\n## Bear Case", bear,
        "\n## CRO Risk Assessment", cro,
        "\n## PM Verdict", pm,
        f"\n---\n*Generated by nightly_swarm.py on {datetime.utcnow().isoformat()}*",
    ])
    path = _VAULT / f"{ticker}-{today}.md"
    path.write_text(md)
    logger.info("wrote research note: %s", path)
    # re-index BM25 so retrieve.py picks it up at market open
    try:
        idx = _RETRIEVE_SCRIPT.parent.parent / "bin" / "bm25-index.py"
        if idx.exists():
            subprocess.run([sys.executable, str(idx), "build"], capture_output=True, timeout=60)
    except Exception:
        pass


# ── main ─────────────────────────────────────────────────────────────────────

def run_committee(ticker: str, client, haiku: str, sonnet: str, dry_run: bool) -> str:
    """Run the full 4-agent committee for one ticker. Returns PM verdict."""
    logger.info("── %s: fetching data ──", ticker)
    data = enrich_ticker(ticker)
    if data["last_close"] is None and not dry_run:
        logger.warning("%s: no price data, skipping", ticker)
        return "SKIP"

    logger.info("%s: running bull + bear in parallel", ticker)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_bull = ex.submit(run_bull, client, haiku, ticker, data, dry_run)
        fut_bear = ex.submit(run_bear, client, haiku, ticker, data, dry_run)
        bull = fut_bull.result()
        bear = fut_bear.result()

    logger.info("%s: running CRO", ticker)
    cro = run_cro(client, haiku, ticker, bull, bear, dry_run)

    logger.info("%s: running PM (final verdict)", ticker)
    pm = run_pm(client, sonnet, ticker, data, bull, bear, cro, dry_run)

    if not dry_run:
        write_research_note(ticker, data, bull, bear, cro, pm)

    logger.info("%s: DONE — %s", ticker, pm[:120].replace("\n", " "))
    return pm


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly investment committee swarm")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (overrides DB)")
    parser.add_argument("--top", type=int, default=5, help="Max candidates from DB (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, print structure only")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = get_recent_candidates(top=args.top)

    if not tickers:
        from config.settings import get_settings
        tickers = get_settings().extra_watchlist_tickers or []
        logger.info("no DB candidates — falling back to extra_watchlist: %s", tickers)

    if not tickers:
        logger.error("no tickers to run — pass --tickers or ensure candidates table has data")
        sys.exit(1)

    from anthropic import Anthropic
    from config.settings import get_settings
    s = get_settings()
    client = Anthropic(api_key=s.anthropic_api_key)
    haiku = s.anthropic_subagent_model   # bull, bear, CRO
    sonnet = s.anthropic_model           # PM final verdict

    logger.info("running nightly swarm on %d tickers: %s", len(tickers), tickers)
    logger.info("models: bull/bear/CRO=%s  PM=%s", haiku, sonnet)

    results: dict[str, str] = {}
    for ticker in tickers:
        try:
            verdict = run_committee(ticker, client, haiku, sonnet, args.dry_run)
            results[ticker] = verdict
        except Exception as exc:
            logger.error("%s failed: %s", ticker, exc)
            results[ticker] = f"ERROR: {exc}"

    print("\n" + "=" * 60)
    print("NIGHTLY SWARM SUMMARY")
    print("=" * 60)
    for ticker, verdict in results.items():
        first_line = verdict.split("\n")[0][:100]
        print(f"{ticker:8s}  {first_line}")
    print(f"\nResearch notes written to: {_VAULT}")


if __name__ == "__main__":
    main()
