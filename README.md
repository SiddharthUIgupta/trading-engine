# trading-engine

An autonomous, multi-agent trading system: deterministic Python filters decide *what's worth looking at*, Claude agents reason only on candidates that clear the bar, deterministic Python guardrails enforce every risk limit, and Alpaca executes — paper trading by default.

## What it does

Every weekday, on a schedule:

1. **Pre-market filter (no LLM)** — for each ticker on the watchlist, pulls real market data (price history, sentiment, fundamentals, SEC filings via OpenBB) and runs it through deterministic technical/fundamental thresholds (e.g. RSI bands, moving-average crossovers, volume spikes, earnings surprises). Only tickers that cross a defined threshold get passed to the agent layer. This is the step that keeps Claude spend and latency proportional to *opportunities*, not watchlist size.
2. **Agent analysis (Claude, candidates only)** — tickers that clear the filter go through 4 agents, each looking at one slice of the picture:
   - **Macro/Sentiment** — news and sentiment tone
   - **Fundamental/SEC** — financial statements, analyst revisions, filings
   - **Technical Analysis** — moving averages, volatility, regime
   - **Risk Compliance Officer** — synthesizes the other three into one proposal, with explicit sign-off
3. **Market-open execution** — submits any approved trade to Alpaca (paper account).
4. **Intraday monitoring (every 15 min, no LLM by default)** — checks portfolio-wide drawdown/profit thresholds with plain Python. Per-position exit checks are rule-based (stop-loss, trailing stop, target hit); an LLM is invoked here **only** if a position needs judgment beyond a fixed rule (e.g. a held position's filter conditions reverse sharply) — and that call is logged and rate-limited like any other agent call.
5. **Post-market logging** — records positions, decisions, filter-pass rate, and Claude API spend for the day.

## Why the filter-first split exists

LLM reasoning is expensive and nondeterministic compared to arithmetic. Math should do the math; the agent layer should only run when there's a real decision to make. Concretely:

- **Pre-filter is deterministic** — no model in the loop, no hallucination risk, near-zero cost, scales to a large watchlist without scaling spend.
- **Agents run only on filtered candidates** — keeps daily Claude spend tied to opportunity count, not ticker count, and keeps each agent's context focused on names that actually matter that day.
- **Exit checks default to rule-based** — a 15-minute LLM call on every held position, every day, is the easiest way to quietly turn a paper-trading experiment into a live decision-making system. Default is plain thresholds; LLM exit review is an opt-in escalation path, not the default loop.

## Why it's hard to make it lose money by accident

The LLM never gets the final word on anything that touches the broker. Every consequential number is re-derived and enforced in plain Python, independent of what any agent says:

- **Max position size** — capped at a % of equity, checked twice (once in the Risk Officer's clamp, once again at the execution boundary) — a bug in one can't bypass the other.
- **Max daily drawdown** — if the account is down more than the configured % in a day, the breaker trips, all positions close, and trading halts until the next day.
- **Daily profit target** — the conservative-by-design counterpart: once the day is up a target dollar amount, the system **stops and banks the gain** rather than risking it chasing more.
- **Wash-sale guard** — blocks a buy-back that would disallow a loss just taken; warns (without blocking) if a sell might itself be a wash sale.
- **Paper trading by default** — going live requires two separate explicit environment variables to agree; any other combination silently stays on the paper sandbox.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full data-flow and layer breakdown.

## Status

Currently running live against a real Alpaca **paper** account (no real money), with real Claude reasoning and real market data. No backtest has been run yet — the filter thresholds and agent logic have not been validated against historical data, only observed live on paper. Treat any paper-trading "results" so far as a live test of plumbing, not a performance track record. Robinhood's agentic-trading product (real money, official MCP integration) is a possible future leg — not yet built, pending real API access.

## Stack

| Layer | Tech |
|---|---|
| Data | [OpenBB Platform](https://openbb.co/) |
| Pre-filter | Plain Python (no LLM) — technical/fundamental threshold checks |
| Analyst/consensus | [LangGraph](https://langchain-ai.github.io/langgraph/) + Anthropic Claude (Sonnet for the Risk Officer, Haiku for the 3 narrow sub-agents) — invoked only on filter-passed candidates |
| Execution | [Alpaca](https://alpaca.markets/) (paper trading), APScheduler, SQLite |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
python main.py
```

Runs the test suite:

```bash
pytest
```

## Cost

Filter pass costs effectively $0 (no LLM call). Each candidate that clears the filter costs roughly $0.01–0.05 to run through the 4-agent consensus (Sonnet for the Risk Officer, Haiku for the 3 sub-agents, prompt caching on). Per-call cost is tracked in SQLite and summarized in the post-market log every day, alongside the filter-pass rate so you can see what fraction of the watchlist is actually reaching the expensive layer.

## Before going live

No backtest exists yet. Before flipping the two live-trading environment variables, define an explicit gate — e.g. a minimum number of paper-trading days, a target win rate or Sharpe ratio over that window, and a maximum observed drawdown — rather than switching on a feeling that it's "been stable."
