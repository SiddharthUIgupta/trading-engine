# trading-engine

An autonomous, multi-agent trading system running 24/7 on a Raspberry Pi 5. Deterministic Python filters decide *what's worth looking at*, Claude agents reason only on candidates that clear the bar, a multi-level circuit breaker enforces every risk limit, and Alpaca executes — paper trading by default.

## What it does

Every weekday, fully automated on a schedule:

### Pre-market (8:00–9:15 AM ET)
- Pulls live US macro indicators (CPI, PMI, NFP, consumer sentiment) via **akshare**
- Assesses market regime — VIX level/trend + SPY SMA + macro news sentiment
- Screens the entire US equity market via 5 **OpenBB** discovery screens (gainers, active, losers, undervalued growth, aggressive small caps) — no hardcoded ticker lists
- Runs every candidate through a deterministic **pre-filter**: RSI, SMA crossover, volume spike, R² trend quality, KBAR candlestick body, VSUMP/VSUMN volume pressure, Amihud illiquidity (qlib Alpha158 factors)

### Five parallel trading tracks

| Track | Strategy | Hold period |
|---|---|---|
| **ORB Equity** | Opening range breakout on volume | Intraday |
| **ORB Options** | Defined-risk calls/puts on directional breakouts | Intraday |
| **Momentum** | Volume spike + price above SMA20 + RSI filter | 1–7 days |
| **Thesis** | 4-agent LLM consensus on pullback candidates | Days–weeks |
| **Swing** | SMA20 > SMA50 + RSI pullback entry | 1–3 weeks |
| **Vol/Premium** | Sells options premium when VIX 18–40 | Until 21 DTE |

### Agent consensus (thesis/swing candidates only)
Four Claude agents each evaluate one lens, then a Risk Officer synthesizes:
- **Macro/Sentiment** — news tone + live economic data
- **Fundamental** — financials, analyst revisions, SEC filings
- **Technical** — moving averages, volatility, regime signals
- **Risk Officer** — synthesizes all three into one proposal

### Intraday monitoring (every 15 min)
Rule-based exit checks (stop-loss, trailing stop, profit target) — LLM only invoked when a position's regime sharply reverses and rules say "hold" but the setup has broken down.

### Learning layer
- **Vowpal Wabbit contextual bandit** — online logistic regression that learns which track × regime × agent signal patterns win. Warm-started on 8,672 historical examples. Win probability injected into every agent prompt.
- **Reflection agent** — after every closed trade, writes a structured lesson stored in SQLite and injected into future consensus prompts for similar setups.

### Shadow signals — measurement only, zero trading influence
Every screened candidate is scored by pluggable signal providers *after* the trade decision is already made — nothing here can affect a trade. Scores are logged alongside each candidate's eventual forward return so `scripts/signal_uplift.py` can measure real predictive power (Spearman IC, n≥300 gate) before any signal is ever considered for promotion into the risk gate — a separate, explicit, future decision (see `CLAUDE.md` → "Signal lifecycle").
- **Kronos-small** — a vendored open-source financial time-series model (24.7M params). Samples 30 Monte Carlo price paths per candidate to compute touch-probability, median return, and dispersion.
- **Short interest / squeeze potential** — real short-interest data (OpenBB/yfinance) and borrow-availability flags (Alpaca), the kind of signal that would have flagged GameStop/AMC-style setups. Tracks metric staleness explicitly (short-interest data is inherently ~20 days lagged — FINRA's settlement cadence) rather than assuming it away.

---

## Risk management

### 5-level circuit breaker (RobustCircuitBreaker)

| Level | Trigger | Response |
|---|---|---|
| 0 | Per-trade too large | Hard block |
| 1 | 3 consecutive losses on a strategy | Position sizes cut to **50%** until next win |
| 2 | 5% daily drawdown on a strategy | That strategy halts for the day |
| 3 | 8% cumulative weekly loss | **All strategies halt until Monday** |
| 4 | 20% drop from all-time equity peak | **Full system halt** — requires manual reset |
| + VIX scaling | VIX rises | Sizing scaled 85% → 70% → 55% → 40% as VIX rises |

### Additional guards
- **Kelly criterion** — position sizes scale with historical win rate. At low win rates, sizes shrink toward zero automatically.
- **Correlation guard** — won't open a position highly correlated to something already held.
- **Wash-sale guard** — blocks buy-backs that would disallow a recent loss.
- **Capital allocation** — each track has an independent capital budget; one track can't crowd out another.
- **Paper trading by default** — two separate env vars must both be set to go live.

---

## Stack

| Component | Tech |
|---|---|
| Hardware | Raspberry Pi 5 (always-on, aarch64) |
| Broker | [Alpaca](https://alpaca.markets/) (paper + live), Level 3 options |
| Market data | [OpenBB Platform](https://openbb.co/) + yfinance |
| Alternative data | [akshare](https://github.com/akfamily/akshare) — US macro indicators, market movers |
| Pre-filter factors | qlib Alpha158 formulas (KBAR, VSUMP/VSUMN, ILLIQ, R², Williams %R) |
| Agent consensus | [LangGraph](https://langchain-ai.github.io/langgraph/) + Anthropic Claude |
| Reinforcement learning | [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) contextual bandit |
| Shadow signals | [Kronos-small](https://github.com/shiyu-coder/Kronos) (vendored, MIT) forecasting model via PyTorch (CPU); short-interest via OpenBB/yfinance + Alpaca |
| Backtesting | [backtrader](https://github.com/mementum/backtrader) walk-forward backtests |
| Scheduling | APScheduler |
| State | SQLite |

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
```

Runs as two independent systemd services rather than a single process — see
[Two-Plane IPC](./ARCHITECTURE.md#two-plane-ipc-order_intents--breaker_state) in
`ARCHITECTURE.md` for why:

```bash
sudo bash install_services.sh                        # installs + enables both units
sudo systemctl start trading-engine-protection.service  # exits/reconciliation — start first
sudo systemctl start trading-engine-alpha.service        # scanning/LLM consensus
```

### Management CLI (uses zcmd)
```bash
python scripts/manage.py status      # service + process status
python scripts/manage.py equity      # P&L summary + open positions
python scripts/manage.py logs 100    # last 100 log lines
python scripts/manage.py follow      # live log tail
python scripts/manage.py restart     # restart the engine
python scripts/manage.py backtest    # run swing backtest (2 years)
```

### Backtesting
```bash
python scripts/backtest.py --strategy swing    --years 2
python scripts/backtest.py --strategy momentum --years 2
python scripts/backtest.py --strategy orb      --years 2
```

### Shadow signals (manually invoked, not scheduled — see README "Shadow signals" above)
```bash
python scripts/kronos_shadow_signal_job.py           # scores un-enriched candidates with Kronos-small
python scripts/short_interest_shadow_signal_job.py   # scores un-enriched candidates with short-interest data
python scripts/signal_uplift.py                      # per-signal IC report + staleness + PROMOTE/DELETE verdict
```

### Tests
```bash
pytest   # 440 tests
```

---

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full data-flow and layer breakdown.

Data flows in one direction only: `data_layer → analyst_layer → execution_layer`. No layer imports from a layer above it. The LLM never gets the final word on anything that touches the broker — every risk limit is re-enforced in plain Python at the execution boundary.

---

## Current status

Running live on a Raspberry Pi 5 against an Alpaca **paper** account. Real Claude reasoning, real market data, no real money. Backtests (swing, momentum, ORB) have been run using backtrader — see `scripts/backtest.py`.
