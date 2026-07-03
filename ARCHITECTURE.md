# Trading Engine — Architecture & Blueprint

## What It Is

An autonomous trading system running 24/7 on a Raspberry Pi. It scans markets each morning, decides what to buy using a multi-agent LLM consensus, manages open positions intraday, closes them according to rule-based exits, and learns from each outcome to improve future decisions. No human needs to approve trades — the system is self-contained from signal to execution to reflection.

Broker: Alpaca (live account, Level 3 options). LLM: Anthropic Claude (Sonnet for risk decisions, Haiku for sub-agents). Market data: OpenBB + Yahoo Finance + Finnhub.

---

## System Map

Two independent processes, deliberately split for fault isolation — a crash in
scanning/LLM logic can no longer take down exit protection:

```
┌───────────────────────────────┐      ┌──────────────────────────────────┐
│   main_alpha.py                │      │   main_protection.py              │
│   Alpha Plane                  │      │   Protection Plane                │
│   Scans, LLM consensus, sizing │      │   Exits, reconciliation, orders   │
│   Writes order_intents (BUY)   │─────►│   Reads order_intents, submits    │
│   Never calls broker.submit_   │ SQLite   bracket orders, marks processed │
│   order() for equity entries   │ IPC   │   Runs even if Alpha is down     │
└───────────────┬────────────────┘      └────────────────┬───────────────────┘
                │                                         │
                │           breaker_state table           │
                │◄─────────── (Protection writes trip,  ─►│
                │              Alpha reads before queuing) │
                │                                         │
        ┌───────▼─────────────────────────────────────────▼───────┐
        │     execution_layer/                                    │
        │     alpha_plane.py       │  ← AlphaRuntime (Alpha proc)  │
        │     protection_plane.py  │  ← ProtectionRuntime (Prot.)  │
        │     runtime.py           │  ← legacy monolith (inactive; │
        │                          │    kept for reference/tests)  │
        │     state_store.py       │  ← SQLite wrapper             │
        │     broker.py            │  ← Alpaca wrapper             │
        │     guardrails.py        │  ← circuit breaker            │
        │     exit_rules.py        │  ← stop/trail logic           │
        │     tax_compliance.py    │  ← wash-sale guard            │
        │     alerting.py          │  ← dead-man alerts + heartbeats│
        │     manual_trigger.py    │  ← dashboard → engine IPC     │
        └───────────────┬──────────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     analyst_layer/            │
        │     graph.py                  │  ← LLM consensus engine
        │     agents/ (8 agents)        │  ← specialist LLM callers
        │     market_regime.py          │  ← daily strategy arm/disarm
        │     macro_news_agent.py       │  ← pre-market Finnhub sentiment
        │     orb_scanner.py            │  ← breakout signal (disabled)
        │     gap_scanner.py            │  ← pre-market gap signal
        │     thesis_scanner.py         │  ← pullback signal
        │     momentum_scanner.py       │  ← float/volume screen
        │     kelly.py                  │  ← position sizing
        │     vw_bandit.py              │  ← VowpalWabbit online bandit
        │     lesson_store.py           │  ← pattern memory
        │     reflection_agent.py       │  ← post-trade LLM review
        │     correlation.py            │  ← portfolio overlap guard
        │     options_structurer.py     │  ← contract selection
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     data_layer/               │
        │     openbb_client.py          │  ← price, fundamentals, chains
        │     finnhub_client.py         │  ← finance-focused news (replaces Google News)
        │     akshare_client.py         │  ← US pre-market movers (gap scanner)
        │     models.py                 │  ← PriceBar, PriceSeries, etc.
        └───────────────────────────────┘
```

### Two-Plane IPC (order_intents / breaker_state)

- **`order_intents` table** — Alpha writes `(client_order_id, strategy, ticker, action, quantity, limit_price, stop_price, status='pending')` after LLM consensus approves a BUY. Protection polls pending rows every 15 min, submits the actual bracket order to Alpaca, and marks the row processed. `client_order_id` is the PRIMARY KEY with `INSERT OR IGNORE`, so a retried write from Alpha can never double-queue the same order.
- **`breaker_state` table** — when Protection trips a circuit breaker, it writes `(breaker_name, state_key='halted', state_value='true')`. Alpha checks this table before queuing any new intent for that bucket, so a halt initiated by Protection is respected by Alpha even though they're separate processes with separate in-memory breaker objects.
- **`client_order_id` idempotency** — `sha1(date|ticker|action|qty)[:32]`. Alpaca dedupes retried submissions on this ID, so a Protection restart mid-submission can't result in a duplicate live order.

### Bracket Orders (OTO)

Protection submits entries as `LimitOrderRequest` with `OrderClass.OTO` + a `StopLossRequest` child leg. The stop rests directly at the Alpaca broker instead of being polled and submitted by our own code — so a stop is enforced even if both planes are down. `amend_stop_order()` moves the resting stop up as the trailing-stop logic activates.

### Dead-Man Alerting

- **Per-job heartbeats** — every scheduled job in both planes writes to `state/heartbeat.json` on completion (and pings healthchecks.io if configured). A missed heartbeat signals the job silently died rather than just finding nothing to do.
- **Zero-BUY streak alert** — if 3+ consecutive sessions produce zero BUYs across all tracks, an email alert fires. Catches the failure mode where the scheduler is technically alive but every scan is silently erroring out before reaching a trade decision.

---

## Daily Schedule (Eastern Time)

Split across the two planes — Alpha decides and queues, Protection executes and protects.

### Alpha Plane (`main_alpha.py`)

| Time | Job | What Happens |
|------|-----|--------------|
| 8:15am | `pre_market_scan` (thesis) | Fetches daily regime (VIX + SPY SMAs). Arms or disarms each strategy track for the day. Runs macro news agent. |
| 8:15am | `thesis_scan_and_trade` | Scans OpenBB's undervalued-growth universe (5 screens, market-wide) for 20–50% pullbacks. Top 20 pass to LLM consensus. Approved BUYs written to `order_intents`, not submitted directly. |
| 9:05am | `gap_scan_and_queue` | Scans akshare US movers + watchlist for pre-market gaps ≥5%. Top 5 candidates pass directly to thesis consensus. |
| 9:30am | `market_open_execution` | Queues any pending approved intents for 9:30 execution. |
| 9:35am | `swing_scan_and_trade` | Scans for swing trade setups (3–6 week holds). |
| 9:00–3:00pm every 15min | `momentum_scan_and_trade` (ORB equity) | Disabled (`ORB_EQUITY_ENABLED=false`); full implementation exists in `alpha_plane.py` for when re-enabled. |
| 9:00–3:00pm every 15min | `options_scan_and_trade` (ORB options) | Disabled (`OPTIONS_TRACK_ENABLED=false`); submits directly (specific option contracts, not routed through `order_intents`). |
| 10:00am + 1:00pm | `vol_options_scan_and_trade` | Short premium (iron condors) when VIX/IV conditions suit. Submits directly — Level 3 iron condor mleg orders, computed per-scan. |
| 4:30pm | `post_market_logging` | Logs cost summary, positions, daily P&L. |
| Every 15s | `check_manual_trigger` | Polls `state/manual_trigger.json` for dashboard-triggered scans. |

### Protection Plane (`main_protection.py`)

| Time | Job | What Happens |
|------|-----|--------------|
| 9:30–4:00pm every 15min | `intraday_monitoring` | Reconciles positions vs broker → ensures day started → checks `breaker_state` → consumes pending `order_intents` (submits bracket orders) → runs all exit checks (intraday, ORB, options, vol options, swing). |
| 3:30pm | `pre_close_orb_exit` | Closes any residual same-day ORB position flat or losing. |

If Alpha crashes, Protection keeps running independently — existing positions stay protected by stop-losses and exit rules every 15 minutes regardless.

---

## Strategy Tracks

### 1. Thesis Pullback (PRIMARY — ENABLED)

**What it is:** Buys quality stocks that have pulled back 20–50% from their 52-week high. Multi-day to multi-week hold.

**Signal pipeline:**
1. OpenBB 5 screens (active, gainers, losers, undervalued_growth, aggressive_small_caps) → raw candidates, market-wide
2. `thesis_scanner.evaluate_thesis_candidate` filters for 20–50% drawdown from 52-week high
3. `thesis_scanner.evaluate_shrink_volume_pullback` checks for MA5 > MA10 > MA20 with low-volume consolidation (score boost if confirmed, not required)
4. Top 20 by score pass to the 4-agent LLM consensus
5. Approved BUYs execute at market open

**Exit rules:**
- Stop loss: 18% below entry
- Trailing stop: activates at +20% gain, trails 10% behind peak
- No fixed take-profit — let winners run

**Backtest results (S&P 500, 3 years, 1,370 closed trades):**

| Scenario | Win Rate | Avg Return | Profit Factor |
|----------|----------|------------|---------------|
| Zero slippage | 57.9% | +6.27% | 1.83 |
| 0.5% entry / 0.3% exit slippage | 56.7% | +5.31% | 1.67 |

Edge survives realistic slippage — the wide trailing stop lets winners run to +23% avg, making entry slippage a rounding error.

---

### 2. Pre-Market Gap Scanner (SECONDARY — ENABLED)

**What it is:** Catches earnings/catalyst gaps before the open. Runs at 9:05 AM, 25 minutes before thesis candidates execute. Finds stocks gapping ≥5% with ≥$5M avg daily dollar volume.

**Why it exists:** The thesis scan runs at 8:15 AM on prior-close data and misses overnight news gaps. META's +10% move in early 2026 was not caught because the gap only appeared pre-market after the scan ran. The gap scanner closes this window.

**Signal pipeline:**
1. akshare `stock_us_famous_spot_em()` → liquid US movers (fast, ~2s)
2. yfinance `fast_info` for watchlist symbols
3. Filter: gap ≥ 5%, avg_dollar_vol ≥ $5M, price > $1
4. Top 5 by gap size pass directly to thesis LLM consensus (bypasses the 20–50% pullback screen — the gap IS the entry signal)
5. Routes through thesis circuit breaker and Kelly sizing (strategy="gap")

---

### 3. ORB Equity (DISABLED — `ORB_EQUITY_ENABLED=false`)

**Why disabled:** Backtest confirmed 46.3% win rate, **-0.07% average return** before slippage. With realistic execution costs, clearly net negative. All intraday capital reallocated to thesis track.

**Previous logic (documented for reference):** Opening Range Breakout on gap ≥ 4%, volume ≥ 2x, SPY green. Intraday only, EOD close. Did not route through LLM consensus — deterministic rule-based entry — which meant no learning loop feedback on ORB losses.

---

### 4. ORB Options (DISABLED — `OPTIONS_TRACK_ENABLED=false`)

**Why disabled:** Signal resolves intraday. 30–45 DTE contracts accumulated theta bleed on a signal with no overnight edge. Generated $8,647 in realized losses before position caps were added.

---

### 5. Vol / Premium Selling (ENABLED — `VOL_OPTIONS_TRACK_ENABLED=true`)

**What it is:** Sells short premium (iron condors, strangles, spreads) when IV Rank is elevated. Based on Natenberg/tastylive framework. Requires Level 3 options approval.

**Arm conditions:** VIX 18–40, not spiking (>15% rise in a week while above 25).

---

## LLM Consensus Engine

Used by thesis, gap, swing, and news-catalyst tracks.

### The 4-Agent Pipeline

```
Ticker + market data
        │
        ├──► MacroSentimentAgent  (Haiku)
        │    Reads recent Finnhub headlines (company_news API, 7 days back).
        │    Score via sentiment lexicon. Returns: stance, confidence, rationale.
        │
        ├──► FundamentalAgent  (Haiku)
        │    Reads P/E, P/B, debt/equity, earnings growth from OpenBB fundamentals.
        │    Returns: stance, confidence, rationale.
        │
        ├──► TechnicalAgent  (Haiku)
        │    Computes RSI, SMA10/30, realized vol, price vs VWAP from price series.
        │    Returns: stance, confidence, rationale.
        │
        └──► RiskOfficerAgent  (Sonnet)
             Receives all three stances + account context (equity, Kelly fraction,
             correlation exposure, existing shares).
             Computes position size. Issues APPROVED / REJECTED / AMENDED verdict.
             AMENDED = size clamped to MAX_POSITION_SIZE_PCT but thesis is sound.
```

All three sub-agents run in parallel via LangGraph. The Risk Officer runs after.

### News Source: Finnhub

Per-ticker sentiment uses Finnhub `/company-news` (7 days, up to 50 articles). Market-wide macro sentiment uses Finnhub `/news?category=general` + `/news?category=merger`. Replaced Google News RSS (generic, unfocused) and yfinance.Search. Falls back to Google/yfinance if no API key configured.

### Kelly Position Sizing

```
f = 0.5 × (p − q/b)
```

Where `p` = historical win rate, `q` = 1-p, `b` = avg_win / avg_loss.

- Requires minimum 15 closed trades before trusting the estimate. Below that, uses 50% of max cap.
- **Exploration floor:** When Kelly computes to 0% (negative edge) and n < 50 trades, uses 1% minimum. Prevents the VW bandit from starving on zero new examples while the system is building its track record.
- Hard-capped at `MAX_POSITION_SIZE_PCT` regardless.

### Correlation Guard

Before the consensus runs, the system computes correlation between the proposed ticker and all current positions. If max correlation exceeds 0.85, the trade is blocked (prevents the portfolio from becoming a single-factor bet).

---

## Macro News Agent

Runs at 8:00 AM before regime assessment. Two outputs:

1. **Market sentiment** (bullish/bearish/neutral + confidence) → adjusts effective VIX used by regime. High-confidence bullish news lowers the effective VIX, potentially arming tracks that would otherwise sit just above a threshold.

2. **News ticker signals** → individual stocks with clear catalysts extracted by Haiku. Bullish tickers bypass the pullback screen and go directly to thesis consensus at 8:15 AM. Bearish tickers are blocked from new entries.

---

## Manual Trigger System (Dashboard → Engine)

The Controls tab in the dashboard lets you fire any scan on demand without waiting for the scheduler:

- **Write path:** Dashboard calls `manual_trigger.write_trigger(scan)` → writes `state/manual_trigger.json` + appends to `state/trigger_history/<date>.jsonl`
- **Read path:** Engine polls `state/manual_trigger.json` every 15 seconds via `check_manual_trigger` job → dispatches to the correct runtime method → deletes the file
- **History:** Trigger history persists to disk by date; dashboard reads it on every page load so scan outputs survive page refreshes

Available triggers: `thesis`, `gap`, `swing`, `momentum`, `options`.

---

## Daily Regime Assessment

Runs at 8:00am. Pure arithmetic — no LLM. Determines which tracks are armed for the day.

**VIX smoothing:** All threshold decisions use a 5-day VIX moving average, not spot VIX. Prevents daily flipping when VIX sits at 29.8 vs 30.2 on adjacent days.

| Condition | Effect |
|-----------|--------|
| VIX(5d avg) > 30 + bearish market | ORB equity DISARMED (moot — ORB disabled) |
| VIX(5d avg) > 30 | Thesis DISARMED |
| VIX(5d avg) < 18 | Vol/premium DISARMED |
| VIX spiking (>15% in a week while above 25) | Vol/premium DISARMED |
| VIX(5d avg) > 40 | Vol/premium DISARMED |

---

## Risk Management

### Circuit Breaker (`guardrails.py`)

Four independent breakers — one per capital allocation bucket (intraday, options, thesis, swing). Each trips on:
- **Drawdown breach:** bucket down ≥ daily drawdown cap → no new trades from that bucket
- **Profit lock:** daily profit target hit → lock in the gain, no new entries
- **Global halt:** weekly or trailing drawdown limit breached → all buckets halted

Resets at midnight each trading day.

### Per-Position Exit Rules

Applied every 15 minutes during intraday monitoring:

| Rule | Thesis / Gap | Swing |
|------|-------------|-------|
| Stop loss | 18% below entry | 8% below entry |
| Trailing activation | +20% gain | +12% gain |
| Trailing distance | 10% behind peak | 7% behind peak |
| EOD close if flat/losing | No (multi-day hold) | No (multi-week hold) |
| Max hold | No limit | 21 calendar days |

### Options Exit Rules

| Rule | Threshold |
|------|-----------|
| Intraday stop (same-day entry, after 3pm ET) | Down ≥ 20% |
| Stop loss | Down ≥ 50% |
| Force close | ≤ 7 DTE remaining |
| Profit target (vol track) | Up ≥ 50% of credit received |

### Position Caps

- Max concurrent equity positions: 15
- Max concurrent options positions: 8

### Wash Sale Guard

Blocks a BUY on any ticker sold at a loss within the past 30 days.

---

## Learning Loop

Active for thesis, gap, swing, and news-catalyst tracks. ORB was deliberately excluded (no LLM consensus = no run_history = no reflection). Now that ORB is disabled, all live trades feed the loop.

### 1. Agent Signal Scoring

Every BUY consensus records each sub-agent's stance and confidence in `agent_signal_log`. After the trade closes, the outcome is written back. Over time this reveals which agents are actually predictive.

### 2. Lesson Journal

After each trade closes, `ReflectionAgent` (Sonnet) generates structured lessons tagged with `setup_tags`. Lessons carry a score: starts at 1.0, +0.1 per win, -0.05 per loss, retired below 0.3. Before the next consensus run for matching conditions, relevant lessons are injected into agent prompts.

### 3. Post-Trade Reflection

`ReflectionAgent` produces a post-mortem with `what_happened`, `root_cause`, `outcome_was_noise`, and `lessons`. Receives full context: regime at entry, all three sub-agent rationales, and outcome.

### 4. Candidate Ledger

Every screened candidate — not just executed trades — is logged to the `candidates` table with its LLM verdict, gate result, and forward returns (1/5/21/63 day) backfilled nightly. This lets the lesson journal's edge be measured against the full candidate population, not just the trades that happened to clear every gate — a much stronger denominator for deciding whether a lesson generalizes.

### 5. Lesson Injection Freeze

`FREEZE_LESSON_INJECTION` (default `true`) stops injecting past lessons into LLM prompts until the candidate ledger has accumulated enough forward-return data to prove a lesson actually predicts anything, rather than the reflection agent's own narrative confidence. Flip to `false` once the candidate ledger shows a lesson's tagged setups have a real edge.

---

## State Database (SQLite)

Location: `./state/trading_engine.sqlite3`

| Table | Purpose |
|-------|---------|
| `positions` | Open equity positions: ticker, qty, avg_entry, stop, target, strategy, regime |
| `option_positions` | Open options: contract symbol, underlying, strike, expiry, qty, entry price |
| `realized_sales` | Closed equity trades with realized P&L |
| `realized_option_sales` | Closed options trades with realized P&L |
| `run_history` | Full JSON of every LLM consensus run |
| `agent_signal_log` | Sub-agent stances at consensus time, scored after close |
| `agent_signal_detail` | Per-agent stance/confidence (child of signal_log) |
| `agent_lessons` | Extracted lessons with score, setup_tags, strategy |
| `lesson_injection_log` | Which lessons were active for which trades |
| `trade_reflections` | Post-mortem narratives |
| `events` | Append-only audit log of all system events |
| `token_usage` | Per-agent LLM token counts and estimated cost |
| `order_intents` | Alpha→Protection IPC: pending BUY intents keyed by `client_order_id` |
| `breaker_state` | Protection→Alpha IPC: which capital buckets are halted |
| `candidates` | Every screened candidate (not just trades) with verdict + forward returns |

---

## Configuration

All settings in `config/settings.py`, overridable via `.env`.

```
# Environment
TRADING_ENV=live
TRADING_LIVE_CONFIRM=I_UNDERSTAND_THIS_IS_LIVE_CAPITAL

# Risk guardrails
MAX_POSITION_SIZE_PCT=0.05          # 5% of equity per position
MAX_DAILY_DRAWDOWN_PCT=0.05         # halt new trades if down 5%
DAILY_PROFIT_TARGET_PCT=0.02        # lock gains after 2% of equity

# Position caps
MAX_OPEN_EQUITY_POSITIONS=15
MAX_OPEN_OPTIONS_POSITIONS=8

# Track switches
THESIS_TRACK_ENABLED=true
ORB_EQUITY_ENABLED=false            # disabled — backtest: -0.07% avg return
OPTIONS_TRACK_ENABLED=false         # ORB options — off
VOL_OPTIONS_TRACK_ENABLED=true      # short premium — on
SWING_TRACK_ENABLED=true
MACRO_NEWS_ENABLED=true

# News
FINNHUB_API_KEY=<key>               # finance-focused news (Finnhub free tier: 60 req/min)

# Gap scanner
GAP_SCAN_MIN_PCT=0.05               # 5% pre-market gap required
GAP_SCAN_MAX_CANDIDATES=5

# Kelly exploration
# When Kelly=0 and n<50 trades, system uses 1% floor to keep learning

# Learning loop safety
FREEZE_LESSON_INJECTION=true        # withhold lessons from prompts until candidate ledger proves edge

# Models
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_SUBAGENT_MODEL=claude-haiku-4-5-20251001
```

---

## Infrastructure

- **Hardware:** Raspberry Pi 5 (ARM64)
- **Process:** Two systemd services — `trading-engine-protection.service` (`Restart=always`, `RestartSec=10`) and `trading-engine-alpha.service` (`Restart=on-failure`, `RestartSec=30`, `BindsTo=trading-engine-protection.service`). The legacy single-process `main.py` / `trading-engine.service` still exists but is inactive.
- **Logs:** `logs/trading_engine_alpha.log`, `logs/trading_engine_protection.log`
- **Dashboard:** Streamlit (`dashboard/app.py`) — positions, P&L, agent signals, regime, Controls tab with manual scan triggers. Reads positions/events/run_history directly from SQLite, so it's agnostic to which plane wrote a given row.
- **Git:** `SiddharthUIgupta/trading-engine`, branch `master`

---

## Backtests

| Strategy | Trades | Win Rate | Avg Return | Notes |
|----------|--------|----------|------------|-------|
| ORB, no filters | 11,275 | 46.3% | -0.07% | Net negative — disabled |
| ORB + 2x volume | ~11,000 | 48% | +0.04% | Break-even pre-slippage, negative after |
| Thesis (zero slippage, current-constituent universe) | 1,370 | 57.9% | +6.27% | Survivorship-biased — universe was today's 503 S&P names applied retroactively |
| Thesis (0.5%/0.3% slippage, current-constituent universe) | 1,385 | 56.7% | +5.31% | Survivorship-biased, same issue |
| **Thesis (0.5%/0.3% slippage, point-in-time universe)** | **1,352 closed** | **54.5%** | **+4.08%**, PF 1.49 | **Current methodology — survivorship bias corrected** |

**Point-in-time (PIT) universe fix:** The current-constituent backtests above applied today's S&P 500 list retroactively — a stock that was later removed (bankruptcy, acquisition, delisting) never had a chance to generate a losing trade, inflating the win rate. `backtest/universe.py` now pulls per-ticker index membership start/end dates from `fja05680/sp500` (`get_pit_membership`), and `_walk_forward` in `backtest/thesis_backtest.py` skips signal generation on any bar where the ticker wasn't actually in the index that day. This expands the universe to 567 tickers (vs 503 current constituents) and lowers the win rate from 56.7% to 54.5% — the true, unbiased number. `run_thesis_backtest(..., use_pit_universe=True)` is now the default; pass `--no-pit` to `backtest/run_thesis_backtest.py` to reproduce the old biased numbers for comparison.

Thesis backtest methodology: 3 years, entry at next-day open, exits via stop-loss (18%) or trailing stop (activates at +20%, trails 10%). No LLM replay — tests the deterministic screen only.

---

## Realized Performance (as of 2026-07-02)

| Category | Trades | P&L |
|----------|--------|-----|
| Equity | 28 | ~-$1,367 |
| Options | 42 | -$8,647 |
| **Total** | **~70** | **~-$10,014** |

**Root cause of equity losses:** A silent `UnboundLocalError` in `_scan_and_run_consensus` caused every thesis BUY proposal to become a HOLD since the RobustCircuitBreaker was introduced. Fixed 2026-07-01. The 7 BUYs placed on 2026-07-02 (TREX, UBER, RIGL, IIIN, ENR, NCLH, BKNG) are effectively the first real thesis trades.

**Root cause of options losses:** ORB options track accumulated 30 simultaneous positions before a position cap was added. Signal resolves intraday; 30–45 DTE contracts bled theta with no edge. Track disabled.

**19 open options positions** expiring July 31 remain from the pre-fix period. Downside capped at zero; no action planned.
