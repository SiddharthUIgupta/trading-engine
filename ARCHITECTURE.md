# Trading Engine — Architecture & Blueprint

## What It Is

An autonomous trading system running 24/7 on a Raspberry Pi. It scans markets each morning, decides what to buy using a multi-agent LLM consensus, manages open positions intraday, closes them according to rule-based exits, and learns from each outcome to improve future decisions. No human needs to approve trades — the system is self-contained from signal to execution to reflection.

Broker: Alpaca (live account, Level 3 options). LLM: Anthropic Claude (Sonnet for risk decisions, Haiku for sub-agents). Market data: OpenBB + Yahoo Finance + Finnhub.

---

## System Map

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│           Boots settings, wires all layers,                 │
│           starts APScheduler, blocks forever.               │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     execution_layer/          │
        │     runtime.py                │  ← the brain
        │     scheduler.py              │  ← the clock
        │     state_store.py            │  ← SQLite wrapper
        │     broker.py                 │  ← Alpaca wrapper
        │     guardrails.py             │  ← circuit breaker
        │     exit_rules.py             │  ← stop/trail logic
        │     tax_compliance.py         │  ← wash-sale guard
        │     manual_trigger.py         │  ← dashboard → engine IPC
        └───────────────┬───────────────┘
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

---

## Daily Schedule (Eastern Time)

| Time | Job | What Happens |
|------|-----|--------------|
| 8:00am | `pre_market_scan` | Fetches daily regime (VIX + SPY SMAs). Arms or disarms each strategy track for the day. Runs macro news agent (Finnhub general/merger feeds → Haiku sentiment scoring). |
| 8:15am | `thesis_scan_and_trade` | Scans OpenBB's undervalued-growth universe (5 screens, market-wide) for 20–50% pullbacks. Top 20 pass to LLM consensus. |
| 9:05am | `gap_scan_and_queue` | Scans akshare US movers + watchlist for pre-market gaps ≥5%. Top 5 candidates pass directly to thesis consensus, bypassing the pullback screen. |
| 9:30am | `market_open_execution` | Fires pending BUY payloads from thesis/gap scans approved pre-market. |
| 9:00–3:00pm every 15min | `intraday_monitoring` | Reconciles positions vs broker, checks stops/trailing stops, checks options exits. |
| 9:00–3:00pm every 30min | `momentum_scan_and_trade` | ORB equity disabled. Job still runs but exits immediately (`ORB_EQUITY_ENABLED=false`). |
| 9:45am | `swing_scan_and_trade` | Scans for swing trade setups (3–6 week holds). |
| 10:00am + 1:00pm | `vol_options_scan_and_trade` | Short premium (iron condors, strangles) when VIX/IV conditions suit. |
| Every 15s | `manual_trigger_watcher` | Polls `state/manual_trigger.json` for dashboard-triggered scans. Dispatches immediately when found. |
| 3:30pm | `pre_close_orb_exit` | Closes any residual ORB position flat or losing (legacy guard, ORB now disabled). |
| 4:30pm | `post_market_logging` | Logs cost summary, positions, daily P&L to the log file. |

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

# Models
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_SUBAGENT_MODEL=claude-haiku-4-5-20251001
```

---

## Infrastructure

- **Hardware:** Raspberry Pi 5 (ARM64)
- **Process:** `nohup python main.py &` — single process, APScheduler manages all timing
- **Logs:** `logs/trading_engine.log`
- **Dashboard:** Streamlit (`dashboard/app.py`) — positions, P&L, agent signals, regime, Controls tab with manual scan triggers
- **Git:** `SiddharthUIgupta/trading-engine`, branch `master`

---

## Backtests

| Strategy | Trades | Win Rate | Avg Return | Notes |
|----------|--------|----------|------------|-------|
| ORB, no filters | 11,275 | 46.3% | -0.07% | Net negative — disabled |
| ORB + 2x volume | ~11,000 | 48% | +0.04% | Break-even pre-slippage, negative after |
| **Thesis (zero slippage)** | **1,370** | **57.9%** | **+6.27%** | **Proven edge** |
| **Thesis (0.5%/0.3% slippage)** | **1,385** | **56.7%** | **+5.31%** | **Edge survives realistic execution costs** |

Thesis backtest methodology: S&P 500 universe, 3 years, entry at next-day open, exits via stop-loss (18%) or trailing stop (activates at +20%, trails 10%). No LLM replay — tests the deterministic screen only.

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
