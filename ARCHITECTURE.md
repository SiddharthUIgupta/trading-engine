# Trading Engine — Architecture & Blueprint

## What It Is

An autonomous trading system running 24/7 on a Raspberry Pi. It scans markets each morning, decides what to buy using a multi-agent LLM consensus, manages open positions intraday, closes them according to rule-based exits, and learns from each outcome to improve future decisions. No human needs to approve trades — the system is self-contained from signal to execution to reflection.

Broker: Alpaca (live account, Level 3 options). LLM: Anthropic Claude (Sonnet for risk decisions, Haiku for sub-agents). Market data: OpenBB + Yahoo Finance.

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
        │     runtime.py (2,200 lines)  │  ← the brain
        │     scheduler.py              │  ← the clock
        │     state_store.py            │  ← SQLite wrapper
        │     broker.py                 │  ← Alpaca wrapper
        │     guardrails.py             │  ← circuit breaker
        │     exit_rules.py             │  ← stop/trail logic
        │     tax_compliance.py         │  ← wash-sale guard
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     analyst_layer/            │
        │     graph.py                  │  ← LLM consensus engine
        │     agents/ (8 agents)        │  ← specialist LLM callers
        │     market_regime.py          │  ← daily strategy arm/disarm
        │     orb_scanner.py            │  ← breakout signal
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
        │     broker.py                 │  ← position/order queries
        │     google_news.py            │  ← headline sentiment
        │     models.py                 │  ← PriceBar, PriceSeries, etc.
        └───────────────────────────────┘
```

---

## Daily Schedule (Eastern Time)

| Time | Job | What Happens |
|------|-----|--------------|
| 8:00am | `pre_market_scan` | Fetches daily regime (VIX + SPY SMAs). Arms or disarms each strategy track for the day. |
| 8:15am | `thesis_scan_and_trade` | Scans OpenBB's undervalued-growth universe for 20-50% pullbacks. Top 20 pass to LLM consensus. |
| 9:30am | `market_open_execution` | Fires pending BUY payloads from the thesis scan that were approved pre-market. |
| 9:00–3:00pm every 15min | `intraday_monitoring` | Reconciles positions vs broker, checks stops/trailing stops, checks options exits. |
| 9:00–3:00pm every 30min | `momentum_scan_and_trade` | Scans active/gainers/losers universe for ORB signals with gap + volume + SPY filters. |
| 3:30pm | `pre_close_orb_exit` | Closes any ORB equity position that is flat or losing. ORB is intraday — no overnight holds on failed breakouts. |
| 4:30pm | `post_market_logging` | Logs cost summary, positions, daily P&L to the log file. |

Options scan (`options_scan_and_trade`) is registered but disabled by default — see Strategy section.

---

## Strategy Tracks

### 1. Thesis Pullback (PRIMARY — ENABLED)

**What it is:** Buys quality stocks that have pulled back 20–50% from their 52-week high. The thesis is that a fundamentally sound company in temporary dislocation will revert. This is a multi-day to multi-week hold.

**Signal pipeline:**
1. OpenBB `aggressive_small_caps` + `undervalued_growth` screens → raw candidates
2. `thesis_scanner.evaluate_thesis_candidate` filters for 20–50% drawdown from 52-week high
3. `thesis_scanner.evaluate_shrink_volume_pullback` checks for MA5 > MA10 > MA20 with low-volume consolidation (score boost if confirmed, not required)
4. Top 20 by score pass to the 4-agent LLM consensus
5. Approved BUYs execute at market open

**Exit rules:**
- Stop loss: 18% below entry
- Trailing stop: activates at +20% gain, trails 10% behind peak
- No fixed take-profit — let winners run

**Backtest results:** 58% win rate, +6.27% average return per trade, +8,670% cumulative across 1,383 trades.

**Why it works:** Pulls back from 52-week high = stock with proven demand history, temporarily oversold. The LLM consensus filters for quality (balance sheet, sentiment, no earnings landmines). The wide stop lets multi-week moves develop instead of shaking out on intraday noise.

---

### 2. ORB Equity (SECONDARY — ENABLED)

**What it is:** Opening Range Breakout. Defines the high/low of the first 15 minutes of trading, then buys when price closes a 5-minute bar above that range. Intraday trade only — closes same day at 3:30pm if flat or losing.

**Quality filters (all must pass):**
- Gap ≥ 4% from prior close — stocks with pre-market catalyst, not random noise
- Volume ≥ 2x the opening range average — institutional participation, not retail
- SPY must be green on the day — long breakouts fail significantly more on red-tape days

**Signal pipeline:**
1. Momentum scan produces mover candidates (OpenBB active/gainers/losers)
2. Fetch SPY intraday once — skip all ORB longs if SPY is red
3. For each candidate: fetch daily closes (for gap calc) + 5m intraday bars
4. `orb_scanner.evaluate_orb` — checks all three filters, returns long/short/none
5. Long signal → immediate order, no LLM (deterministic execution)

**Exit rules:**
- 3:30pm EOD exit for any position at or below entry price
- Stop loss: opening range low
- Target: entry + 2× risk (2:1 R/R)
- Trailing stop activates at +15% gain, trails 7% behind peak

**Backtest results:** With 1.5x volume filter: 48% win rate, +0.04% average return. Marginal positive expectancy. Without filters: -0.07% average, destroys capital.

**Important limitation:** ORB does NOT go through LLM consensus — it's a deterministic rule-based entry. This means it generates no `run_history` entries and the reflection/lesson loop does not fire on ORB trades.

---

### 3. ORB Options (DISABLED — `OPTIONS_TRACK_ENABLED=false`)

**What it was:** Same ORB signal as above but expressed as a call (long signal) or put (short signal) instead of shares. 30–45 DTE contracts.

**Why disabled:** The signal resolves intraday. Paying 30–45 days of theta for a trade that either works in 2 hours or doesn't work at all is a structural mismatch. The system accumulated 30 simultaneous positions before a position cap was added, generating $8,647 in realized losses and ~$20K in unrealized losses.

**Re-enable:** Set `OPTIONS_TRACK_ENABLED=true` in `.env`. Not recommended without backtesting first.

---

### 4. Vol / Premium Selling (DISABLED — `VOL_OPTIONS_TRACK_ENABLED=false`)

**What it is:** Sells short premium (iron condors, strangles, spreads) when IV Rank is elevated. Based on Natenberg/tastylive framework. Requires Level 3 options approval.

**Arm conditions:** VIX 18–40, not spiking (>15% rise in a week while above 25), `VOL_OPTIONS_TRACK_ENABLED=true` in `.env`.

**Why disabled by default:** Undefined risk on uncovered legs. Requires explicit opt-in.

---

## LLM Consensus Engine

Used by the thesis track. This is where the intelligence lives.

### The 4-Agent Pipeline

```
Ticker + market data
        │
        ├──► MacroSentimentAgent  (Haiku)
        │    Reads recent headlines via Google News + sentiment lexicon.
        │    Returns: stance (BUY/HOLD/SELL), confidence, rationale.
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

All three sub-agents run in parallel via LangGraph. The Risk Officer runs after. Output is a `ConsensusPayload` stored in `run_history`.

### Kelly Position Sizing

```
f = 0.5 × (p − q/b)
```

Where `p` = historical win rate, `q` = 1-p, `b` = avg_win / avg_loss. Requires minimum 15 closed trades before trusting the estimate. Below that threshold, uses a conservative fixed fraction. Hard-capped at `MAX_POSITION_SIZE_PCT` regardless.

### Correlation Guard

Before the consensus runs, the system computes correlation between the proposed ticker's daily closes and all currently held positions. High correlation → Kelly fraction is reduced. If max correlation exceeds 0.85, the trade is blocked entirely (prevents the portfolio from becoming a single-factor bet).

---

## Daily Regime Assessment

Runs at 8:00am. Pure arithmetic — no LLM. Determines which tracks are armed for the day.

**VIX smoothing:** All threshold decisions use a 5-day VIX moving average, not spot VIX. Prevents daily flipping when VIX sits at 29.8 vs 30.2 on adjacent days.

**SPY trend:** SMA10 vs SMA30 with 1% minimum separation. Under 1% = neutral (not directional enough).

| Condition | Effect |
|-----------|--------|
| VIX(5d avg) > 30 + bearish market | ORB equity DISARMED |
| SPY SMA10/SMA30 within 1% | ORB options DISARMED |
| VIX(5d avg) > 30 | Thesis DISARMED |
| VIX(5d avg) < 18 | Vol/premium DISARMED |
| VIX spiking (>15% in a week while above 25) | Vol/premium DISARMED |
| VIX(5d avg) > 40 | Vol/premium DISARMED |

---

## Risk Management

### Circuit Breaker (`guardrails.py`)

Central kill switch. Trips on:
- **Drawdown breach:** portfolio down ≥ 5% from day-start equity → all new stock trades blocked
- **Profit lock:** portfolio up ≥ daily profit target → no new entries (lock in the gain)
- **Options halt:** separate from stock halt; options tracks have their own breakers

Resets at midnight each trading day.

### Per-Position Exit Rules

Applied every 15 minutes during intraday monitoring:

| Rule | Equity ORB | Thesis |
|------|-----------|--------|
| Stop loss | 7% below entry | 18% below entry |
| Trailing activation | +15% gain | +20% gain |
| Trailing distance | 7% behind peak | 10% behind peak |
| EOD close if flat/losing | Yes (3:30pm) | No (multi-day hold) |

### Options Exit Rules

| Rule | Threshold |
|------|-----------|
| Intraday stop (same-day entry, after 3pm ET) | Down ≥ 20% |
| Stop loss | Down ≥ 50% |
| Force close | ≤ 7 DTE remaining |

### Position Caps

- Max concurrent equity positions: 15
- Max concurrent options positions: 8 (applies if re-enabled)

### Wash Sale Guard

Blocks a BUY on any ticker sold at a loss within the past 30 days. Warns before any SELL that would create a wash sale violation.

---

## Learning Loop

The system is designed to improve over time. Currently active only for thesis trades (ORB trades bypass the consensus and generate no run_history).

### 1. Agent Signal Scoring

Every BUY consensus run records each sub-agent's stance and confidence in `agent_signal_log`. After the trade closes, the outcome is written back. Over time this reveals which agents are actually predictive, surfaced via `get_agent_accuracy(track, regime)`.

### 2. Lesson Journal

After each trade closes, `ReflectionAgent` (Sonnet) generates structured lessons tagged with `setup_tags` (bull_regime, high_rsi, volume_spike, gap_up, etc.). Lessons carry a score: starts at 1.0, +0.1 per win, -0.05 per loss, retired below 0.3 (~14 net losses).

Before the next consensus run for matching conditions, relevant lessons are retrieved and injected into the agent prompts as context.

### 3. Post-Trade Reflection

`ReflectionAgent` produces a post-mortem after each trade with:
- `what_happened`: what the signals said vs. what the price did
- `root_cause`: why the trade won or lost
- `outcome_was_noise`: random vs. real signal failure
- `lessons`: extracted lessons fed into the journal

The reflection receives full market context: regime at entry, entry price, shares, risk verdict, risk reasons, entry date, and all three sub-agent rationales.

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
DAILY_PROFIT_TARGET_USD=50.0        # lock gains after $50 profit

# Position caps
MAX_OPEN_EQUITY_POSITIONS=15
MAX_OPEN_OPTIONS_POSITIONS=8

# ORB quality filters
ORB_MIN_GAP_PCT=0.04                # 4% pre-market gap required
ORB_VOLUME_CONFIRMATION_MULTIPLE=2.0
ORB_REQUIRE_SPY_POSITIVE=true

# Thesis
THESIS_MAX_DAILY_CANDIDATES=20

# Track switches
THESIS_TRACK_ENABLED=true
OPTIONS_TRACK_ENABLED=false         # ORB options — off
VOL_OPTIONS_TRACK_ENABLED=false     # short premium — off

# Models
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_SUBAGENT_MODEL=claude-haiku-4-5-20251001
```

---

## Infrastructure

- **Hardware:** Raspberry Pi (ARM64)
- **Process:** `nohup python main.py &` — single process, APScheduler manages all timing
- **Logs:** `logs/trading_engine.log`
- **Dashboard:** Streamlit (`dashboard/app.py`) — positions, P&L, agent signals, regime. Floating orange Refresh button bottom-right.
- **Git:** `SiddharthUIgupta/trading-engine`, branch `master`

---

## Backtests

| Strategy | Result |
|----------|--------|
| ORB, no filters | 46% win rate, -0.07% avg — net negative |
| ORB + 1.5x volume | 48% win rate, +0.04% avg — marginally positive |
| **Thesis pullback** | **58% win rate, +6.27% avg — the proven edge** |

ORB backtest exit breakdown (long trades with vol filter): 2,371 EOD closes, 492 stops, 129 target hits. The 3.8:1 stop-to-target ratio is why quality filters matter — fewer bad entries = fewer stops.

---

## Realized Performance (as of 2026-06-29)

| Category | Trades | P&L |
|----------|--------|-----|
| Equity | 21 | -$1,367 |
| Options | 42 | -$8,647 |
| **Total** | **63** | **-$10,014** |

LLM API spend to date: ~$1.79 (Haiku sub-agents + Sonnet risk officer).

**Root cause of losses:** Before position caps and ORB filters were added, the system placed 57 pending equity orders exhausting buying power, and accumulated 30 simultaneous long options positions on ORB signals with no same-day exit. The options losses dominate because the DTE (30–45 days) was mismatched to the signal (intraday). These issues are now fixed.

**30 open options positions** expiring July 31 remain from the pre-fix period. No action planned — downside is capped at zero, a few may recover.
