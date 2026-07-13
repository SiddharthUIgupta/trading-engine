# Trading Engine — Claude Instructions

---

## Session Handoff — 2026-07-12 (PROTECTED)

This section is the starting context for new sessions. Read it before anything else.

### System overview

This is one of 5 integrated repos at `~/Projects/`. They are **not independent** — the goal is to wire them together into a single AI-driven trading loop:

```
claude-obsidian  →  Vibe-Trading  →  trading-engine  →  Telegram alerts
      ↑                                     ↓
      └──────── trade post-mortems ←─────────┘
```

| Repo | Path | Source | Role | Status |
|------|------|--------|------|--------|
| trading-engine | `~/Projects/trading-engine` | SiddharthUIgupta/trading-engine | Live executor (Pi) | Paper trading, STOPPED |
| Vibe-Trading | `~/Projects/Vibe-Trading` | HKUDS/Vibe-Trading | Research / backtest lab, 460 alpha factors | NOT configured (no API keys yet) |
| claudian | `~/Projects/claudian` | YishenTu/claudian | Claude agent framework | NOT wired yet |
| claude-obsidian | `~/Projects/claude-obsidian` | AgriciDaniel/claude-obsidian | Knowledge wiki (/autoresearch, BM25 retrieval) | NOT wired to trading |
| cpr-compress-preserve-resume | `~/Projects/cpr-compress-preserve-resume` | EliaAlberti/cpr-compress-preserve-resume | Session context preservation (/preserve /compress /resume) | Installed here (see `.claude/commands/`) |

### Phase 1 — COMPLETE (committed 2026-07-12, commit 8fafd44)

All changes are on `master`. Pull on Pi, then do the Pi-only steps below.

**What was fixed:**
| File | Change |
|------|--------|
| `pyproject.toml` | Added `vowpalwabbit>=9.11.2` (was missing → VW silently disabled on Pi) and `akshare>=1.0.0` (MacroSnapshot broken) |
| `execution_layer/protection_plane.py` | Fixed VW bandit: `learn()` now fires for **every** closed trade — was gated behind `if not agent_signals` early-return, so 0 examples were ever recorded |
| `config/settings.py` | `orb_equity_enabled` default `True→False` (PF 0.92, losing); `options_track_enabled` default `True→False` (30 losing positions); added `telegram_bot_token`, `telegram_chat_id`, `extra_watchlist_tickers` fields |
| `execution_layer/alerting.py` | Added `_send_telegram()` + `_broadcast()` — all 9 alert functions now send to Telegram (primary) + Gmail (optional) |
| `main_alpha.py` | Removed hardcoded `WATCHLIST` (CLAUDE.md violation); wired `GlobalRiskState` into all 4 Alpha Plane `RobustCircuitBreaker` instances (was missing, defence-in-depth gap vs Protection Plane) |
| `.env.example` | Added `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `EXTRA_WATCHLIST_TICKERS` |

**Pi-only steps still required (cannot be done remotely):**
```bash
# ── 0. Clone the other repos (only needed once) ─────────────────────────────
mkdir -p ~/Projects && cd ~/Projects
git clone https://github.com/HKUDS/Vibe-Trading.git
git clone https://github.com/YishenTu/claudian.git
git clone https://github.com/AgriciDaniel/claude-obsidian.git
git clone https://github.com/EliaAlberti/cpr-compress-preserve-resume.git

# ── 1. Pull trading-engine changes ──────────────────────────────────────────
cd ~/Projects/trading-engine && git pull

# ── 2. Add to .env (get bot token from @BotFather on Telegram) ─────────────
#    TELEGRAM_BOT_TOKEN=<token>
#    TELEGRAM_CHAT_ID=<your chat id>
#    EXTRA_WATCHLIST_TICKERS=SHAZ,NNBR,LAES,WULF,MARA   # Sid's manual watchlist

# ── 3. Install new deps ──────────────────────────────────────────────────────
.venv/bin/pip install vowpalwabbit==9.11.2 akshare

# ── 4. Bootstrap VW bandit from trade history (run once before restart) ─────
.venv/bin/python scripts/vw_warmup.py

# ── 5. Restart both services ─────────────────────────────────────────────────
sudo systemctl restart trading-engine-alpha trading-engine-protection
```

### Phase 2 — Next (NOT started)

Goal: connect the research layer so Vibe-Trading's 460 alpha factors feed the VW bandit as richer features.

1. **Configure Vibe-Trading**: copy API keys into `~/Projects/Vibe-Trading/.env` (Alpaca, Anthropic, Finnhub). Run `pip install -e .` in that repo.
2. **Factor provider bridge**: create `analyst_layer/factor_provider.py` — thin wrapper that calls `Registry().compute(alpha_id, panel)` from Vibe-Trading and returns a feature dict for the VW bandit's `_full_features()`.
3. **Enrich VW features**: in `vw_bandit.py`, call `factor_provider` in `_full_features()` to add e.g. `alpha101_001`, `gtja191_030`, `qlib158_vol` as VW namespace features.
4. **Run alpha bench on thesis universe**: use Vibe-Trading's backtest CLI to identify which of the 460 factors have Spearman IC > 0.03 on the thesis universe — these become the promoted feature set.
5. **VW `predict_full` → sizing**: `predict_full()` currently returns a win probability but the result is never used for sizing. Wire it: pass `vw_prob` to the Kelly sizing formula in `alpha_plane.py` as a multiplier cap (e.g. `kelly_size * min(1.0, vw_prob / 0.55)`).

### Phase 3 — Future

Goal: close the knowledge loop — trade outcomes feed back into the research base.

1. **Trading vault in claude-obsidian**: create `~/Projects/claude-obsidian/vaults/trading/` with strategy pages and a post-mortem template.
2. **Auto post-mortem ingest**: after `_run_reflection()` completes, write a markdown post-mortem to the vault (ticker, entry/exit, regime, what the agents said, P&L). Claude-obsidian's BM25 index will pick it up.
3. **Wiki-retrieve in Risk Officer prompt**: before consensus, call `scripts/retrieve.py "ticker sector regime"` from the claude-obsidian repo and prepend the top-3 results to the Risk Officer system prompt. This gives the LLM memory of prior trades in the same setup.

### Key architecture facts (do not re-derive without reading the code)

- **Only thesis track has a proven backtest edge** (PF 1.49, 58% WR). ORB equity is disabled (PF 0.92). Options disabled (30 losing positions). Thesis + swing + recovery are the active tracks.
- **VW bandit**: `analyst_layer/vw_bandit.py`. `_MIN_EXAMPLES = 20` — predictions suppressed below this count. `warm_start()` bootstraps from DB on first run. After Phase 1 fix + warmup script, should finally accumulate examples.
- **Lesson store**: frozen (`FREEZE_LESSON_INJECTION=True` in settings) until 100 closed trades on fixed config. Do not unfreeze.
- **GlobalRiskState**: DB-level halt for weekly/trailing drawdown. Now wired in both planes (as of Phase 1). The weekly halt resets on Monday; trailing requires manual `reset()`.
- **Two-plane IPC**: Alpha writes `order_intents` table → Protection reads and executes. Never bypass with direct broker calls from Alpha. Never import scanners/LLM into Protection.
- **CLAUDE.md invariant**: no hardcoded ticker lists — universe from OpenBB screens. Extra tickers go in `EXTRA_WATCHLIST_TICKERS` env var.

---

## The most important rule

**If the user says something, do exactly that. Never assume you know better. If anything is unclear, ask a question before writing a single line of code.**

Examples of when to ask instead of assume:
- User says "integrate X" → ask: where should it plug in? what should it replace or supplement?
- User says "fix the universe" → ask: what should the new universe be? screens only, or something else?
- User says "look into X" → ask: do you want me to install it, just research it, or build something with it?
- User says "make it better" → ask: better in what specific way?

Never interpret a vague instruction as permission to redesign, refactor, or add features the user didn't ask for.

---

## Confirm before acting

- **Never implement from a summary, overview, or "take a look at" prompt.** Those are research requests, not build requests.
- Always wait for an explicit go-ahead ("yes", "do it", "build it", "start now") before writing code.
- If the user approves approach A, do not silently implement approach B because you think it's better.

---

## Risk invariants — violating any of these is a critical bug, not a style choice

Each of these was learned from a real production defect. Any diff that weakens one must be called out explicitly and confirmed before writing it.

1. **Breakers gate ENTRIES only. Exits are unconditional.** No halted/tripped/profit-locked state may ever block, skip, or delay a stop-loss or exit check.
2. **Risk guards are monotonic.** No signal — LLM sentiment, news score, anything — may ever LOOSEN a guard (lower effective VIX, raise a cap, re-arm a track). Tighten or no-op only.
3. **Profit lock = no new entries. Never liquidation.** The thesis edge is winners running for weeks; any rule that force-flattens the book is a different, untested strategy.
4. **Never submit a software SELL on a bracketed position without cancelling the resting stop leg first** (`_release_bracket_stop`). Alpaca reserves the shares against the open stop.
5. **A live rule must exist in the backtest, and a backtested rule must exist live.** If a diff adds an execution-time rule (lock, filter, exit) with no backtest counterpart, say so.
6. **No strategy may be armed in config without a linked, passing point-in-time backtest artifact.** Current-constituent universes are banned for anything that buys drawdowns.
7. **No `except Exception` around decision paths.** Adapters return Ok/Empty/Failed; Empty and Failed are different states and Failed alerts. Silence is an alarm, not a default.
8. **Two-process rules:** all Alpha↔Protection IPC goes through DB tables; any state written on an event (trip, lock) must have a write-back/sync path that resets it, or it is a permanent silent halt; SQLite stays in WAL with a busy timeout; Protection never imports scanners, scrapers, or LLM clients.

---

## Financial methodology — when touching strategy, sizing, or backtest code

- Point-in-time membership always; report skipped-ticker counts so residual survivorship is visible.
- No parameter tuning, lesson injection, or Kelly changes on n < 100 closed trades on a frozen config. Log, don't act.
- Slippage assumptions must be justified against the actual live universe (small caps ≠ S&P slippage).
- The candidate ledger is the measurement instrument: every screened candidate gets a row whether traded or not. Never bypass it.
- LLM layers are advisory until the ledger shows uplift of approved vs rejected candidates. "Shadow mode" is the default for anything new.

### Signal lifecycle

A signal (any `analyst_layer/shadow_signals.py` provider) is always in exactly one of three states — no permanent shadow residents:

1. **Shadow** — logged to `signal_values` for every candidate, influences nothing. Default state for anything new.
2. **Promoted** — used in the risk gate. Requires an explicit, separate, approved task — never bundled into the same change that adds the signal.
3. **Deleted** — removed once `scripts/signal_uplift.py` shows no edge at n>=300.

`scripts/signal_uplift.py`'s verdict at n>=300 is authoritative: `PROMOTE-CANDIDATE` or `DELETE-CANDIDATE`. Below n=300, "INSUFFICIENT SAMPLE" — no conclusions, no promotion, no deletion.

Current-snapshot-only signals (no free point-in-time history — e.g. `short_interest`, which only has a ~20-day-stale "current" settlement snapshot, not a queryable historical series like price-based signals) report `median_staleness_days` alongside their verdict. The n>=300 gate is necessary but not sufficient for these — any `PROMOTE-CANDIDATE` verdict with non-zero staleness needs manual review before it's trusted, since the measurement may have leaked future information into the correlation.

---

## Evidence protocol — how to make claims

- Tag claims: **Certain** (read the code / ran it), **[Likely]** (strong inference), **[Guessing]** (filling gaps). If a reply is mostly guessing, say so first.
- **Master rule: a [Guessing] tag on any external or verifiable fact is illegal — it converts into a tool call.** Guessing is legal only for the genuinely unknowable (future returns, behavior that verification failed to settle).
- Never claim behavior of code you haven't opened this session. Cite file and line for bug claims.

### Verification gates — you may not say it until you've done it

| Before asserting…                                   | Required action this session                          |
|-----------------------------------------------------|-------------------------------------------------------|
| What THIS repo does (any behavior claim)            | Open the file. Cite file:line. Never memory, never docs — ARCHITECTURE.md has lied before. |
| Any Alpaca/broker semantic (order classes, cancels, replace, options levels, rate limits) | Fetch docs.alpaca.markets and/or run it against the paper API. Memory of broker APIs is stale by construction. |
| A library works here                                | Install + import on this machine (aarch64). "It should work" is banned — ARM wheels have burned us. |
| Any market fact (price, halt, split, earnings date, corporate action) | Live fetch via the system's own data adapters first, ad-hoc search second. Training data has no date on it. |
| Any number about the strategy (win rate, PF, exposure, P&L) | Query the DB / candidate ledger / run the backtest. Never estimate a number that is sitting in SQLite. |
| Why production did X                                | `journalctl -u trading-engine-*`, the events table, and the DB — logs before code. Code says what CAN happen; logs say what DID. |
| A fix works                                         | Run the test, revert the fix, confirm red, restore, confirm green. |

### Escalation ladder — cheapest sufficient tool, stop at the first that answers

1. grep / read the repo — internal behavior questions end here
2. run code locally (REPL, pytest, scripts/) — running beats reasoning about running
3. query state (sqlite3, journalctl, heartbeat.json) — production questions end here
4. paper Alpaca API call — broker semantics end here; NEVER verify against live
5. web search — external, current, versioned things
6. web fetch primary source — docs > snippets > blogs
7. ask Sid — only for intent/preference; never for anything tools 1–6 can answer

Escalate only when the current rung can't answer. Three failed attempts on one rung = stop and report what you tried; don't thrash.

### Financial opinions require a data pull first

Any financial judgment — sizing, strategy change, "should we exit X", "is this track working" — must be preceded by pulling: current regime (DB), open positions (broker), relevant candidate-ledger stats, and realized P&L for the bucket. An opinion without those four is speculation and must be labeled as such in its first line. Sample-size gate applies to advice too: no recommendation from n < 100 closed trades on frozen config — say "insufficient sample" instead of extrapolating.

When a bucket has a `strategy_version` split (currently: options, via `CURRENT_OPTIONS_STRATEGY_VERSION` in `state_store.py`), pull P&L split by version. Only the current version's stats are decision-relevant. Older versions may be shown for context but must be labeled "prior strategy — not evidence about current agents" and never cited for or against a new trade.

---

## Proving work is actually done

- Show actual running output — never "I've integrated it". `python -c "import X; print(X.__version__)"`, a real call with real output, and `python -m pytest tests/ -q` passing.
- **A regression test must fail when the fix is reverted.** Revert, run, confirm red, restore, confirm green. A test that passes either way is decoration.
- **Every new seam gets a test the day it's born** — new IPC table, new broker call pattern, new process boundary. "All N tests pass" means nothing about code the N tests never touch.
- If a change affects Protection-plane behavior, state whether the paper fire drill (bracketed entry → forced exit → `kill -9` alpha → stop survives at broker) needs re-running.

---

## Architecture constraints

Two-plane, three-layer. Imports flow one direction:

```
data_layer → analyst_layer → execution_layer
Alpha Plane (alpha_plane.py):           scans, LLM, risk gate → writes order_intents
Protection Plane (protection_plane.py): consumes intents, brackets, exits, reconcile
IPC: SQLite (WAL) — order_intents, breaker_state, events
```

- `data_layer` never imports from `analyst_layer` or `execution_layer`
- `analyst_layer` never imports from `execution_layer`
- Protection Plane stays boring: no scrapers, no scanners, no LLM calls except the exit-escalation agent.

Circuit breaker limits (never change without explicit instruction):
- Intraday (momentum/ORB equity/news): 35% daily drawdown cap
- Options (ORB options/vol): 35% daily drawdown cap
- Thesis/recovery: 10% daily drawdown cap
- Swing: 20% daily drawdown cap

---

## What not to do

- Do not add features, refactors, or abstractions beyond what was asked
- Do not add comments that explain what code does — only add a comment if the WHY is non-obvious
- Do not create markdown files or documentation unless explicitly asked
- Do not push to GitHub unless explicitly told to
- Do not restart the trading engine services unless explicitly told to
- Do not add error handling for scenarios that can't happen
- Do not add backwards-compatibility shims for removed code
- Do not hardcode ticker lists anywhere — universe comes from live discovery screens (all 5 OpenBB screens, market-wide); empty screen → warn and skip, never fall back to a list

---

## Stack / environment

- Raspberry Pi 5, aarch64 — some packages (e.g. pyqlib) do not install on ARM, check before promising
- Python 3.13, virtualenv at `.venv/`
- Broker: Alpaca (paper + live), Level 3 options enabled
- DB: `state/trading_engine.sqlite3` (WAL — `-wal`/`-shm` sidecars are normal; checkpoint before file-copy backups)
- Logs: `logs/trading_engine.log`
- Services: `trading-engine-alpha`, `trading-engine-protection` (systemd)
- Key libraries: vowpalwabbit 9.11.2, backtrader 1.9.78, akshare 1.18.64, openbb, yfinance, anthropic

---

## Before marking any task done

1. Run the relevant code and show the output
2. Run `python -m pytest tests/ -q` — must still pass (407 as of 2026-07-03; update this number when it changes)
3. If it touches imports, run `python -c "from module import thing"` to confirm no import errors
4. If it touches exit paths, breakers, brackets, or IPC: confirm `tests/test_exit_priority.py` and `tests/test_protection_plane.py` still pass and state which invariant (1–8 above) the change interacts with
