# Trading Engine — Claude Instructions

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
