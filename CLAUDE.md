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

## Proving work is actually done

Whenever a library is installed or integrated, **show actual running output** — not just "I've integrated it". Required proof:
- `python -c "import X; print(X.__version__)"` — library is installed
- A real function call with real output — the integration actually works
- `python -m pytest tests/ -q` passes — nothing is broken

**Never describe an integration as complete if it only exists in comments.** If qlib formulas are referenced in a comment but the library is not imported, that is NOT integration.

---

## Universe rules — strict

- **Never hardcode a list of tickers anywhere in the codebase.** Not as a fallback, not as a default, not "temporarily".
- The thesis/swing universe must always come from live discovery screens (OpenBB gainers, active, losers, undervalued_growth, aggressive_small_caps — all 5, market-wide).
- "Consider everything" means all market caps — not just small caps, not just large caps.
- If a screen returns no results, log a warning and skip. Do not fall back to a hardcoded list.

---

## Architecture constraints

Three-layer architecture — imports only flow one direction:

```
data_layer → analyst_layer → execution_layer
```

- `data_layer` never imports from `analyst_layer` or `execution_layer`
- `analyst_layer` never imports from `execution_layer`
- `execution_layer/runtime.py` is the only module that imports from all three layers

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
- Do not restart the trading engine unless explicitly told to
- Do not add error handling for scenarios that can't happen
- Do not add backwards-compatibility shims for removed code

---

## Stack / environment

- Raspberry Pi 5, aarch64 — some packages (e.g. pyqlib) do not install on ARM, check before promising
- Python 3.13, virtualenv at `.venv/`
- Broker: Alpaca (paper + live), Level 3 options enabled
- DB: `state/trading_engine.sqlite3`
- Logs: `logs/trading_engine.log`
- Service: `trading-engine` (systemd)
- Key libraries: vowpalwabbit 9.11.2, backtrader 1.9.78, akshare 1.18.64, openbb, yfinance, anthropic

---

## Before marking any task done

1. Run the relevant code and show the output
2. Run `python -m pytest tests/ -q` — must still pass (currently 390 tests)
3. If it touches imports, run `python -c "from module import thing"` to confirm no import errors
