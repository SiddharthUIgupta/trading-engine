# Architecture

Three layers, one direction of dependency: `data_layer -> analyst_layer -> execution_layer`.
`analyst_layer` imports `data_layer` (for the Pydantic contracts it consumes).
`execution_layer` imports both (it's the only module — `execution_layer/runtime.py`
— allowed to import from all three). Neither `data_layer` nor `analyst_layer`
ever imports from `execution_layer`. This means the broker, the circuit
breaker, and the state store can change completely without either upstream
layer noticing.

## Data flow, boundary by boundary

```
OpenBB providers (yfinance, polygon, fmp, benzinga, sec, ...)
        │
        ▼
data_layer/openbb_client.py   — the ONLY module that imports `openbb`
        │  validates every field against data_layer/models.py
        │  (PriceBar, OrderBookSnapshot, SentimentSnapshot,
        │   FilingSummary, FundamentalsSnapshot)
        │  raises DataValidationError / ProviderFetchError on any
        │  malformed or missing field — nothing downstream ever sees
        │  a raw DataFrame or an OBBject
        ▼
analyst_layer/prefilter.py::evaluate_ticker()   — deterministic, zero-LLM
        │  RSI band, SMA-crossover flip, volume spike, |sentiment| threshold,
        │  recent 8-K — pure arithmetic over the data above, no model call
        │  FilterSignal.passed == False → ticker is skipped entirely this
        │  cycle (logged, never reaches the agents). This is what keeps
        │  Claude spend tied to opportunity count, not watchlist size.
        ▼  (only for tickers that passed)
analyst_layer/graph.py        — LangGraph StateGraph, ConsensusState
        │
        ├─ macro_sentiment_agent   ─┐
        ├─ fundamental_sec_agent   ─┼─ fan-out from START, run independently,
        ├─ technical_analysis_agent┘  blind to each other's output
        │       each agent's Claude call is forced through a single
        │       tool_choice={"type":"tool", ...} — there is no path for
        │       the model to answer in free text (analyst_layer/agents/base.py)
        │       each result is validated into AgentSignal before merging
        │       into ConsensusState.signals (Annotated[..., operator.add])
        ▼
        risk_compliance_officer_agent   — fan-in, runs only after all
        three signals (or their failures) have landed
            1. LLM drafts a TradeProposal from the signals (still just a draft)
            2. analyst_layer/agents/risk_officer_agent.py::_clamp_to_limits
               deterministically re-derives max_notional = equity * MAX_POSITION_SIZE_PCT
               and clamps/rejects the draft in plain Python — the LLM's
               own quantity is never trusted past this point
        ▼
ConsensusPayload                — risk_review.verdict ∈ {approved, amended, rejected}
        │  is_executable == (verdict == APPROVED and proposal.action != HOLD)
        │  schema-enforced invariant: zero signals can never carry APPROVED
        ▼
execution_layer/runtime.py::TradingRuntime   — the only cross-layer module
        │
        ├─ pre_market_scan()        for each watchlist ticker: pull data,
        │                           run prefilter.evaluate_ticker() — only
        │                           on a PASS does it call run_consensus();
        │                           stash pending payloads, record_run() to
        │                           StateStore, log the filter-pass rate
        │
        ├─ market_open_execution()  for each pending payload:
        │     - skip if not is_executable
        │     - WashSaleGuard.check_before_buy()  ← BUY only. Blocks the
        │       trade outright if a loss on this ticker was realized
        │       within the lookback window (default 30 days) — buying
        │       back now would disallow that loss for tax purposes.
        │       WashSaleGuard.warn_before_sell()  ← SELL only. Logs a
        │       warning (does not block) if this sale would itself be a
        │       wash sale against a recent buy of the same ticker.
        │     - CircuitBreaker.validate_position_size()  ← SECOND,
        │       independent re-check of the same position-size limit
        │       the risk officer already applied. A bug or
        │       prompt-injection in Layer 2 still cannot reach the
        │       broker without passing this gate too.
        │     - AlpacaBroker.submit_order(proposal)  — paper-forced
        │       (see below), accepts ONLY the typed TradeProposal,
        │       never a string
        │     - upsert_position() into StateStore (BUY also stamps
        │       last_buy_at; SELL also calls record_realized_sale(),
        │       which is what WashSaleGuard reads on the next BUY)
        │
        ├─ intraday_monitoring()    polls equity every 15 min;
        │     CircuitBreaker.check_drawdown() / check_profit_target()
        │     compare against the day's starting equity; on breach:
        │       execute_global_shutdown() → broker.close_all_positions()
        │       + halt_callback() → scheduler.pause() (wired in main.py)
        │     this is cross-trade state the analyst layer has no
        │     visibility into, so it can only live here
        │     then _check_intraday_exits() — per held position, no LLM
        │     by default (see "Exit checks" below)
        │
        └─ post_market_logging()    summarizes positions/runs/breaker
                                     state and Claude cost into StateStore
```

## Filter-first: why the agents don't see every ticker

`analyst_layer/prefilter.py` runs before any LLM call, on every ticker, every
cycle. No model in the loop — RSI band, SMA-crossover flip, volume spike vs.
trailing average, `|sentiment.score|` threshold, and a recent-8-K check (the
closest proxy to "earnings surprise" available without an analyst EPS
estimate to diff against). `evaluate_ticker()` returns a `FilterSignal` with
`passed: bool` and the `reasons` that tripped it; `pre_market_scan()` skips
`run_consensus()` entirely on a `False`, and logs the day's pass rate. This
is what keeps Claude spend proportional to *opportunities*, not watchlist
size — a 200-ticker watchlist costs the same as a 3-ticker one on a quiet
day, because the filter is pure arithmetic.

## Exit checks: rule-based by default, LLM only on a sharp reversal

`execution_layer/exit_rules.py::evaluate_exit()` is the default intraday
exit path — three plain thresholds (`EXIT_STOP_LOSS_PCT`,
`EXIT_TAKE_PROFIT_PCT`, `EXIT_TRAILING_STOP_PCT`) compared against
`positions.avg_entry_price` and `positions.high_water_mark` (bumped every
tick via `StateStore.update_high_water_mark`). No LLM call, no API cost,
runs on every held position every 15 minutes without hesitation.

The LLM exit-review agent (`analyst_layer/agents/intraday_exit_agent.py`)
only fires as an escalation: the rules above said "hold," *and* the
ticker's regime (`prefilter.compute_regime`) has flipped since the position
was opened (`positions.entry_regime`, stamped at the BUY fill) — e.g.
bought on a bullish crossover, now showing bearish. Even then, it's rate
limited to once per position per day (`StateStore.has_intraday_escalation_today`)
regardless of how many ticks fire. A 15-minute LLM call on every held
position, every day, is the easiest way to quietly turn a paper-trading
experiment into a live decision-making system — this is the guard against
that, not an optimization.

## Why the guardrails can't be talked around

Max position size is checked twice for a **BUY** — once inside the analyst
layer (`risk_officer_agent._clamp_to_limits`) and once again inside the
execution layer (`guardrails.CircuitBreaker.validate_position_size`). Both
re-derive the limit from `equity * MAX_POSITION_SIZE_PCT` independently;
neither trusts a number the LLM produced. A **SELL** is deliberately exempt
from this check — the cap bounds *new* exposure, and must never block
exiting a position that grew past the cap through price appreciation alone.

The daily-drawdown and profit-target breakers have no analyst-layer
equivalent because they need state (`day_start_equity`) that only exists
once trading has begun — they live solely in the execution layer and are
checked on every `intraday_monitoring()` tick.

Paper-trading default: `config/settings.py::Settings` only flips to live
mode if **both** `TRADING_ENV=live` **and** `TRADING_LIVE_CONFIRM` equals the
exact literal token `I_UNDERSTAND_THIS_IS_LIVE_CAPITAL`. Any other
combination — including `TRADING_ENV=live` with the confirm var unset or
wrong — is silently forced back to `paper` inside a `model_validator`.
`AlpacaBroker.from_settings` then passes `paper=not settings.is_live` to
`TradingClient`, and double-checks the same invariant before constructing
the client (`LiveTradingBlockedError`) as defense in depth.

## Tax compliance: the wash-sale guard

`execution_layer/tax_compliance.py::WashSaleGuard` is a deterministic check,
not an LLM judgment call — same reasoning as the position-size guardrail.
It tracks two things in `StateStore`: `positions.last_buy_at` (stamped only
on a BUY fill) and a `realized_sales` ledger (written on every SELL fill,
with the realized P&L computed from the prior `avg_entry_price`).

- **Before a BUY**: if a loss was realized on that ticker within the
  lookback window (default 30 days, `WashSaleGuard.lookback_days`), the
  buy is **blocked outright** and logged to `StateStore.events` as
  `wash_sale_blocked`. New discretionary exposure is the cheap thing to
  refuse.
- **Before a SELL**: if the current lot was bought within the lookback
  window and selling now would realize a loss, a **warning is logged but
  the sale proceeds**. Blocking an exit for tax-bookkeeping reasons would
  itself be a risk-management hazard — losing the tax deduction is a much
  smaller cost than being stuck in a position you wanted to close.

**Known limitation, by design**: "substantially identical security" is
approximated as "same ticker." It does not catch options or ETFs on the
same underlying, and it only sees trades placed through this system — a
position or loss-sale from outside it is invisible. This is a guard
against the *system* creating new wash sales going forward, not a
complete tax-lot accounting engine.

## State persistence

`execution_layer/state_store.py` is a single SQLite file
(`STATE_DB_PATH`, default `./state/trading_engine.sqlite3`) with five
tables: `positions` (mirrors broker state for fast local reads — plus
`entry_regime` and `high_water_mark`, the two fields the exit-rules/escalation
logic above depends on), `realized_sales` (the wash-sale guard's ledger),
`run_history` (every `ConsensusPayload`, JSON-serialized, for audit/replay),
`token_usage` (per-agent, per-model cost — `model`, `input_tokens`,
`output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`,
`estimated_cost_usd`, wired up via `BaseAgent`'s `usage_callback` →
`TradingRuntime._record_usage` → `analyst_layer/pricing.py::estimate_cost_usd`),
and `events` (breaker trips, wash-sale blocks, prefilter pass-rate summaries,
LLM exit escalations, post-market summaries). It survives process restarts;
the broker is still the source of truth for actual positions.

## Test strategy

`tests/` mocks at exactly one seam per layer: `OpenBBDataClient._obb`
(skips the real `openbb` import entirely), `Anthropic.messages.create`
(no network calls, no API key needed to run the suite), and
`execution_layer.broker.TradingClient` (patched at construction).
`tests/test_consensus_graph.py` is the one integration test that runs the
real `StateGraph` end-to-end — it's what would catch a broken edge or a
state-merge bug that the per-agent unit tests can't see on their own.
`tests/test_prefilter.py` and `tests/test_exit_rules.py` test the two
zero-LLM decision points directly (no mocking needed — they're pure
functions); `tests/test_intraday_exit.py` covers the rule-vs-escalation
branch in `TradingRuntime._check_intraday_exits`, including the once-per-day
rate limit on the LLM path.
