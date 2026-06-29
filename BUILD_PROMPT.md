# Build prompt: autonomous multi-layer trading engine

Paste this whole file as your first message to Claude Code (or another
coding agent) in an empty project directory to build your own copy of this
system. It's a requirements spec, not a tutorial — the agent should make
its own implementation decisions where this doesn't dictate one.

## What to build

A three-layer, fully autonomous paper-trading system:

1. **Data layer** — OpenBB Platform for market data (prices, fundamentals,
   sentiment, SEC filings, discovery screens). Free `yfinance` provider by
   default, no paid keys required.
2. **Analyst layer** — a multi-agent Claude consensus system (LangGraph) that
   turns raw data into a BUY/SELL/HOLD trade proposal with a risk sign-off.
3. **Execution layer** — Alpaca paper-trading API, gated by deterministic,
   code-level guardrails the LLM can never bypass.

Strict one-directional dependency: data layer → analyst layer → execution
layer. Only the top-level runtime/orchestrator is allowed to import from all
three. Validate every cross-layer boundary with Pydantic models in strict
mode (numbers as numbers, enums as enums) — reject malformed data at the
boundary rather than downstream.

## Non-negotiable guardrails (code-level, not prompted)

These must be enforced in plain Python that the LLM's output passes through,
not requested of the LLM:

- **Paper-trading forced by default.** A "live" mode may exist, but it must
  require *two* independent, explicit signals to agree (e.g. an env var set
  to `live` *and* a literal confirmation string env var) before any live
  client is constructed. Any disagreement between the two silently forces
  paper. This isn't a runtime warning — it should be structurally impossible
  to flip live with a single flag or default.
- **Position-size cap** — reject/clamp any BUY whose notional exceeds a
  configured % of account equity (default 5%). Only check this on BUY — a
  position that's grown past the cap purely from price appreciation must
  still be sellable.
- **Daily drawdown circuit breaker** — track equity at the start of each
  trading day; if intraday equity drops more than a configured % (default
  2%) from that baseline, halt all new orders for the rest of the day.
- **Daily profit-target lock-in** — symmetric to the drawdown breaker: once
  today's gain (mark-to-market, not just realized trades) crosses a target
  in dollars (default $50), stop trading and don't keep risking the gain
  chasing more.
- **Wash-sale guard** — block a same-ticker BUY within N days of a realized
  loss sale on that ticker (warn/log, don't silently swallow).
- A SELL proposal against zero existing shares must be deterministically
  clamped to HOLD, never crash the pipeline.

## Analyst layer: 4-agent consensus

Per-candidate, fan out to 3 narrow agents in parallel via LangGraph, then
fan in to one risk sign-off:

- **Sentiment/macro agent** — news/sentiment data only.
- **Fundamentals/SEC agent** — fundamentals + recent filings only.
- **Technical analysis agent** — price series, moving averages, regime
  classification (bullish/bearish crossover) only.
- **Risk & Compliance Officer agent** — synthesizes the three signals into a
  single `TradeProposal` (action/quantity/limit price), checks it against
  account context (equity, current price, existing shares, drawdown %), and
  emits a `RiskReview` verdict (approved/amended/rejected) with reasons.
  This is the only agent whose output can actually become a trade — wrap its
  call in a `try/except` that falls back to a forced HOLD on any schema
  validation failure, so a malformed LLM response degrades safely instead of
  crashing the whole run.

**Model tiering**: cheap/fast model (e.g. Haiku-tier) for the 3 narrow
agents — they're just interpreting an already-computed signal. More capable
model (e.g. Sonnet-tier) for the Risk Officer — it's the one consequential
decision. Use prompt caching (cache the system prompt + tool schema) since
these run repeatedly per cycle.

Every run — regardless of verdict — gets persisted (ticker, signals,
proposal, risk review, timestamp) for audit/replay, plus per-call token
usage and estimated cost.

## Filter-first design

Don't run the expensive 4-agent consensus on every ticker in the universe.
Gate it behind cheap, deterministic, code-only screens so Claude spend is
proportional to *opportunity count*, not watchlist size. Build the universe
dynamically from OpenBB discovery screens (active movers, gainers, losers,
small caps, etc.) rather than a fixed ticker list, then narrow with one or
both of the two strategy-specific screens below before any LLM call happens.

### Deterministic prefilter (Track A + B)

The prefilter fires on any of these triggers (any one is enough to pass):
- MA crossover just flipped (10/30 day SMA pair)
- RSI outside [30, 70]
- Volume spike ≥ 2× 10-day average
- Sentiment score magnitude ≥ threshold
- Recent 8-K filing (material event proxy)
- Turnover rate ≥ 3% (daily volume ÷ shares float): more manipulation-resistant
  than headline sentiment — a meaningful fraction of the float changed hands.
  Best-effort: fetch `shares_float` per ticker; skip gracefully if unavailable.
- **qlib Alpha158 factors** (all computable from daily OHLCV, no extra fetch):
  - R² ≥ 0.80 over a 20-day linear fit: clean, sustained directional trend —
    stronger signal than a single SMA crossover which can fire on one noisy candle.
  - N-day range position (Williams %R variant, 30-day window): where today's close
    sits in the 30-day high/low band. ≤15% = at low end; ≥85% = extended. More
    context-aware than RSI because it anchors to actual price extremes.
  - Return-volume correlation (20-day rolling): positive (≥0.30) = volume rises on
    up days (trend confirmation); negative (≤-0.30) = volume rises on down days
    (distribution signal). Catches the classic "looks fine on price, institutions
    quietly unloading" pattern.

## Three independent strategy tracks

Same agents, same guardrails, same execution path — but different entry
screens and different exit-rule brackets, since they're targeting
structurally different setups. Tag every position with which track opened
it (a `strategy` column) so exit logic routes correctly per-position.

### Track A — Opening Range Breakout, equity (intraday, fast)

Signal: price breaks above (long) or below (short, skipped — long only) the
range formed in the first 15 minutes of trading, confirmed by volume ≥1.5×
the opening-range average. Evaluate per-tick on 5-minute bars from OpenBB's
market-movers discovery screen (active, gainers, losers).

Run every ~30 min during market hours — **not** pre-market, since VWAP/EMA
need real intraday bars that don't exist before the open.

Exit bracket: fixed price levels from the opening range itself (not percentages):
stop = opening range low, target = entry + 2× (entry − stop). Force-close
any position still open from a prior day regardless of price — ORB is a
day-trade by design; overnight hold is a different, untested risk profile.
Tag these positions `strategy="orb"`.

### Track A.options — Opening Range Breakout, options (intraday, fast)

Same ORB signal as Track A, expressed as long calls (bullish breakout) or
long puts (bearish breakdown) instead of shares. Runs on the same market-movers
universe at a different cadence offset (e.g. :15/:45 vs :00/:30) so both
tracks' OpenBB fetches don't fire simultaneously.

Sizing: off premium at risk (max loss on a long option), not share notional —
the options multiplier (×100) makes the equity cap the wrong denominator.
DTE band: 5–10 days (enough time for the move without excessive theta).
Stop: 40% premium loss. Force-close 2 days before expiration regardless of P&L.
Tag these positions `strategy="orb_options"`.

The `_check_options_exits` method handles this track. It must explicitly skip
`vol_short` positions — those have their own exit method. Use the `strategy`
column to route: `if position["strategy"] == "vol_short": continue`.

### Track B — quality pullback (daily, slow)

Inverse shape of Track A: looks for fundamentally sound names having a
*quiet* pullback, not stocks already moving fast.

**Primary screen:** price is 20%–50% off its 52-week high. The floor excludes
noise; the ceiling excludes names that are probably genuinely impaired rather
than temporarily out of favor.

**Secondary screen — shrink-volume retest (ranking boost, not hard gate):**
After a candidate passes the pullback screen, fetch 30 days of daily price
history and run three conjunctive conditions:
1. MA5 > MA10 > MA20 — uptrend confirmed across multiple timeframes
2. Price within 2% of MA5 or MA10 — retesting support, not breaking down
3. Today's volume < 70% of prior 5-day average — sellers absent on the dip

Candidates passing all three get a +0.5 boost to their ranking score so they
surface above plain pullbacks. Candidates that don't pass are **not** blocked —
a genuine dislocation thesis doesn't require an intact uptrend. This pattern
is from ZhuLinsen/daily_stock_analysis's `shrink_pullback.yaml` strategy.

- Universe: should include names outside any major index — the dynamic
  discovery-screen universe already covers this.
- Run once daily (fundamentals don't change intraday), feeds the same
  4-agent consensus.

Exit bracket: wide and asymmetric — wide stop-loss (e.g. 18%), **no fixed
take-profit** (let a winner run), trailing stop (e.g. 10%) that only
*activates* once the position is already up significantly (e.g. +20%) so it
doesn't choke off a big move early.

### Track A/B momentum exit bracket

For momentum (Track A equity) and non-thesis positions:
- Stop-loss: 2% below entry
- **No hard take-profit cap** — a fixed cap like 3% chops winners (a stock up
  3% that continues to +12% is sold at 3% for no reason).
- Trailing stop: 1.5% behind the high-water-mark, **activating only once the
  position is up 3%** (the activation threshold). Below 3% gain, the trailing
  stop is dormant; once the position crosses 3%, it starts trailing.

This means a winner is held as long as it keeps advancing. The trailing stop
only fires on a meaningful pullback from the peak — not the moment profit
appears. The thesis track uses the same no-hard-cap design with a higher
activation threshold (20%) to match its longer time horizon.

### Track C — vol options / short premium (daily, defined-risk)

Implements the Natenberg/tastylive short-premium framework.

**Universe: dynamically screened for options liquidity, seeded by a static watchlist.**
Start with a seed of liquid, high-options-volume names (`["AAPL", "MSFT", "NVDA",
"SPY", "QQQ", "AMZN", "META", "TSLA"]` is a reasonable floor). Before each scan
cycle, augment the seed with that day's market movers, then filter the combined
pool for three hard liquidity gates:
1. ATM call and put both have open interest ≥ threshold (default 500)
2. ATM bid/ask spread ≤ max fraction of mid (default 10%) — ensures mid-price fills work
3. At least one expiration exists in the target DTE window (default 21–60)
Sort survivors by average ATM OI descending (deepest liquidity first) and cap at
max_size (default 20). If nothing passes the screen (off-hours, API outage), fall
back to the seed unchanged — the seed is guaranteed liquid.

Do NOT use raw market movers without the liquidity screen — movers can be thinly
traded names where premium selling would face blown-out spreads. The screen, not
the seed, is what actually enforces liquidity.

**Entry signal — IV Rank (IVR) > 50**:
IVR = (current_IV30 − 52wk_low_IV) / (52wk_high_IV − 52wk_low_IV) × 100.
IVR > 50 means current implied vol is elevated vs. its own recent history
→ variance risk premium is available → sell premium.

**HARD GATE — no premium selling through earnings:**
Before running any LLM agent, check `vol_snapshot.earnings_within_dte`. If
True, skip the ticker entirely — do not call the LLM, do not enter a position.
This is a code-level gate, not a prompt instruction. Selling premium into an
earnings event is a fundamentally different risk (binary event, not structural
premium) and the LLM must not be able to override it.

**Structure: iron condor** (defined risk, requires broker Level 3):
- SELL short call (OTM, ~1 standard deviation above current price)
- SELL short put (OTM, ~1 standard deviation below current price)
- BUY long call wing (further OTM, ~2 SDs above — limits loss on call side)
- BUY long put wing (further OTM, ~2 SDs below — limits loss on put side)
Target 30–45 DTE at entry.

**Mid-price entry (better fills):**
Do not use natural prices (bid for shorts, ask for longs) — that's the
worst-case spread. Compute the mid-price credit:
```
mid_credit = (short_call.mid + short_put.mid) - (long_call.mid + long_put.mid)
# where mid = (bid + ask) / 2 for each leg
```
Submit the mleg at `limit_price = -round(mid_credit, 2)`. On liquid names
(SPY, AAPL, QQQ) mid-price fills frequently. The order is DAY TIF so if it
doesn't fill it expires at close and the system retries the next entry window.
Fall back to natural credit only if mid_credit ≤ 0 (wide spread makes mid a
net debit).

**tastylive management rules (mechanical, no LLM)**:
- Close at **50% of credit received** (captures half the premium early, avoids
  gamma risk into expiration).
- Close when cost-to-close reaches **2× the original credit** (loss limit).
- Close at **21 DTE** regardless of P&L (roll before gamma accelerates; this
  is the one that trips most often).

**Vol agent graph** (separate from the momentum consensus graph):
LangGraph fan-out to 3 independent agents, then fan-in to a Greeks Risk Officer:
- `iv_surface_agent` — analyzes IV30/HV30 spread, IV percentile, term structure.
- `event_risk_agent` — checks for upcoming earnings, FDA dates, FOMC within DTE.
- `vol_regime_agent` — reads VIX level/trend, VIX vs VIX3M contango/backwardation.
All three run in parallel with **no cross-anchoring** (they don't see each other's
output; parallel isolation prevents anchoring bias).
Fan-in: `greeks_risk_node` synthesizes all three signals, builds the actual
OptionsProposal (strikes, expiration, DTE, net credit) via `options_structurer`,
then a Greek Risk Officer agent reviews portfolio-level vega/delta limits before
emitting the final `VolConsensusPayload`.

`allow_uncovered` gate — **requires two independent env vars**, same ceremony
as the live-trading switch. Naked short calls have theoretically unbounded
loss on the call side; this is a risk-profile change, not a config tweak.
Both must be set to enable strangles:
```
VOL_OPTIONS_ALLOW_UNCOVERED=true
VOL_OPTIONS_UNCOVERED_CONFIRM=I_UNDERSTAND_STRANGLES_HAVE_UNBOUNDED_RISK
```
Any disagreement between the two silently degrades to iron condor. Implement
with a model-level validator (same pattern as `_enforce_paper_default`) that
force-sets `vol_options_allow_uncovered=False` when the confirm token is wrong
or absent. Expose as an `is_uncovered_allowed` property (parallel to `is_live`)
so call sites read the property, not the raw field.

**CRITICAL: iron condor must use Alpaca's mleg order, not 4 individual orders.**
Submitting 4 separate single-leg orders fails for iron condors even with Level 3:
Alpaca evaluates each SELL independently (not yet seeing the BUY hedge), so
short legs are rejected as "uncovered" before the long wings are recognized.
The fix — submit as a single atomic multi-leg order:
```python
from alpaca.trading.enums import OrderClass, PositionIntent
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

legs = [
    OptionLegRequest(symbol=short_call_sym, ratio_qty=1.0, position_intent=PositionIntent.SELL_TO_OPEN),
    OptionLegRequest(symbol=short_put_sym,  ratio_qty=1.0, position_intent=PositionIntent.SELL_TO_OPEN),
    OptionLegRequest(symbol=long_call_sym,  ratio_qty=1.0, position_intent=PositionIntent.BUY_TO_OPEN),
    OptionLegRequest(symbol=long_put_sym,   ratio_qty=1.0, position_intent=PositionIntent.BUY_TO_OPEN),
]
# Mid-price credit: (bid+ask)/2 for each leg
mid_credit = round(
    (sc.bid+sc.ask)/2 + (sp.bid+sp.ask)/2 - (lc.bid+lc.ask)/2 - (lp.bid+lp.ask)/2, 2
)
natural_credit = round(sc.bid + sp.bid - lc.ask - lp.ask, 2)
net_credit = mid_credit if mid_credit > 0 else natural_credit
order = LimitOrderRequest(
    order_class=OrderClass.MLEG,
    qty=contracts,
    time_in_force=TimeInForce.DAY,
    limit_price=round(-net_credit, 2),  # MUST round to 2 decimal places or API rejects
    legs=legs,
    # no top-level `symbol` or `side` — those go on each leg for mleg
)
```
For mleg orders, `symbol` is omitted at the order level (each leg carries its
own symbol). The `limit_price` MUST be rounded to 2 decimal places (code 42210000
is "limit price must be limited to 2 decimal places").

**Double-entry guard — dedup across state store AND pending broker orders:**
Build `existing_vol_tickers` before scanning each cycle:
```python
existing_vol_tickers = {
    p["underlying_symbol"]
    for p in state_store.get_option_positions()
    if p["strategy"] == "vol_short" and p["quantity"] != 0  # qty=0 = pending/expired, not a live position
}
# Also check broker open orders — a pending mleg has qty=0 in local state until it fills
for order in broker.get_open_orders():
    if order.get("legs"):
        for leg in order["legs"]:
            parsed = parse_occ_symbol(leg["symbol"]) if leg.get("symbol") else None
            if parsed:
                existing_vol_tickers.add(parsed.underlying_symbol)
```
The `qty != 0` filter is critical: a mleg submitted but not yet filled records
qty=0 in the local state store. Without the qty filter, expired unfilled mlg
orders would permanently block that ticker from re-entry. The open-orders check
covers the pending case; once the order expires, neither check blocks it.

**Short option position accounting:**
Alpaca reports short positions as **negative qty**, and `avg_entry_price` = the
credit originally received (not a cost). To close: submit a BUY order (not SELL).
P&L on close = (credit_received − close_cost) × contracts × 100.
Record this correctly in the realized P&L table — the sign is the opposite of
a long option close.

**Strategy tagging** — `option_positions` table needs a `strategy TEXT` column:
- `"vol_short"` — opened by the vol track
- `"orb_options"` — opened by the ORB options track
The two tracks' exit checks must be mutually exclusive:
- `_check_options_exits` (ORB): `if position["strategy"] == "vol_short": continue`
- `_check_vol_options_exits`: `if position["strategy"] != "vol_short": continue`

**Reconciliation for vol positions:** The `_reconcile_option_positions` intraday
tick must run when *either* options track is enabled (not just the ORB options
track). mleg fills are asynchronous; quantity stays 0 in local state until the
next intraday reconcile syncs it from the broker.

## Intraday exit monitoring

Every position, every ~15 min during market hours:
1. **Reconcile first.** Re-fetch every locally-tracked position's real
   quantity/avg-price from the broker and correct any drift before doing
   anything else — see "Known pitfalls" below for why this has to be a
   standing periodic step, not just a one-time check at fill time.
2. Evaluate the position's exit rule (stop-loss / trailing stop, using
   whichever bracket corresponds to its `strategy` tag) — plain
   deterministic Python, no LLM call, no per-tick API cost.
3. **LLM escalation only as a rate-limited fallback**: if the deterministic
   rules say "hold" but the position's technical regime has sharply
   reversed since entry (e.g. bullish → bearish crossover), escalate to a
   cheap LLM-based exit-review agent for a second opinion — at most once per
   position per day, logged like any other agent call.
4. Any resulting SELL goes through the exact same guardrail + broker path
   as an entry order (circuit breaker check, position-size check, wash-sale
   warn, submit, record fill, record event).

## Scheduler

Cron-style jobs, market timezone (e.g. `America/New_York`), weekdays only:
- Pre-market scan (~8:00) — records the day's starting equity baseline
  (needed for the profit-target/drawdown breaker *and* for an accurate
  "today's P&L" elsewhere — see pitfalls).
- Daily pullback-track scan (~8:15).
- Market-open execution (~9:30) — submits orders for whatever the
  pre-market/overnight consensus runs approved.
- **Vol options scan — two daily entry windows:**
  - Morning (~10:00) — 30 min after open, gives opening-hour vol spikes time
    to settle before selling premium.
  - Afternoon (~13:00) — catches tickers whose IV spiked from a mid-day catalyst
    (competitor earnings, macro event) that weren't eligible at 10 AM.
  - Both windows use the same `vol_options_scan_and_trade` function. The
    double-entry guard (existing positions + open orders) prevents re-entering
    a ticker already opened in the morning window. Do NOT use a "scanned today"
    set to skip the afternoon scan — that defeats the purpose. Let the
    existing_vol_tickers guard do the dedup.
- Intraday monitoring, every ~15 min during market hours — reconciliation +
  exit checks (see above).
- Intraday momentum-track scan, every ~30 min during market hours.
- Post-market logging (~16:30) — summarize the day.

## State persistence

Single SQLite file as the source of truth, read concurrently by the live
trading process and any read-only dashboard. Plain SQL, no ORM needed at
this scale. Minimum tables:
- `positions` — current holdings, including a `strategy` tag and a
  trailing-stop high-water-mark column, upserted with "only overwrite
  fields that were actually passed" semantics (a SELL-driven update
  shouldn't reset the entry regime a BUY set).
- `realized_sales` — every closed trade (both wins and losses), used by the
  wash-sale guard and by trade-history reporting.
- `run_history` — every consensus run, full payload as JSON, regardless of
  outcome.
- `token_usage` — per-call agent name, model, token counts (split out
  cache-creation vs. cache-read), estimated cost.
- `events` — a generic append-only log (scan summaries, breaker trips,
  order submissions/fills, wash-sale blocks, escalations) — give order
  executions their own clearly-named event type so they're easy to find
  without parsing `run_history` JSON.

## Dashboard

Read-only visualization, separate process from the trading engine, reading
the same SQLite file plus live broker state. Should never write anything
except for one explicitly advisory, explicitly non-automatic feature:

- Overview: live equity, market open/closed + next open/close time, today's
  Claude spend (call out clearly that this is real money billed separately
  from the paper-trading equity number next to it), and today's P&L against
  the profit target (using the actual recorded start-of-day baseline, not
  any all-time number).
- Open positions, with live unrealized P&L pulled from the broker, not
  recomputed locally.
- **Pending orders tab:** mleg orders show as pending between submission and
  fill. The dashboard must handle `symbol=None` at the order level (mleg
  orders carry symbols on legs, not the top level). Expand each mleg into
  individual leg rows using `order["legs"]`, showing per-leg symbol, side,
  and position_intent. Crashing on `parse_occ_symbol(None)` is a real bug
  to avoid.
- Trade history with win rate and cumulative realized P&L.
- Every consensus run ever made (not just executed ones), with each agent's
  stance/confidence/rationale and the risk verdict.
- Scanner activity (both tracks).
- Cost tracking, totals and by-agent breakdown.
- Full events log.
- **On-demand single-ticker analysis**: a text input + button that runs the
  exact same 4-agent consensus on any user-typed ticker, bypassing both
  scanner screens entirely, and displays the result. This must be advisory
  only — it must never place an order itself. Still record the run to
  `run_history`/`token_usage` like any other consensus call.

## Known pitfalls — design around these from the start

These are real bugs found and fixed while building this system the first
time. Building the system correctly from the start beats finding them again:

- **Limit-order fill races.** Submitting an order and immediately reading
  back "current position quantity" can read stale state — a broker's fill
  can lag the submit call returning by anywhere from milliseconds to
  minutes. Two layers of defense, not one: (a) after submitting, poll the
  order's own status for a few seconds before trusting a position read; (b)
  separately, *every* periodic monitoring tick should re-reconcile local
  position records against the broker's actual state regardless of what
  the local record currently says — a slow-to-fill order that the poll
  window missed will otherwise stay wrong forever, since nothing else ever
  goes back to check it.
- **"Today's P&L" needs an explicit recorded baseline.** An account's
  equity has an all-time opening balance that is *not* the same number as
  "what the account was worth at the start of today" — record an explicit
  start-of-day equity event each morning and diff against that, not against
  any hardcoded or all-time figure.
- **Mark-to-market vs. realized P&L are different numbers, both legitimate.**
  Account equity moves continuously with unrealized gains/losses on open
  positions, with no sale required — this is correct and is exactly what
  the drawdown/profit-target breakers should be watching. Realized P&L only
  changes when a position actually closes. Don't conflate the two.
- **Percent fields from data providers aren't always on the scale you'd
  guess** — verify directly (e.g. is "percent_change" already a fraction
  like 0.07, or a whole percentage like 7.0) before computing thresholds
  against it, rather than assuming.
- **Pydantic field validators run in declaration order** — a cross-field
  validator on field B that needs field A to already exist must be a
  model-level (`mode="after"`) validator, not a per-field validator on B,
  if A is declared after B.
- **Strict-mode type validation can be too strict for real tool-call
  JSON** — if using a strict base model config, make sure it still allows
  normal string→Enum coercion for fields typed as Enums/Literals, since a
  live LLM tool-call response arrives as JSON strings.
- **Date arithmetic**: don't subtract raw days from a date's `.day`
  component (breaks at month boundaries) — use `timedelta`.
- **Timezone consistency**: if timestamps are stored via UTC, every query
  that filters "since today" must compute "today" in the same UTC frame,
  not local time — a local-vs-UTC mismatch silently drops or includes the
  wrong rows depending on time of day.
- **Iron condor: never submit legs individually.** Even with a Level 3 Alpaca
  account (spreads enabled), submitting the short call before the long call wing
  is filled results in "account not eligible to trade uncovered option contracts"
  — Alpaca evaluates each order independently. The only fix is one atomic mleg
  order with all 4 legs; see the Track C section for the exact SDK call.
- **mleg limit_price sign convention.** For Alpaca's mleg order class: positive
  `limit_price` = you are paying (debit). Negative `limit_price` = you are
  receiving (credit). Iron condors collect premium, so `limit_price` is negative.
  Round to exactly 2 decimal places or Alpaca rejects with error code 42210000.
- **Short option qty is negative in Alpaca.** The broker reports short positions
  as `qty < 0`. BUY (not SELL) to close. P&L sign is inverted versus a long
  option: `(credit_received - close_cost) × contracts × 100`.
- **mleg fills are async.** A submitted mleg order returns "accepted" before the
  individual legs fill and show up as positions. Call `get_position_detail` on
  each leg symbol during intraday reconciliation (not immediately after submit)
  to get the actual fill qty/price. If reconciliation is gated on
  `options_track_enabled`, also run it when `vol_options_track_enabled` is set.
- **21-DTE roll check runs intraday (every 15 min), not in a separate daily job.**
  `_check_vol_options_exits` is called from `intraday_monitoring`, which fires
  every 15 min during market hours. DTE doesn't change within a day so the check
  is idempotent, but the right place for it is in the intraday loop — not a
  separate once-daily cron. If you put it only in the vol scan (10:00 daily),
  a position that crosses the 21-DTE threshold between market-open and 10:00 would
  not be caught until the next day's scan, which is a full missed cycle.
- **IVR is a position within a range, not IV level.** IVR of 80 means current
  IV30 is in the 80th percentile of its 52-week range — it says nothing about
  absolute IV. Two tickers can both have IVR=80 but wildly different actual IV.
  The premium-selling signal is IVR, not raw IV level.
- **Never sell premium through earnings.** This is a hard code-level gate, not
  a prompt instruction. Check `earnings_within_dte` on the volatility snapshot
  before running the vol consensus. If True: skip, log, move on. Do not let the
  LLM agents decide — they should not have veto authority over a structural risk
  constraint like this.
- **Vol scan dedup: two distinct mechanisms for two distinct cases.**
  (a) Already-filled position in state store: filter `qty != 0`. A qty=0 entry
  means the mleg was submitted but not yet filled (or has already expired) — a
  permanent block here would prevent re-entry after an expired unfilled order.
  (b) Pending mleg (submitted, not yet filled): check `broker.get_open_orders()`
  for mleg legs. A pending mleg has qty=0 in local state until it fills, so the
  state store check alone misses it.
  Combining both: `existing_vol_tickers` covers live positions (qty ≠ 0) AND
  pending orders (from open_orders), giving correct dedup in all states.
- **Hard take-profit caps chop winners.** A 3% take-profit sells a stock that
  proceeds to +12%. Replace fixed take-profit with a trailing stop that activates
  once the position reaches the profit threshold (e.g. 3%). Below the threshold
  the trailing stop is dormant; once crossed, it trails the peak. This lets
  genuine winners compound while still protecting against reversal.
- **yfinance returns `float('nan')` for missing data, not `None`.** Pydantic's
  `gt=0` / `ge=0` validators reject NaN with "Input should be greater than 0
  [input_value=nan]". The `is not None` guard everyone reaches for doesn't help:
  `float('nan') is not None` is `True`. Build a `_safe_float(value) -> float | None`
  helper that calls `math.isnan()` and returns `None` for NaN, and use it everywhere
  a numeric field from a DataFrame row is assigned to an Optional model field.
  ```python
  def _safe_float(value) -> float | None:
      if value is None: return None
      try:
          f = float(value)
          return None if math.isnan(f) else f
      except (TypeError, ValueError):
          return None
  ```
- **pandas NaN in date columns appears as `float('nan')` OR the string `'nan'`**
  depending on column dtype — both must be treated as missing. Build an `_is_nan`
  helper that checks `isinstance(value, float) and math.isnan(value)` AND
  `isinstance(value, str) and value.strip().lower() in ('nan', 'nat', 'none', '')`.
  Use it as a pre-check before calling `date.fromisoformat()` or constructing a
  `date` field — otherwise you get "Invalid isoformat string: 'nan'" deep in the
  validation stack with no hint about which field or which ticker caused it.
- **`float('nan')` is truthy.** `bool(float('nan'))` is `True`, so
  `row.get("field_a") or row.get("field_b")` does NOT fall back to `field_b`
  when `field_a` is NaN — it returns the NaN. The same trap hits `if value:`
  guards on optional numeric fields. Use `_is_nan(value)` instead of `if value`
  or `if value is not None` for any field that may come from a pandas DataFrame.
- **Wrap DataFrame row iteration per-row, not with a single outer try/except.**
  A `try/except` around the entire `[Model(...) for row in records]` list
  comprehension aborts all remaining rows the moment one row is malformed. Prefer
  an explicit loop with per-row `try/except` that logs at DEBUG and `continue`s —
  one bad row from yfinance (e.g. a filing with a missing date, a price bar with
  a NaN close during a trading halt) should not fail the entire fetch. Only raise
  `DataValidationError` if the result is completely empty when it shouldn't be.
- **Two daily vol scan windows, not one.** A single 10 AM scan misses tickers
  whose IV spikes from a mid-day catalyst. Add a second scan at ~1 PM. The
  double-entry guard handles dedup automatically — do NOT use a "scanned today"
  set to suppress the second scan; that defeats the purpose.

## Performance analytics

Compute post-trade performance metrics from the `realized_sales` table.
Aggregate trades by sale date into daily P&L before computing ratio statistics
(avoids inflating Sharpe by treating same-day trades as independent observations):

- **Sharpe ratio** — annualized: `mean(daily_pnl) / std(daily_pnl, ddof=1) * sqrt(252)`.
  Zero risk-free rate (paper account). Return `None` if std is 0 or fewer than 2
  trading days of data.
- **Sortino ratio** — same as Sharpe but denominator uses only negative days
  (MAR=0). Compute downside std over the full sample (not just the subset of down
  days). Return `None` if there are no down days.
- **Calmar ratio** — `annualized_pnl / max_drawdown`. Annualize from the date
  range of actual trades. Return `None` if max drawdown is 0.
- **Max drawdown** — peak-to-trough on the cumulative realized P&L curve (not
  equity curve). Dollar amount, ≥ 0.
- **Profit factor** — `sum(wins) / abs(sum(losses))`. `inf` if no losing trades.

Show these in a dedicated Performance tab in the dashboard, broken out by
track (equity/options/combined).

## GARCH(1,1) realized vol forecast

The IV-surface agent's `VRP = IV30 − HV30` is backward-looking (HV30 uses the
past 30 days). Supplement it with a forward-looking GARCH(1,1) forecast so the
agent sees both:

**Model**: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` (Bollerslev 1986).
Use **variance targeting** to set ω: `ω = σ̄²·(1 − α − β)` where σ̄² is the
sample variance of log returns. This avoids numerical MLE and gives a stable
long-run variance anchor. Default parameters: α=0.10, β=0.85 (persistence=0.95).

**h-step ahead forecast**:
```
VL = ω / (1 − α − β)           # long-run variance
σ²_{T+h} = VL + (α+β)^h · (σ²_T − VL)
avg_daily_var = VL + (σ²_T − VL) · persistence · (1 − persistence^h) / (1 − persistence) / h
annualized_garch_rv = sqrt(avg_daily_var * 252)
```
Require at least 31 bars (30 log returns) to return a result; return `None`
for shorter series.

Store the forecast as `garch_rv_forecast: float | None` on `VolatilitySnapshot`
(optional field, default None — the field must be in the model from the start
so the snapshot is a self-contained package when the agent receives it).

**VRP (GARCH)** = `iv_30 − garch_rv_forecast`. Positive = options overpriced vs
expected realized vol = the structural short-premium edge. Surface this in the
IV surface agent's prompt alongside the backward-looking `iv_30 − hv_30` spread,
labeled clearly so the agent can weight the more predictive forward-looking signal
appropriately.

## Tech stack (use these unless you have a strong reason not to)

OpenBB Platform (data), LangGraph (agent graph), official `anthropic` SDK
(LLM calls — see the model/thinking guidance separately if your agent has
access to current Claude API docs), `alpaca-py` (broker), `pydantic` v2 +
`pydantic-settings` (config/validation), `apscheduler` (cron jobs),
`streamlit` + `plotly` + `pandas` (dashboard), plain `sqlite3` (state),
`pytest` (tests — write real unit tests per layer, including regression
tests for the race conditions above, not just smoke tests).

## What "done" looks like

- Settings/config object is the single source of truth for every risk
  limit and the paper/live switch — nothing downstream redefines its own
  copy of a threshold.
- A full test suite covering: the paper/live gate, position-size and
  drawdown guardrails, both scanners' criteria (including edge cases at
  exact thresholds), both exit-rule brackets (including trailing stop
  activation threshold — verify that a position at +10% with no pullback
  is held, not sold), state-store round-trips, fill-race + reconciliation
  behavior, vol track consensus → mleg submit → position tagging, and all
  three tastylive exit triggers (profit target, loss limit, DTE roll).
- Vol track tests verify the iron condor submits exactly one `submit_spread_order`
  call (not individual `submit_option_order` calls), at mid-price credit, and
  that all 4 legs are stored with `strategy='vol_short'`.
- Prefilter tests cover the qlib Alpha158 factors: R², range position, and
  return-volume correlation each trigger correctly at their thresholds and
  don't trigger below them.
- The dashboard runs against both an empty state DB and a populated one
  without throwing, including when mleg orders with `symbol=None` are present.
- A dry run (mocked broker/LLM) exercises the full scheduled-job pipeline
  end to end without manual intervention.
- End-to-end live test (paper account) confirms the iron condor mleg order
  is accepted by Alpaca (status "accepted", no rejection error). Fill shows
  up in position state after the next intraday reconciliation cycle.
