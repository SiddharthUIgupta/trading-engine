# Peer Review: Short-Interest / Squeeze-Potential Shadow Signal

**Status:** uncommitted, ready for review. Not yet merged/pushed.

## Summary

Adds a second shadow-signal provider (alongside the already-shipped Kronos-small forecast signal) that scores every screened trading candidate with real short-interest data — the kind of metric that would have flagged GameStop/AMC-style setups. It is **measurement-only**: nothing here can influence a trade. It logs to the same `signal_values` table Kronos uses, and a signal only ever becomes "live" through a separate, explicit, future approval (see `CLAUDE.md` → "Signal lifecycle").

## Why

The trading engine had no visibility into short interest at all. The request was to build this "using free resources" — three real data sources were evaluated live (not from memory) before writing any code:

| Source | What it gives | Cost | Freshness |
|---|---|---|---|
| OpenBB (`obb.equity.ownership.share_statistics`, yfinance provider) | `short_percent_of_float`, `days_to_cover`, month-over-month short interest | Free, already a dependency | ~20 days stale (FINRA's bi-weekly settlement cadence — inherent to the source, not fixable) |
| Alpaca (`TradingClient.get_asset`) | `shortable`, `easy_to_borrow` (booleans) | Free, already integrated, zero new dependency | Live/current, no history |
| Ortex / S3 Partners / iBorrowDesk (actual borrow *rate*) | The single most valuable squeeze signal | **Paywalled everywhere checked** (403s confirmed) | N/A — not used |

## The one design problem worth your attention

Kronos's signal only needs historical price bars, which have a genuine point-in-time structure: "what was the price history as of candidate_date" is always answerable with no ambiguity. **Short interest data does not have this property** — there is no free historical time series, only a "current" snapshot that's itself already lagged by design. Naively stamping an old candidate with "today's" short-interest number would leak future information into the uplift measurement (`scripts/signal_uplift.py`'s correlation calculation), quietly inflating apparent predictive power.

**Fix:** added a nullable `metric_as_of` column to `signal_values` (`execution_layer/state_store.py`), representing when a metric was *actually true*, separate from when the row was written. For Kronos, `metric_as_of == candidate_date` always (zero behavioral change — verified via `test_pit_clean_signal_reports_zero_staleness`). For short interest, it's the real FINRA settlement date. `scripts/signal_uplift.py` now computes and prints `median_staleness_days` next to every verdict, so this is a visible, queryable fact rather than a documentation-only caveat. `CLAUDE.md`'s "Signal lifecycle" section now requires manual staleness review before any `PROMOTE-CANDIDATE` verdict on a signal with non-zero staleness.

**Please check:** is a nullable column + reporting-time surfacing sufficient, or would you want this enforced harder (e.g. `signal_uplift.py` refusing to print a verdict at all above some staleness threshold)?

## Architecture-boundary decision

This repo enforces one-directional imports: `data_layer → analyst_layer → execution_layer`. The Alpaca `shortable`/`easy_to_borrow` flags come from `execution_layer.AlpacaBroker`'s underlying client — reusing that class directly from `analyst_layer` would have violated the rule. Instead: a new, standalone `data_layer/alpaca_reference_client.py` builds its own read-only `TradingClient` straight from `Settings`, exactly how `OpenBBDataClient` already takes `settings.openbb_pat` directly. It never imports `execution_layer` — guarded by `test_alpaca_reference_client.py::test_alpaca_reference_client_never_imports_execution_layer` (via `ast`, not string-matching — see next section for why that distinction mattered).

**Please check:** agree this is the right boundary, or would you rather this data come from `AlpacaBroker` via some other injection path?

## A mistake I made and caught, worth flagging so you don't repeat it

The first version of both new "never imports X" guard tests used plain substring search (`"protection_plane" not in source`). This false-positived immediately — the module's own docstring explains *why* it doesn't import something, using the forbidden word in prose. Rewrote both using `ast.parse` + walking real `Import`/`ImportFrom` nodes, which can't be fooled by comments. If you're adding similar guard tests elsewhere in this repo, use the `ast` version in `tests/test_alpaca_reference_client.py` or `tests/test_shadow_signals.py` as the template, not string search.

## Design choice: raw metrics, no composite score

Emits `short_percent_of_float`, `days_to_cover`, `short_interest_mom_change`, `shortable`, `easy_to_borrow` as five independent metrics — no single "squeeze_score." Reasoning: `short_percent_of_float` and forward return can have opposite-signed relationships depending on regime (rising short interest + rising price = squeeze building = bullish; rising short interest + falling price = just bearish conviction). A composite score would bake in an arbitrary weighting before any IC evidence justifies one — same philosophy Kronos already follows (`p_touch_win`/`med_ret_21d`/`path_dispersion` are separate, not blended).

## Files

**New:** `data_layer/alpaca_reference_client.py`, `analyst_layer/short_interest_provider.py`, `scripts/short_interest_shadow_signal_job.py`, `tests/test_alpaca_reference_client.py`, `tests/test_short_interest_provider.py`

**Modified:** `data_layer/models.py` (+`ShortInterestSnapshot`, `ShortableStatus`), `data_layer/openbb_client.py` (+`get_short_interest_snapshot`), `execution_layer/state_store.py` (+`metric_as_of` column/migration, `record_signal_values` gains the param), `analyst_layer/shadow_signals.py` (harness passes through `metric_as_of` via `getattr` — optional, doesn't change the required `SignalProvider` shape), `config/settings.py` (+short-interest block, 7-day default lookback vs Kronos's 30, reflecting the staleness concern), `scripts/signal_uplift.py` (+staleness computation/reporting), `CLAUDE.md` (+staleness-review caveat)

**Zero changes** (confirmed via `git diff --stat`): `execution_layer/protection_plane.py`, `execution_layer/guardrails.py`, `execution_layer/broker.py`, and `log_candidate()` itself.

## Verification

- 440 tests passing (14 new for this feature), 1 opt-in slow test skipped by default (unrelated, pre-existing Kronos integration test).
- One regression verified red-when-reverted: temporarily removed the `getattr(provider, "get_metric_as_of", ...)` harness logic, confirmed `test_provider_with_get_metric_as_of_uses_custom_value` failed, restored, confirmed green.
- Ran the real batch job against a real, live-seeded GME candidate (cleaned up after): all 5 metrics landed correctly, `metric_as_of='2026-06-15'` vs `candidate_date='2026-07-05'` — 20 days of real, correctly-captured staleness, not silently defaulted to "today."
- Ran `scripts/signal_uplift.py` for real — correctly reports `INSUFFICIENT SAMPLE` (n=0, candidate too fresh for a 21-session forward return yet) for all 5 metrics.

## Known limitations / explicitly out of scope

- No true borrow-rate data — free sources don't have it. `shortable`/`easy_to_borrow` are a binary proxy, not a rate.
- FINRA's free *daily* short-sale-*volume* file (a different, faster-updating metric than cumulative short interest) was evaluated but not built — flagged as a natural future extension, not built now to keep this scoped.
- 7-day default lookback window means this signal will accumulate `n` much more slowly than Kronos's 30-day window — expect `INSUFFICIENT SAMPLE` for a while.
