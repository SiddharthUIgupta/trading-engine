#!/usr/bin/env python3
"""Signal uplift report — does a shadow signal actually predict anything?

Usage:
    source .venv/bin/activate
    python scripts/signal_uplift.py

For each (signal_name, signal_version, metric_name) found in signal_values,
joins against candidates.fwd_ret_21d and reports:
  - n (sample size, status='ok' rows with a non-NULL forward return)
  - raw Spearman IC of the metric value vs fwd_ret_21d (rank-transform +
    numpy.corrcoef — NOT pandas.Series.corr(method="spearman"), which looks
    scipy-free but actually imports scipy.stats internally; scipy isn't
    installed here and isn't worth adding just for this)
  - incremental IC: Spearman IC after residualizing both the metric value and
    fwd_ret_21d on screen_score via a 1-variable OLS (numpy.polyfit)

n < 300 -> "INSUFFICIENT SAMPLE", no IC computed, no verdict — this is the
expected, correct output while a signal is new, not a bug.

At n >= 300: PROMOTE-CANDIDATE if abs(incremental_ic) >= 0.03, else
DELETE-CANDIDATE. This script only ever reports a verdict — promoting a
signal into the risk gate is a separate, explicit, future task (see
CLAUDE.md "Signal lifecycle").

Lookahead exclusion: any row where metric_as_of > candidate_date (the
metric is dated AFTER the candidate it's attached to — real information
leakage, e.g. a batch job backfilling an old candidate with a live-fetched
flag) is hard-excluded from n and every IC calculation, structurally, not
just flagged for manual review. Ordinary staleness (metric_as_of BEFORE
candidate_date — e.g. a 20-day-old FINRA settlement number, which is
exactly what the live system would have had at decision time) is not a
leak and is not excluded — it's honest, expected, and reported as context.
"""
from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.settings import get_settings
from execution_layer.state_store import StateStore

_MIN_SAMPLE = 300
_PROMOTE_THRESHOLD = 0.03


def _residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """1-variable OLS residuals of y on x — closed-form, no scipy/sklearn."""
    slope, intercept = np.polyfit(x.values, y.values, 1)
    return y - (slope * x + intercept)


def _split_lookahead_contaminated(group: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Hard-excludes rows where metric_as_of > candidate_date — the metric
    is dated AFTER the candidate it's attached to, which is real information
    leakage (e.g. a batch job backfilling an old candidate with a value
    fetched live today). This is structural, not a manual-review flag: a
    human remembering to check is exactly the kind of control this repo's
    own history shows fails (see CLAUDE.md evidence protocol). NaT
    (metric_as_of never recorded) is NOT treated as contaminated — that's
    "unknown provenance", a different and separately-reported condition
    (see the None-staleness case below), not a known future-dated leak.
    Returns (clean_rows, n_excluded).
    """
    as_of = pd.to_datetime(group["metric_as_of"], errors="coerce")
    candidate_date = pd.to_datetime(group["candidate_date"], errors="coerce")
    contaminated = as_of > candidate_date  # NaT comparisons evaluate to False, not excluded
    return group[~contaminated], int(contaminated.sum())


def _median_staleness_days(group: pd.DataFrame) -> float | None:
    """candidate_date - metric_as_of, in days, over already-lookahead-clean
    rows. For PIT-clean sources (e.g. Kronos, where metric_as_of ==
    candidate_date always) this is always 0 — a useful confirmation, not
    just an assumption. For current-snapshot-only sources (e.g. short
    interest) this surfaces real staleness next to the verdict rather than
    silently ignoring it — this is honest lag, not a leak, and is reported
    as context, never used to exclude a row. None if metric_as_of was never
    recorded (a provider that predates this tracking, or a bug).
    """
    as_of = pd.to_datetime(group["metric_as_of"], errors="coerce")
    candidate_date = pd.to_datetime(group["candidate_date"], errors="coerce")
    if as_of.isna().all():
        return None
    staleness = (candidate_date - as_of).dt.days
    return float(staleness.median())


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation via rank + Pearson — pandas.Series.corr(method=
    "spearman") looks scipy-free but actually imports scipy.stats.spearmanr
    internally (confirmed by running it — it raised ModuleNotFoundError on
    this machine). scipy isn't installed and isn't worth adding for this;
    rank-transform + numpy.corrcoef is the exact same statistic with no
    scipy/sklearn dependency at all.
    """
    ra, rb = a.rank().values, b.rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_uplift(store: StateStore) -> list[dict]:
    with closing(store._connect()) as conn:
        # LEFT JOIN, not INNER: a signal can have plenty of 'ok' rows with no
        # fwd_ret_21d yet (candidate too recent — 21 sessions haven't passed).
        # That's INSUFFICIENT SAMPLE (n=0), a normal early-life state — not
        # "no data at all", which an INNER JOIN would silently collapse into.
        all_ok = pd.read_sql_query(
            "SELECT DISTINCT signal_name, signal_version, metric_name FROM signal_values WHERE status='ok'",
            conn,
        )
        joined = pd.read_sql_query(
            """SELECT sv.signal_name, sv.signal_version, sv.metric_name, sv.value, sv.metric_as_of,
                      c.screen_score, c.fwd_ret_21d, c.candidate_date
               FROM signal_values sv
               JOIN candidates c ON c.id = sv.candidate_id
               WHERE sv.status = 'ok' AND sv.value IS NOT NULL AND c.fwd_ret_21d IS NOT NULL""",
            conn,
        )

    results = []
    if all_ok.empty:
        return results

    grouped = dict(list(joined.groupby(["signal_name", "signal_version", "metric_name"]))) if not joined.empty else {}

    for _, key_row in all_ok.iterrows():
        key = (key_row["signal_name"], key_row["signal_version"], key_row["metric_name"])
        signal_name, signal_version, metric_name = key
        raw_group = grouped.get(key)
        group, n_excluded = _split_lookahead_contaminated(raw_group) if raw_group is not None else (None, 0)
        n = len(group) if group is not None else 0
        row = {
            "signal_name": signal_name,
            "signal_version": signal_version,
            "metric_name": metric_name,
            "n": n,
            "n_excluded_lookahead": n_excluded,
            "median_staleness_days": _median_staleness_days(group) if group is not None and n > 0 else None,
        }
        if n < _MIN_SAMPLE:
            row["status"] = "INSUFFICIENT SAMPLE"
            results.append(row)
            continue

        # screen_score is currently always NULL at the one live log_candidate
        # call site — incremental IC degrades gracefully to raw IC in that case
        # since residualizing against an all-NULL/constant column is meaningless.
        has_screen_score = group["screen_score"].notna().all() and group["screen_score"].nunique() > 1

        raw_ic = _spearman(group["value"], group["fwd_ret_21d"])
        if has_screen_score:
            resid_value = _residualize(group["value"], group["screen_score"])
            resid_ret = _residualize(group["fwd_ret_21d"], group["screen_score"])
            incremental_ic = _spearman(resid_value, resid_ret)
        else:
            incremental_ic = raw_ic

        verdict = "PROMOTE-CANDIDATE" if abs(incremental_ic) >= _PROMOTE_THRESHOLD else "DELETE-CANDIDATE"
        row.update({
            "status": "ok",
            "raw_ic": round(raw_ic, 4),
            "incremental_ic": round(incremental_ic, 4),
            "verdict": verdict,
        })
        results.append(row)

    return results


def main() -> None:
    settings = get_settings()
    store = StateStore(settings.state_db_path)
    results = compute_uplift(store)

    if not results:
        print("No signal_values data yet.")
        return

    for r in results:
        header = f"{r['signal_name']} / {r['signal_version']} / {r['metric_name']}  (n={r['n']})"
        print(header)
        if r.get("n_excluded_lookahead"):
            print(f"  EXCLUDED {r['n_excluded_lookahead']} row(s): metric_as_of > candidate_date (lookahead leak, not counted in n or IC)")
        staleness = r.get("median_staleness_days")
        staleness_str = f"{staleness:.0f}d" if staleness is not None else "n/a"
        print(f"  median staleness (candidate_date - metric_as_of), clean rows only: {staleness_str}")
        if r["status"] == "INSUFFICIENT SAMPLE":
            print("  INSUFFICIENT SAMPLE — no conclusions")
        else:
            print(f"  raw IC={r['raw_ic']:+.4f}  incremental IC={r['incremental_ic']:+.4f}  -> {r['verdict']}")
            if staleness and staleness > 0:
                print("  NOTE: non-zero staleness — per CLAUDE.md Signal lifecycle, any PROMOTE-CANDIDATE here needs manual staleness review before it's trusted.")
        print()


if __name__ == "__main__":
    sys.exit(main())
