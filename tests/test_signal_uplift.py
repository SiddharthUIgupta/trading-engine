from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from execution_layer.state_store import StateStore
from scripts.signal_uplift import compute_uplift, _median_staleness_days, _residualize, _spearman, _MIN_SAMPLE, _PROMOTE_THRESHOLD


def _seed(
    store: StateStore, n: int, signal_name: str = "kronos_small", signal_version: str = "v1",
    metric_name: str = "p_touch_win", value_fn=None, screen_score_fn=None, metric_as_of_fn=None,
) -> None:
    rng = np.random.default_rng(42)
    for i in range(n):
        d = date(2024, 1, 1) + timedelta(days=i)
        screen_score = screen_score_fn(i) if screen_score_fn else None
        store.log_candidate(
            candidate_date=d, strategy="thesis", ticker=f"T{i}",
            llm_verdict="BUY", gate_result="APPROVED", traded=True,
            screen_score=screen_score,
        )
        cid = store.get_candidate_id(d, "thesis", f"T{i}")
        fwd_ret = rng.normal(0, 0.1)
        store.update_candidate_forward_return(cid, 21, fwd_ret)
        value = value_fn(i, fwd_ret) if value_fn else rng.normal(0, 0.1)
        metric_as_of = metric_as_of_fn(d) if metric_as_of_fn else d.isoformat()
        store.record_signal_values(
            cid, signal_name, signal_version, {metric_name: value}, status="ok", metric_as_of=metric_as_of,
        )


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "uplift_test.sqlite3")


def test_below_min_sample_reports_insufficient(store: StateStore):
    _seed(store, n=50)
    results = compute_uplift(store)
    assert len(results) == 1
    assert results[0]["status"] == "INSUFFICIENT SAMPLE"
    assert results[0]["n"] == 50
    assert "raw_ic" not in results[0]
    assert "verdict" not in results[0]


def test_perfect_correlation_gives_ic_near_one_and_promotes(store: StateStore):
    # value == fwd_ret_21d exactly -> perfect rank correlation
    _seed(store, n=_MIN_SAMPLE + 10, value_fn=lambda i, fwd_ret: fwd_ret)
    results = compute_uplift(store)
    assert len(results) == 1
    r = results[0]
    assert r["n"] >= _MIN_SAMPLE
    assert r["raw_ic"] == pytest.approx(1.0, abs=1e-6)
    assert r["verdict"] == "PROMOTE-CANDIDATE"


def test_uncorrelated_signal_deletes(store: StateStore):
    rng = np.random.default_rng(7)
    # independent noise, no relationship to fwd_ret_21d. n is well above
    # _MIN_SAMPLE so the expected noise correlation (~1/sqrt(n)) is
    # reliably under _PROMOTE_THRESHOLD regardless of random seed.
    _seed(store, n=3000, value_fn=lambda i, fwd_ret: rng.normal(0, 1))
    results = compute_uplift(store)
    assert len(results) == 1
    r = results[0]
    assert abs(r["incremental_ic"]) < _PROMOTE_THRESHOLD
    assert r["verdict"] == "DELETE-CANDIDATE"


def test_incremental_ic_controls_for_screen_score():
    """Hand-computed reference: y and x both driven by screen_score plus
    independent noise. Raw IC should pick up the shared screen_score
    component; incremental IC (residualized on screen_score) should be
    much smaller, since once screen_score is controlled for, x and y are
    independent.
    """
    rng = np.random.default_rng(3)
    n = 500
    screen_score = rng.normal(0, 1, n)
    noise_y = rng.normal(0, 0.05, n)
    noise_x = rng.normal(0, 0.05, n)
    y = 0.8 * screen_score + noise_y
    x = 0.8 * screen_score + noise_x

    import pandas as pd
    y_s, x_s, ss_s = pd.Series(y), pd.Series(x), pd.Series(screen_score)

    raw_ic = _spearman(x_s, y_s)
    resid_x = _residualize(x_s, ss_s)
    resid_y = _residualize(y_s, ss_s)
    incremental_ic = _spearman(resid_x, resid_y)

    assert raw_ic > 0.5, "raw IC should be inflated by the shared screen_score driver"
    assert abs(incremental_ic) < 0.2, "incremental IC should be much smaller once screen_score is controlled for"


def test_pit_clean_signal_reports_zero_staleness(store: StateStore):
    """A Kronos-like signal where metric_as_of == candidate_date always
    (the default, when a provider has no get_metric_as_of) should report
    exactly 0 staleness — a confirmation, not an assumption.
    """
    _seed(store, n=_MIN_SAMPLE + 10)  # default metric_as_of_fn = candidate_date
    results = compute_uplift(store)
    assert len(results) == 1
    assert results[0]["median_staleness_days"] == 0.0


def test_current_snapshot_signal_reports_real_staleness(store: StateStore):
    """A short-interest-like signal where metric_as_of consistently lags
    candidate_date (e.g. by 20 days, matching FINRA's settlement cadence)
    must have that staleness surfaced, not silently reported as 0.
    """
    _seed(
        store, n=_MIN_SAMPLE + 10,
        metric_as_of_fn=lambda candidate_date: (candidate_date - timedelta(days=20)).isoformat(),
    )
    results = compute_uplift(store)
    assert len(results) == 1
    assert results[0]["median_staleness_days"] == pytest.approx(20.0)


def test_median_staleness_days_returns_none_when_never_recorded():
    import pandas as pd
    group = pd.DataFrame({"metric_as_of": [None, None], "candidate_date": ["2026-01-01", "2026-01-02"]})
    assert _median_staleness_days(group) is None


def test_screen_score_populated_uses_incremental_path(store: StateStore):
    rng = np.random.default_rng(11)
    screen_scores = rng.normal(0, 1, _MIN_SAMPLE + 10)

    def value_fn(i, fwd_ret):
        return 0.8 * screen_scores[i] + rng.normal(0, 0.05)

    # fwd_ret_21d itself is independent random noise (seeded in _seed), but we
    # want fwd_ret correlated with screen_score too for a meaningful check —
    # simplest: just confirm the code path runs and produces a verdict when
    # screen_score has real variance (has_screen_score branch is exercised).
    _seed(store, n=_MIN_SAMPLE + 10, value_fn=value_fn, screen_score_fn=lambda i: screen_scores[i])
    results = compute_uplift(store)
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert "incremental_ic" in results[0]
