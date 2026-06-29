from __future__ import annotations

from analyst_layer.thesis_scanner import evaluate_thesis_candidate
from data_layer.models import ThesisCandidate

_KWARGS = dict(min_pullback_pct=0.20, max_pullback_pct=0.50)


def _candidate(price: float, year_high: float) -> ThesisCandidate:
    return ThesisCandidate(symbol="TEST", price=price, year_high=year_high, year_low=year_high * 0.5)


def test_passes_when_pullback_within_band():
    candidate = _candidate(price=80.0, year_high=100.0)  # 20% off highs
    signal = evaluate_thesis_candidate(candidate, **_KWARGS)
    assert signal.passed is True
    assert signal.score == 0.20


def test_fails_when_pullback_below_minimum():
    candidate = _candidate(price=95.0, year_high=100.0)  # only 5% off highs
    signal = evaluate_thesis_candidate(candidate, **_KWARGS)
    assert signal.passed is False
    assert signal.score == 0.0


def test_fails_when_pullback_beyond_ceiling():
    """Down 80% — statistically more likely genuinely impaired than a
    quality name temporarily out of favor. Must not pass even though it
    clears the floor."""
    candidate = _candidate(price=20.0, year_high=100.0)
    signal = evaluate_thesis_candidate(candidate, **_KWARGS)
    assert signal.passed is False
    assert signal.score == 0.0
    assert any("ceiling" in r for r in signal.reasons)


def test_passes_at_exact_ceiling():
    candidate = _candidate(price=50.0, year_high=100.0)  # exactly 50% off highs
    signal = evaluate_thesis_candidate(candidate, **_KWARGS)
    assert signal.passed is True
    assert signal.score == 0.50


def test_score_scales_with_pullback_magnitude_within_band_for_ranking():
    shallow = evaluate_thesis_candidate(_candidate(85.0, 100.0), min_pullback_pct=0.10, max_pullback_pct=0.50)
    deep = evaluate_thesis_candidate(_candidate(60.0, 100.0), min_pullback_pct=0.10, max_pullback_pct=0.50)
    assert deep.score > shallow.score


def test_at_new_high_does_not_pass():
    candidate = _candidate(price=100.0, year_high=100.0)
    signal = evaluate_thesis_candidate(candidate, **_KWARGS)
    assert signal.passed is False
