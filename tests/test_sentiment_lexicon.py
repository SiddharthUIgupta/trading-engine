from __future__ import annotations

from data_layer.sentiment_lexicon import score_headline, score_headlines


def test_score_headline_returns_none_with_no_recognized_terms():
    assert score_headline("Tesla unveils new factory tour for investors") is None


def test_score_headline_positive_terms():
    assert score_headline("Company beats estimates and raises guidance") == 1.0


def test_score_headline_negative_terms():
    assert score_headline("Firm under investigation amid fraud allegations") == -1.0


def test_score_headline_mixed_terms_partially_cancel():
    score = score_headline("Stock surges after earnings beat but warns on guidance cuts")
    assert -1.0 < score < 1.0


def test_score_headlines_treats_no_signal_as_neutral_not_excluded():
    """Regression test: a single strongly negative headline among many
    unrelated ones must not dominate the aggregate the way it would if
    no-signal headlines were dropped from the denominator instead of
    counted as a neutral 0.
    """
    texts = [
        "Fatal incident under federal investigation",
        "Company hosts annual shareholder meeting",
        "New product line announced",
        "Quarterly report scheduled for next week",
        "Factory tour highlights production process",
    ]
    score = score_headlines(texts)
    assert -0.3 < score < 0  # negative, but far from the full -1.0 single-headline value


def test_score_headlines_empty_list_is_neutral():
    assert score_headlines([]) == 0.0


def test_score_headlines_all_neutral_is_zero():
    texts = ["Quarterly report scheduled", "Annual meeting held", "Product tour announced"]
    assert score_headlines(texts) == 0.0
