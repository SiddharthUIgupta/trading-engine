"""Deterministic, keyword-based headline sentiment scoring.

Exists because no free OpenBB news provider returns a pre-computed
sentiment field — `benzinga`/`fmp`/`intrinio`/`tiingo` do, but all require
paid API keys, and `yfinance` (the only free provider) returns raw
headlines with no sentiment column at all. Computing it here from headline
text keeps the data layer self-contained and free, rather than silently
defaulting every score to a constant 0.0 "neutral" placeholder, which masks
a real signal as a real read.

Deliberately not an LLM call: the data layer must not depend on the
analyst layer, and re-running an LLM per headline just to feed a single
score into another LLM call downstream would be redundant cost for a
narrow, well-suited-to-keywords task.
"""
from __future__ import annotations

import re

_POSITIVE_TERMS = [
    "surge", "surges", "surged", "soar", "soars", "soared", "rally", "rallies", "rallied",
    "beat", "beats", "beating", "outperform", "outperforms", "upgrade", "upgrades", "upgraded",
    "record high", "all-time high", "raises guidance", "raised guidance", "raises forecast",
    "strong demand", "strong growth", "buyback", "buybacks", "share repurchase",
    "expands", "expansion", "partnership", "approval", "approved", "breakthrough",
    "profit jumps", "profit rises", "revenue jumps", "revenue beats", "earnings beat",
    "bullish", "tops estimates", "exceeds expectations", "wins contract", "secures deal",
    "acquisition", "merger agreement", "dividend increase", "raises dividend",
]

_NEGATIVE_TERMS = [
    "plunge", "plunges", "plunged", "tumble", "tumbles", "tumbled", "slump", "slumps", "slumped",
    "miss", "misses", "missing estimates", "downgrade", "downgrades", "downgraded",
    "lawsuit", "sues", "sued", "recall", "recalls", "investigation", "investigated", "probe",
    "bankruptcy", "bankrupt", "layoffs", "lays off", "job cuts", "guidance cut", "cuts guidance",
    "cuts forecast", "lowers forecast", "decline", "declines", "weak demand", "fraud",
    "bearish", "sell-off", "selloff", "warns", "warning", "misses expectations",
    "default", "delisted", "delisting", "downturn", "loss widens", "losses widen",
    "shortfall", "scandal", "resigns", "ousted", "halted", "suspends",
]

_POS_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t in _POSITIVE_TERMS) + r")\b", re.IGNORECASE)
_NEG_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t in _NEGATIVE_TERMS) + r")\b", re.IGNORECASE)


def score_headline(text: str) -> float | None:
    """Returns a score in [-1, 1], or None if the headline contains no
    recognized sentiment-bearing terms at all (a real "no signal" rather
    than a fabricated neutral reading).
    """
    pos = len(_POS_PATTERN.findall(text))
    neg = len(_NEG_PATTERN.findall(text))
    if pos == 0 and neg == 0:
        return None
    return (pos - neg) / (pos + neg)


def score_headlines(texts: list[str]) -> float:
    """Average across ALL headlines, not just ones with a keyword match —
    a headline with no recognized term contributes 0 (no detected signal),
    rather than being dropped from the denominator. Dropping them would let
    one off-topic but sentiment-bearing headline (e.g. a lone "investigation"
    among nine unrelated stories) swing the whole aggregate to a full +-1.0,
    overstating conviction the underlying sample doesn't support.
    """
    if not texts:
        return 0.0
    scores = [score_headline(t) or 0.0 for t in texts]
    return sum(scores) / len(scores)
