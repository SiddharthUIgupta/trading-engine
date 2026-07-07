"""Portfolio correlation guard.

Prevents the system from adding a position that is nearly identical to
one it already holds. On a watchlist of AAPL, MSFT, NVDA, SPY, QQQ,
AMZN, META, TSLA, the pairwise correlations are high (SPY/QQQ ≈ 0.97,
NVDA/AAPL ≈ 0.75, etc.) — without this check the system could easily
double up on the same effective exposure.

Two thresholds:
    HARD_BLOCK  (> 0.85) — reject outright. Adding this ticker gives
                the portfolio essentially zero new independent exposure.
                e.g. buying QQQ when already long SPY.

    SOFT_REDUCE (> 0.70) — allow but reduce Kelly fraction by CORR_PENALTY.
                The position is still meaningful but partially overlapping,
                so we size it down to account for reduced diversification.

Correlation is computed on daily log returns over the overlapping history.
Requires MIN_BARS overlapping bars; returns 0.0 (no correlation) when
price history is too short to be meaningful.
"""
from __future__ import annotations

import math

MIN_BARS = 20
HARD_BLOCK_THRESHOLD = 0.85
SOFT_REDUCE_THRESHOLD = 0.70
CORR_PENALTY = 0.30   # reduce Kelly by 30% when soft threshold is crossed


def _log_returns(closes: list[float]) -> list[float]:
    result = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            result.append(math.log(closes[i] / closes[i - 1]))
    return result


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < MIN_BARS:
        return None
    a, b = a[:n], b[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    denom = math.sqrt(var_a * var_b)
    if denom == 0.0:
        return None
    return cov / denom


def pairwise_correlation(closes_a: list[float], closes_b: list[float]) -> float | None:
    """Pearson correlation of daily log returns. None if insufficient history."""
    n = min(len(closes_a), len(closes_b))
    if n < MIN_BARS + 1:
        return None
    ret_a = _log_returns(closes_a[-n:])
    ret_b = _log_returns(closes_b[-n:])
    return _pearson(ret_a, ret_b)


def check_portfolio_correlation(
    proposed_closes: list[float],
    held_closes: dict[str, list[float]],   # ticker -> daily closes
) -> tuple[float, str]:
    """Compute the highest (signed) correlation between the proposed ticker
    and all currently held positions.

    Signed, not absolute: this guard exists to catch duplicate exposure
    (e.g. buying QQQ when already long SPY), not to flag hedges. A strong
    NEGATIVE correlation is an inverse/hedging relationship — it reduces net
    portfolio risk and should never be treated the same as a near-duplicate
    long position.

    Returns
    -------
    (max_correlation, description)
        max_correlation: highest signed r found (most positive, i.e. most
                         duplicate-like); 0.0 if no held positions,
                         insufficient history, or the highest value found
                         is negative (nothing to warn about).
        description:     human-readable summary for logging / risk officer.
    """
    if not held_closes:
        return 0.0, "no existing equity positions"

    results: list[tuple[str, float]] = []
    for ticker, closes in held_closes.items():
        r = pairwise_correlation(proposed_closes, closes)
        if r is not None:
            results.append((ticker, r))

    if not results:
        return 0.0, "insufficient price history for correlation check"

    max_ticker, max_r = max(results, key=lambda x: x[1])
    all_str = ", ".join(
        f"{t}={r:+.2f}" for t, r in sorted(results, key=lambda x: -x[1])
    )
    desc = f"highest correlation: {max_ticker}={max_r:+.2f} (all held: {all_str})"
    return max(max_r, 0.0), desc


def apply_correlation_adjustment(
    kelly_fraction: float,
    max_correlation: float,
    correlation_description: str,
) -> tuple[float, str, bool]:
    """Adjust Kelly fraction based on portfolio correlation.

    Returns
    -------
    (adjusted_fraction, reason, hard_blocked)
        hard_blocked=True means the trade should be rejected outright.
    """
    if max_correlation > HARD_BLOCK_THRESHOLD:
        return 0.0, (
            f"HARD BLOCK: {correlation_description} — "
            f"correlation {max_correlation:.2f} > {HARD_BLOCK_THRESHOLD} adds near-zero "
            "independent exposure; rejecting to avoid concentrated duplicate"
        ), True

    if max_correlation > SOFT_REDUCE_THRESHOLD:
        adjusted = kelly_fraction * (1.0 - CORR_PENALTY)
        return adjusted, (
            f"correlation reduction: {correlation_description} — "
            f"r={max_correlation:.2f} > {SOFT_REDUCE_THRESHOLD}; "
            f"Kelly {kelly_fraction:.1%} → {adjusted:.1%} (-{CORR_PENALTY:.0%})"
        ), False

    return kelly_fraction, (
        f"correlation acceptable: {correlation_description} (r={max_correlation:.2f})"
    ), False
