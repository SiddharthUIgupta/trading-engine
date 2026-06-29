"""Lesson retrieval and prompt injection for agent memory.

Three responsibilities:
  1. derive_setup_tags  — compute setup tags from pre-LLM data so retrieval
     can happen before any Claude call.
  2. get_relevant_lessons — query the state store for lessons whose tags
     overlap with the current setup, scored by overlap count.
  3. format_for_prompt — render retrieved lessons as a [LESSONS] block ready
     to append to any agent's user prompt.
"""
from __future__ import annotations

import json
import logging

from data_layer.models import PriceSeries

logger = logging.getLogger(__name__)


def derive_setup_tags(
    price_series: PriceSeries,
    strategy: str,
    *,
    iv_rank: float | None = None,
    earnings_within_dte: bool = False,
) -> list[str]:
    """Derive setup tags from pre-LLM analysis context.

    Tags are used to retrieve relevant past lessons before running the consensus.
    Kept deliberately simple — the goal is recall, not precision.
    """
    tags: list[str] = [strategy]
    closes = [b.close for b in price_series.bars]
    volumes = [b.volume for b in price_series.bars]

    # RSI (14-period approximation from last 15 closes)
    if len(closes) >= 15:
        gains, losses = [], []
        for i in range(1, 15):
            delta = closes[-15 + i] - closes[-15 + i - 1]
            (gains if delta >= 0 else losses).append(abs(delta))
        avg_gain = sum(gains) / 14 if gains else 0.0
        avg_loss = sum(losses) / 14 if losses else 1e-9
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        if rsi >= 65:
            tags.append("high_rsi")
        elif rsi <= 35:
            tags.append("low_rsi")

    # Trend regime from SMA 10 vs SMA 30
    if len(closes) >= 30:
        sma10 = sum(closes[-10:]) / 10
        sma30 = sum(closes[-30:]) / 30
        tags.append("bull_regime" if sma10 > sma30 else "bear_regime")
    elif len(closes) >= 5:
        tags.append("neutral_regime")

    # Volume spike vs 10-day average
    if len(volumes) >= 11:
        avg_vol = sum(volumes[-11:-1]) / 10
        if avg_vol > 0:
            ratio = volumes[-1] / avg_vol
            if ratio >= 2.0:
                tags.append("volume_spike")
            elif ratio <= 0.5:
                tags.append("low_volume")

    # Gap from prior close
    if len(closes) >= 2 and closes[-2] > 0:
        gap_pct = (closes[-1] - closes[-2]) / closes[-2]
        if gap_pct >= 0.03:
            tags.append("gap_up")
        elif gap_pct <= -0.03:
            tags.append("gap_down")

    # Vol-track specific
    if iv_rank is not None:
        if iv_rank >= 70:
            tags.append("high_iv_rank")
        elif iv_rank <= 30:
            tags.append("low_iv_rank")
    if earnings_within_dte:
        tags.append("earnings_adjacent")

    return tags


def get_relevant_lessons(
    state_store,
    strategy: str,
    setup_tags: list[str],
    limit: int = 5,
) -> list[dict]:
    """Return the most setup-relevant recent lessons, scored by tag overlap."""
    all_lessons = state_store.get_lessons(strategy=strategy, limit=200)
    current_tags = set(setup_tags)

    scored: list[tuple[int, dict]] = []
    for lesson in all_lessons:
        if lesson.get("score", 1.0) < 0.3:
            continue  # retired lesson — consistently correlated with losses
        lesson_tags = set(json.loads(lesson.get("setup_tags_json", "[]")))
        overlap = len(lesson_tags & current_tags)
        if overlap > 0:
            scored.append((overlap, lesson))

    # Within same overlap tier, higher-scored lessons rank first
    scored.sort(key=lambda x: (x[0], x[1].get("score", 1.0)), reverse=True)
    return [lesson for _, lesson in scored[:limit]]


def format_for_prompt(lessons: list[dict]) -> str:
    """Format retrieved lessons into a prompt block for agent injection."""
    if not lessons:
        return ""
    lines = [
        "[LESSONS FROM SIMILAR PAST TRADES]",
        "These are real outcomes from trades in similar setups. Apply them when reasoning.",
    ]
    for i, lesson in enumerate(lessons, 1):
        date_str = lesson.get("created_at", "")[:10]
        win_loss = "WIN" if lesson.get("outcome_was_win") else "LOSS"
        score = lesson.get("score", 1.0)
        score_str = f" score={score:.1f}" if score != 1.0 else ""
        lines.append(f"{i}. ({date_str}, {win_loss}{score_str}) {lesson['lesson']}")
    lines.append("")
    return "\n".join(lines)
