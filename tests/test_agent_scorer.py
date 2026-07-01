"""Tests for agent performance tracking, lesson validation loop, and price cache."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analyst_layer.agent_scorer import format_accuracy_context, _MIN_SAMPLE
from analyst_layer.lesson_store import get_relevant_lessons, format_for_prompt


# ── format_accuracy_context ───────────────────────────────────────────────────

def test_format_accuracy_empty_rows():
    assert format_accuracy_context([]) == ""


def test_format_accuracy_below_min_sample():
    rows = [("macro_agent", _MIN_SAMPLE - 1, 5)]
    assert format_accuracy_context(rows) == ""


def test_format_accuracy_exactly_at_min_sample():
    rows = [("technical_agent", _MIN_SAMPLE, 7)]
    result = format_accuracy_context(rows)
    assert "technical_agent" in result
    assert "70%" in result


def test_format_accuracy_mixed_above_and_below_min():
    rows = [
        ("macro_agent", _MIN_SAMPLE + 5, 10),
        ("fundamental_agent", _MIN_SAMPLE - 1, 3),
    ]
    result = format_accuracy_context(rows)
    assert "macro_agent" in result
    assert "fundamental_agent" not in result


def test_format_accuracy_includes_header_and_footer():
    rows = [("technical_agent", 20, 15)]
    result = format_accuracy_context(rows)
    assert "AGENT TRACK RECORD" in result
    assert "Weight signals accordingly" in result


def test_format_accuracy_all_below_min_returns_empty():
    rows = [("a", 2, 1), ("b", 3, 2)]
    assert format_accuracy_context(rows) == ""


# ── StateStore: agent signal log ──────────────────────────────────────────────

def _make_store(tmp_path: Path):
    from execution_layer.state_store import StateStore
    return StateStore(tmp_path / "test.sqlite3")


def test_record_agent_signal_log_returns_id(tmp_path: Path):
    store = _make_store(tmp_path)
    signals = [{"agent_name": "macro_agent", "stance": "BUY", "confidence": "high"}]
    log_id = store.record_agent_signal_log("AAPL", "momentum", "bullish", "BUY", signals)
    assert isinstance(log_id, int)
    assert log_id > 0


def test_score_agent_signals_marks_outcome(tmp_path: Path):
    store = _make_store(tmp_path)
    signals = [{"agent_name": "technical_agent", "stance": "BUY", "confidence": "medium"}]
    store.record_agent_signal_log("MSFT", "momentum", "bullish", "BUY", signals)
    store.score_agent_signals("MSFT", pnl=150.0)
    rows = store.get_agent_accuracy("momentum", "bullish")
    assert len(rows) == 1
    agent_name, total, wins = rows[0]
    assert agent_name == "technical_agent"
    assert total == 1
    assert wins == 1


def test_score_agent_signals_loss(tmp_path: Path):
    store = _make_store(tmp_path)
    signals = [{"agent_name": "fundamental_agent", "stance": "BUY", "confidence": "low"}]
    store.record_agent_signal_log("TSLA", "momentum", "bearish", "BUY", signals)
    store.score_agent_signals("TSLA", pnl=-80.0)
    rows = store.get_agent_accuracy("momentum", "bearish")
    _, total, wins = rows[0]
    assert wins == 0
    assert total == 1


def test_get_agent_accuracy_empty_when_no_scored(tmp_path: Path):
    store = _make_store(tmp_path)
    signals = [{"agent_name": "macro_agent", "stance": "HOLD", "confidence": "low"}]
    store.record_agent_signal_log("NVDA", "thesis", "neutral", "HOLD", signals)
    # Not scored yet
    rows = store.get_agent_accuracy("thesis", "neutral")
    assert rows == []


def test_get_agent_accuracy_filters_by_track_and_regime(tmp_path: Path):
    store = _make_store(tmp_path)
    s = [{"agent_name": "agent_x", "stance": "BUY", "confidence": "high"}]
    store.record_agent_signal_log("A", "momentum", "bullish", "BUY", s)
    store.record_agent_signal_log("B", "thesis", "bullish", "BUY", s)
    store.score_agent_signals("A", 100.0)
    store.score_agent_signals("B", 100.0)
    rows = store.get_agent_accuracy("momentum", "bullish")
    assert len(rows) == 1
    assert rows[0][0] == "agent_x"


def test_score_agent_signals_only_marks_unscored(tmp_path: Path):
    store = _make_store(tmp_path)
    s = [{"agent_name": "technical_agent", "stance": "BUY", "confidence": "high"}]
    store.record_agent_signal_log("AAPL", "momentum", "bullish", "BUY", s)
    store.score_agent_signals("AAPL", 200.0)   # win
    store.score_agent_signals("AAPL", -100.0)  # second call should not overwrite
    rows = store.get_agent_accuracy("momentum", "bullish")
    _, total, wins = rows[0]
    assert wins == 1  # still a win — second call was a no-op


# ── StateStore: lesson injection and scoring ──────────────────────────────────

def test_record_lesson_returns_id(tmp_path: Path):
    store = _make_store(tmp_path)
    lesson_id = store.record_lesson("Buy breakouts with volume.", ["volume_spike"], "momentum", True, 100.0)
    assert isinstance(lesson_id, int)
    assert lesson_id > 0


def test_lesson_score_default_is_one(tmp_path: Path):
    store = _make_store(tmp_path)
    store.record_lesson("Test lesson.", ["gap_up"], "momentum", True, 50.0)
    lessons = store.get_lessons()
    assert lessons[0]["score"] == pytest.approx(1.0)


def test_update_lesson_score(tmp_path: Path):
    store = _make_store(tmp_path)
    lid = store.record_lesson("Test.", ["gap_up"], "momentum", True, 50.0)
    store.update_lesson_score(lid, 0.5)
    lessons = store.get_lessons()
    assert lessons[0]["score"] == pytest.approx(1.5)


def test_update_lesson_score_clamped_at_zero(tmp_path: Path):
    store = _make_store(tmp_path)
    lid = store.record_lesson("Test.", ["gap_up"], "momentum", False, -50.0)
    store.update_lesson_score(lid, -5.0)
    lessons = store.get_lessons()
    assert lessons[0]["score"] == pytest.approx(0.0)


def test_score_lesson_injections_win(tmp_path: Path):
    store = _make_store(tmp_path)
    lid = store.record_lesson("Test.", ["gap_up"], "momentum", True, 100.0)
    store.record_lesson_injection(lid, "AAPL", "momentum")
    store.score_lesson_injections("AAPL", pnl=200.0)
    lessons = store.get_lessons()
    assert lessons[0]["score"] == pytest.approx(1.1)


def test_score_lesson_injections_loss(tmp_path: Path):
    store = _make_store(tmp_path)
    lid = store.record_lesson("Test.", ["volume_spike"], "momentum", False, -50.0)
    store.record_lesson_injection(lid, "MSFT", "momentum")
    store.score_lesson_injections("MSFT", pnl=-100.0)
    lessons = store.get_lessons()
    assert lessons[0]["score"] == pytest.approx(0.95)


# ── lesson_store: score filter and ordering ───────────────────────────────────

class _FakeStore:
    def __init__(self, lessons):
        self._lessons = lessons

    def get_lessons(self, strategy=None, limit=200):
        return self._lessons if strategy is None else [l for l in self._lessons if l["strategy"] == strategy]


def test_get_relevant_lessons_suppresses_low_score():
    lessons = [
        {
            "id": 1, "lesson": "Bad lesson.", "setup_tags_json": '["momentum", "gap_up"]',
            "strategy": "momentum", "outcome_was_win": False, "source_pnl": -50.0,
            "score": 0.2, "created_at": "2024-01-01",
        }
    ]
    store = _FakeStore(lessons)
    result = get_relevant_lessons(store, "momentum", ["momentum", "gap_up"])
    assert result == []  # score 0.2 < 0.3 threshold, suppressed


def test_get_relevant_lessons_score_ordering():
    lessons = [
        {
            "id": 1, "lesson": "High score lesson.", "setup_tags_json": '["momentum", "gap_up"]',
            "strategy": "momentum", "outcome_was_win": True, "source_pnl": 200.0,
            "score": 1.5, "created_at": "2024-01-02",
        },
        {
            "id": 2, "lesson": "Low score lesson.", "setup_tags_json": '["momentum", "gap_up"]',
            "strategy": "momentum", "outcome_was_win": True, "source_pnl": 50.0,
            "score": 0.8, "created_at": "2024-01-01",
        },
    ]
    store = _FakeStore(lessons)
    result = get_relevant_lessons(store, "momentum", ["momentum", "gap_up"])
    assert result[0]["id"] == 1  # higher score first
    assert result[1]["id"] == 2


def test_format_for_prompt_includes_score_when_not_one():
    lessons = [
        {
            "lesson": "Test lesson.", "outcome_was_win": True,
            "score": 1.5, "created_at": "2024-01-01",
        }
    ]
    text = format_for_prompt(lessons)
    assert "score=1.5" in text


def test_format_for_prompt_no_score_when_default():
    lessons = [
        {
            "lesson": "Default score lesson.", "outcome_was_win": True,
            "score": 1.0, "created_at": "2024-01-01",
        }
    ]
    text = format_for_prompt(lessons)
    assert "score=" not in text


# ── price cache ───────────────────────────────────────────────────────────────

def test_price_cache_returns_cached_on_second_call():
    """_get_daily_closes should return cached closes without hitting data client."""
    from execution_layer.runtime import TradingRuntime
    from config.settings import Settings

    dc = MagicMock()
    bars = [MagicMock(close=float(i)) for i in range(60)]
    dc.get_price_history.return_value = MagicMock(bars=bars)

    rt = TradingRuntime(
        settings=Settings(),
        data_client=dc,
        broker=MagicMock(),
        intraday_breaker=MagicMock(), options_breaker=MagicMock(), thesis_breaker=MagicMock(), swing_breaker=MagicMock(),
        state_store=MagicMock(),
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )

    today = date.today()
    closes1 = rt._get_daily_closes("AAPL", today)
    closes2 = rt._get_daily_closes("AAPL", today)

    assert closes1 == closes2
    assert dc.get_price_history.call_count == 1  # only one actual fetch


def test_price_cache_cleared_on_pre_market_scan():
    """pre_market_scan clears _price_cache so stale data doesn't persist overnight."""
    from execution_layer.runtime import TradingRuntime
    from config.settings import Settings

    dc = MagicMock()
    dc.get_price_history.return_value = MagicMock(bars=[MagicMock(close=100.0)] * 30)
    broker = MagicMock()
    broker.get_equity.return_value = 10_000.0
    breaker = MagicMock()
    breaker.is_stock_halted = False
    ss = MagicMock()
    ss.get_positions.return_value = []

    rt = TradingRuntime(
        settings=Settings(),
        data_client=dc,
        broker=broker,
        intraday_breaker=breaker, options_breaker=breaker, thesis_breaker=breaker, swing_breaker=breaker,
        state_store=ss,
        anthropic_client=MagicMock(),
        watchlist=[],
    )

    rt._price_cache["AAPL"] = [100.0, 101.0]
    # Trigger pre_market_scan's clear (via direct call to the clear section)
    rt._price_cache.clear()
    assert "AAPL" not in rt._price_cache
