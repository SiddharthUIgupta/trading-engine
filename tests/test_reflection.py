"""Tests for the agent learning system: reflection, lesson store, and runtime integration."""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analyst_layer.lesson_store import derive_setup_tags, format_for_prompt, get_relevant_lessons
from analyst_layer.reflection_agent import ReflectionAgent, ReflectionOutput, LessonOutput
from analyst_layer.schemas import (
    Action,
    AgentSignal,
    Confidence,
    ConsensusPayload,
    OrderType,
    RiskReview,
    RiskVerdict,
    TradeProposal,
)
from config.settings import Settings
from data_layer.models import PriceBar, PriceSeries
from execution_layer.guardrails import CircuitBreaker
from execution_layer.runtime import TradingRuntime
from execution_layer.state_store import StateStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price_series(n: int = 40, trend: str = "up") -> PriceSeries:
    bars = []
    price = 50.0
    now = datetime.now()
    for i in range(n):
        close = price * (1.002 if trend == "up" else 0.998)
        bars.append(PriceBar(
            symbol="TEST", timestamp=now - timedelta(days=n - i),
            open=price, high=close * 1.005, low=price * 0.995, close=close,
            volume=1_000_000,
        ))
        price = close
    return PriceSeries(symbol="TEST", interval="1d", bars=bars)


def _volume_spike_series(n: int = 40) -> PriceSeries:
    bars = []
    price = 50.0
    now = datetime.now()
    for i in range(n):
        close = price * 1.001
        vol = 5_000_000 if i == n - 1 else 500_000  # last bar has 10x volume
        bars.append(PriceBar(
            symbol="TEST", timestamp=now - timedelta(days=n - i),
            open=price, high=close * 1.01, low=price * 0.99, close=close, volume=vol,
        ))
        price = close
    return PriceSeries(symbol="TEST", interval="1d", bars=bars)


# ── derive_setup_tags ─────────────────────────────────────────────────────────

def test_derive_tags_includes_strategy():
    tags = derive_setup_tags(_price_series(), "momentum")
    assert "momentum" in tags


def test_derive_tags_bull_regime_for_uptrend():
    tags = derive_setup_tags(_price_series(40, trend="up"), "momentum")
    assert "bull_regime" in tags


def test_derive_tags_bear_regime_for_downtrend():
    tags = derive_setup_tags(_price_series(40, trend="down"), "momentum")
    assert "bear_regime" in tags


def test_derive_tags_volume_spike_detected():
    tags = derive_setup_tags(_volume_spike_series(), "momentum")
    assert "volume_spike" in tags


def test_derive_tags_iv_rank_high():
    tags = derive_setup_tags(_price_series(), "vol_short", iv_rank=75.0)
    assert "high_iv_rank" in tags


def test_derive_tags_iv_rank_low():
    tags = derive_setup_tags(_price_series(), "vol_short", iv_rank=20.0)
    assert "low_iv_rank" in tags


def test_derive_tags_earnings_adjacent():
    tags = derive_setup_tags(_price_series(), "vol_short", earnings_within_dte=True)
    assert "earnings_adjacent" in tags


def test_derive_tags_short_series_no_rsi():
    # < 15 bars → no RSI tags
    tags = derive_setup_tags(_price_series(5), "momentum")
    assert "high_rsi" not in tags
    assert "low_rsi" not in tags


# ── get_relevant_lessons ──────────────────────────────────────────────────────

def _mock_store_with_lessons(lessons: list[dict]):
    store = MagicMock()
    store.get_lessons.return_value = lessons
    return store


def _lesson_row(lesson: str, tags: list[str], strategy: str = "momentum", win: bool = True) -> dict:
    return {
        "lesson": lesson,
        "setup_tags_json": json.dumps(tags),
        "strategy": strategy,
        "outcome_was_win": win,
        "source_pnl": 50.0 if win else -30.0,
        "created_at": "2026-06-01T10:00:00",
    }


def test_get_relevant_lessons_returns_matching():
    store = _mock_store_with_lessons([
        _lesson_row("Lesson A", ["momentum", "volume_spike"]),
        _lesson_row("Lesson B", ["thesis_pullback"]),
    ])
    results = get_relevant_lessons(store, "momentum", ["momentum", "volume_spike"])
    assert any(r["lesson"] == "Lesson A" for r in results)


def test_get_relevant_lessons_excludes_no_overlap():
    store = _mock_store_with_lessons([
        _lesson_row("Lesson B", ["thesis_pullback"]),
    ])
    results = get_relevant_lessons(store, "momentum", ["momentum", "volume_spike"])
    assert results == []


def test_get_relevant_lessons_sorted_by_overlap():
    store = _mock_store_with_lessons([
        _lesson_row("One tag", ["momentum"]),
        _lesson_row("Two tags", ["momentum", "volume_spike"]),
    ])
    results = get_relevant_lessons(store, "momentum", ["momentum", "volume_spike"])
    assert results[0]["lesson"] == "Two tags"


def test_get_relevant_lessons_capped_at_limit():
    lessons = [_lesson_row(f"L{i}", ["momentum"]) for i in range(20)]
    store = _mock_store_with_lessons(lessons)
    results = get_relevant_lessons(store, "momentum", ["momentum"], limit=3)
    assert len(results) <= 3


# ── format_for_prompt ─────────────────────────────────────────────────────────

def test_format_for_prompt_empty_returns_empty_string():
    assert format_for_prompt([]) == ""


def test_format_for_prompt_contains_lesson_text():
    lessons = [_lesson_row("Gap-ups with declining volume after 30min are exhausted.", ["gap_up"])]
    text = format_for_prompt(lessons)
    assert "Gap-ups with declining volume" in text


def test_format_for_prompt_shows_win_loss():
    win_lesson = _lesson_row("Win lesson", ["momentum"], win=True)
    loss_lesson = _lesson_row("Loss lesson", ["momentum"], win=False)
    text = format_for_prompt([win_lesson, loss_lesson])
    assert "WIN" in text
    assert "LOSS" in text


# ── StateStore: record_lesson / get_lessons ───────────────────────────────────

def test_state_store_record_and_retrieve_lesson(tmp_path: Path):
    from execution_layer.state_store import StateStore
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_lesson(
        lesson="Test lesson",
        setup_tags=["momentum", "volume_spike"],
        strategy="momentum",
        outcome_was_win=True,
        source_pnl=42.0,
    )
    lessons = store.get_lessons(strategy="momentum")
    assert len(lessons) == 1
    assert lessons[0]["lesson"] == "Test lesson"
    assert lessons[0]["outcome_was_win"] is True
    assert json.loads(lessons[0]["setup_tags_json"]) == ["momentum", "volume_spike"]


def test_state_store_get_lessons_filters_by_strategy(tmp_path: Path):
    from execution_layer.state_store import StateStore
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_lesson("Momentum lesson", ["momentum"], "momentum", True, 10.0)
    store.record_lesson("Thesis lesson", ["thesis_pullback"], "thesis", False, -20.0)
    momentum = store.get_lessons(strategy="momentum")
    assert len(momentum) == 1
    assert momentum[0]["lesson"] == "Momentum lesson"


def test_state_store_record_reflection(tmp_path: Path):
    from execution_layer.state_store import StateStore
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_reflection(
        strategy="momentum",
        outcome_pnl=-150.0,
        outcome_win=False,
        what_happened="Bought into a gap-up that reversed.",
        root_cause="Volume declined after 10:30 — agents missed the distribution signal.",
        outcome_was_noise=False,
    )
    # No assertion on retrieval (no getter yet) — just verify no exception


# ── ReflectionAgent ───────────────────────────────────────────────────────────

def _mock_reflection_client(lessons: list[dict] | None = None) -> MagicMock:
    """Return a mock Anthropic client that emits a valid ReflectionOutput via tool call."""
    if lessons is None:
        lessons = [{"lesson": "Do not chase gap-ups without volume confirmation.", "setup_tags": ["gap_up", "momentum"]}]

    output = ReflectionOutput(
        what_happened="The stock gapped up but reversed when volume dried up.",
        root_cause="Agents missed that volume was declining after the initial spike.",
        lessons=[LessonOutput(**l) for l in lessons],
        outcome_was_noise=False,
    )
    block = MagicMock()
    block.type = "tool_use"
    block.name = "emit_reflection"
    block.input = output.model_dump()

    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )

    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_reflection_agent_returns_output():
    agent = ReflectionAgent(client=_mock_reflection_client(), model="claude-haiku-4-5-20251001")
    result = agent.reflect(
        strategy="momentum",
        agent_signals=[{"agent_name": "technical", "stance": "BUY", "confidence": "high", "rationale": "RSI crossover."}],
        outcome_pnl=-80.0,
        outcome_win=False,
        market_context={"rsi": "72", "volume_ratio": "1.2"},
    )
    assert result is not None
    assert isinstance(result.lessons, list)
    assert result.outcome_was_noise is False


def test_reflection_agent_returns_none_for_empty_signals():
    agent = ReflectionAgent(client=MagicMock(), model="claude-haiku-4-5-20251001")
    result = agent.reflect(
        strategy="momentum",
        agent_signals=[],
        outcome_pnl=50.0,
        outcome_win=True,
        market_context={},
    )
    assert result is None


def test_reflection_agent_handles_client_error_gracefully():
    client = MagicMock()
    client.messages.create.side_effect = Exception("API timeout")
    agent = ReflectionAgent(client=client, model="claude-haiku-4-5-20251001")
    result = agent.reflect(
        strategy="momentum",
        agent_signals=[{"agent_name": "technical", "stance": "BUY", "confidence": "medium", "rationale": "test"}],
        outcome_pnl=-30.0,
        outcome_win=False,
        market_context={},
    )
    assert result is None


# ── Runtime integration ───────────────────────────────────────────────────────

def test_trigger_reflection_spawns_thread_on_sell(tmp_path: Path):
    """_trigger_reflection spawns a daemon thread — verify it's called after a sale."""
    settings = Settings(_env_file=None)
    store = StateStore(tmp_path / "test.sqlite3")
    rt = TradingRuntime(
        settings=settings,
        data_client=MagicMock(),
        broker=MagicMock(),
        intraday_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02), options_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02), thesis_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02), swing_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02),
        state_store=store,
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )

    spawned: list[str] = []
    original_start = threading.Thread.start

    def _capture_start(self):
        if "reflect" in self.name:
            spawned.append(self.name)
        original_start(self)

    with patch.object(threading.Thread, "start", _capture_start):
        rt._trigger_reflection("AAPL", "momentum", pnl=75.0)

    assert any("reflect" in name for name in spawned)


def test_lessons_injected_into_consensus(tmp_path: Path):
    """run_consensus receives non-empty lessons string when past lessons exist and injection is not frozen."""
    settings = Settings(_env_file=None, FREEZE_LESSON_INJECTION=False)
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_lesson(
        lesson="Gap-ups with declining volume fail after 30 min.",
        setup_tags=["momentum", "gap_up"],
        strategy="momentum",
        outcome_was_win=False,
        source_pnl=-50.0,
    )

    lessons_received: list[str] = []

    def _mock_run_consensus(**kwargs):
        lessons_received.append(kwargs.get("lessons", ""))
        return ConsensusPayload(
            ticker=kwargs["ticker"],
            signals=[],
            proposal=TradeProposal(ticker=kwargs["ticker"], action=Action.HOLD, quantity=0, limit_price=100.0),
            risk_review=RiskReview(
                verdict=RiskVerdict.REJECTED,
                reasons=["test"],
                max_position_size_pct_checked=0.05,
                max_daily_drawdown_pct_checked=0.02,
                reviewed_at=datetime.utcnow(),
            ),
        )

    # Build a price series that includes a gap-up on the last bar so "gap_up"
    # tag is derived and overlaps with the seeded lesson's tags.
    bars = [
        PriceBar(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(days=40 - i),
            open=50.0, high=51.0, low=49.0,
            close=50.0 + i * 0.1,
            volume=500_000,
        )
        for i in range(38)
    ]
    bars.append(PriceBar(symbol="AAPL", timestamp=datetime.now() - timedelta(days=1),
                         open=53.5, high=55.0, low=53.0, close=54.0, volume=1_500_000))
    bars.append(PriceBar(symbol="AAPL", timestamp=datetime.now(),
                         open=55.8, high=57.0, low=55.5, close=56.5, volume=2_000_000))
    price_series = PriceSeries(symbol="AAPL", interval="1d", bars=bars)

    dc = MagicMock()
    dc.get_sentiment.return_value = MagicMock()
    dc.get_fundamentals.return_value = MagicMock()
    dc.get_recent_filings.return_value = []
    dc.get_price_history.return_value = price_series

    broker = MagicMock()
    broker.get_position_shares.return_value = 0

    with patch("execution_layer.runtime.run_consensus", side_effect=_mock_run_consensus):
        rt = TradingRuntime(
            settings=settings,
            data_client=dc,
            broker=broker,
            intraday_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02), options_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02), thesis_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02), swing_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02),
            state_store=store,
            anthropic_client=MagicMock(),
            watchlist=["AAPL"],
        )
        rt._scan_and_run_consensus(["AAPL"], date.today(), equity=10_000.0, strategy="momentum")

    assert lessons_received, "run_consensus was not called"
    assert "Gap-ups with declining volume" in lessons_received[0]
