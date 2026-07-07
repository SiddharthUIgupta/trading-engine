from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from data_layer.models import PriceBar, PriceSeries, ThesisCandidate
from execution_layer.alpha_plane import AlphaRuntime
from execution_layer.guardrails import CircuitBreaker
from execution_layer.state_store import StateStore


@pytest.fixture
def alpha(tmp_path: Path) -> AlphaRuntime:
    settings = Settings(_env_file=None)
    broker = MagicMock()
    broker.get_equity.return_value = 100_000.0

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    breaker.start_trading_day(equity=100_000.0, today=date.today())

    store = StateStore(tmp_path / "alpha_test.sqlite3")

    return AlphaRuntime(
        settings=settings,
        data_client=MagicMock(),
        broker=broker,
        intraday_breaker=breaker, options_breaker=breaker, thesis_breaker=breaker, swing_breaker=breaker,
        state_store=store,
        anthropic_client=MagicMock(),
        watchlist=["AAPL"],
    )


# ── _build_thesis_candidates ────────────────────────────────────────────────
# Regression for a real production bug: this previously called a nonexistent
# `thesis_scanner.build_thesis_candidate` and would raise AttributeError on
# every single invocation, silently zeroing out the entire thesis track.

def test_build_thesis_candidates_passes_real_pullback(alpha: AlphaRuntime):
    passing = ThesisCandidate(symbol="AAPL", price=75.0, year_high=100.0, year_low=60.0)
    too_shallow = ThesisCandidate(symbol="MSFT", price=98.0, year_high=100.0, year_low=60.0)

    result = alpha._build_thesis_candidates(date.today(), universe=[passing, too_shallow])

    assert result == ["AAPL"]


def test_build_thesis_candidates_empty_universe_returns_empty(alpha: AlphaRuntime):
    assert alpha._build_thesis_candidates(date.today(), universe=[]) == []


# ── _build_recovery_candidates ──────────────────────────────────────────────
# Regression for a real production bug: this previously called
# evaluate_recovery_candidate(symbol, data_client, today) — completely wrong
# argument types against the real (price_series, min_pullback_pct, ...)
# signature — and would raise TypeError on every invocation.

def _make_series(symbol: str, closes: list[float], volumes: list[int]) -> PriceSeries:
    bars = [
        PriceBar(
            symbol=symbol, timestamp=datetime(2026, 1, 1) + timedelta(days=i),
            open=c, high=c * 1.01, low=c * 0.99, close=c, volume=v,
        )
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]
    return PriceSeries(symbol=symbol, interval="1d", bars=bars)


def _recovery_closes() -> list[float]:
    return [100.0] * 5 + [75.0] * 15 + [76.0, 77.0, 78.0, 80.0, 82.0]


def test_build_recovery_candidates_passes_real_bounce(alpha: AlphaRuntime):
    closes = _recovery_closes()
    volumes = [1_000_000] * 22 + [1_500_000] * 3
    series = _make_series("PASS", closes, volumes)

    alpha._data_client.get_price_history.return_value = series
    candidate = ThesisCandidate(symbol="PASS", price=closes[-1], year_high=100.0, year_low=60.0)

    result = alpha._build_recovery_candidates(date.today(), universe=[candidate])

    assert result == ["PASS"]


def test_build_recovery_candidates_fails_without_volume_pickup(alpha: AlphaRuntime):
    closes = _recovery_closes()
    volumes = [1_000_000] * 25  # flat volume — no 1.2x pickup in the last 3 days
    series = _make_series("NOVOL", closes, volumes)

    alpha._data_client.get_price_history.return_value = series
    candidate = ThesisCandidate(symbol="NOVOL", price=closes[-1], year_high=100.0, year_low=60.0)

    result = alpha._build_recovery_candidates(date.today(), universe=[candidate])

    assert result == []


# ── _record_usage ────────────────────────────────────────────────────────────
# Regression for a real production bug: this dropped cache_creation_input_tokens,
# cache_read_input_tokens, and estimated_cost_usd when calling
# StateStore.record_token_usage(), which requires them — every sub-agent call
# in every consensus run raised TypeError, forcing every candidate to a
# no-signals HOLD regardless of what the LLM agents actually concluded.

class _FakeUsage:
    input_tokens = 100
    output_tokens = 50
    cache_creation_input_tokens = 10
    cache_read_input_tokens = 5


def test_record_usage_writes_full_token_usage_row(alpha: AlphaRuntime):
    alpha._record_usage("technical_analysis_agent", alpha._settings.anthropic_model, _FakeUsage())

    summary = alpha._state_store.get_cost_summary()
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50


# ── Global halt gating (audit finding #7) ───────────────────────────────────
# Regression: GlobalRiskState was never wired into the live system at all.
# Alpha doesn't get its own GlobalRiskState instance (Protection is the one
# that actually computes/detects the halt); instead Alpha reads the halted
# flag Protection persists to the shared breaker_state table before queuing
# any entry — both through the order-intent path and the direct-submission
# options/vol_options/ORB paths that bypass it.

from datetime import datetime as _dt

from analyst_layer.schemas import (
    Action as _Action,
    AgentSignal as _AgentSignal,
    Confidence as _Confidence,
    ConsensusPayload as _ConsensusPayload,
    RiskReview as _RiskReview,
    RiskVerdict as _RiskVerdict,
    TradeProposal as _TradeProposal,
)
from execution_layer.guardrails import GlobalRiskState as _GlobalRiskState


def _buy_payload(ticker: str) -> _ConsensusPayload:
    signal = _AgentSignal(
        agent_name="x", ticker=ticker, stance=_Action.BUY, confidence=_Confidence.HIGH,
        rationale="r", generated_at=_dt.utcnow(),
    )
    proposal = _TradeProposal(ticker=ticker, action=_Action.BUY, quantity=5, limit_price=100.0)
    review = _RiskReview(
        verdict=_RiskVerdict.APPROVED, reasons=["ok"],
        max_position_size_pct_checked=0.05, max_daily_drawdown_pct_checked=0.02,
        reviewed_at=_dt.utcnow(),
    )
    return _ConsensusPayload(ticker=ticker, signals=[signal], proposal=proposal, risk_review=review)


def test_queue_pending_as_intents_skips_when_globally_halted(alpha: AlphaRuntime):
    _GlobalRiskState().sync_to_db(alpha._state_store)  # not halted yet — baseline sanity
    alpha._pending_payloads["AAPL"] = _buy_payload("AAPL")
    alpha._pending_strategies["AAPL"] = "thesis"

    gs = _GlobalRiskState(max_trailing_drawdown_pct=0.20)
    gs.update(100_000.0, date.today())
    gs.update(75_000.0, date.today())  # -25%, past trailing limit
    gs.sync_to_db(alpha._state_store)

    alpha._queue_pending_as_intents(date.today())

    assert alpha._state_store.get_pending_order_intents() == []


def test_assert_not_globally_halted_raises_when_halted(alpha: AlphaRuntime):
    from execution_layer.guardrails import CircuitBreakerTripped

    gs = _GlobalRiskState(max_trailing_drawdown_pct=0.20)
    gs.update(100_000.0, date.today())
    gs.update(70_000.0, date.today())
    gs.sync_to_db(alpha._state_store)

    with pytest.raises(CircuitBreakerTripped):
        alpha._assert_not_globally_halted()


def test_assert_not_globally_halted_noop_when_not_halted(alpha: AlphaRuntime):
    _GlobalRiskState().sync_to_db(alpha._state_store)
    alpha._assert_not_globally_halted()  # must not raise


# ── Halted state must gate entries only, never discretionary sells ─────────
# Regression: breaker.is_stock_halted was checked uniformly for BUY and SELL
# order intents. A legitimate LLM-driven discretionary SELL ("thesis broken,
# exit early") could be silently skipped during a halt — invariant #1 says
# exits must be unconditional.

def _sell_payload(ticker: str) -> _ConsensusPayload:
    signal = _AgentSignal(
        agent_name="x", ticker=ticker, stance=_Action.SELL, confidence=_Confidence.HIGH,
        rationale="thesis broken", generated_at=_dt.utcnow(),
    )
    proposal = _TradeProposal(ticker=ticker, action=_Action.SELL, quantity=5, limit_price=90.0)
    review = _RiskReview(
        verdict=_RiskVerdict.APPROVED, reasons=["ok"],
        max_position_size_pct_checked=0.05, max_daily_drawdown_pct_checked=0.02,
        reviewed_at=_dt.utcnow(),
    )
    return _ConsensusPayload(ticker=ticker, signals=[signal], proposal=proposal, risk_review=review)


def test_queue_pending_as_intents_allows_sell_even_when_breaker_halted(alpha: AlphaRuntime):
    alpha._thesis_breaker._tripped = True  # simulate a halted breaker
    alpha._pending_payloads["AAPL"] = _sell_payload("AAPL")
    alpha._pending_strategies["AAPL"] = "thesis"

    alpha._queue_pending_as_intents(date.today())

    intents = alpha._state_store.get_pending_order_intents()
    assert len(intents) == 1
    assert intents[0]["action"] == "SELL"


def test_queue_pending_as_intents_still_blocks_buy_when_breaker_halted(alpha: AlphaRuntime):
    alpha._thesis_breaker._tripped = True
    alpha._pending_payloads["AAPL"] = _buy_payload("AAPL")
    alpha._pending_strategies["AAPL"] = "thesis"

    alpha._queue_pending_as_intents(date.today())

    assert alpha._state_store.get_pending_order_intents() == []


# ── _assess_market_regime (audit findings #13) ──────────────────────────────
# Regression: this called assess_daily_regime(self._data_client, vix_bearish_
# threshold=..., vix_neutral_threshold=..., macro_snapshot=...) — completely
# wrong against the real (spy_closes: list[float], vix_bars: list[PriceBar],
# macro_sentiment=..., ...) signature. Every call raised TypeError, silently
# caught, meaning self._daily_regime was always None and the swing bearish-
# news-skip filter (which reads self._daily_regime.news_ticker_signals) was
# permanently dead.

def test_assess_market_regime_calls_real_signature_and_returns_regime(alpha: AlphaRuntime):
    alpha._settings.macro_news_enabled = False  # skip the LLM macro path for this test
    closes = [100.0 + i * 0.1 for i in range(60)]
    series = _make_series("SPY", closes, [1_000_000] * 60)
    vix_closes = [15.0 + (i % 5) for i in range(45)]
    vix_series = _make_series("^VIX", vix_closes, [0] * 45)

    def _price_history(symbol, **kwargs):
        return series if symbol == "SPY" else vix_series
    alpha._data_client.get_price_history.side_effect = _price_history

    regime = alpha._assess_market_regime(date.today())

    assert regime is not None
    assert regime.vix_current == vix_closes[-1]


def test_assess_market_regime_records_parseable_news_signals_event(alpha: AlphaRuntime):
    alpha._settings.macro_news_enabled = False
    closes = [100.0 + i * 0.1 for i in range(60)]
    series = _make_series("SPY", closes, [1_000_000] * 60)
    vix_series = _make_series("^VIX", [20.0] * 45, [0] * 45)

    def _price_history(symbol, **kwargs):
        return series if symbol == "SPY" else vix_series
    alpha._data_client.get_price_history.side_effect = _price_history

    alpha._assess_market_regime(date.today())

    events = alpha._state_store.get_events(event_type_like="daily_regime_news_signals", limit=1)
    assert len(events) == 1
    import json
    parsed = json.loads(events[0]["detail"])
    assert parsed == []  # no macro news agent run in this test -> empty list, but valid JSON
