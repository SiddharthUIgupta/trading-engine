from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analyst_layer.schemas import Action, ConsensusPayload, RiskReview, RiskVerdict, TradeProposal
from config.settings import LIVE_CONFIRM_TOKEN, UNCOVERED_CONFIRM_TOKEN, Settings
from execution_layer.broker import AlpacaBroker, LiveTradingBlockedError
from execution_layer.state_store import StateStore


# ---- Settings paper/live gate ----

def test_settings_default_to_paper():
    settings = Settings(_env_file=None)
    assert settings.is_live is False
    assert settings.trading_env == "paper"


def test_settings_force_paper_if_confirm_token_missing():
    settings = Settings(_env_file=None, TRADING_ENV="live", TRADING_LIVE_CONFIRM="")
    assert settings.trading_env == "paper"
    assert settings.is_live is False


def test_settings_allow_live_only_with_full_explicit_override():
    settings = Settings(_env_file=None, TRADING_ENV="live", TRADING_LIVE_CONFIRM=LIVE_CONFIRM_TOKEN)
    assert settings.trading_env == "live"
    assert settings.is_live is True


# ---- Settings uncovered gate ----

def test_settings_uncovered_default_to_false():
    settings = Settings(_env_file=None)
    assert settings.vol_options_allow_uncovered is False
    assert settings.is_uncovered_allowed is False


def test_settings_uncovered_single_flag_is_not_enough():
    """VOL_OPTIONS_ALLOW_UNCOVERED=true alone must NOT enable strangles."""
    settings = Settings(_env_file=None, VOL_OPTIONS_ALLOW_UNCOVERED=True)
    assert settings.vol_options_allow_uncovered is False
    assert settings.is_uncovered_allowed is False


def test_settings_uncovered_requires_both_signals():
    settings = Settings(
        _env_file=None,
        VOL_OPTIONS_ALLOW_UNCOVERED=True,
        VOL_OPTIONS_UNCOVERED_CONFIRM=UNCOVERED_CONFIRM_TOKEN,
    )
    assert settings.vol_options_allow_uncovered is True
    assert settings.is_uncovered_allowed is True


def test_settings_uncovered_wrong_confirm_token_stays_false():
    settings = Settings(
        _env_file=None,
        VOL_OPTIONS_ALLOW_UNCOVERED=True,
        VOL_OPTIONS_UNCOVERED_CONFIRM="wrong token",
    )
    assert settings.vol_options_allow_uncovered is False
    assert settings.is_uncovered_allowed is False


# ---- AlpacaBroker forced-paper construction ----

@patch("execution_layer.broker.TradingClient")
def test_broker_from_settings_constructs_with_paper_true_by_default(mock_trading_client):
    settings = Settings(_env_file=None, ALPACA_API_KEY="k", ALPACA_SECRET_KEY="s")
    AlpacaBroker.from_settings(settings)
    _, kwargs = mock_trading_client.call_args
    assert kwargs["paper"] is True


@patch("execution_layer.broker.TradingClient")
def test_broker_from_settings_live_requires_full_override(mock_trading_client):
    settings = Settings(
        _env_file=None,
        ALPACA_API_KEY="k",
        ALPACA_SECRET_KEY="s",
        TRADING_ENV="live",
        TRADING_LIVE_CONFIRM=LIVE_CONFIRM_TOKEN,
    )
    broker = AlpacaBroker.from_settings(settings)
    _, kwargs = mock_trading_client.call_args
    assert kwargs["paper"] is False
    assert broker.is_live is True


def _hold_payload() -> ConsensusPayload:
    from datetime import datetime

    proposal = TradeProposal(ticker="AAPL", action=Action.HOLD, quantity=0, limit_price=100.0)
    review = RiskReview(
        verdict=RiskVerdict.APPROVED,
        reasons=["no signals"],
        max_position_size_pct_checked=0.05,
        max_daily_drawdown_pct_checked=0.02,
        reviewed_at=datetime.utcnow(),
    )
    from analyst_layer.schemas import AgentSignal, Confidence

    signal = AgentSignal(
        agent_name="x", ticker="AAPL", stance=Action.HOLD, confidence=Confidence.LOW, rationale="r", generated_at=datetime.utcnow()
    )
    return ConsensusPayload(ticker="AAPL", signals=[signal], proposal=proposal, risk_review=review)


def test_broker_submit_order_skips_hold():
    broker = AlpacaBroker(trading_client=MagicMock(), is_live=False)
    result = broker.submit_order(_hold_payload().proposal)
    assert result["status"] == "skipped"


def _order_state(status: str, filled_qty: float = 0.0, filled_avg_price: float | None = None):
    return MagicMock(id="order-123", status=status, filled_qty=filled_qty, filled_avg_price=filled_avg_price)


def test_broker_submit_order_calls_client_for_buy():
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_state("new")
    mock_client.get_order_by_id.return_value = _order_state("filled", filled_qty=5, filled_avg_price=150.0)
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=5, limit_price=150.0)
    result = broker.submit_order(proposal, poll_for_fill_seconds=1.0)

    assert result["status"] == "submitted"
    assert result["order_status"] == "filled"
    assert result["filled_qty"] == 5
    assert result["filled_avg_price"] == 150.0
    mock_client.submit_order.assert_called_once()


def test_broker_submit_order_skips_polling_when_window_is_zero():
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_state("new")
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=5, limit_price=150.0)
    result = broker.submit_order(proposal, poll_for_fill_seconds=0)

    mock_client.get_order_by_id.assert_not_called()
    assert result["order_status"] == "new"
    assert result["filled_qty"] == 0.0


def test_broker_submit_order_catches_fill_that_happens_during_poll_window():
    """Regression test for the exact race that caused a real bug: a limit
    order that's still "new" the instant submit_order returns, but fills
    moments later — submit_order must not report 0 filled shares just
    because that's what was true at the very first instant.
    """
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_state("new")
    mock_client.get_order_by_id.side_effect = [
        _order_state("new", filled_qty=0),
        _order_state("new", filled_qty=0),
        _order_state("filled", filled_qty=5, filled_avg_price=150.0),
    ]
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    proposal = TradeProposal(ticker="AAPL", action=Action.BUY, quantity=5, limit_price=150.0)
    with patch("execution_layer.broker.time.sleep"):  # don't actually wait in tests
        result = broker.submit_order(proposal, poll_for_fill_seconds=5.0)

    assert result["order_status"] == "filled"
    assert result["filled_qty"] == 5


def test_broker_submit_order_reports_still_open_after_poll_timeout():
    """A genuinely slow-to-fill limit order (the AZTA case) must not hang
    forever — and must honestly report 0 filled shares, since that's
    actually true at that point.
    """
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_state("new")
    mock_client.get_order_by_id.return_value = _order_state("new", filled_qty=0)
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    proposal = TradeProposal(ticker="AZTA", action=Action.BUY, quantity=1, limit_price=22.47)
    with patch("execution_layer.broker.time.sleep"):
        result = broker.submit_order(proposal, poll_for_fill_seconds=0.2)

    assert result["order_status"] == "new"
    assert result["filled_qty"] == 0.0


def test_broker_submit_option_order_calls_client_with_occ_symbol():
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_state("new")
    mock_client.get_order_by_id.return_value = _order_state("filled", filled_qty=2, filled_avg_price=3.65)
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    result = broker.submit_option_order(
        "AAPL260629C00295000", side=Action.BUY, contracts=2, limit_price=3.60, poll_for_fill_seconds=1.0
    )

    assert result["order_status"] == "filled"
    assert result["filled_qty"] == 2
    assert result["filled_avg_price"] == 3.65
    order_request = mock_client.submit_order.call_args[0][0]
    assert order_request.symbol == "AAPL260629C00295000"
    assert order_request.qty == 2
    assert order_request.limit_price == 3.60


def test_broker_get_last_fill_price_returns_most_recent_filled_order():
    filled = MagicMock(status=MagicMock(value="filled"), filled_avg_price="16.81")
    mock_client = MagicMock()
    mock_client.get_orders.return_value = [filled]
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    assert broker.get_last_fill_price("WWW") == 16.81


def test_broker_get_last_fill_price_skips_non_filled_orders():
    canceled = MagicMock(status=MagicMock(value="canceled"), filled_avg_price=None)
    filled = MagicMock(status=MagicMock(value="filled"), filled_avg_price="22.97")
    mock_client = MagicMock()
    mock_client.get_orders.return_value = [canceled, filled]
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    assert broker.get_last_fill_price("AZTA") == 22.97


def test_broker_get_last_fill_price_returns_none_with_no_filled_orders():
    mock_client = MagicMock()
    mock_client.get_orders.return_value = []
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    assert broker.get_last_fill_price("AZTA") is None


def test_broker_get_open_orders_returns_pending_orders():
    mock_order = MagicMock(
        id="order-abc-123", symbol="RIVN260702C00015000", side=MagicMock(value="buy"), qty="13", limit_price="0.73",
        status=MagicMock(value="accepted"), submitted_at="2026-06-23 22:51:52+00:00", legs=None,
    )
    mock_client = MagicMock()
    mock_client.get_orders.return_value = [mock_order]
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    orders = broker.get_open_orders()

    assert orders == [{
        "order_id": "order-abc-123", "symbol": "RIVN260702C00015000", "side": "buy", "qty": 13.0, "limit_price": 0.73,
        "status": "accepted", "submitted_at": "2026-06-23 22:51:52+00:00", "legs": None,
    }]


def test_broker_close_position_calls_client():
    mock_client = MagicMock()
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)
    broker.close_position("WWW")
    mock_client.close_position.assert_called_once_with(symbol_or_asset_id="WWW")


def test_broker_close_position_does_not_raise_if_already_flat():
    mock_client = MagicMock()
    mock_client.close_position.side_effect = Exception("position does not exist")
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)
    broker.close_position("WWW")  # must not raise


def test_broker_cancel_order_calls_client():
    mock_client = MagicMock()
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)
    broker.cancel_order("order-123")
    mock_client.cancel_order_by_id.assert_called_once_with("order-123")


def test_broker_get_position_detail_returns_live_snapshot():
    mock_client = MagicMock()
    mock_client.get_open_position.return_value = MagicMock(
        qty="10", avg_entry_price="100.0", current_price="110.0", unrealized_plpc="0.10"
    )
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    detail = broker.get_position_detail("AAPL")

    assert detail == {"qty": 10.0, "avg_entry_price": 100.0, "current_price": 110.0, "unrealized_plpc": 0.10}


def test_broker_get_position_detail_returns_none_with_no_position():
    mock_client = MagicMock()
    mock_client.get_open_position.side_effect = RuntimeError("no position")
    broker = AlpacaBroker(trading_client=mock_client, is_live=False)

    assert broker.get_position_detail("AAPL") is None


# ---- StateStore round-trip ----

def test_state_store_records_and_reads_positions(tmp_path: Path):
    store = StateStore(tmp_path / "test.sqlite3")
    store.upsert_position("AAPL", 10, 150.0)
    store.upsert_position("AAPL", 15, 151.0)  # update existing

    positions = store.get_positions()
    assert len(positions) == 1
    assert positions[0]["quantity"] == 15


def test_state_store_records_run_history(tmp_path: Path):
    store = StateStore(tmp_path / "test.sqlite3")
    payload = _hold_payload()
    store.record_run(payload)

    history = store.get_run_history(ticker="AAPL")
    assert len(history) == 1
    assert history[0]["payload"]["ticker"] == "AAPL"
    assert history[0]["is_executable"] is False


def test_state_store_records_and_summarizes_token_usage(tmp_path: Path):
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_token_usage(
        agent_name="technical_analysis_agent",
        model="claude-haiku-4-5-20251001",
        input_tokens=500,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        estimated_cost_usd=0.0008,
    )
    store.record_token_usage(
        agent_name="risk_compliance_officer_agent",
        model="claude-sonnet-4-6",
        input_tokens=800,
        output_tokens=100,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=1000,
        estimated_cost_usd=0.0045,
    )

    summary = store.get_cost_summary()

    assert summary["total_calls"] == 2
    assert summary["total_cost_usd"] == pytest.approx(0.0053)
    assert summary["total_cache_read_input_tokens"] == 1000
    by_agent_names = {row["agent_name"] for row in summary["by_agent"]}
    assert by_agent_names == {"technical_analysis_agent", "risk_compliance_officer_agent"}
    # most expensive agent first
    assert summary["by_agent"][0]["agent_name"] == "risk_compliance_officer_agent"


def test_state_store_cost_summary_filters_by_since_date(tmp_path: Path):
    from datetime import datetime, timedelta

    store = StateStore(tmp_path / "test.sqlite3")
    store.record_token_usage(
        agent_name="x", model="claude-sonnet-4-6", input_tokens=1, output_tokens=1,
        cache_creation_input_tokens=0, cache_read_input_tokens=0, estimated_cost_usd=1.0,
    )

    # created_at is stored via datetime.utcnow() — the cutoff must use the same
    # clock as the test fixture (see runtime.py post_market_logging for why).
    today_utc = datetime.utcnow().date()

    summary_future = store.get_cost_summary(since=today_utc + timedelta(days=1))
    assert summary_future["total_calls"] == 0
    assert summary_future["total_cost_usd"] == 0

    summary_today = store.get_cost_summary(since=today_utc)
    assert summary_today["total_calls"] == 1


def test_get_all_realized_sales_includes_wins_and_losses(tmp_path: Path):
    from datetime import date

    store = StateStore(tmp_path / "test.sqlite3")
    store.record_realized_sale("AAPL", sale_date=date.today(), quantity=10, sale_price=120.0, cost_basis=100.0)
    store.record_realized_sale("MSFT", sale_date=date.today(), quantity=5, sale_price=90.0, cost_basis=100.0)

    sales = store.get_all_realized_sales()

    assert len(sales) == 2
    pnls = {s["ticker"]: s["realized_pnl"] for s in sales}
    assert pnls["AAPL"] == 200.0
    assert pnls["MSFT"] == -50.0


def test_get_events_filters_by_event_type_like(tmp_path: Path):
    store = StateStore(tmp_path / "test.sqlite3")
    store.record_event(event_type="wash_sale_blocked", detail="a")
    store.record_event(event_type="intraday_llm_exit_escalation:AAPL", detail="b")
    store.record_event(event_type="intraday_llm_exit_escalation:MSFT", detail="c")

    all_events = store.get_events()
    assert len(all_events) == 3

    escalations = store.get_events(event_type_like="intraday_llm_exit_escalation:%")
    assert len(escalations) == 2
