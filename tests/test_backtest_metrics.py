from __future__ import annotations

from datetime import date

import pytest

from backtest.metrics import Trade, summarize


def _closed_trade(entry: float, exit_: float) -> Trade:
    return Trade(ticker="TEST", entry_date=date(2025, 1, 1), entry_price=entry, exit_date=date(2025, 1, 5), exit_price=exit_)


def test_trade_return_pct_for_a_win():
    t = _closed_trade(100.0, 110.0)
    assert t.return_pct == 0.10


def test_trade_return_pct_for_a_loss():
    t = _closed_trade(100.0, 95.0)
    assert t.return_pct == -0.05


def test_trade_not_closed_has_no_return():
    t = Trade(ticker="TEST", entry_date=date(2025, 1, 1), entry_price=100.0)
    assert t.is_closed is False
    assert t.return_pct is None


def test_summarize_empty_trade_list():
    report = summarize([])
    assert report.total_trades == 0
    assert report.win_rate is None
    assert "No closed trades" in report.confidence_note


def test_summarize_computes_win_rate_and_averages():
    trades = [
        _closed_trade(100.0, 110.0),  # +10%
        _closed_trade(100.0, 120.0),  # +20%
        _closed_trade(100.0, 90.0),   # -10%
        _closed_trade(100.0, 95.0),   # -5%
    ]
    report = summarize(trades)
    assert report.total_trades == 4
    assert report.win_rate == 0.5
    assert report.avg_win_pct == pytest.approx(0.15)  # (0.10 + 0.20) / 2
    assert report.avg_loss_pct == pytest.approx(-0.075)  # (-0.10 + -0.05) / 2
    assert report.profit_factor == pytest.approx(0.30 / 0.15)  # gross win / gross loss


def test_summarize_excludes_still_open_trades_from_closed_stats():
    trades = [
        _closed_trade(100.0, 110.0),
        Trade(ticker="OPEN", entry_date=date(2025, 1, 1), entry_price=100.0),  # never closed
    ]
    report = summarize(trades)
    assert report.total_trades == 1
    assert report.still_open_at_backtest_end == 1


def test_summarize_profit_factor_is_infinite_with_no_losses():
    trades = [_closed_trade(100.0, 110.0), _closed_trade(100.0, 105.0)]
    report = summarize(trades)
    assert report.profit_factor == float("inf")


def test_summarize_confidence_note_scales_with_sample_size():
    small = summarize([_closed_trade(100.0, 101.0) for _ in range(5)])
    assert "too small" in small.confidence_note

    medium = summarize([_closed_trade(100.0, 101.0) for _ in range(30)])
    assert "thin sample" in medium.confidence_note

    large = summarize([_closed_trade(100.0, 101.0) for _ in range(60)])
    assert "reasonable sample" in large.confidence_note
