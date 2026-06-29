from __future__ import annotations

from datetime import date

from data_layer.occ_symbol import parse_occ_symbol


def test_parses_a_real_call_symbol():
    parsed = parse_occ_symbol("RIVN260702C00015000")
    assert parsed.underlying_symbol == "RIVN"
    assert parsed.expiration == date(2026, 7, 2)
    assert parsed.option_type == "call"
    assert parsed.strike == 15.0


def test_parses_a_put_symbol():
    parsed = parse_occ_symbol("SOFI260702P00017500")
    assert parsed.option_type == "put"
    assert parsed.strike == 17.5


def test_plain_equity_ticker_returns_none():
    assert parse_occ_symbol("AAPL") is None


def test_garbage_input_returns_none():
    assert parse_occ_symbol("not-a-symbol-123") is None
