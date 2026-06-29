from __future__ import annotations

from analyst_layer.universe import prerank_movers
from data_layer.models import MarketMover


def _mover(symbol: str, percent_change: float) -> MarketMover:
    return MarketMover(symbol=symbol, price=100.0, change=percent_change * 100, percent_change=percent_change, volume=1000)


def test_prerank_movers_sorts_by_absolute_percent_change():
    movers = [_mover("A", 0.05), _mover("B", -0.20), _mover("C", 0.10)]
    ranked = prerank_movers(movers, limit=3)
    assert ranked == ["B", "C", "A"]


def test_prerank_movers_respects_limit():
    movers = [_mover(f"T{i}", 0.01 * i) for i in range(10)]
    ranked = prerank_movers(movers, limit=3)
    assert len(ranked) == 3
    assert ranked == ["T9", "T8", "T7"]


def test_prerank_movers_dedupes_by_symbol_keeping_one_entry():
    movers = [_mover("A", 0.05), _mover("A", 0.99)]
    ranked = prerank_movers(movers, limit=10)
    assert ranked == ["A"]
