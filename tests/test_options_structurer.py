from __future__ import annotations

from datetime import date

from analyst_layer.options_structurer import select_contract
from analyst_layer.schemas import Action
from data_layer.models import OptionContract, OptionType


def _contract(option_type: OptionType, dte: int, strike: float, underlying_price: float = 100.0) -> OptionContract:
    return OptionContract(
        contract_symbol=f"TEST{dte}{option_type.value[0].upper()}{int(strike)}",
        underlying_symbol="TEST",
        underlying_price=underlying_price,
        expiration=date(2026, 1, 1),
        dte=dte,
        strike=strike,
        option_type=option_type,
        bid=1.0,
        ask=1.1,
        implied_volatility=0.3,
        open_interest=100,
        volume=10,
    )


def test_hold_direction_selects_nothing():
    chain = [_contract(OptionType.CALL, dte=7, strike=100.0)]
    result = select_contract(chain, Action.HOLD, min_dte=5, max_dte=10)
    assert result.selected is False


def test_buy_direction_picks_a_call_not_a_put():
    chain = [_contract(OptionType.CALL, dte=7, strike=100.0), _contract(OptionType.PUT, dte=7, strike=100.0)]
    result = select_contract(chain, Action.BUY, min_dte=5, max_dte=10)
    assert result.selected is True
    assert result.contract.option_type == OptionType.CALL


def test_sell_direction_picks_a_put():
    chain = [_contract(OptionType.CALL, dte=7, strike=100.0), _contract(OptionType.PUT, dte=7, strike=100.0)]
    result = select_contract(chain, Action.SELL, min_dte=5, max_dte=10)
    assert result.selected is True
    assert result.contract.option_type == OptionType.PUT


def test_excludes_contracts_below_the_dte_floor():
    """The whole point of this track vs. a 0-1 DTE one — a 2-day contract
    must never be selectable even if it's otherwise the closest fit.
    """
    chain = [_contract(OptionType.CALL, dte=2, strike=100.0), _contract(OptionType.CALL, dte=7, strike=100.0)]
    result = select_contract(chain, Action.BUY, min_dte=5, max_dte=10)
    assert result.selected is True
    assert result.contract.dte == 7


def test_excludes_contracts_beyond_the_dte_ceiling():
    chain = [_contract(OptionType.CALL, dte=30, strike=100.0)]
    result = select_contract(chain, Action.BUY, min_dte=5, max_dte=10)
    assert result.selected is False


def test_picks_nearest_expiration_to_the_floor_not_the_ceiling():
    chain = [_contract(OptionType.CALL, dte=6, strike=100.0), _contract(OptionType.CALL, dte=9, strike=100.0)]
    result = select_contract(chain, Action.BUY, min_dte=5, max_dte=10)
    assert result.contract.dte == 6


def test_picks_strike_nearest_at_the_money():
    chain = [
        _contract(OptionType.CALL, dte=7, strike=90.0, underlying_price=100.0),
        _contract(OptionType.CALL, dte=7, strike=101.0, underlying_price=100.0),
        _contract(OptionType.CALL, dte=7, strike=120.0, underlying_price=100.0),
    ]
    result = select_contract(chain, Action.BUY, min_dte=5, max_dte=10)
    assert result.contract.strike == 101.0


def test_excludes_illiquid_contracts_with_no_quote():
    illiquid = _contract(OptionType.CALL, dte=7, strike=100.0)
    illiquid = illiquid.model_copy(update={"bid": 0.0, "ask": 0.0})
    result = select_contract([illiquid], Action.BUY, min_dte=5, max_dte=10)
    assert result.selected is False


def test_no_candidates_in_band_reports_reason():
    chain = [_contract(OptionType.CALL, dte=20, strike=100.0)]
    result = select_contract(chain, Action.BUY, min_dte=5, max_dte=10)
    assert result.selected is False
    assert "no liquid call" in result.reasons[0]
