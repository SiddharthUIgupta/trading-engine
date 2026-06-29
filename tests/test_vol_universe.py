"""Tests for analyst_layer.vol_universe — options liquidity screener."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from analyst_layer.vol_universe import VolUniverseResult, screen_vol_universe, _find_best_expiration
from data_layer.exceptions import DataLayerError
from data_layer.models import MarketMover, OptionContract, OptionType


# ── Helpers ───────────────────────────────────────────────────────────────────

_EXP_35D = date.today() + timedelta(days=35)  # inside 21-60 DTE window
_EXP_5D = date.today() + timedelta(days=5)    # outside window


def _contract(
    ticker: str,
    option_type: OptionType,
    strike: float,
    bid: float = 2.00,
    ask: float = 2.20,
    oi: int = 1000,
    dte: int = 35,
    underlying_price: float = 100.0,
    expiration: date | None = None,
) -> OptionContract:
    exp = expiration or (date.today() + timedelta(days=dte))
    return OptionContract(
        contract_symbol=f"{ticker}{dte}{option_type.value[0].upper()}{int(strike)}",
        underlying_symbol=ticker,
        underlying_price=underlying_price,
        expiration=exp,
        dte=dte,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        implied_volatility=0.30,
        open_interest=oi,
        volume=200,
    )


def _liquid_chain(ticker: str = "AAPL") -> list[OptionContract]:
    """Chain that passes all three liquidity criteria."""
    return [
        _contract(ticker, OptionType.CALL, 105.0, bid=2.50, ask=2.60, oi=1500),
        _contract(ticker, OptionType.PUT, 95.0, bid=2.20, ask=2.30, oi=1200),
    ]


def _illiquid_chain_low_oi(ticker: str = "ILLIQ") -> list[OptionContract]:
    """ATM OI below threshold."""
    return [
        _contract(ticker, OptionType.CALL, 105.0, bid=2.50, ask=2.60, oi=50),
        _contract(ticker, OptionType.PUT, 95.0, bid=2.20, ask=2.30, oi=50),
    ]


def _wide_spread_chain(ticker: str = "WIDE") -> list[OptionContract]:
    """Spread > 10% of mid — excluded."""
    # mid = (0.50 + 3.00) / 2 = 1.75, spread = 2.50 / 1.75 = 143% >> 10%
    return [
        _contract(ticker, OptionType.CALL, 105.0, bid=0.50, ask=3.00, oi=2000),
        _contract(ticker, OptionType.PUT, 95.0, bid=2.20, ask=2.30, oi=2000),
    ]


def _wrong_dte_chain(ticker: str = "WRONG") -> list[OptionContract]:
    """Only has a 5-DTE expiration — outside the 21-60 DTE window."""
    return [
        _contract(ticker, OptionType.CALL, 105.0, dte=5, expiration=_EXP_5D),
        _contract(ticker, OptionType.PUT, 95.0, dte=5, expiration=_EXP_5D),
    ]


def _mock_data_client(chains: dict[str, list[OptionContract]], movers: list[str] | None = None):
    dc = MagicMock()
    def _chain(ticker):
        if ticker in chains:
            return chains[ticker]
        raise DataLayerError(f"no chain for {ticker}")
    dc.get_option_chain.side_effect = _chain

    mover_list = []
    for sym in (movers or []):
        m = MagicMock()
        m.symbol = sym
        mover_list.append(m)
    dc.get_market_movers.return_value = mover_list
    return dc


# ── screen_vol_universe ───────────────────────────────────────────────────────

def test_liquid_seed_passes_screen():
    dc = _mock_data_client({"AAPL": _liquid_chain("AAPL")})
    result = screen_vol_universe(dc, seed=["AAPL"], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert "AAPL" in result.passed
    assert result.fallback_used is False


def test_low_oi_ticker_excluded():
    # Pass ILLIQ as a mover (not seed) so the fallback doesn't rescue it
    dc = _mock_data_client({"ILLIQ": _illiquid_chain_low_oi("ILLIQ")}, movers=["ILLIQ"])
    result = screen_vol_universe(dc, seed=[], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert "ILLIQ" not in result.passed


def test_wide_spread_ticker_excluded():
    dc = _mock_data_client({"WIDE": _wide_spread_chain("WIDE")}, movers=["WIDE"])
    result = screen_vol_universe(dc, seed=[], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert "WIDE" not in result.passed


def test_wrong_dte_window_excluded():
    dc = _mock_data_client({"WRONG": _wrong_dte_chain("WRONG")}, movers=["WRONG"])
    result = screen_vol_universe(dc, seed=[], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert "WRONG" not in result.passed


def test_chain_fetch_failure_excludes_ticker():
    # FAIL is only a mover, not in seed, and its chain fetch will fail
    dc = _mock_data_client({}, movers=["FAIL"])
    result = screen_vol_universe(dc, seed=[], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert "FAIL" not in result.passed


def test_fallback_to_seed_when_all_fail():
    """When no tickers pass the screen, return the seed as fallback."""
    dc = _mock_data_client({})  # every chain fetch fails
    result = screen_vol_universe(dc, seed=["AAPL", "SPY"], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert result.fallback_used is True
    assert result.passed == ["AAPL", "SPY"]


def test_market_movers_added_to_pool():
    """Tickers from market_movers that are not in seed should be screened."""
    dc = _mock_data_client(
        chains={"MOVER": _liquid_chain("MOVER")},
        movers=["MOVER"],
    )
    result = screen_vol_universe(dc, seed=[], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    assert "MOVER" in result.passed


def test_movers_fetch_failure_falls_back_to_seed_only():
    """If market_movers fetch fails, screen the seed only — don't crash."""
    dc = MagicMock()
    dc.get_market_movers.side_effect = DataLayerError("network error")
    dc.get_option_chain.return_value = _liquid_chain("AAPL")
    result = screen_vol_universe(dc, seed=["AAPL"], min_option_oi=500, max_spread_pct=0.10,
                                  min_dte=21, max_dte=60, max_size=20)
    # Seed was screened and passed
    assert "AAPL" in result.passed


def test_max_size_cap():
    chains = {f"T{i}": _liquid_chain(f"T{i}") for i in range(10)}
    dc = _mock_data_client(chains)
    result = screen_vol_universe(dc, seed=list(chains.keys()), min_option_oi=500,
                                  max_spread_pct=0.10, min_dte=21, max_dte=60, max_size=3)
    assert len(result.passed) <= 3


def test_higher_oi_ranked_first():
    """Tickers are sorted by ATM OI descending — deeper liquidity first."""
    chains = {
        "LOW": [
            _contract("LOW", OptionType.CALL, 105.0, oi=600),
            _contract("LOW", OptionType.PUT, 95.0, oi=600),
        ],
        "HIGH": [
            _contract("HIGH", OptionType.CALL, 105.0, oi=5000),
            _contract("HIGH", OptionType.PUT, 95.0, oi=5000),
        ],
    }
    dc = _mock_data_client(chains)
    result = screen_vol_universe(dc, seed=["LOW", "HIGH"], min_option_oi=500,
                                  max_spread_pct=0.10, min_dte=21, max_dte=60, max_size=20)
    assert result.passed[0] == "HIGH"
    assert result.passed[1] == "LOW"


def test_screened_count_reported():
    chains = {"AAPL": _liquid_chain("AAPL"), "ILLIQ": _illiquid_chain_low_oi("ILLIQ")}
    dc = _mock_data_client(chains)
    result = screen_vol_universe(dc, seed=["AAPL", "ILLIQ"], min_option_oi=500,
                                  max_spread_pct=0.10, min_dte=21, max_dte=60, max_size=20)
    assert result.screened == 2


# ── Runtime integration ───────────────────────────────────────────────────────

def test_refresh_vol_universe_updates_runtime(tmp_path: Path):
    """refresh_vol_universe() replaces _vol_universe with the screened list."""
    import os
    from config.settings import Settings
    from execution_layer.guardrails import CircuitBreaker
    from execution_layer.runtime import TradingRuntime
    from execution_layer.state_store import StateStore

    os.environ["VOL_OPTIONS_TRACK_ENABLED"] = "true"
    settings = Settings(_env_file=None)
    del os.environ["VOL_OPTIONS_TRACK_ENABLED"]

    dc = MagicMock()
    dc.get_market_movers.return_value = []
    dc.get_option_chain.return_value = _liquid_chain("NVDA")

    breaker = CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02)
    store = StateStore(tmp_path / "test.sqlite3")

    rt = TradingRuntime(
        settings=settings, data_client=dc, broker=MagicMock(),
        circuit_breaker=breaker, state_store=store,
        anthropic_client=MagicMock(), watchlist=["NVDA"],
    )
    assert rt._vol_universe == ["NVDA"]  # initialized to watchlist

    rt.refresh_vol_universe()

    # After refresh, _vol_universe reflects the screened result
    assert isinstance(rt._vol_universe, list)
    assert len(rt._vol_universe) >= 1
    assert "NVDA" in rt._vol_universe


def test_refresh_vol_universe_no_ops_when_track_disabled(tmp_path: Path):
    """refresh_vol_universe() is a no-op when vol track is disabled."""
    from config.settings import Settings
    from execution_layer.guardrails import CircuitBreaker
    from execution_layer.runtime import TradingRuntime
    from execution_layer.state_store import StateStore

    import os
    os.environ["VOL_OPTIONS_TRACK_ENABLED"] = "false"
    settings = Settings(_env_file=None)
    del os.environ["VOL_OPTIONS_TRACK_ENABLED"]
    dc = MagicMock()

    rt = TradingRuntime(
        settings=settings, data_client=dc, broker=MagicMock(),
        circuit_breaker=CircuitBreaker(max_position_size_pct=0.05, max_daily_drawdown_pct=0.02),
        state_store=StateStore(tmp_path / "test.sqlite3"),
        anthropic_client=MagicMock(), watchlist=["AAPL"],
    )
    rt.refresh_vol_universe()

    dc.get_option_chain.assert_not_called()
    dc.get_market_movers.assert_not_called()
