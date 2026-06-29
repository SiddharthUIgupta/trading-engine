"""Dynamic universe builder for the vol options track.

Short-premium strategies are highly sensitive to options liquidity. Selling
premium on a name with wide bid/ask spreads or thin open interest means:
  - The iron condor net credit is eaten by the spread (mid-price fills fail)
  - IV calculations are unreliable (sparse quotes distort the surface)
  - Exit fills are worse than entry (doubled spread on close)

This module screens a candidate pool for three hard requirements:
  1. At least one expiration in the target DTE window exists in the chain
  2. ATM call and put both have open interest ≥ threshold
  3. ATM bid/ask spread as a fraction of mid ≤ threshold

Candidates that clear all three are sorted by average OI (deeper OI = more
liquid = better fills) so the top-N are the highest-quality names available
that day.

Seed logic:
  The static watchlist (former hardcoded list) is always included in the
  candidate pool — those names are guaranteed liquid and serve as a floor.
  Market movers from the discovery screens are added on top; they tend to be
  higher-volume names that coincidentally have active options markets.
  Together they form a 50-100 ticker pool that is screened down to max_size.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from data_layer.exceptions import DataLayerError
from data_layer.models import OptionContract, OptionType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolUniverseResult:
    passed: list[str]      # tickers that cleared the liquidity screen
    screened: int          # total candidates evaluated
    fallback_used: bool    # True if screen returned empty and seed was returned


def _spread_pct(contract: OptionContract) -> float | None:
    """Bid/ask spread as a fraction of the mid-price. None if quotes are absent."""
    mid = (contract.bid + contract.ask) / 2.0
    if mid <= 0:
        return None
    return (contract.ask - contract.bid) / mid


def screen_vol_universe(
    data_client,
    seed: list[str],
    min_option_oi: int,
    max_spread_pct: float,
    min_dte: int,
    max_dte: int,
    max_size: int,
) -> VolUniverseResult:
    """Screen a candidate pool and return the most liquid options names.

    Parameters
    ----------
    data_client:
        OpenBBDataClient instance. `get_market_movers()` and
        `get_option_chain(ticker)` must be available.
    seed:
        Tickers that are always included in the candidate pool (the former
        static watchlist). Returned as fallback if the screen yields nothing.
    min_option_oi:
        Minimum open interest required on the ATM call AND put at the
        best expiration in the target DTE window.
    max_spread_pct:
        Maximum bid/ask spread as a fraction of mid-price for the ATM call
        and put (e.g. 0.10 = 10%). Names above this threshold are excluded.
    min_dte / max_dte:
        DTE window for the target expiration (e.g. 21-60 to cover the
        30-45 DTE tastylive sweet spot with some buffer).
    max_size:
        Cap on the number of tickers in the returned universe.
    """
    # Build candidate pool: seed (guaranteed liquid) + market movers (dynamic)
    candidates: list[str] = list(seed)
    try:
        movers = data_client.get_market_movers()
        for m in movers:
            if m.symbol not in candidates:
                candidates.append(m.symbol)
    except DataLayerError as exc:
        logger.warning("vol universe: market movers fetch failed — screening seed only: %s", exc)

    logger.info("vol universe: screening %d candidates for options liquidity", len(candidates))

    passed: list[tuple[str, float]] = []  # (ticker, avg_atm_oi) for ranking

    for ticker in candidates:
        try:
            chain = data_client.get_option_chain(ticker)
        except DataLayerError as exc:
            logger.debug("%s: option chain fetch failed — excluded from vol universe: %s", ticker, exc)
            continue

        # Find contracts in the target DTE window
        in_window = [c for c in chain if min_dte <= c.dte <= max_dte]
        if not in_window:
            logger.debug("%s: no expiration in %d-%d DTE window — excluded", ticker, min_dte, max_dte)
            continue

        underlying_price = in_window[0].underlying_price
        calls = [c for c in in_window if c.option_type == OptionType.CALL]
        puts = [c for c in in_window if c.option_type == OptionType.PUT]
        if not calls or not puts:
            logger.debug("%s: missing call or put leg in DTE window — excluded", ticker)
            continue

        # Use the expiration with the best ATM liquidity if multiple exist in window
        best_exp = _find_best_expiration(calls, puts, underlying_price, min_option_oi, max_spread_pct)
        if best_exp is None:
            logger.debug(
                "%s: no expiration in window clears OI ≥ %d and spread ≤ %.0f%% — excluded",
                ticker, min_option_oi, max_spread_pct * 100,
            )
            continue

        avg_oi = best_exp
        passed.append((ticker, avg_oi))
        logger.debug("%s: passed liquidity screen (avg ATM OI=%.0f)", ticker, avg_oi)

    # Sort by OI descending — deeper liquidity first
    passed.sort(key=lambda t: t[1], reverse=True)
    result = [ticker for ticker, _ in passed[:max_size]]

    if not result:
        logger.warning(
            "vol universe: no tickers cleared liquidity screen — falling back to seed %s", seed
        )
        return VolUniverseResult(passed=list(seed[:max_size]), screened=len(candidates), fallback_used=True)

    logger.info(
        "vol universe: %d/%d candidates passed → universe=%s",
        len(result), len(candidates), result,
    )
    return VolUniverseResult(passed=result, screened=len(candidates), fallback_used=False)


def _find_best_expiration(
    calls: list[OptionContract],
    puts: list[OptionContract],
    underlying_price: float,
    min_option_oi: int,
    max_spread_pct: float,
) -> float | None:
    """Find the expiration in the call/put lists that best satisfies the
    liquidity requirements. Returns the average ATM OI of the best expiration,
    or None if no expiration qualifies.

    Tries each unique expiration date and picks the one with the highest
    average ATM OI that also passes the spread filter.
    """
    expirations = sorted({c.expiration for c in calls} & {p.expiration for p in puts})

    best_avg_oi: float | None = None
    for exp in expirations:
        exp_calls = [c for c in calls if c.expiration == exp]
        exp_puts = [p for p in puts if p.expiration == exp]

        atm_call = min(exp_calls, key=lambda c: abs(c.strike - underlying_price))
        atm_put = min(exp_puts, key=lambda p: abs(p.strike - underlying_price))

        if atm_call.open_interest < min_option_oi or atm_put.open_interest < min_option_oi:
            continue

        call_spread = _spread_pct(atm_call)
        put_spread = _spread_pct(atm_put)
        if call_spread is None or put_spread is None:
            continue
        if call_spread > max_spread_pct or put_spread > max_spread_pct:
            continue

        avg_oi = (atm_call.open_interest + atm_put.open_interest) / 2.0
        if best_avg_oi is None or avg_oi > best_avg_oi:
            best_avg_oi = avg_oi

    return best_avg_oi
