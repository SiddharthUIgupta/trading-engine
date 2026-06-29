"""Historical backtest universes.

Neither track's live universe is reconstructable historically: OpenBB's
discovery screens (active/gainers/losers, aggressive_small_caps) are live
snapshots with no historical query — there's no way to ask "who was in
today's gainers screen on 2024-03-15." So backtesting needs its own,
honestly-disclosed universe choice instead of pretending to replay the
live mechanism bar-for-bar.

Thesis track: S&P 500 constituents (free via Wikipedia) — a reasonable,
non-cherry-picked universe of the "established business" names the
pullback-from-52-week-high screen is actually aimed at.

Momentum track: today's live discovery-screen movers (active/gainers/
losers), the same function the live system already uses. This is NOT a
true historical reconstruction (today's volatile names weren't
necessarily volatile two months ago) — it's the best honestly-available
proxy for "a population of names that behave the way this screen is
looking for," used because there is no free alternative.
"""
from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "Mozilla/5.0 (compatible; trading-engine-backtest/1.0)"


def get_sp500_universe() -> list[str]:
    response = requests.get(_WIKIPEDIA_SP500_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    # Yahoo/yfinance uses '-' where Wikipedia uses '.' for share classes (e.g. BRK.B -> BRK-B).
    return [str(s).replace(".", "-") for s in table["Symbol"].tolist()]


def get_momentum_backtest_universe(data_client, limit: int = 150) -> list[str]:
    movers = data_client.get_market_movers()
    seen: list[str] = []
    for mover in movers:
        if mover.symbol not in seen:
            seen.append(mover.symbol)
        if len(seen) >= limit:
            break
    return seen
