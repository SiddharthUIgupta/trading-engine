"""Parses OCC-format option contract symbols (e.g. "AAPL260629C00295000")
into their underlying/expiration/type/strike — used wherever a raw
contract symbol needs to be shown to a human instead of as an opaque
string. Returns None for anything that isn't OCC-shaped (a plain equity
ticker), so callers can handle both order types without a separate
asset-class lookup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_OCC_PATTERN = re.compile(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


@dataclass(frozen=True)
class ParsedOptionSymbol:
    underlying_symbol: str
    expiration: date
    option_type: str  # "call" or "put"
    strike: float


def parse_occ_symbol(symbol: str) -> ParsedOptionSymbol | None:
    match = _OCC_PATTERN.match(symbol)
    if not match:
        return None
    underlying, yy, mm, dd, cp, strike_raw = match.groups()
    try:
        expiration = date(2000 + int(yy), int(mm), int(dd))
    except ValueError:
        return None
    return ParsedOptionSymbol(
        underlying_symbol=underlying,
        expiration=expiration,
        option_type="call" if cp == "C" else "put",
        strike=int(strike_raw) / 1000.0,
    )
