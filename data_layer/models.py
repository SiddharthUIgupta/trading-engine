"""Strict Pydantic contracts for everything that crosses the Data Layer
boundary into the Analyst & Intel Layer. The analyst layer is only ever
allowed to import these types from data_layer — never the raw OpenBB
DataFrame/Obbject responses.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class PriceBar(StrictModel):
    symbol: str
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def _high_is_extreme(self) -> "PriceBar":
        # mode="after" (not field_validator) so both `high` and `low` are
        # guaranteed populated — field_validator("high") would run before
        # "low" exists in info.data, since "high" is declared first.
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class PriceSeries(StrictModel):
    symbol: str
    interval: str
    bars: list[PriceBar] = Field(min_length=1)


class OrderBookLevel(StrictModel):
    price: float = Field(gt=0)
    size: float = Field(ge=0)


class OrderBookSnapshot(StrictModel):
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel] = Field(min_length=1)
    asks: list[OrderBookLevel] = Field(min_length=1)

    @field_validator("asks")
    @classmethod
    def _asks_above_bids(cls, asks: list[OrderBookLevel], info) -> list[OrderBookLevel]:
        bids = info.data.get("bids")
        if bids and asks and min(a.price for a in asks) <= max(b.price for b in bids):
            raise ValueError("best ask must be strictly above best bid (crossed book)")
        return asks


class SentimentPolarity(str, Enum):
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"


class SentimentSnapshot(StrictModel):
    symbol: str
    as_of: datetime
    source: str
    score: float = Field(ge=-1.0, le=1.0)
    polarity: SentimentPolarity
    headline_count: int = Field(ge=0)


class FilingType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    OTHER = "other"


class FilingSummary(StrictModel):
    symbol: str
    filing_type: FilingType
    filed_on: date
    period_end: date | None = None
    summary: str = Field(min_length=1)
    url: str | None = None


class AnalystRevision(StrictModel):
    symbol: str
    as_of: date
    firm: str
    rating: str
    target_price: float | None = Field(default=None, gt=0)


class FundamentalsSnapshot(StrictModel):
    symbol: str
    as_of: date
    eps: float | None = None
    revenue: float | None = Field(default=None, ge=0)
    pe_ratio: float | None = Field(default=None, gt=0)
    revisions: list[AnalystRevision] = Field(default_factory=list)


class ShortInterestSnapshot(StrictModel):
    """FINRA bi-weekly settlement data, republished by yfinance/OpenBB.
    `as_of` is the actual settlement date this data reflects — always
    materially in the past (~20 days is typical), never "today". Use this
    for any point-in-time bookkeeping, never the fetch time.
    """
    symbol: str
    as_of: date
    shares_short: int = Field(ge=0)
    short_percent_of_float: float | None = Field(default=None, ge=0)
    days_to_cover: float | None = Field(default=None, ge=0)
    shares_short_prior_month: int | None = Field(default=None, ge=0)


class ShortableStatus(StrictModel):
    """Alpaca's current view of borrow availability — a live snapshot with
    no historical time series, unlike ShortInterestSnapshot's settlement
    date. `as_of` is always "now" (fetch time), by construction.
    """
    symbol: str
    as_of: datetime
    shortable: bool
    easy_to_borrow: bool


class MarketMover(StrictModel):
    """One row from an OpenBB discovery screen (active/gainers/losers) —
    cheap, already-computed price/volume stats with no extra fetch needed,
    used to prerank a large dynamic universe before the heavier per-ticker
    calls the prefilter needs.
    """

    symbol: str
    price: float = Field(gt=0)
    change: float
    percent_change: float
    volume: int = Field(ge=0)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionContract(StrictModel):
    """One row from OpenBB's options chain (yfinance provider) — a single
    expiration/strike/type combination for one underlying.
    """

    contract_symbol: str
    underlying_symbol: str
    underlying_price: float = Field(gt=0)
    expiration: date
    dte: int = Field(ge=0)
    strike: float = Field(gt=0)
    option_type: OptionType
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    implied_volatility: float | None = Field(default=None, ge=0)
    open_interest: int = Field(ge=0)
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def _ask_not_below_bid(self) -> "OptionContract":
        if self.ask > 0 and self.bid > 0 and self.ask < self.bid:
            raise ValueError("ask must be >= bid (crossed quote)")
        return self


class VolatilitySnapshot(StrictModel):
    """Computed volatility surface data for one underlying.

    iv_rank: where current 30-day IV sits within its 52-week high/low range
             (0–100). Natenberg's core signal: elevated IV → options are priced
             above what realized vol will likely be → sell premium.
    iv_percentile: fraction of trading days in the past year where IV was below
                   today's level (0–100). Tastylive's preferred metric because
                   it's less sensitive to a single extreme spike distorting the range.
    iv_30: current 30-day implied volatility, annualized decimal (0.30 = 30%).
    hv_20: 20-day realized/historical volatility, annualized decimal.
    hv_30: 30-day realized/historical volatility, annualized decimal.
    iv_hv_spread: iv_30 − hv_30. Positive = options overpriced vs what the stock
                  actually did — the variance risk premium, the structural edge.
    term_structure_ratio: front-month IV / back-month IV. > 1.0 = backwardation
                          (short-term fear elevated); < 1.0 = contango (normal).
    put_skew: 25-delta put IV − 25-delta call IV. Positive (put skew) is normal;
              extreme values signal tail-risk pricing or strong directional bias.
    earnings_within_dte: True if a known earnings event falls within the default
                         30–45 DTE trade window. Earnings inflate IV artificially —
                         you're not selling structural premium, you're selling event
                         risk, which is a different and harder game.
    next_earnings_date: the earnings date if known.
    """

    symbol: str
    as_of: datetime
    iv_rank: float = Field(ge=0, le=100)
    iv_percentile: float = Field(ge=0, le=100)
    iv_30: float = Field(ge=0)
    hv_20: float = Field(ge=0)
    hv_30: float = Field(ge=0)
    iv_hv_spread: float
    term_structure_ratio: float | None = None
    put_skew: float | None = None
    earnings_within_dte: bool = False
    next_earnings_date: date | None = None
    # Forward-looking realized vol estimate from GARCH(1,1). Set by the runtime
    # after fetching price history; None if the fetch failed or data was
    # insufficient. Used by the IVSurfaceAgent for a sharper VRP signal.
    garch_rv_forecast: float | None = Field(default=None, ge=0)


class ThesisCandidate(StrictModel):
    """One row from OpenBB's discovery screens (undervalued_growth,
    aggressive_small_caps, gainers, active — any market cap). Unlike
    MarketMover, these screens already carry the fields the thesis screen
    needs (year_high, moving averages, basic earnings context), so no
    extra per-ticker fetch is needed before the LLM consensus run.
    """

    symbol: str
    price: float = Field(gt=0)
    year_high: float = Field(gt=0)
    year_low: float = Field(gt=0)
    ma50: float | None = None
    ma200: float | None = None
    eps_ttm: float | None = None
    eps_forward: float | None = None
    pe_forward: float | None = None
    market_cap: float | None = None
