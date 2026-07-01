"""Thin, validating wrapper around the OpenBB Platform SDK.

This is the *only* module in the codebase allowed to import `openbb`.
Every public method here returns one of the Pydantic models in
data_layer.models — raw OpenBB `OBBject`/DataFrame objects never leave
this file. Anything that fails validation is raised as
DataValidationError rather than passed downstream malformed.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime

from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_layer.exceptions import DataValidationError, ProviderFetchError
from data_layer.google_news import fetch_headlines
from data_layer.models import (
    AnalystRevision,
    FilingSummary,
    FilingType,
    FundamentalsSnapshot,
    MarketMover,
    OptionContract,
    OptionType,
    OrderBookLevel,
    OrderBookSnapshot,
    PriceBar,
    PriceSeries,
    SentimentPolarity,
    SentimentSnapshot,
    ThesisCandidate,
    VolatilitySnapshot,
)
from data_layer.sentiment_lexicon import score_headlines

logger = logging.getLogger(__name__)

# Market-cap-agnostic quality universe for the thesis and swing tracks.
# Spans mega-cap tech, mid-cap growth, healthcare, financials, consumer,
# energy, and defense. The thesis pre-filter (pullback %, shrink-volume)
# and the swing scanner (SMA/RSI/ADV) winnow this down each session —
# agents only run on names that clear those mechanical screens.
# Replaces the old aggressive_small_caps + undervalued_growth screeners
# which biased toward small-cap value traps with no catalyst.
_QUALITY_UNIVERSE: list[str] = [
    # Mega cap tech / AI
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "NFLX",
    # Semiconductors
    "AMD", "QCOM", "MU", "AMAT", "TXN", "LRCX", "KLAC", "ARM", "MRVL", "ON", "INTC",
    # Cloud / Software
    "CRM", "ADBE", "NOW", "INTU", "WDAY", "VEEV", "SNOW", "DDOG", "MDB", "HUBS", "ZM",
    # Cybersecurity
    "PANW", "CRWD", "FTNT", "ZS", "OKTA", "NET",
    # AI / Data infrastructure
    "PLTR", "AI", "PATH", "GTLB", "CFLT",
    # Healthcare / Biotech
    "LLY", "UNH", "ABBV", "MRK", "AMGN", "GILD", "REGN", "VRTX", "ISRG", "TMO",
    "DHR", "DXCM", "IDXX", "SYK", "ELV", "HCA",
    # Financials
    "JPM", "BAC", "GS", "MS", "BLK", "V", "MA", "AXP", "COF", "SCHW", "PYPL", "SQ",
    # Consumer / Retail
    "COST", "WMT", "HD", "TGT", "NKE", "LULU", "SBUX", "MCD", "CMG",
    "BKNG", "ABNB", "UBER", "LYFT",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",
    # Industrials / Defense / Space
    "CAT", "DE", "HON", "RTX", "LMT", "NOC", "GE", "AXON", "ROP", "SPCE",
    # Media / Entertainment / Streaming
    "DIS", "SPOT", "RBLX", "TTD",
    # EV / Clean Energy
    "RIVN", "LCID", "ENPH", "SEDG", "FSLR",
    # Broad market ETFs — captured here so swing scanner can use SPY/QQQ
    "SPY", "QQQ", "IWM",
]

_RETRYABLE = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(ProviderFetchError),
)


class OpenBBDataClient:
    """Normalizes OpenBB Platform calls across providers into strict
    Pydantic contracts. One instance per process is fine — the
    underlying `obb` SDK is stateless aside from credential config.
    """

    def __init__(self, pat: str | None = None) -> None:
        self._pat = pat
        self._obb = None  # lazy import/init so tests never need a real OpenBB hub login

    def _client(self):
        if self._obb is None:
            from openbb import obb  # imported lazily — heavy import, and only needed live

            self._obb = obb
        return self._obb

    @_RETRYABLE
    def get_price_history(
        self, symbol: str, start_date: date, end_date: date, interval: str = "1d", provider: str = "yfinance"
    ) -> PriceSeries:
        obb = self._client()
        try:
            result = obb.equity.price.historical(
                symbol=symbol, start_date=start_date, end_date=end_date, interval=interval, provider=provider
            )
            records = result.to_df().reset_index().to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001 — any upstream/provider failure is a fetch error
            raise ProviderFetchError(f"price history fetch failed for {symbol}: {exc}") from exc

        bars = []
        for row in records:
            try:
                o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                if any(math.isnan(v) or v <= 0 for v in (o, h, l, c)):
                    logger.debug("%s: skipping bar with missing/zero OHLC at %s", symbol, row.get("date"))
                    continue
                bars.append(PriceBar(
                    symbol=symbol,
                    timestamp=_coerce_datetime(row["date"]),
                    open=o, high=h, low=l, close=c,
                    volume=max(0, int(row.get("volume") or 0)),
                ))
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                logger.debug("%s: skipping malformed price bar — %s", symbol, exc)
        if not bars:
            raise DataValidationError(f"price history for {symbol} failed validation: no valid bars in response")
        return PriceSeries(symbol=symbol, interval=interval, bars=bars)

    @_RETRYABLE
    def get_order_book(self, symbol: str, provider: str = "yfinance") -> OrderBookSnapshot:
        obb = self._client()
        try:
            result = obb.equity.price.quote(symbol=symbol, provider=provider)
            row = result.to_df().reset_index().iloc[0]
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"order book fetch failed for {symbol}: {exc}") from exc

        try:
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            if not bid or not ask:
                raise DataValidationError(f"order book for {symbol} has missing bid/ask (bid={bid}, ask={ask})")
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=_coerce_datetime(row.get("date", datetime.utcnow())),
                bids=[OrderBookLevel(price=bid, size=float(row.get("bid_size") or 0))],
                asks=[OrderBookLevel(price=ask, size=float(row.get("ask_size") or 0))],
            )
        except DataValidationError:
            raise
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"order book for {symbol} failed validation: {exc}") from exc

    @_RETRYABLE
    def get_sentiment(self, symbol: str) -> SentimentSnapshot:
        """Headlines come from Google News' public RSS search (see
        data_layer/google_news.py), not OpenBB's `news.company` — the free
        yfinance provider behind that call returns a far smaller (~10),
        less financially-targeted set, with no sentiment field of its own.
        """
        try:
            headlines = fetch_headlines(symbol)
        except ProviderFetchError as exc:
            raise ProviderFetchError(f"sentiment fetch failed for {symbol}: {exc}") from exc

        try:
            score = score_headlines(headlines)
            polarity = (
                SentimentPolarity.BULLISH
                if score > 0.15
                else SentimentPolarity.BEARISH if score < -0.15 else SentimentPolarity.NEUTRAL
            )
            return SentimentSnapshot(
                symbol=symbol,
                as_of=datetime.utcnow(),
                source="google_news",
                score=max(-1.0, min(1.0, score)),
                polarity=polarity,
                headline_count=len(headlines),
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"sentiment for {symbol} failed validation: {exc}") from exc

    @_RETRYABLE
    def get_recent_filings(self, symbol: str, limit: int = 5, provider: str = "sec") -> list[FilingSummary]:
        obb = self._client()
        try:
            result = obb.equity.fundamental.filings(symbol=symbol, provider=provider, limit=limit)
            records = result.to_df().reset_index().to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"filings fetch failed for {symbol}: {exc}") from exc

        summaries = []
        for row in records:
            try:
                filing_date_raw = row.get("filing_date")
                if _is_nan(filing_date_raw):
                    logger.debug("%s: skipping filing row with missing filing_date", symbol)
                    continue
                raw_type = str(row.get("report_type", "other")).upper()
                filing_type = FilingType(raw_type) if raw_type in {t.value for t in FilingType} else FilingType.OTHER
                report_date_raw = row.get("report_date")
                summaries.append(
                    FilingSummary(
                        symbol=symbol,
                        filing_type=filing_type,
                        filed_on=_coerce_date(filing_date_raw),
                        period_end=_coerce_date(report_date_raw) if not _is_nan(report_date_raw) else None,
                        summary=str(row.get("report_type", "filing")),
                        url=str(row["report_url"]) if row.get("report_url") and not _is_nan(row.get("report_url")) else None,
                    )
                )
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                logger.debug("%s: skipping malformed filing row — %s (row keys: %s)", symbol, exc, list(row.keys()))
        return summaries

    @_RETRYABLE
    def get_fundamentals(self, symbol: str, provider: str = "yfinance") -> FundamentalsSnapshot:
        obb = self._client()
        try:
            metrics = obb.equity.fundamental.metrics(symbol=symbol, provider=provider).to_df().reset_index().iloc[0]
            estimates_df = obb.equity.estimates.consensus(symbol=symbol, provider=provider).to_df().reset_index()
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"fundamentals fetch failed for {symbol}: {exc}") from exc

        # `equity.fundamental.metrics` (yfinance) never carries raw eps/revenue —
        # only ratios/margins computed from them. The actual figures live on the
        # income statement. Fetched separately and best-effort: a transient
        # failure here shouldn't sink the whole fundamentals call when pe_ratio,
        # filings, and sentiment can still inform a decision without it.
        eps = None
        revenue = None
        try:
            income_row = (
                obb.equity.fundamental.income(symbol=symbol, provider=provider, period="annual", limit=1)
                .to_df()
                .reset_index()
                .iloc[0]
            )
            revenue = _safe_float(income_row.get("total_revenue"))
            eps = _safe_float(income_row.get("diluted_earnings_per_share")) or _safe_float(income_row.get("basic_earnings_per_share"))
        except Exception:  # noqa: BLE001
            logger.warning("income statement fetch failed for %s — eps/revenue will be unavailable", symbol)

        try:
            # `estimates.consensus` (yfinance) returns one aggregated row per
            # symbol — recommendation/target_consensus/number_of_analysts —
            # not a per-firm "date"/"firm"/"rating"/"target_price" shape. Reading
            # those nonexistent keys used to silently produce rating="n/a" and
            # target_price=None for every ticker regardless of what Wall Street
            # actually thinks, which is exactly backwards for a stock with a
            # strong analyst consensus the rest of the system can't see.
            revisions = [
                AnalystRevision(
                    symbol=symbol,
                    as_of=date.today(),
                    firm="consensus",
                    rating=_format_recommendation(row),
                    target_price=_safe_float(row.get("target_consensus")) or None,
                )
                for row in estimates_df.to_dict(orient="records")
            ]
            return FundamentalsSnapshot(
                symbol=symbol,
                as_of=date.today(),
                eps=eps,
                revenue=revenue,
                pe_ratio=_safe_float(metrics.get("pe_ratio")),
                revisions=revisions,
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"fundamentals for {symbol} failed validation: {exc}") from exc

    @_RETRYABLE
    def get_shares_float(self, symbol: str, provider: str = "yfinance") -> int:
        obb = self._client()
        try:
            row = obb.equity.profile(symbol=symbol, provider=provider).to_df().reset_index().iloc[0]
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"profile fetch failed for {symbol}: {exc}") from exc

        try:
            return int(row["shares_float"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"shares_float for {symbol} failed validation: {exc}") from exc

    @_RETRYABLE
    def get_option_chain(self, symbol: str, provider: str = "yfinance") -> list[OptionContract]:
        obb = self._client()
        try:
            records = obb.derivatives.options.chains(symbol=symbol, provider=provider).to_df().reset_index().to_dict(
                orient="records"
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"option chain fetch failed for {symbol}: {exc}") from exc

        contracts: list[OptionContract] = []
        for row in records:
            try:
                contracts.append(
                    OptionContract(
                        contract_symbol=str(row["contract_symbol"]),
                        underlying_symbol=str(row["underlying_symbol"]),
                        underlying_price=float(row["underlying_price"]),
                        expiration=_coerce_date(row["expiration"]),
                        dte=int(row["dte"]),
                        strike=float(row["strike"]),
                        option_type=OptionType(str(row["option_type"]).lower()),
                        bid=float(row["bid"] or 0.0),
                        ask=float(row["ask"] or 0.0),
                        implied_volatility=float(row["implied_volatility"]) if row.get("implied_volatility") is not None else None,
                        open_interest=int(row["open_interest"] or 0),
                        volume=int(row["volume"] or 0),
                    )
                )
            except (ValidationError, KeyError, TypeError, ValueError):
                continue  # one malformed contract in a chain of hundreds shouldn't sink the whole chain
        return contracts

    @_RETRYABLE
    def get_market_movers(self, provider: str = "yfinance") -> list[MarketMover]:
        """Combines OpenBB's active/gainers/losers discovery screens into one
        deduplicated pool — the dynamic candidate universe, in place of a
        fixed watchlist. Each screen already includes price/volume/% change,
        so no extra per-ticker fetch is needed for this step.
        """
        obb = self._client()
        seen: dict[str, MarketMover] = {}
        try:
            for screen in (obb.equity.discovery.active, obb.equity.discovery.gainers, obb.equity.discovery.losers):
                records = screen(provider=provider).to_df().reset_index().to_dict(orient="records")
                for row in records:
                    symbol = str(row["symbol"])
                    if symbol in seen:
                        continue
                    try:
                        seen[symbol] = MarketMover(
                            symbol=symbol,
                            price=float(row["price"]),
                            change=float(row["change"]),
                            percent_change=float(row["percent_change"]),
                            volume=int(row["volume"]),
                        )
                    except (ValidationError, KeyError, TypeError, ValueError):
                        continue  # one malformed row in a 200-row screen shouldn't sink the whole pool
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"market movers fetch failed: {exc}") from exc

        return list(seen.values())

    def get_thesis_universe(self) -> list[ThesisCandidate]:
        """Builds thesis/swing candidates from a quality, market-cap-agnostic universe.

        Downloads 1 year of daily OHLCV for _QUALITY_UNIVERSE in a single
        batched yfinance call, then computes price, 52-week high/low, MA50,
        and MA200 for each ticker. The thesis pre-filter (pullback %) and
        swing scanner (SMA/RSI/ADV) winnow this down — agents only run on
        names that pass those mechanical screens.

        Replaces the old aggressive_small_caps + undervalued_growth screeners
        which returned small-cap value traps with no catalyst.
        """
        import yfinance as yf

        try:
            raw = yf.download(
                _QUALITY_UNIVERSE,
                period="1y",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"thesis universe download failed: {exc}") from exc

        if raw.empty:
            raise ProviderFetchError("thesis universe: yfinance returned empty data")

        try:
            closes = raw["Close"]
            highs = raw["High"]
            lows = raw["Low"]
        except KeyError as exc:
            raise ProviderFetchError(f"thesis universe: unexpected yfinance columns: {exc}") from exc

        candidates: list[ThesisCandidate] = []
        for symbol in _QUALITY_UNIVERSE:
            try:
                sym_closes = closes[symbol].dropna()
                if len(sym_closes) < 20:
                    continue
                current_price = float(sym_closes.iloc[-1])
                if current_price < 5.0:  # skip penny stocks / delisted
                    continue
                year_high = float(highs[symbol].dropna().max())
                year_low = float(lows[symbol].dropna().min())
                ma50 = float(sym_closes.iloc[-50:].mean()) if len(sym_closes) >= 50 else None
                ma200 = float(sym_closes.iloc[-200:].mean()) if len(sym_closes) >= 200 else None
                candidates.append(ThesisCandidate(
                    symbol=symbol,
                    price=current_price,
                    year_high=year_high,
                    year_low=year_low,
                    ma50=ma50,
                    ma200=ma200,
                    eps_ttm=None,
                    eps_forward=None,
                    pe_forward=None,
                    market_cap=None,
                ))
            except (KeyError, IndexError, TypeError, ValueError, ValidationError):
                continue

        if not candidates:
            raise ProviderFetchError("thesis universe: no valid candidates after price fetch")

        logger.info("Thesis universe: %d quality candidates built from %d symbols", len(candidates), len(_QUALITY_UNIVERSE))
        return candidates

    @_RETRYABLE
    def get_volatility_snapshot(
        self,
        symbol: str,
        iv_history_days: int = 252,
        provider: str = "yfinance",
    ) -> VolatilitySnapshot:
        """Compute the volatility surface inputs the options analyst agents need.

        IVR and IV percentile are derived from yfinance's options chain across
        the available expirations. HV is computed from close-to-close log returns
        on the daily price history. Earnings date comes from the equity profile.

        This is intentionally a single, bounded fetch per ticker — one chain
        pull, one price history pull, one profile pull — so the cost per
        candidate is predictable before the LLM agents run.
        """
        import math

        obb = self._client()
        as_of = datetime.now()

        # ── 1. Options chain → front/back month IV, skew ──────────────────────
        chain = self.get_option_chain(symbol, provider=provider)
        if not chain:
            raise DataValidationError(f"empty options chain for {symbol}")

        # Separate into expirations, sorted by DTE
        expirations: dict[int, list[OptionContract]] = {}
        for c in chain:
            expirations.setdefault(c.dte, []).append(c)
        sorted_dtes = sorted(expirations.keys())

        # Front month: nearest expiration with at least 7 DTE and liquid options
        front_dte = next((d for d in sorted_dtes if d >= 7), None)
        back_dte = next((d for d in sorted_dtes if d >= 30), None)

        def _atm_iv(contracts: list[OptionContract]) -> float | None:
            liquid = [c for c in contracts if c.implied_volatility and c.bid > 0 and c.ask > 0]
            if not liquid:
                return None
            # Use call + put nearest ATM (averaging the two removes put/call parity noise)
            underlying = liquid[0].underlying_price
            calls = sorted([c for c in liquid if c.option_type == OptionType.CALL],
                           key=lambda c: abs(c.strike - underlying))
            puts = sorted([c for c in liquid if c.option_type == OptionType.PUT],
                          key=lambda c: abs(c.strike - underlying))
            ivs = [c.implied_volatility for c in (calls[:1] + puts[:1]) if c.implied_volatility]
            return sum(ivs) / len(ivs) if ivs else None

        front_iv = _atm_iv(expirations[front_dte]) if front_dte else None
        back_iv = _atm_iv(expirations[back_dte]) if back_dte and back_dte != front_dte else None
        iv_30 = back_iv or front_iv or 0.0
        term_structure_ratio = (front_iv / back_iv) if (front_iv and back_iv and back_iv > 0) else None

        # Skew: 25-delta proxy = strike roughly 0.5 SD OTM for the front month
        put_skew: float | None = None
        if front_dte and iv_30 > 0:
            underlying_price = chain[0].underlying_price
            sd_move = underlying_price * iv_30 * math.sqrt(front_dte / 365)
            otm_put_target = underlying_price - 0.5 * sd_move
            otm_call_target = underlying_price + 0.5 * sd_move
            front_contracts = expirations[front_dte]
            otm_puts = [c for c in front_contracts if c.option_type == OptionType.PUT and c.implied_volatility and c.bid > 0]
            otm_calls = [c for c in front_contracts if c.option_type == OptionType.CALL and c.implied_volatility and c.bid > 0]
            if otm_puts and otm_calls:
                closest_put = min(otm_puts, key=lambda c: abs(c.strike - otm_put_target))
                closest_call = min(otm_calls, key=lambda c: abs(c.strike - otm_call_target))
                if closest_put.implied_volatility and closest_call.implied_volatility:
                    put_skew = closest_put.implied_volatility - closest_call.implied_volatility

        # ── 2. Price history → HV and IV rank over past year ──────────────────
        from datetime import timedelta
        end = date.today()
        start = end - timedelta(days=iv_history_days + 10)
        try:
            price_series = self.get_price_history(symbol, start_date=start, end_date=end)
        except Exception as exc:  # noqa: BLE001
            raise ProviderFetchError(f"price history fetch for vol snapshot failed ({symbol}): {exc}") from exc

        closes = [b.close for b in price_series.bars]
        if len(closes) < 22:
            raise DataValidationError(f"insufficient price history for HV ({symbol}): {len(closes)} bars")

        def _hv(closes: list[float], window: int) -> float:
            if len(closes) < window + 1:
                return 0.0
            log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - window, len(closes))]
            mean = sum(log_returns) / len(log_returns)
            variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
            return math.sqrt(variance * 252)

        hv_20 = _hv(closes, 20)
        hv_30 = _hv(closes, min(30, len(closes) - 1))

        # IV rank: compute a proxy by using the chain's IV range over today only
        # (true 52-week IV history requires a paid data source; this uses
        # realized vol range as a disclosed proxy for the IV range).
        # For a free implementation: IV rank ≈ position of iv_30 within the
        # annualized HV range observed over the lookback period.
        hv_windows = [_hv(closes[:i], 20) for i in range(25, len(closes), 5) if i <= len(closes)]
        if hv_windows:
            hv_min = min(hv_windows)
            hv_max = max(hv_windows)
            iv_rank = ((iv_30 - hv_min) / (hv_max - hv_min) * 100) if hv_max > hv_min else 50.0
            iv_rank = max(0.0, min(100.0, iv_rank))
            days_below = sum(1 for h in hv_windows if h < iv_30)
            iv_percentile = (days_below / len(hv_windows)) * 100
        else:
            iv_rank = 50.0
            iv_percentile = 50.0

        # ── 3. Earnings date ───────────────────────────────────────────────────
        next_earnings: date | None = None
        earnings_within_dte = False
        try:
            profile_df = obb.equity.profile(symbol=symbol, provider=provider).to_df().reset_index()
            if not profile_df.empty:
                row = profile_df.iloc[0]
                raw_earnings = row.get("next_earnings_date")
                if _is_nan(raw_earnings):
                    raw_earnings = row.get("earnings_date")
                if not _is_nan(raw_earnings):
                    next_earnings = _coerce_date(raw_earnings)
                    days_to_earnings = (next_earnings - date.today()).days
                    earnings_within_dte = 0 < days_to_earnings <= 45
        except Exception:  # noqa: BLE001
            pass  # earnings date is best-effort; don't fail the whole snapshot

        return VolatilitySnapshot(
            symbol=symbol,
            as_of=as_of,
            iv_rank=round(iv_rank, 1),
            iv_percentile=round(iv_percentile, 1),
            iv_30=round(iv_30, 4),
            hv_20=round(hv_20, 4),
            hv_30=round(hv_30, 4),
            iv_hv_spread=round(iv_30 - hv_30, 4),
            term_structure_ratio=round(term_structure_ratio, 3) if term_structure_ratio else None,
            put_skew=round(put_skew, 4) if put_skew is not None else None,
            earnings_within_dte=earnings_within_dte,
            next_earnings_date=next_earnings,
        )


def _coerce_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _format_recommendation(row: dict) -> str:
    """`recommendation_mean` isn't always present on the consensus response
    (e.g. when `recommendation` itself is the literal "none" — no analyst
    consensus rating at all) — read defensively rather than assume every
    field that's usually there always is.
    """
    recommendation = row.get("recommendation")
    if recommendation is None:
        return "n/a"
    analysts = _safe_float(row.get("number_of_analysts"))
    analysts_text = f"{int(analysts)} analysts" if analysts is not None else "analyst count unknown"
    mean = _safe_float(row.get("recommendation_mean"))
    mean_text = f", mean {mean:.2f}" if mean is not None else ""
    return f"{recommendation} ({analysts_text}{mean_text})"


_MISSING_STRINGS = frozenset(("nan", "nat", "none", ""))


def _safe_float(value) -> float | None:
    """Return float(value), or None if value is None, NaN, or unconvertible."""
    if _is_nan(value):
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _is_nan(value) -> bool:
    """True for None, float NaN, and the string representations pandas emits for missing dates."""
    if value is None:
        return True
    if isinstance(value, float):
        try:
            return math.isnan(value)
        except (TypeError, ValueError):
            return False
    if isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS:
        return True
    return False


def _coerce_date(value) -> date:
    if _is_nan(value):
        raise ValueError(f"date value is missing (got {value!r})")
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    s = str(value)[:10]
    if s.strip().lower() in _MISSING_STRINGS:
        raise ValueError(f"date string is missing (got {value!r})")
    return date.fromisoformat(s)
