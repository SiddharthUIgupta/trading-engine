"""The orchestrator that ties all three layers together.

This is the only module that imports from data_layer, analyst_layer,
AND execution_layer simultaneously — by design, neither of the other
two layers imports from each other directly. Data flows
data_layer -> analyst_layer -> (TradeProposal) -> execution_layer,
and every hop is re-validated at the boundary rather than trusted from
the layer before it.

Filter-first: the expensive 4-agent consensus only runs on tickers that
clear analyst_layer.prefilter's deterministic, zero-LLM screen. Intraday
exits default to execution_layer.exit_rules (also zero-LLM); the LLM
exit-review agent is an opt-in escalation, rate-limited to once per
position per day, used only when the rules say "hold" but the position's
regime has sharply reversed since entry.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

from anthropic import Anthropic

from analyst_layer import agent_scorer, lesson_store, momentum_scanner, options_structurer, orb_scanner, prefilter, thesis_scanner, universe, vol_analytics, vol_universe
from analyst_layer.correlation import apply_correlation_adjustment, check_portfolio_correlation
from analyst_layer.kelly import kelly_fraction_from_pnl_history
from analyst_layer.market_regime import DailyRegime, assess_daily_regime
from analyst_layer.reflection_agent import ReflectionAgent
from analyst_layer.agents.greeks_risk_officer import PortfolioGreeks
from analyst_layer.agents.intraday_exit_agent import IntradayExitAgent
from analyst_layer.agents.risk_officer_agent import AccountContext
from analyst_layer.agents.vol_regime_agent import VixContext
from analyst_layer.graph import run_consensus
from analyst_layer.pricing import estimate_cost_usd
from analyst_layer.schemas import Action, ConsensusPayload, StructureType, TradeProposal, VolConsensusPayload
from analyst_layer.vol_graph import run_vol_consensus
from config.settings import Settings
from data_layer.exceptions import DataLayerError
from data_layer.models import OptionContract, OptionType
from data_layer.occ_symbol import parse_occ_symbol
from data_layer.openbb_client import OpenBBDataClient
from execution_layer import exit_rules
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import CircuitBreaker, CircuitBreakerTripped, execute_global_shutdown
from execution_layer.state_store import StateStore
from execution_layer.tax_compliance import WashSaleGuard

logger = logging.getLogger(__name__)


class TradingRuntime:
    def __init__(
        self,
        settings: Settings,
        data_client: OpenBBDataClient,
        broker: AlpacaBroker,
        circuit_breaker: CircuitBreaker,
        state_store: StateStore,
        anthropic_client: Anthropic,
        watchlist: list[str],
        halt_callback: Callable[[], None] | None = None,
        wash_sale_guard: WashSaleGuard | None = None,
    ) -> None:
        self._settings = settings
        self._data_client = data_client
        self._broker = broker
        self._breaker = circuit_breaker
        self._state_store = state_store
        self._anthropic = anthropic_client
        self._watchlist = watchlist
        self._halt_callback = halt_callback
        self._wash_sale_guard = wash_sale_guard or WashSaleGuard(state_store)
        self._pending_payloads: dict[str, ConsensusPayload] = {}
        self._pending_regimes: dict[str, str] = {}
        self._pending_strategies: dict[str, str] = {}
        self._scanned_tickers_today: set[str] = set()
        self._executed_tickers_today: set[str] = set()
        # Independent from _scanned_tickers_today (the equity momentum track's
        # dedup set) — a ticker passing both screens on the same day can get
        # an equity buy AND an options buy; they're not coupled by default.
        self._scanned_options_tickers_today: set[str] = set()
        # Vol track runs once daily per ticker (premium-selling, not intraday signal)
        self._scanned_vol_tickers_today: set[str] = set()
        self._exit_agent = IntradayExitAgent(
            anthropic_client, settings.anthropic_subagent_model, usage_callback=self._record_usage
        )
        self._reflection_agent = ReflectionAgent(
            anthropic_client, settings.anthropic_subagent_model, usage_callback=self._record_usage
        )
        # Vol track uses a dynamically screened universe rather than the static
        # watchlist. Initialized to the watchlist so the system works on day one
        # before the first pre-market refresh runs.
        self._vol_universe: list[str] = list(watchlist)
        # Assessed once per day in pre_market_scan(). None until first assessment;
        # each track checks this before running — capability flags gate infrastructure
        # availability, regime gates whether today's conditions suit the strategy.
        self._daily_regime: DailyRegime | None = None
        # 60-day daily closes per ticker, reset each pre-market scan. Prevents
        # redundant data-layer calls when the same ticker appears in both the
        # correlation guard and the consensus loop, or across multiple candidates.
        self._price_cache: dict[str, list[float]] = {}

    def _assess_market_regime(self, today: date) -> DailyRegime | None:
        """Fetch SPY and VIX daily bars and run the zero-LLM regime assessment.

        Returns None on any data failure — callers treat None as 'use safe defaults'
        (arm directional tracks, disarm vol selling).
        """
        try:
            spy_series = self._data_client.get_price_history(
                "SPY", start_date=today - timedelta(days=60), end_date=today
            )
            vix_series = self._data_client.get_price_history(
                "^VIX", start_date=today - timedelta(days=45), end_date=today
            )
            spy_closes = [b.close for b in spy_series.bars]
            regime = assess_daily_regime(spy_closes, vix_series.bars)
            logger.info("%s", regime.log_summary())
            self._state_store.record_event(
                event_type="daily_regime_assessed",
                detail=regime.log_summary(),
            )
            return regime
        except DataLayerError as exc:
            logger.warning(
                "Market regime assessment failed — tracks fall back to safe defaults: %s", exc
            )
            return None

    def refresh_vol_universe(self) -> None:
        """Screen a dynamic candidate pool for options liquidity and update
        the vol track's universe. Runs pre-market so both the 10 AM and 1 PM
        vol scans use a fresh, liquidity-verified list.

        Falls back to the static watchlist on any failure — the system keeps
        trading even if the refresh can't reach the data provider.
        """
        if not self._settings.vol_options_track_enabled:
            return
        try:
            result = vol_universe.screen_vol_universe(
                data_client=self._data_client,
                seed=list(self._watchlist),
                min_option_oi=self._settings.vol_universe_min_option_oi,
                max_spread_pct=self._settings.vol_universe_max_spread_pct,
                min_dte=self._settings.vol_options_min_dte,
                max_dte=self._settings.vol_options_max_dte,
                max_size=self._settings.vol_universe_max_size,
            )
            self._vol_universe = result.passed
            self._state_store.record_event(
                event_type="vol_universe_refreshed",
                detail=(
                    f"screened={result.screened} passed={len(result.passed)} "
                    f"fallback={result.fallback_used} universe={result.passed}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("vol universe refresh failed — keeping existing universe: %s", exc)

    # ---- Phase 1: Pre-market bookkeeping (+ static-mode scan) ----
    def pre_market_scan(self) -> None:
        equity = self._broker.get_equity()
        today = date.today()
        self._breaker.start_trading_day(equity=equity, today=today)
        # Persisted so anything outside this process (e.g. the dashboard) can
        # compute *today's* P&L correctly, instead of assuming the account's
        # original opening balance is "today's starting point" — it isn't,
        # after the first day.
        self._state_store.record_event(event_type="day_start_equity", detail=f"{equity:.2f}")
        self._pending_payloads.clear()
        self._pending_regimes.clear()
        self._pending_strategies.clear()
        self._scanned_tickers_today.clear()
        self._executed_tickers_today.clear()
        self._scanned_options_tickers_today.clear()
        self._scanned_vol_tickers_today.clear()
        self._price_cache.clear()

        self._daily_regime = self._assess_market_regime(today)

        # Only refresh the vol universe when the regime actually supports premium
        # selling today — no point screening for liquidity if the track won't fire.
        vol_regime_armed = self._daily_regime.arm_vol if self._daily_regime else False
        if self._settings.vol_options_track_enabled and vol_regime_armed:
            self.refresh_vol_universe()
        elif self._settings.vol_options_track_enabled and not vol_regime_armed:
            logger.info("Vol universe refresh skipped — regime disarmed vol track for today")

        if not self._settings.dynamic_universe_enabled:
            logger.info("=== PRE-MARKET SCAN (static watchlist) ===")
            candidates = self._filter_static_watchlist(today)
            self._scan_and_run_consensus(candidates, today, equity)
        else:
            # The momentum scanner needs TODAY's intraday bars (VWAP, 9/20 EMA),
            # which don't exist yet pre-market — there's nothing valid to scan
            # until the market has been open a while. See momentum_scan_and_trade(),
            # which runs on the intraday cron instead.
            logger.info("=== PRE-MARKET: dynamic universe mode — momentum scan runs intraday instead ===")

    # ---- Runs on the intraday cron in dynamic-universe mode ----
    #
    # Previously a 7-criteria low-float scanner feeding the 4-agent
    # consensus — proven (backtest/momentum_backtest.py) to fire on almost
    # nothing regardless of universe (confirmed live: 0/50 every single
    # 15-minute tick, all week). Replaced with Opening Range Breakout, the
    # same signal now driving the options track, which backtested with
    # real, frequent signal generation. Deliberately deterministic, no
    # LLM/consensus: ORB's edge in backtest was on the bare mechanical
    # rule, and by the time a multi-second consensus call finished, an
    # intraday breakout's entry window would often already be gone.
    def momentum_scan_and_trade(self) -> None:
        orb_armed = self._daily_regime.arm_orb_equity if self._daily_regime else True
        if not self._settings.dynamic_universe_enabled or not orb_armed or self._breaker.is_stock_halted:
            if self._settings.dynamic_universe_enabled and not orb_armed:
                logger.info("ORB equity scan skipped — regime disarmed for today")
            return
        today = date.today()
        try:
            movers = self._data_client.get_market_movers()
        except DataLayerError as exc:
            logger.error("Momentum scan: market movers fetch failed: %s", exc)
            return
        candidates = [m.symbol for m in movers]
        new_candidates = [c for c in candidates if c not in self._scanned_tickers_today]
        if not new_candidates:
            return

        logger.info("=== INTRADAY MOMENTUM SCAN (ORB signal): %d new candidate(s) ===", len(new_candidates))
        equity = self._broker.get_equity()
        self._scan_and_trade_orb_equities(new_candidates, today, equity)

    def _scan_and_trade_orb_equities(self, candidates: list[str], today: date, equity: float) -> None:
        """Long-only: a confirmed short breakdown is a real ORB signal,
        but this system has no short-selling infrastructure (margin
        requirements, unbounded loss profile, none of the existing
        guardrail math accounts for it) — the options track already
        expresses ORB's bearish signals safely via buying puts, which
        is the intended outlet for those, not equities.
        """
        signals_found = 0
        for ticker in candidates:
            self._scanned_tickers_today.add(ticker)
            try:
                intraday = self._data_client.get_price_history(ticker, start_date=today, end_date=today, interval="5m")
            except DataLayerError as exc:
                logger.debug("%s: skipped for ORB equity scan (%s)", ticker, exc)
                continue

            signal = orb_scanner.evaluate_orb(intraday, opening_range_minutes=15, volume_confirmation_multiple=1.5)
            if signal.direction != "long":
                continue

            signals_found += 1
            logger.info("%s: ORB long signal (%s)", ticker, "; ".join(signal.reasons))
            self._open_orb_equity_position(ticker, signal, equity, today)

        self._state_store.record_event(
            event_type="momentum_orb_scan_summary",
            detail=f"{signals_found} long ORB signal(s) found across {len(candidates)} candidates",
        )

    def _open_orb_equity_position(self, ticker: str, signal, equity: float, today: date) -> None:
        current_price = signal.opening_range_high  # the breakout level itself, a defensible proxy for entry price ahead of the actual fill
        stop_price = signal.opening_range_low
        risk_per_share = current_price - stop_price
        if risk_per_share <= 0:
            logger.info("%s: ORB signal has non-positive risk per share — skipping", ticker)
            return
        target_price = current_price + 2 * risk_per_share

        max_notional = equity * self._settings.max_position_size_pct
        quantity = math.floor(max_notional / current_price)
        if quantity <= 0:
            logger.info("%s: max position size insufficient for 1 share at %.2f", ticker, current_price)
            return

        proposal = TradeProposal(ticker=ticker, action=Action.BUY, quantity=quantity, limit_price=current_price)
        try:
            self._breaker.assert_not_tripped()
            self._breaker.validate_position_size(proposal, equity)
            result = self._broker.submit_order(proposal)
            logger.info("%s: ORB equity order result=%s", ticker, result)
            self._record_order_event(ticker, proposal, result)
            self._record_fill(ticker, proposal, today, entry_regime="bullish_crossover", strategy="orb")
            self._state_store.upsert_position(ticker, int(self._broker.get_position_shares(ticker)), current_price, stop_price=stop_price, target_price=target_price)

            new_equity = self._broker.get_equity()
            if self._breaker.check_profit_target(new_equity):
                self._lock_in_profit(reason=f"daily profit target reached after ORB trade on {ticker}, equity={new_equity:.2f}")
        except CircuitBreakerTripped as exc:
            logger.error("Circuit breaker blocked ORB equity order for %s: %s", ticker, exc)
            self._trip_breaker(reason=str(exc))

    # ---- Options track: Opening Range Breakout signal, expressed via
    # calls/puts instead of shares. Independently toggleable from the
    # equity momentum track — see options_track_enabled.
    #
    # Previously mirrored the low-float momentum scanner's signal — but
    # that screen was proven (backtest/momentum_backtest.py) to fire on
    # almost nothing regardless of universe, which meant this track
    # inherited the same near-total silence (confirmed live: zero
    # autonomous fires all week). ORB backtested with real, frequent
    # signal generation and a roughly-breakeven-to-positive edge with
    # volume confirmation — a much more honest signal source. ----
    def options_scan_and_trade(self) -> None:
        orb_opt_armed = self._daily_regime.arm_orb_options if self._daily_regime else True
        if not self._settings.options_track_enabled or not orb_opt_armed or self._breaker.is_options_halted:
            if self._settings.options_track_enabled and not orb_opt_armed:
                logger.info("ORB options scan skipped — regime disarmed for today (neutral market)")
            return
        today = date.today()
        try:
            movers = self._data_client.get_market_movers()
        except DataLayerError as exc:
            logger.error("Options scan: market movers fetch failed: %s", exc)
            return
        candidates = [m.symbol for m in movers]
        new_candidates = [c for c in candidates if c not in self._scanned_options_tickers_today]
        if not new_candidates:
            return

        logger.info("=== OPTIONS SCAN (ORB signal): %d new candidate(s) ===", len(new_candidates))
        equity = self._broker.get_equity()
        self._scan_and_trade_options_orb(new_candidates, today, equity)

    def _scan_and_trade_options_orb(self, candidates: list[str], today: date, equity: float) -> None:
        signals_found = 0
        for ticker in candidates:
            self._scanned_options_tickers_today.add(ticker)
            try:
                intraday = self._data_client.get_price_history(ticker, start_date=today, end_date=today, interval="5m")
            except DataLayerError as exc:
                logger.debug("%s: skipped for options ORB scan (%s)", ticker, exc)
                continue

            signal = orb_scanner.evaluate_orb(intraday, opening_range_minutes=15, volume_confirmation_multiple=1.5)
            if signal.direction == "none":
                continue

            signals_found += 1
            logger.info("%s: ORB signal -> %s (%s)", ticker, signal.direction, "; ".join(signal.reasons))
            direction = Action.BUY if signal.direction == "long" else Action.SELL
            self._open_option_position(ticker, direction, equity, today)

        self._state_store.record_event(
            event_type="options_orb_scan_summary",
            detail=f"{signals_found} ORB signal(s) found across {len(candidates)} candidates",
        )

    def _open_option_position(self, ticker: str, direction: Action, equity: float, today: date) -> None:
        try:
            chain = self._data_client.get_option_chain(ticker)
        except DataLayerError as exc:
            logger.error("%s: option chain fetch failed: %s", ticker, exc)
            return

        selection = options_structurer.select_contract(
            chain, direction, min_dte=self._settings.options_min_dte, max_dte=self._settings.options_max_dte
        )
        if not selection.selected:
            logger.info("%s: no contract selected (%s)", ticker, "; ".join(selection.reasons))
            return

        contract = selection.contract
        logger.info("%s: %s", ticker, "; ".join(selection.reasons))

        # Sized off premium at risk (max loss on a long option), not share
        # notional — options_max_risk_pct is deliberately much smaller than
        # the equity cap; see config/settings.py.
        max_risk_dollars = equity * self._settings.options_max_risk_pct
        contracts = math.floor(max_risk_dollars / (contract.ask * 100)) if contract.ask > 0 else 0
        if contracts <= 0:
            logger.info(
                "%s: max risk %.2f insufficient for 1 contract at ask %.2f (x100 multiplier)",
                ticker, max_risk_dollars, contract.ask,
            )
            return

        try:
            self._breaker.assert_options_trading_allowed()
            result = self._broker.submit_option_order(
                contract.contract_symbol, side=Action.BUY, contracts=contracts, limit_price=contract.ask
            )
            logger.info("%s: options order result=%s", contract.contract_symbol, result)
            self._record_option_order_event(contract.contract_symbol, Action.BUY, contracts, contract.ask, result)
            self._record_option_fill(
                contract.contract_symbol, contract.underlying_symbol, contract.option_type.value,
                contract.strike, contract.expiration.isoformat(), Action.BUY, contracts, today,
            )

            new_equity = self._broker.get_equity()
            if self._breaker.check_profit_target(new_equity):
                self._lock_in_profit(reason=f"daily profit target reached after options trade on {ticker}, equity={new_equity:.2f}")
        except CircuitBreakerTripped as exc:
            logger.error("Circuit breaker blocked options order for %s: %s", ticker, exc)
            self._trip_breaker(reason=str(exc))

    # ---- Runs once daily — fundamentals don't change intraday ----
    def thesis_scan_and_trade(self) -> None:
        thesis_armed = self._daily_regime.arm_thesis if self._daily_regime else True
        if not self._settings.thesis_track_enabled or not thesis_armed or self._breaker.is_stock_halted:
            if self._settings.thesis_track_enabled and not thesis_armed:
                logger.info("Thesis scan skipped — regime disarmed (VIX > 30, fear environment)")
            return
        today = date.today()
        candidates = self._build_thesis_candidates(today)
        new_candidates = [c for c in candidates if c not in self._scanned_tickers_today]
        if not new_candidates:
            return

        logger.info("=== THESIS SCAN: %d new candidate(s) ===", len(new_candidates))
        equity = self._broker.get_equity()
        self._scan_and_run_consensus(new_candidates, today, equity, strategy="thesis")
        self._execute_pending_payloads(today)

    def _build_thesis_candidates(self, today: date) -> list[str]:
        """Scans OpenBB's aggressive_small_caps/undervalued_growth discovery
        screens for names pulled back from their 52-week high, capped at
        `thesis_max_daily_candidates`. Deliberately not gated by the momentum
        scanner's float/volume/price-action criteria — opposite use case.
        """
        try:
            candidates = self._data_client.get_thesis_universe()
        except DataLayerError as exc:
            logger.error("Thesis universe fetch failed: %s", exc)
            return []

        passed: list[tuple[str, float]] = []
        for candidate in candidates:
            signal = thesis_scanner.evaluate_thesis_candidate(
                candidate,
                min_pullback_pct=self._settings.thesis_min_pullback_pct,
                max_pullback_pct=self._settings.thesis_max_pullback_pct,
            )
            if signal.passed:
                score = signal.score
                # Shrink-volume retest check: MA5>MA10>MA20 + quiet pullback volume.
                # Passing boosts the ranking score so these surface above plain pullbacks.
                # Non-passing candidates are NOT excluded — a dislocation thesis doesn't
                # require an intact uptrend.
                try:
                    sv_series = self._data_client.get_price_history(
                        candidate.symbol, start_date=today - timedelta(days=30), end_date=today
                    )
                    sv_signal = thesis_scanner.evaluate_shrink_volume_pullback(sv_series)
                    if sv_signal.passed:
                        score += 0.5
                        logger.info(
                            "%s: shrink-volume confirmed (%s) — score boosted to %.2f",
                            candidate.symbol, "; ".join(sv_signal.reasons), score,
                        )
                    else:
                        logger.debug("%s: shrink-volume not confirmed (%s)", candidate.symbol, "; ".join(sv_signal.reasons))
                except DataLayerError as exc:
                    logger.debug("%s: shrink-volume price fetch failed — no boost: %s", candidate.symbol, exc)
                logger.info("%s: thesis scan PASSED (%s)", candidate.symbol, "; ".join(signal.reasons))
                passed.append((candidate.symbol, score))

        passed.sort(key=lambda pair: pair[1], reverse=True)
        capped = [ticker for ticker, _ in passed[: self._settings.thesis_max_daily_candidates]]
        self._state_store.record_event(
            event_type="thesis_scan_summary",
            detail=f"passed={len(passed)}/{len(candidates)} thesis-universe candidates, capped to {len(capped)}",
        )
        return capped

    def _get_daily_closes(self, ticker: str, today: date) -> list[float] | None:
        """Fetch 60-day daily closes, using the session cache to avoid duplicate requests."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        try:
            series = self._data_client.get_price_history(
                ticker, start_date=today - timedelta(days=60), end_date=today
            )
            closes = [b.close for b in series.bars]
            self._price_cache[ticker] = closes
            return closes
        except DataLayerError:
            return None

    def _scan_and_run_consensus(
        self, candidates: list[str], today: date, equity: float, strategy: str = "momentum"
    ) -> None:
        for ticker in candidates:
            self._scanned_tickers_today.add(ticker)
            self._pending_strategies[ticker] = strategy
            try:
                sentiment = self._data_client.get_sentiment(ticker)
                fundamentals = self._data_client.get_fundamentals(ticker)
                filings = self._data_client.get_recent_filings(ticker)
                price_series = self._data_client.get_price_history(
                    ticker, start_date=today - timedelta(days=60), end_date=today
                )
            except DataLayerError as exc:
                logger.error("Skipping %s: %s", ticker, exc)
                continue

            regime = prefilter.compute_regime(
                [bar.close for bar in price_series.bars],
                self._settings.filter_sma_short_window,
                self._settings.filter_sma_long_window,
            )
            self._pending_regimes[ticker] = regime
            logger.info("%s: cleared screening — running consensus", ticker)

            # Cache this ticker's closes for correlation check + future reuse
            proposed_closes = [b.close for b in price_series.bars]
            self._price_cache[ticker] = proposed_closes

            # Retrieve relevant lessons; prepend agent accuracy track record
            setup_tags = lesson_store.derive_setup_tags(price_series, strategy)
            relevant_lessons = lesson_store.get_relevant_lessons(
                self._state_store, strategy, setup_tags
            )
            if relevant_lessons:
                logger.debug("%s: injecting %d past lessons into consensus", ticker, len(relevant_lessons))

            accuracy_rows = self._state_store.get_agent_accuracy(strategy, regime)
            accuracy_context = agent_scorer.format_accuracy_context(accuracy_rows)
            lessons_text = accuracy_context + lesson_store.format_for_prompt(relevant_lessons)

            existing_shares = self._broker.get_position_shares(ticker)

            # ── Kelly position sizing ─────────────────────────────────────────
            pnl_history = self._state_store.get_pnl_history()
            kelly_fraction, kelly_reason = kelly_fraction_from_pnl_history(
                pnl_history, self._settings.max_position_size_pct
            )

            # ── Correlation guard (parallel fetches, session cache) ───────────
            active_held = [
                pos["ticker"] for pos in self._state_store.get_positions()
                if pos["ticker"] != ticker and pos.get("quantity", 0) > 0
            ]
            held_closes: dict[str, list[float]] = {}
            if active_held:
                def _fetch(ht: str) -> tuple[str, list[float]]:
                    closes = self._get_daily_closes(ht, today)
                    if closes is None:
                        raise DataLayerError(f"no closes for {ht}")
                    return ht, closes

                with ThreadPoolExecutor(max_workers=min(4, len(active_held))) as pool:
                    futures = {pool.submit(_fetch, ht): ht for ht in active_held}
                    for fut in as_completed(futures):
                        try:
                            ht, closes = fut.result()
                            held_closes[ht] = closes
                        except Exception:
                            pass  # correlation check is best-effort

            max_corr, corr_desc = check_portfolio_correlation(proposed_closes, held_closes)
            kelly_fraction, corr_reason, corr_blocked = apply_correlation_adjustment(
                kelly_fraction, max_corr, corr_desc
            )
            logger.debug("%s: Kelly=%s, correlation=%s", ticker, kelly_reason, corr_reason)

            account = AccountContext(
                equity=equity,
                current_price=price_series.bars[-1].close,
                existing_shares=existing_shares,
                max_daily_drawdown_pct=self._breaker.max_daily_drawdown_pct,
                kelly_fraction=kelly_fraction,
                kelly_reason=kelly_reason,
                correlation_hard_blocked=corr_blocked,
                correlation_reason=corr_reason,
            )
            payload = run_consensus(
                client=self._anthropic,
                model=self._settings.anthropic_model,
                max_position_size_pct=self._settings.max_position_size_pct,
                ticker=ticker,
                sentiment=sentiment,
                fundamentals=fundamentals,
                filings=filings,
                price_series=price_series,
                account=account,
                subagent_model=self._settings.anthropic_subagent_model,
                usage_callback=self._record_usage,
                lessons=lessons_text,
            )

            # Record signal log so agent accuracy can be scored after the trade closes
            self._state_store.record_agent_signal_log(
                ticker=ticker,
                track=strategy,
                regime=regime,
                proposed_action=payload.proposal.action.value,
                signals=[
                    {
                        "agent_name": s.agent_name,
                        "stance": s.stance.value,
                        "confidence": s.confidence.value,
                    }
                    for s in payload.signals
                ],
            )
            # Record which lessons were active so their scores update after close
            for lesson in relevant_lessons:
                if "id" in lesson:
                    self._state_store.record_lesson_injection(lesson["id"], ticker, strategy)

            self._pending_payloads[ticker] = payload
            self._state_store.record_run(payload)
            logger.info(
                "%s: proposal=%s verdict=%s", ticker, payload.proposal.action.value, payload.risk_review.verdict.value
            )

    def _build_momentum_candidates(self, today: date) -> list[str]:
        """Scans OpenBB's active/gainers/losers discovery screens through
        the low-float momentum scanner, capped at `max_daily_candidates`.
        """
        try:
            movers = self._data_client.get_market_movers()
        except DataLayerError as exc:
            logger.error("Market movers fetch failed, falling back to static watchlist: %s", exc)
            return self._filter_static_watchlist(today)

        movers_by_symbol = {m.symbol: m for m in movers}
        preranked = universe.prerank_movers(movers, self._settings.universe_prerank_limit)

        passed: list[tuple[str, float]] = []
        for ticker in preranked:
            mover = movers_by_symbol[ticker]
            try:
                intraday_series = self._data_client.get_price_history(
                    ticker, start_date=today, end_date=today, interval="5m"
                )
                shares_float = self._data_client.get_shares_float(ticker)
                # Ends yesterday, not today — including today's still-partial
                # bar would understate the average and inflate relative volume.
                volume_history = self._data_client.get_price_history(
                    ticker,
                    start_date=today - timedelta(days=self._settings.momentum_volume_lookback_days * 2),
                    end_date=today - timedelta(days=1),
                )
            except DataLayerError as exc:
                logger.debug("%s: skipped during momentum scan (%s)", ticker, exc)
                continue

            recent_bars = volume_history.bars[-self._settings.momentum_volume_lookback_days :]
            average_daily_volume = sum(bar.volume for bar in recent_bars) / len(recent_bars) if recent_bars else 0.0

            signal = momentum_scanner.evaluate_low_float_momentum(
                intraday_series=intraday_series,
                shares_float=shares_float,
                today_percent_change=mover.percent_change,
                today_volume=mover.volume,
                average_daily_volume=average_daily_volume,
                max_float_shares=self._settings.momentum_max_float_shares,
                ema_short_period=self._settings.momentum_ema_short_period,
                ema_long_period=self._settings.momentum_ema_long_period,
                min_daily_gain_pct=self._settings.momentum_min_daily_gain_pct,
                clean_body_dominance_threshold=self._settings.momentum_clean_body_dominance_threshold,
                clean_lookback_bars=self._settings.momentum_clean_lookback_bars,
                min_relative_volume=self._settings.momentum_min_relative_volume,
                price_min=self._settings.momentum_price_min,
                price_max=self._settings.momentum_price_max,
            )
            if signal.passed:
                logger.info("%s: momentum scan PASSED (%s)", ticker, "; ".join(signal.reasons))
                passed.append((ticker, signal.score))
            else:
                logger.debug("%s: momentum scan failed (%s)", ticker, "; ".join(signal.reasons))

        passed.sort(key=lambda pair: pair[1], reverse=True)
        capped = [ticker for ticker, _ in passed[: self._settings.max_daily_candidates]]
        self._state_store.record_event(
            event_type="momentum_scan_summary",
            detail=f"passed={len(passed)}/{len(preranked)} preranked movers, capped to {len(capped)}",
        )
        return capped

    def _filter_static_watchlist(self, today: date) -> list[str]:
        passed: list[str] = []
        for ticker in self._watchlist:
            try:
                sentiment = self._data_client.get_sentiment(ticker)
                filings = self._data_client.get_recent_filings(ticker)
                price_series = self._data_client.get_price_history(
                    ticker, start_date=today - timedelta(days=60), end_date=today
                )
            except DataLayerError as exc:
                logger.error("Skipping %s during pre-market scan: %s", ticker, exc)
                continue

            shares_float: int | None = None
            try:
                shares_float = self._data_client.get_shares_float(ticker)
            except DataLayerError:
                pass  # turnover rate is best-effort; don't fail the whole prefilter pass

            signal = prefilter.evaluate_ticker(
                price_series=price_series,
                sentiment=sentiment,
                filings=filings,
                today=today,
                rsi_period=self._settings.filter_rsi_period,
                rsi_oversold=self._settings.filter_rsi_oversold,
                rsi_overbought=self._settings.filter_rsi_overbought,
                sma_short_window=self._settings.filter_sma_short_window,
                sma_long_window=self._settings.filter_sma_long_window,
                volume_spike_multiplier=self._settings.filter_volume_spike_multiplier,
                sentiment_abs_threshold=self._settings.filter_sentiment_abs_threshold,
                recent_filing_days=self._settings.filter_recent_filing_days,
                shares_float=shares_float,
            )
            if signal.passed:
                logger.info("%s: filter PASSED (%s)", ticker, "; ".join(signal.reasons))
                passed.append(ticker)
            else:
                logger.info("%s: filtered out (%s)", ticker, "; ".join(signal.reasons))

        self._state_store.record_event(
            event_type="prefilter_summary", detail=f"passed={len(passed)}/{len(self._watchlist)} watchlist tickers"
        )
        return passed

    # ---- Phase 2: Market-open execution ----
    def market_open_execution(self) -> None:
        logger.info("=== MARKET-OPEN EXECUTION ===")
        if self._breaker.is_stock_halted:
            logger.warning("Stock trading halted for today (tripped=%s, profit_locked=%s) — skipping execution.",
                            self._breaker.is_tripped, self._breaker.is_profit_locked)
            return
        self._execute_pending_payloads(date.today())

    def _execute_pending_payloads(self, today: date) -> None:
        """Shared by market_open_execution (static-watchlist mode, fires
        once at 9:30) and momentum_scan_and_trade (dynamic mode, fires
        every intraday tick) — `_executed_tickers_today` stops the latter
        from re-submitting an order for a ticker it already acted on
        earlier in the day.
        """
        if self._breaker.is_stock_halted:
            return
        equity = self._broker.get_equity()
        for ticker, payload in self._pending_payloads.items():
            if ticker in self._executed_tickers_today:
                continue
            if self._breaker.is_stock_halted:
                logger.info("Stock trading halted mid-batch — skipping remaining tickers.")
                break
            self._executed_tickers_today.add(ticker)
            if not payload.is_executable:
                logger.info("%s: not executable (verdict=%s) — skipping.", ticker, payload.risk_review.verdict.value)
                continue

            proposal = payload.proposal
            if proposal.action == Action.BUY:
                violation = self._wash_sale_guard.check_before_buy(ticker, today)
                if violation is not None:
                    logger.warning("Blocking BUY for %s — wash sale: %s", ticker, violation.reason)
                    self._state_store.record_event(event_type="wash_sale_blocked", detail=violation.reason)
                    continue
            elif proposal.action == Action.SELL:
                self._wash_sale_guard.warn_before_sell(ticker, proposal.limit_price, today)

            try:
                self._breaker.assert_not_tripped()
                self._breaker.validate_position_size(proposal, equity)
                result = self._broker.submit_order(proposal)
                logger.info("%s: order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(
                    ticker,
                    proposal,
                    today,
                    entry_regime=self._pending_regimes.get(ticker),
                    strategy=self._pending_strategies.get(ticker),
                )

                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after trading {ticker}, equity={equity:.2f}")
                    break
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked order for %s: %s", ticker, exc)
                self._trip_breaker(reason=str(exc))
                break

    # ---- Phase 3: Intraday monitoring ----
    def intraday_monitoring(self) -> None:
        if self._breaker.is_tripped:
            return  # real risk breach -- a true full halt, nothing runs
        self._reconcile_positions()
        if self._settings.options_track_enabled or self._settings.vol_options_track_enabled:
            self._reconcile_option_positions()
        equity = self._broker.get_equity()
        logger.info("Intraday check: equity=%.2f", equity)
        if self._breaker.check_drawdown(equity):
            self._trip_breaker(reason=f"intraday drawdown breach at equity={equity:.2f}")
            return
        # Profit target is checked only while stocks aren't already halted —
        # check_profit_target() re-flags True on every call once banked, so
        # without this guard a later tick would re-trigger _lock_in_profit
        # again for no reason.
        if not self._breaker.is_stock_halted and self._breaker.check_profit_target(equity):
            self._lock_in_profit(reason=f"daily profit target reached, equity={equity:.2f}")
        if not self._breaker.is_stock_halted:
            self._check_intraday_exits(equity)
            self._check_orb_exits(equity)
        if self._settings.options_track_enabled and not self._breaker.is_options_halted:
            self._check_options_exits(equity)
        if self._settings.vol_options_track_enabled and not self._breaker.is_options_halted:
            self._check_vol_options_exits(equity)

    def _reconcile_positions(self) -> None:
        """Catches the case `submit_order`'s fill-poll doesn't: a limit
        order that's genuinely slow to fill (minutes, not seconds) gets
        correctly recorded as 0 shares at submission time, and nothing
        ever re-checks it once it eventually does fill. Runs every
        intraday tick (15 min) against every ticker we have any local
        record for, regardless of current recorded quantity, and corrects
        drift from the broker's actual state.
        """
        for position in self._state_store.get_positions():
            ticker = position["ticker"]
            detail = self._broker.get_position_detail(ticker)
            real_qty = int(detail["qty"]) if detail else 0
            if real_qty != position["quantity"]:
                logger.warning(
                    "%s: local quantity %d out of sync with broker's %d — reconciling",
                    ticker, position["quantity"], real_qty,
                )
                self._state_store.upsert_position(
                    ticker, real_qty, detail["avg_entry_price"] if detail else position["avg_entry_price"]
                )

    def _reconcile_option_positions(self) -> None:
        """Same gap, same fix, options side: a slow-to-fill options order
        (paper-account options fills can lag well behind equities — seen
        live) gets correctly recorded as 0 contracts at submission time,
        and nothing else ever re-checks it once it eventually does fill.
        """
        for position in self._state_store.get_option_positions():
            contract_symbol = position["contract_symbol"]
            detail = self._broker.get_position_detail(contract_symbol)
            real_qty = int(detail["qty"]) if detail else 0
            if real_qty != position["quantity"]:
                logger.warning(
                    "%s: local options quantity %d out of sync with broker's %d — reconciling",
                    contract_symbol, position["quantity"], real_qty,
                )
                self._state_store.upsert_option_position(
                    contract_symbol,
                    position["underlying_symbol"],
                    position["option_type"],
                    position["strike"],
                    position["expiration"],
                    real_qty,
                    detail["avg_entry_price"] if detail else position["avg_entry_price"],
                )

    def _check_intraday_exits(self, equity: float) -> None:
        """Default path is plain Python thresholds (execution_layer.exit_rules)
        — no LLM, no per-tick API cost. The LLM exit-review agent only fires
        as a rate-limited escalation when the rules say "hold" but the
        position's regime has sharply reversed since entry.
        """
        today = date.today()
        for position in self._state_store.get_positions():
            if self._breaker.is_stock_halted:
                return
            ticker = position["ticker"]
            if position["quantity"] <= 0 or position["strategy"] == "orb":
                continue  # ORB positions use their own fixed-price stop/target + EOD close — see _check_orb_exits

            detail = self._broker.get_position_detail(ticker)
            if detail is None or detail["qty"] <= 0:
                continue

            self._state_store.update_high_water_mark(ticker, detail["current_price"])
            high_water_mark = max(position["high_water_mark"], detail["current_price"])

            decision = exit_rules.evaluate_exit(
                avg_entry_price=detail["avg_entry_price"],
                current_price=detail["current_price"],
                high_water_mark=high_water_mark,
                **self._exit_params_for(position["strategy"]),
            )

            proposal: TradeProposal | None = None
            if decision.should_exit:
                logger.info("%s: rule-based exit — %s", ticker, decision.reason)
                proposal = TradeProposal(
                    ticker=ticker, action=Action.SELL, quantity=int(detail["qty"]), limit_price=detail["current_price"]
                )
            elif self._should_escalate_to_llm(ticker, position, today):
                proposal = self._exit_agent.review(
                    ticker=ticker,
                    quantity=detail["qty"],
                    avg_entry_price=detail["avg_entry_price"],
                    current_price=detail["current_price"],
                    unrealized_plpc=detail["unrealized_plpc"],
                )
                self._state_store.record_event(
                    event_type=f"intraday_llm_exit_escalation:{ticker}",
                    detail=f"regime reversed since entry; agent decided {proposal.action.value}",
                )
                logger.info("%s: LLM exit-review escalation -> %s", ticker, proposal.action.value)

            if proposal is None or proposal.action != Action.SELL:
                continue

            self._wash_sale_guard.warn_before_sell(ticker, proposal.limit_price, today)
            try:
                self._breaker.assert_not_tripped()
                self._breaker.validate_position_size(proposal, equity)
                result = self._broker.submit_order(proposal)
                logger.info("%s: intraday exit order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(ticker, proposal, today)
                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after intraday exit of {ticker}, equity={equity:.2f}")
                    return
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked intraday exit for %s: %s", ticker, exc)
                self._trip_breaker(reason=str(exc))
                return

    def _exit_params_for(self, strategy: str) -> dict:
        """Which exit-rule thresholds apply to a position is entirely a
        function of which track opened it — momentum gets a tight, fast
        bracket; thesis gets a wide stop, no fixed target, and a trailing
        stop that only engages once it's already up significantly.
        """
        if strategy == "thesis":
            return {
                "stop_loss_pct": self._settings.thesis_stop_loss_pct,
                "take_profit_pct": None,
                "trailing_stop_pct": self._settings.thesis_trailing_stop_pct,
                "trailing_stop_activation_pct": self._settings.thesis_trailing_stop_activation_pct,
            }
        return {
            "stop_loss_pct": self._settings.exit_stop_loss_pct,
            "take_profit_pct": None,  # no hard cap — trailing stop rides winners
            "trailing_stop_pct": self._settings.exit_trailing_stop_pct,
            "trailing_stop_activation_pct": self._settings.exit_trailing_stop_activation_pct,
        }

    def _check_orb_exits(self, equity: float) -> None:
        """ORB's stop/target are fixed price levels from the opening range
        at entry time, not percentages off entry — a different shape than
        exit_rules.evaluate_exit, hence its own check. Also force-closes
        anything still open from a PRIOR day regardless of price: ORB is a
        day-trade by design (verified in the backtest as same-session
        exits only) — carrying a position overnight is a different, untested
        risk profile this track was never validated for.
        """
        today = date.today()
        for position in self._state_store.get_positions():
            if self._breaker.is_stock_halted:
                return
            ticker = position["ticker"]
            if position["quantity"] <= 0 or position["strategy"] != "orb":
                continue

            detail = self._broker.get_position_detail(ticker)
            if detail is None or detail["qty"] <= 0:
                continue
            current_price = detail["current_price"]

            should_exit = False
            reason = ""
            if position["last_buy_at"] != today.isoformat():
                should_exit = True
                reason = f"ORB position held past its entry day ({position['last_buy_at']}) — force-closing, day-trade only"
            elif position["stop_price"] is not None and current_price <= position["stop_price"]:
                should_exit = True
                reason = f"stop hit: {current_price:.2f} <= {position['stop_price']:.2f}"
            elif position["target_price"] is not None and current_price >= position["target_price"]:
                should_exit = True
                reason = f"target hit: {current_price:.2f} >= {position['target_price']:.2f}"

            if not should_exit:
                continue
            logger.info("%s: ORB exit — %s", ticker, reason)

            proposal = TradeProposal(ticker=ticker, action=Action.SELL, quantity=int(detail["qty"]), limit_price=current_price)
            self._wash_sale_guard.warn_before_sell(ticker, proposal.limit_price, today)
            try:
                self._breaker.assert_not_tripped()
                self._breaker.validate_position_size(proposal, equity)
                result = self._broker.submit_order(proposal)
                logger.info("%s: ORB exit order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(ticker, proposal, today)
                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after ORB exit of {ticker}, equity={equity:.2f}")
                    return
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked ORB exit for %s: %s", ticker, exc)
                self._trip_breaker(reason=str(exc))
                return

    def _check_options_exits(self, equity: float) -> None:
        """No fixed take-profit (let a winning premium run, same philosophy
        as the thesis track) — just two hard, mechanical rules: a stop-loss
        on premium, and a force-close floor near expiration that fires
        regardless of P&L. The latter is what actually keeps this track out
        of 0-1 DTE-style risk; the stop-loss alone wouldn't reliably catch
        the sharp theta/gamma acceleration in the contract's final days.
        """
        today = date.today()
        for position in self._state_store.get_option_positions():
            if self._breaker.is_options_halted:
                return
            contract_symbol = position["contract_symbol"]
            # vol_short positions use their own tastylive exit rules — see _check_vol_options_exits
            if position.get("strategy") == "vol_short":
                continue
            if position["quantity"] <= 0:
                continue

            detail = self._broker.get_position_detail(contract_symbol)
            if detail is None or detail["qty"] <= 0:
                continue

            expiration = date.fromisoformat(position["expiration"])
            dte_remaining = (expiration - today).days
            avg_entry_price = detail["avg_entry_price"]
            current_price = detail["current_price"]
            premium_drawdown_pct = (avg_entry_price - current_price) / avg_entry_price if avg_entry_price > 0 else 0.0

            should_exit = False
            reason = ""
            if dte_remaining <= self._settings.options_force_close_days_before_expiration:
                should_exit = True
                reason = f"{dte_remaining}d to expiration <= {self._settings.options_force_close_days_before_expiration}d force-close floor"
            elif premium_drawdown_pct >= self._settings.options_stop_loss_pct:
                should_exit = True
                reason = f"premium down {premium_drawdown_pct:.1%} >= {self._settings.options_stop_loss_pct:.1%} stop-loss"

            if not should_exit:
                continue
            logger.info("%s: options exit — %s", contract_symbol, reason)

            try:
                self._breaker.assert_options_trading_allowed()
                contracts = int(detail["qty"])
                result = self._broker.submit_option_order(
                    contract_symbol, side=Action.SELL, contracts=contracts, limit_price=current_price
                )
                logger.info("%s: options exit order result=%s", contract_symbol, result)
                self._record_option_order_event(contract_symbol, Action.SELL, contracts, current_price, result)
                self._record_option_fill(
                    contract_symbol, position["underlying_symbol"], position["option_type"],
                    position["strike"], position["expiration"], Action.SELL, contracts, today,
                    sale_price=current_price,
                )

                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after options exit of {contract_symbol}, equity={equity:.2f}")
                    return
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked options exit for %s: %s", contract_symbol, exc)
                self._trip_breaker(reason=str(exc))
                return

    # ---- Vol options track (short premium — Natenberg/tastylive) ----

    def vol_options_scan_and_trade(self) -> None:
        """Runs once daily. For each watchlist ticker, fetches a volatility
        snapshot and runs the 3-agent + risk-officer vol consensus. If the
        consensus is executable, submits the short-premium structure as
        individual leg orders via the broker.

        Universe: the static watchlist — premium selling requires liquid,
        high-options-volume names (SPY, AAPL, etc.), not the momentum movers
        universe used by the ORB tracks.
        """
        vol_armed = self._daily_regime.arm_vol if self._daily_regime else False
        if not self._settings.vol_options_track_enabled or not vol_armed or self._breaker.is_options_halted:
            if self._settings.vol_options_track_enabled and not vol_armed:
                logger.info("Vol options scan skipped — regime disarmed (VIX conditions unfavorable)")
            return
        today = date.today()
        # Use the dynamically screened vol universe (refreshed pre-market).
        # Falls back to the static watchlist if the refresh hasn't run yet.
        # Both the 10 AM and 1 PM scans use the same universe; the double-entry
        # guard (existing_vol_tickers) handles per-ticker dedup within a day.
        candidates = list(self._vol_universe)

        logger.info("=== VOL OPTIONS SCAN: %d candidate(s) ===", len(candidates))
        equity = self._broker.get_equity()
        vix_context = self._fetch_vix_context(today)
        portfolio = self._build_portfolio_greeks(equity)

        # Build a set of tickers that already have an open or pending vol_short leg
        # so we never double-enter the same underlying in the same cycle.
        # qty != 0 filter is intentional: a qty=0 entry means the mleg was submitted
        # but not yet filled (or already expired). The open-orders check below handles
        # the pending case; an expired unfilled mleg (qty=0, no open order) should NOT
        # permanently block re-entry.
        existing_vol_tickers = {
            p["underlying_symbol"]
            for p in self._state_store.get_option_positions()
            if p.get("strategy") == "vol_short" and p["quantity"] != 0
        }
        # Also check Alpaca open orders: a pending mleg has qty=0 in state until it fills,
        # so we must treat any mleg leg's underlying as already-entered.
        for order in self._broker.get_open_orders():
            if order.get("legs"):
                for leg in order["legs"]:
                    parsed = parse_occ_symbol(leg["symbol"]) if leg.get("symbol") else None
                    if parsed:
                        existing_vol_tickers.add(parsed.underlying_symbol)

        for ticker in candidates:
            if self._breaker.is_options_halted:
                return
            if ticker in existing_vol_tickers:
                logger.info("%s: vol scan skipped — existing vol_short position or pending order", ticker)
                continue
            try:
                vol_snapshot = self._data_client.get_volatility_snapshot(ticker)
                chain = self._data_client.get_option_chain(ticker)
            except DataLayerError as exc:
                logger.error("%s: vol scan data fetch failed — skipping: %s", ticker, exc)
                continue

            # Hard gate: never sell premium through an earnings event.
            # The LLM agents see earnings_within_dte but we enforce it here too
            # so a miscalibrated agent can't override a fundamental risk constraint.
            if vol_snapshot.earnings_within_dte:
                logger.info(
                    "%s: vol scan skipped — earnings within expiration window (next: %s)",
                    ticker, vol_snapshot.next_earnings_date,
                )
                self._state_store.record_event(
                    event_type="vol_scan_skipped_earnings",
                    detail=f"{ticker}: earnings {vol_snapshot.next_earnings_date} within DTE window — no new positions",
                )
                continue

            # GARCH(1,1) realized vol forecast — best-effort, enriches the IVSurfaceAgent's
            # prompt with a forward-looking VRP signal rather than the backward-looking HV.
            # Failure is silent: HV-based VRP in the snapshot is the fallback.
            try:
                price_hist = self._data_client.get_price_history(
                    ticker, start_date=today - timedelta(days=90), end_date=today
                )
                garch_rv = vol_analytics.estimate_garch_rv(
                    price_hist, forecast_horizon=self._settings.vol_options_target_dte
                )
                if garch_rv is not None:
                    vol_snapshot = vol_snapshot.model_copy(update={"garch_rv_forecast": garch_rv})
                    vrp_garch = vol_snapshot.iv_30 - garch_rv
                    logger.info(
                        "%s: running vol consensus (IVR=%.1f, IV30=%.1f%%, HV30=%.1f%%, GARCH_RV=%.1f%%, VRP_GARCH=%+.1f%%)",
                        ticker, vol_snapshot.iv_rank, vol_snapshot.iv_30 * 100, vol_snapshot.hv_30 * 100,
                        garch_rv * 100, vrp_garch * 100,
                    )
                else:
                    logger.info(
                        "%s: running vol consensus (IVR=%.1f, IV30=%.1f%%, HV30=%.1f%%)",
                        ticker, vol_snapshot.iv_rank, vol_snapshot.iv_30 * 100, vol_snapshot.hv_30 * 100,
                    )
            except DataLayerError as exc:
                logger.debug("%s: GARCH price fetch failed — using HV-based VRP only: %s", ticker, exc)
                logger.info(
                    "%s: running vol consensus (IVR=%.1f, IV30=%.1f%%, HV30=%.1f%%)",
                    ticker, vol_snapshot.iv_rank, vol_snapshot.iv_30 * 100, vol_snapshot.hv_30 * 100,
                )

            payload = run_vol_consensus(
                client=self._anthropic,
                model=self._settings.anthropic_model,
                ticker=ticker,
                vol_snapshot=vol_snapshot,
                option_chain=chain,
                vix_context=vix_context,
                portfolio=portfolio,
                max_position_size_pct=self._settings.max_position_size_pct,
                allow_uncovered=self._settings.is_uncovered_allowed,
                subagent_model=self._settings.anthropic_subagent_model,
                usage_callback=self._record_usage,
            )

            structure_name = payload.proposal.structure.value if payload.proposal else "none"
            verdict = payload.risk_review.verdict.value if payload.risk_review else "none"
            self._state_store.record_event(
                event_type="vol_consensus_result",
                detail=f"{ticker}: executable={payload.is_executable} structure={structure_name} verdict={verdict}",
            )
            logger.info(
                "%s: vol consensus → %s / %s (executable=%s)",
                ticker, structure_name, verdict, payload.is_executable,
            )

            if not payload.is_executable:
                continue

            self._open_vol_options_position(ticker, payload, chain, equity, today)
            # Refresh portfolio Greeks so the next candidate sees the updated book
            portfolio = self._build_portfolio_greeks(equity)

    def _fetch_vix_context(self, today: date) -> VixContext:
        """Best-effort VIX fetch from yfinance. On any failure, falls back to
        VixContext(vix_current=18.0) — this puts the regime agent in STABLE
        rather than EXPANSION, allowing the other vol agents to proceed with
        their own assessments. A fallback that silently allows all trades is
        the wrong failure mode here; STABLE is conservative enough that the
        IV surface and event risk agents still have full veto authority.
        """
        try:
            vix_series = self._data_client.get_price_history(
                "^VIX", start_date=today - timedelta(days=45), end_date=today
            )
            bars = vix_series.bars
            if not bars:
                return VixContext(vix_current=18.0)

            vix_current = bars[-1].close
            vix_1w_ago = bars[-5].close if len(bars) >= 5 else None
            vix_1m_ago = bars[-21].close if len(bars) >= 21 else None

            vix3m_current: float | None = None
            try:
                vix3m_series = self._data_client.get_price_history(
                    "^VIX3M", start_date=today - timedelta(days=5), end_date=today
                )
                if vix3m_series.bars:
                    vix3m_current = vix3m_series.bars[-1].close
            except DataLayerError:
                pass  # VIX3M optional — regime agent still works without it

            return VixContext(
                vix_current=vix_current,
                vix_1w_ago=vix_1w_ago,
                vix_1m_ago=vix_1m_ago,
                vix3m_current=vix3m_current,
            )
        except DataLayerError as exc:
            logger.warning("VIX fetch failed — defaulting to stable VIX context: %s", exc)
            return VixContext(vix_current=18.0)

    def _build_portfolio_greeks(self, equity: float) -> PortfolioGreeks:
        """Estimates portfolio-level Greeks from the current open vol short
        positions. Uses rough tastylive-calibrated estimates rather than
        pulling per-contract Greeks from the chain — good enough for the
        Greeks Risk Officer's portfolio-level limit checks.
        """
        option_positions = self._state_store.get_option_positions()
        vol_short_positions = [
            p for p in option_positions
            if p.get("strategy") == "vol_short" and p["quantity"] != 0
        ]
        n = len(vol_short_positions)
        # Rough per-leg estimates for 16-delta short premium positions:
        # Each short leg contributes approximately -$10 net vega (sensitivity
        # to a 1-vol-point move) and +$1 theta (daily decay income). These are
        # intentionally conservative — if the LLM sees more vega than reality,
        # it flags the limit sooner, which is the right failure mode.
        return PortfolioGreeks(
            net_delta=0.0,
            net_vega=-10.0 * n,
            net_theta=1.0 * n,
            portfolio_value=equity,
            num_open_positions=n,
        )

    def _find_option_contract(
        self,
        chain: list[OptionContract],
        option_type: OptionType,
        expiration: date,
        strike: float,
    ) -> OptionContract | None:
        for c in chain:
            if c.option_type == option_type and c.expiration == expiration and abs(c.strike - strike) < 0.01:
                return c
        return None

    def _open_vol_options_position(
        self,
        ticker: str,
        payload: VolConsensusPayload,
        chain: list[OptionContract],
        equity: float,
        today: date,
    ) -> None:
        """Submits the vol options structure to the broker.

        Iron condors are submitted as a single atomic mleg order so Alpaca
        evaluates all legs together — never sees the short legs as uncovered.
        Other structures (strangle, single-leg) use individual limit orders.
        """
        proposal = payload.proposal

        if proposal.structure == StructureType.IRON_CONDOR:
            self._open_iron_condor(ticker, proposal, chain, today)
            return

        # All other structures: individual limit orders per leg
        legs: list[tuple[OptionContract, Action]] = []

        if proposal.structure == StructureType.SHORT_STRANGLE:
            call = self._find_option_contract(chain, OptionType.CALL, proposal.expiration, proposal.short_call_strike)
            put = self._find_option_contract(chain, OptionType.PUT, proposal.expiration, proposal.short_put_strike)
            if call:
                legs.append((call, Action.SELL))
            if put:
                legs.append((put, Action.SELL))

        elif proposal.structure in (StructureType.SHORT_PUT, StructureType.SHORT_PUT_SPREAD):
            put = self._find_option_contract(chain, OptionType.PUT, proposal.expiration, proposal.single_strike)
            if put:
                legs.append((put, Action.SELL))

        elif proposal.structure in (StructureType.SHORT_CALL, StructureType.SHORT_CALL_SPREAD):
            call = self._find_option_contract(chain, OptionType.CALL, proposal.expiration, proposal.single_strike)
            if call:
                legs.append((call, Action.SELL))

        else:
            logger.info("%s: structure %s not yet supported in vol execution", ticker, proposal.structure.value)
            return

        if not legs:
            logger.error(
                "%s: no matching contracts found in chain for %s (exp=%s)",
                ticker, proposal.structure.value, proposal.expiration,
            )
            return

        for contract, side in legs:
            limit_price = contract.bid if side == Action.SELL else contract.ask
            if limit_price <= 0:
                logger.warning(
                    "%s: %s contract %s has zero bid/ask — skipping leg",
                    ticker, side.value, contract.contract_symbol,
                )
                continue
            try:
                self._breaker.assert_options_trading_allowed()
                result = self._broker.submit_option_order(
                    contract.contract_symbol, side=side,
                    contracts=proposal.quantity, limit_price=limit_price,
                )
                logger.info(
                    "%s: vol %s %s @ %.2f → %s",
                    ticker, side.value, contract.contract_symbol,
                    limit_price, result.get("order_status", "unknown"),
                )
                self._record_option_order_event(
                    contract.contract_symbol, side, proposal.quantity, limit_price, result
                )
                self._record_option_fill(
                    contract.contract_symbol, ticker, contract.option_type.value,
                    contract.strike, contract.expiration.isoformat(),
                    side, proposal.quantity, today, strategy="vol_short",
                )
                self._state_store.record_event(
                    event_type="vol_options_opened",
                    detail=(
                        f"{ticker}: {side.value} {contract.option_type.value} {contract.strike:.2f} "
                        f"exp={contract.expiration} DTE={contract.dte} "
                        f"structure={proposal.structure.value}"
                    ),
                )
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked vol options order for %s (%s): %s", ticker, contract.contract_symbol, exc)
                self._trip_breaker(reason=str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                # Broker-level rejection (e.g. account not approved for naked shorts,
                # market closed, or margin insufficient). Log and abort the whole
                # structure — a half-entered strangle is worse than no position.
                logger.error(
                    "%s: broker rejected %s order for %s — aborting structure: %s",
                    ticker, side.value, contract.contract_symbol, exc,
                )
                self._state_store.record_event(
                    event_type="vol_options_broker_rejected",
                    detail=f"{ticker}: {side.value} {contract.contract_symbol} — {exc}",
                )
                return

    def _open_iron_condor(
        self,
        ticker: str,
        proposal,
        chain: list[OptionContract],
        today: date,
    ) -> None:
        """Submit an iron condor as a single atomic mleg order.

        Alpaca's mleg order evaluates all 4 legs together, so the short call
        and short put are never momentarily uncovered — the root cause of the
        Level 3 rejections when submitting legs sequentially.
        """
        short_call = self._find_option_contract(chain, OptionType.CALL, proposal.expiration, proposal.short_call_strike)
        short_put = self._find_option_contract(chain, OptionType.PUT, proposal.expiration, proposal.short_put_strike)
        long_call = self._find_option_contract(chain, OptionType.CALL, proposal.expiration, proposal.long_call_strike)
        long_put = self._find_option_contract(chain, OptionType.PUT, proposal.expiration, proposal.long_put_strike)

        missing = [name for name, c in (
            ("short_call", short_call), ("short_put", short_put),
            ("long_call", long_call), ("long_put", long_put),
        ) if c is None]
        if missing:
            logger.error("%s: iron condor missing legs in chain: %s", ticker, missing)
            self._state_store.record_event(
                event_type="vol_options_broker_rejected",
                detail=f"{ticker}: iron condor missing chain legs: {missing}",
            )
            return

        def _mid(bid: float, ask: float) -> float:
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return bid or ask or 0.0

        mid_credit = round(
            _mid(short_call.bid, short_call.ask) + _mid(short_put.bid, short_put.ask)
            - _mid(long_call.bid, long_call.ask) - _mid(long_put.bid, long_put.ask),
            2,
        )
        natural_credit = round(
            (short_call.bid or 0.0) + (short_put.bid or 0.0)
            - (long_call.ask or 0.0) - (long_put.ask or 0.0),
            2,
        )
        # Prefer mid-price credit — asks for better fills than natural (bid/ask).
        # Liquid names (AAPL, SPY, QQQ) fill at mid frequently. If mid <= 0
        # (wide bid/ask makes the spread a debit at mid), fall back to natural.
        # The mleg is DAY TIF so an unfilled mid-price order expires at close.
        net_credit = mid_credit if mid_credit > 0 else natural_credit
        if net_credit <= 0:
            logger.info(
                "%s: iron condor net credit %.2f <= 0 (mid=%.2f, natural=%.2f) — skipping",
                ticker, net_credit, mid_credit, natural_credit,
            )
            return
        logger.info(
            "%s: iron condor credit: mid=%.2f natural=%.2f submitting at %.2f",
            ticker, mid_credit, natural_credit, net_credit,
        )

        spread_legs = [
            (short_call.contract_symbol, Action.SELL),
            (short_put.contract_symbol, Action.SELL),
            (long_call.contract_symbol, Action.BUY),
            (long_put.contract_symbol, Action.BUY),
        ]

        try:
            self._breaker.assert_options_trading_allowed()
            result = self._broker.submit_spread_order(
                legs=spread_legs,
                contracts=proposal.quantity,
                net_credit=net_credit,
            )
            logger.info(
                "%s: iron condor mleg → %s (net_credit=%.2f)",
                ticker, result.get("order_status", "unknown"), net_credit,
            )
            self._state_store.record_event(
                event_type="vol_options_opened",
                detail=(
                    f"{ticker}: iron_condor mleg "
                    f"{short_call.strike:.0f}C/{short_put.strike:.0f}P short "
                    f"{long_call.strike:.0f}C/{long_put.strike:.0f}P long "
                    f"exp={proposal.expiration} DTE={proposal.dte} "
                    f"net_credit={net_credit:.2f}"
                ),
            )
            # Record each leg individually in the state store so exits can
            # manage them leg-by-leg with get_position_detail()
            for contract, side in (
                (short_call, Action.SELL), (short_put, Action.SELL),
                (long_call, Action.BUY), (long_put, Action.BUY),
            ):
                self._record_option_fill(
                    contract.contract_symbol, ticker, contract.option_type.value,
                    contract.strike, contract.expiration.isoformat(),
                    side, proposal.quantity, today, strategy="vol_short",
                )
        except CircuitBreakerTripped as exc:
            logger.error("Circuit breaker blocked iron condor for %s: %s", ticker, exc)
            self._trip_breaker(reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: iron condor mleg rejected: %s", ticker, exc)
            self._state_store.record_event(
                event_type="vol_options_broker_rejected",
                detail=f"{ticker}: iron condor mleg — {exc}",
            )

    def _check_vol_options_exits(self, equity: float) -> None:
        """tastylive's three management rules for short premium positions:
        - Close at 50% of credit received (capture half the theta without gamma risk)
        - Close when cost-to-close reaches 2x credit (loss limit, not stop-loss-style)
        - Close when DTE hits roll level (default 21d) to avoid gamma acceleration

        For short options, qty is negative (short position). We BUY to close.
        P&L = (credit_received - cost_to_close) × contracts × 100.
        """
        today = date.today()
        for position in self._state_store.get_option_positions():
            if position.get("strategy") != "vol_short":
                continue
            if self._breaker.is_options_halted:
                return
            contract_symbol = position["contract_symbol"]
            if position["quantity"] == 0:
                continue

            detail = self._broker.get_position_detail(contract_symbol)
            if detail is None or detail["qty"] == 0:
                continue

            expiration = date.fromisoformat(position["expiration"])
            dte_remaining = (expiration - today).days
            credit_received = position["avg_entry_price"]
            cost_to_close = detail["current_price"]

            should_exit = False
            reason = ""

            if dte_remaining <= self._settings.vol_options_roll_dte:
                should_exit = True
                reason = f"{dte_remaining}d DTE at/below roll level {self._settings.vol_options_roll_dte}d — close or roll"
            elif credit_received > 0:
                pnl_pct = (credit_received - cost_to_close) / credit_received
                if pnl_pct >= self._settings.vol_options_profit_target_pct:
                    should_exit = True
                    reason = f"profit target: captured {pnl_pct:.1%} of credit (${credit_received:.2f} → ${cost_to_close:.2f})"
                elif cost_to_close >= credit_received * self._settings.vol_options_loss_limit_multiplier:
                    should_exit = True
                    reason = (
                        f"loss limit: cost to close ${cost_to_close:.2f} >= "
                        f"{self._settings.vol_options_loss_limit_multiplier:.0f}x credit ${credit_received:.2f}"
                    )

            if not should_exit:
                continue

            logger.info("%s: vol options exit — %s", contract_symbol, reason)
            qty = abs(int(detail["qty"]))  # positive count for the BUY-to-close order

            try:
                self._breaker.assert_options_trading_allowed()
                result = self._broker.submit_option_order(
                    contract_symbol, side=Action.BUY, contracts=qty, limit_price=cost_to_close
                )
                logger.info("%s: vol options close order result=%s", contract_symbol, result)
                self._record_option_order_event(contract_symbol, Action.BUY, qty, cost_to_close, result)

                # For short options, P&L = credit_received - cost_to_close (per share)
                # Passing them swapped relative to the long-options convention achieves
                # the right sign: (sale_price - cost_basis) = (credit_received - close_cost)
                prior = self._state_store.get_option_position(contract_symbol)
                if prior is not None:
                    self._state_store.record_realized_option_sale(
                        contract_symbol=contract_symbol,
                        underlying_symbol=position["underlying_symbol"],
                        sale_date=today,
                        contracts=qty,
                        sale_price=prior["avg_entry_price"],  # credit originally received
                        cost_basis=cost_to_close,             # cost to close the short
                    )

                # After buy-to-close, update local state to qty=0
                closed_detail = self._broker.get_position_detail(contract_symbol)
                closed_qty = int(closed_detail["qty"]) if closed_detail else 0
                self._state_store.upsert_option_position(
                    contract_symbol,
                    position["underlying_symbol"],
                    position["option_type"],
                    position["strike"],
                    position["expiration"],
                    closed_qty,
                    position["avg_entry_price"],
                    strategy=position.get("strategy", "vol_short"),
                )
                self._state_store.record_event(
                    event_type="vol_options_closed",
                    detail=f"{contract_symbol}: {reason}",
                )

                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(
                        reason=f"daily profit target reached after vol options exit of {contract_symbol}, equity={equity:.2f}"
                    )
                    return
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked vol options exit for %s: %s", contract_symbol, exc)
                self._trip_breaker(reason=str(exc))
                return

    def _should_escalate_to_llm(self, ticker: str, position: dict, today: date) -> bool:
        if not self._settings.intraday_llm_escalation_enabled:
            return False
        entry_regime = position.get("entry_regime")
        if not entry_regime or entry_regime == "neutral":
            return False
        if self._state_store.has_intraday_escalation_today(ticker, today):
            return False

        current_regime = self._current_regime(ticker, today)
        sharp_reversal = current_regime not in ("neutral", entry_regime)
        if sharp_reversal:
            logger.info("%s: regime reversed %s -> %s — escalating to LLM exit review", ticker, entry_regime, current_regime)
        return sharp_reversal

    def _current_regime(self, ticker: str, today: date) -> str:
        try:
            price_series = self._data_client.get_price_history(
                ticker, start_date=today - timedelta(days=60), end_date=today
            )
        except DataLayerError:
            return "neutral"
        closes = [bar.close for bar in price_series.bars]
        return prefilter.compute_regime(closes, self._settings.filter_sma_short_window, self._settings.filter_sma_long_window)

    def _record_order_event(self, ticker: str, proposal: TradeProposal, result: dict) -> None:
        """Human-readable trail of every order actually submitted — the
        run_history table has the agents' reasoning, but a dashboard
        showing "what did the system actually buy/sell, and when" needs
        this dedicated, easy-to-query event instead of reconstructing it
        from JSON payloads.
        """
        self._state_store.record_event(
            event_type=f"order_{proposal.action.value.lower()}",
            detail=(
                f"{ticker}: {proposal.action.value} x{proposal.quantity} @ {proposal.limit_price:.2f} "
                f"-> {result.get('order_status', 'unknown')} "
                f"(filled {result.get('filled_qty', 0)}@{result.get('filled_avg_price')})"
            ),
        )

    def _record_fill(
        self,
        ticker: str,
        proposal: TradeProposal,
        today: date,
        entry_regime: str | None = None,
        strategy: str | None = None,
    ) -> None:
        """Shared bookkeeping after ANY order actually reaches the broker —
        used by both the morning batch and intraday exit checks so position
        state, realized P&L, and last_buy_at stay consistent either way.
        `strategy` tags which exit-rule parameters apply to this position
        (see _check_intraday_exits) — only set on a BUY, same as entry_regime.

        Reads `get_position_detail` (not `get_position_shares` +
        `proposal.limit_price`) because the broker already tracks the
        correct blended quantity/cost-basis across every fill on this
        ticker — reconstructing it from just this one fill is exactly
        wrong the moment this is the second-or-later buy on an existing
        position: it understated quantity (this fill's count, not the
        running total) and overwrote cost basis with only this fill's
        price, discarding whatever was paid for shares already held.
        """
        detail = self._broker.get_position_detail(ticker)
        shares = detail["qty"] if detail else 0.0
        avg_price = detail["avg_entry_price"] if detail else proposal.limit_price
        if proposal.action == Action.BUY:
            prior_position = self._state_store.get_position(ticker)
            # A new high print on THIS fill can raise the peak; a fill below
            # the existing peak (averaging into a dip) must never lower it —
            # high_water_mark is a running peak for the trailing-stop, not
            # "the price of the most recent buy."
            prior_hwm = (prior_position or {}).get("high_water_mark") or avg_price
            self._state_store.upsert_position(
                ticker,
                int(shares),
                avg_price,
                last_buy_at=today.isoformat(),
                entry_regime=entry_regime,
                high_water_mark=max(prior_hwm, proposal.limit_price),
                strategy=strategy,
            )
        else:
            prior_position = self._state_store.get_position(ticker)
            if prior_position is not None:
                pnl = self._state_store.record_realized_sale(
                    ticker=ticker,
                    sale_date=today,
                    quantity=proposal.quantity,
                    sale_price=proposal.limit_price,
                    cost_basis=prior_position["avg_entry_price"],
                )
                self._trigger_reflection(ticker, strategy, pnl)
            self._state_store.upsert_position(ticker, int(shares), avg_price)

    def _trigger_reflection(self, ticker: str, strategy: str, pnl: float) -> None:
        """Fire-and-forget: extract lessons from a closed trade in a daemon thread.

        Never blocks execution. Any failure is logged at DEBUG and silently dropped —
        the reflection system enriches future decisions but must never affect the current one.
        """
        threading.Thread(
            target=self._run_reflection,
            args=(ticker, strategy, pnl),
            daemon=True,
            name=f"reflect-{ticker}",
        ).start()

    def _run_reflection(self, ticker: str, strategy: str, pnl: float) -> None:
        try:
            # Find the most recent executable BUY run for this ticker
            runs = self._state_store.get_run_history(ticker=ticker, limit=10)
            agent_signals: list[dict] = []
            for run in runs:
                payload_dict = run["payload"]
                if payload_dict.get("proposal", {}).get("action") == "BUY":
                    agent_signals = [
                        {
                            "agent_name": s.get("agent_name", "unknown"),
                            "stance": s.get("stance", "HOLD"),
                            "confidence": s.get("confidence", "low"),
                            "rationale": s.get("rationale", ""),
                        }
                        for s in payload_dict.get("signals", [])
                    ]
                    break

            if not agent_signals:
                logger.debug("_run_reflection: no BUY run found for %s — skipping", ticker)
                return

            reflection = self._reflection_agent.reflect(
                strategy=strategy,
                agent_signals=agent_signals,
                outcome_pnl=pnl,
                outcome_win=pnl > 0,
                market_context={"strategy": strategy, "realized_pnl": f"${pnl:+.2f}"},
            )
            if reflection is None:
                return

            self._state_store.record_reflection(
                strategy=strategy,
                outcome_pnl=pnl,
                outcome_win=pnl > 0,
                what_happened=reflection.what_happened,
                root_cause=reflection.root_cause,
                outcome_was_noise=reflection.outcome_was_noise,
            )

            # Score agent signals and lesson injections based on this trade's outcome
            self._state_store.score_agent_signals(ticker, pnl)
            self._state_store.score_lesson_injections(ticker, pnl)

            if not reflection.outcome_was_noise:
                for lesson_out in reflection.lessons:
                    self._state_store.record_lesson(
                        lesson=lesson_out.lesson,
                        setup_tags=lesson_out.setup_tags,
                        strategy=strategy,
                        outcome_was_win=pnl > 0,
                        source_pnl=pnl,
                    )
                logger.info(
                    "Reflection complete for %s (P&L $%+.2f): %d lesson(s) stored",
                    ticker, pnl, len(reflection.lessons),
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Post-trade reflection failed for %s: %s", ticker, exc)

    def _record_option_order_event(
        self, contract_symbol: str, side: Action, contracts: int, limit_price: float, result: dict
    ) -> None:
        self._state_store.record_event(
            event_type=f"option_order_{side.value.lower()}",
            detail=(
                f"{contract_symbol}: {side.value} x{contracts} @ {limit_price:.2f} "
                f"-> {result.get('order_status', 'unknown')} "
                f"(filled {result.get('filled_qty', 0)}@{result.get('filled_avg_price')})"
            ),
        )

    def _record_option_fill(
        self,
        contract_symbol: str,
        underlying_symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        action: Action,
        contracts: int,
        today: date,
        sale_price: float | None = None,
        strategy: str = "orb_options",
    ) -> None:
        """Mirrors _record_fill's broker-is-authoritative approach — reads
        the blended quantity/cost-basis back from the broker rather than
        reconstructing it from just this fill, for the same reason: a
        second buy on an already-open contract must accumulate, not
        overwrite.
        """
        detail = self._broker.get_position_detail(contract_symbol)
        qty = detail["qty"] if detail else 0.0
        avg_price = detail["avg_entry_price"] if detail else (sale_price or 0.0)
        if action == Action.BUY:
            self._state_store.upsert_option_position(
                contract_symbol, underlying_symbol, option_type, strike, expiration,
                int(qty), avg_price, opened_at=today.isoformat(), strategy=strategy,
            )
        else:
            prior = self._state_store.get_option_position(contract_symbol)
            if prior is not None and sale_price is not None:
                self._state_store.record_realized_option_sale(
                    contract_symbol=contract_symbol,
                    underlying_symbol=underlying_symbol,
                    sale_date=today,
                    contracts=contracts,
                    sale_price=sale_price,
                    cost_basis=prior["avg_entry_price"],
                )
            self._state_store.upsert_option_position(
                contract_symbol, underlying_symbol, option_type, strike, expiration,
                int(qty), avg_price, strategy=strategy,
            )

    # ---- Phase 4: Post-market logging ----
    def post_market_logging(self) -> None:
        logger.info("=== POST-MARKET LOGGING ===")
        positions = self._state_store.get_positions()
        history = self._state_store.get_run_history(limit=len(self._watchlist) or 50)
        # token_usage.created_at is stored via datetime.utcnow() (state_store.py),
        # so the cutoff must use the same clock — date.today() is local time and
        # can disagree with UTC by a day near midnight, depending on timezone.
        cost_summary = self._state_store.get_cost_summary(since=datetime.utcnow().date())
        self._state_store.record_event(
            event_type="post_market_summary",
            detail=(
                f"positions={len(positions)} runs_logged={len(history)} "
                f"breaker_tripped={self._breaker.is_tripped} profit_locked={self._breaker.is_profit_locked} "
                f"claude_cost_usd={cost_summary['total_cost_usd']:.4f}"
            ),
        )
        logger.info("Post-market summary: %d open positions, %d runs logged today.", len(positions), len(history))
        logger.info(
            "Claude spend today: $%.4f across %d calls (%d input, %d output, %d cache-read tokens)",
            cost_summary["total_cost_usd"],
            cost_summary["total_calls"],
            cost_summary["total_input_tokens"],
            cost_summary["total_output_tokens"],
            cost_summary["total_cache_read_input_tokens"],
        )
        for row in cost_summary["by_agent"]:
            logger.info(
                "  %-30s %d calls  $%.4f",
                row["agent_name"],
                row["calls"],
                row["cost_usd"],
            )

    # ---- internals ----
    def _record_usage(self, agent_name: str, model: str, usage) -> None:
        cost = estimate_cost_usd(model, usage)
        self._state_store.record_token_usage(
            agent_name=agent_name,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            estimated_cost_usd=cost,
        )

    def _trip_breaker(self, reason: str) -> None:
        """A real risk breach (drawdown) — a true emergency stop. Closes
        and halts BOTH tracks; pauses the whole scheduler.
        """
        self._close_all_and_reconcile(reason)
        if self._halt_callback is not None:
            self._halt_callback()

    def _lock_in_profit(self, reason: str) -> None:
        """Hitting the daily $ target is a banked win, not a risk breach
        — by design (per explicit instruction), this stops STOCKS only.
        Options keep trading on their own already-bounded per-trade risk
        limits, so the scheduler is deliberately NOT paused here — only
        the stock-side methods self-gate via is_stock_halted.
        """
        self._close_stocks_and_reconcile(reason)

    def _close_stocks_and_reconcile(self, reason: str) -> None:
        today = date.today()
        equity_positions = [p for p in self._state_store.get_positions() if p["quantity"] > 0]

        logger.info("STOCK-ONLY PROFIT LOCK: %s", reason)
        self._state_store.record_event(event_type="daily_profit_target_reached", detail=reason)

        for order in self._broker.get_open_orders():
            sym = order["symbol"]
            # mleg orders have symbol=None (symbol lives on each leg); treat as options — don't cancel.
            if sym is not None and parse_occ_symbol(sym) is None:  # plain equity order only
                self._broker.cancel_order(order["order_id"])

        for position in equity_positions:
            self._broker.close_position(position["ticker"])
        for position in equity_positions:
            self._wait_until_flat(position["ticker"])

        for position in equity_positions:
            ticker = position["ticker"]
            fill_price = self._broker.get_last_fill_price(ticker) or position["avg_entry_price"]
            self._state_store.record_realized_sale(
                ticker=ticker, sale_date=today, quantity=position["quantity"],
                sale_price=fill_price, cost_basis=position["avg_entry_price"],
            )
            self._state_store.upsert_position(ticker, 0, position["avg_entry_price"])

    def _wait_until_flat(self, ticker: str, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._broker.get_position_detail(ticker) is None:
                return
            time.sleep(0.5)
        logger.warning("%s: close_position did not confirm flat within %.1fs", ticker, timeout_seconds)

    def _close_all_and_reconcile(self, reason: str, event_type: str = "circuit_breaker_shutdown") -> None:
        """execute_global_shutdown closes every position for real on the
        broker, but has no way to know the resulting fill prices or to
        update our own bookkeeping — without this, a real, successful
        closeout left every position looking "still open" in the local
        DB, and the realized P&L from a real overnight session (found
        live: +$592.53 across 4 positions) was never recorded anywhere.
        """
        today = date.today()
        equity_positions = [p for p in self._state_store.get_positions() if p["quantity"] > 0]
        option_positions = [p for p in self._state_store.get_option_positions() if p["quantity"] > 0]

        execute_global_shutdown(self._broker, self._state_store, reason, event_type=event_type)

        for position in equity_positions:
            ticker = position["ticker"]
            fill_price = self._broker.get_last_fill_price(ticker) or position["avg_entry_price"]
            self._state_store.record_realized_sale(
                ticker=ticker, sale_date=today, quantity=position["quantity"],
                sale_price=fill_price, cost_basis=position["avg_entry_price"],
            )
            self._state_store.upsert_position(ticker, 0, position["avg_entry_price"])

        for position in option_positions:
            contract_symbol = position["contract_symbol"]
            fill_price = self._broker.get_last_fill_price(contract_symbol) or position["avg_entry_price"]
            self._state_store.record_realized_option_sale(
                contract_symbol=contract_symbol, underlying_symbol=position["underlying_symbol"],
                sale_date=today, contracts=position["quantity"], sale_price=fill_price,
                cost_basis=position["avg_entry_price"],
            )
            self._state_store.upsert_option_position(
                contract_symbol, position["underlying_symbol"], position["option_type"],
                position["strike"], position["expiration"], 0, position["avg_entry_price"],
            )
