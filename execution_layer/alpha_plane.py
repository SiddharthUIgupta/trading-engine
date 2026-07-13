"""Alpha Plane — scanning, LLM consensus, sizing. Writes BUY intents to the DB.

Does NOT submit orders to the broker directly. Instead, approved BUY payloads are
written to the `order_intents` table, where the Protection Plane reads and executes them.

IPC: order_intents table
  Alpha writes: (client_order_id, strategy, ticker, action, quantity, limit_price, stop_price)
  Protection reads, submits bracket orders, marks as processed

Process isolation: this module runs as trading-engine-alpha systemd unit.
If it crashes (LLM timeout, scan failure), the Protection Plane keeps running
and all existing positions stay protected.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

from anthropic import Anthropic

from analyst_layer import (
    agent_scorer, lesson_store, momentum_scanner, options_structurer,
    orb_scanner, prefilter, swing_scanner, thesis_scanner, universe,
    vol_analytics, vol_universe,
)
from analyst_layer.vw_bandit import VWSignalBandit
from analyst_layer.correlation import apply_correlation_adjustment, check_portfolio_correlation
from analyst_layer.kelly import kelly_fraction_from_pnl_history
from analyst_layer.macro_news_agent import assess_macro_sentiment
from analyst_layer.market_regime import DailyRegime, assess_daily_regime
from analyst_layer.recovery_scanner import evaluate_recovery_candidate
from analyst_layer.reflection_agent import ReflectionAgent
from analyst_layer.agents.greeks_risk_officer import PortfolioGreeks
from analyst_layer.agents.risk_officer_agent import AccountContext
from analyst_layer.agents.vol_regime_agent import VixContext
from analyst_layer.graph import run_consensus
from analyst_layer.pricing import estimate_cost_usd
from analyst_layer.schemas import (
    Action, AgentSignal, Confidence, ConsensusPayload, OrderType,
    RiskReview, RiskVerdict, StructureType, TradeProposal, VolConsensusPayload,
)
from analyst_layer.vol_graph import run_vol_consensus
from config.settings import Settings
from data_layer.akshare_client import MacroSnapshot, get_macro_snapshot
from data_layer.exceptions import DataLayerError
from data_layer.models import OptionContract, OptionType
from data_layer.occ_symbol import parse_occ_symbol
from data_layer.openbb_client import OpenBBDataClient
from execution_layer import alerting, exit_rules
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import CircuitBreaker, CircuitBreakerTripped, GlobalRiskState
from execution_layer.state_store import StateStore
from execution_layer.tax_compliance import WashSaleGuard

logger = logging.getLogger(__name__)

_OBSIDIAN_RETRIEVE = (
    __import__("pathlib").Path.home() / "Projects" / "claude-obsidian" / "scripts" / "retrieve.py"
)


def _fetch_trade_memory(ticker: str, strategy: str, regime: str) -> str:
    """Return a formatted snippet block from claude-obsidian for similar past setups.

    Returns empty string when the vault is not provisioned or retrieval fails.
    Calls are subprocess-isolated with a hard 5-second timeout so a hung index
    never stalls the consensus loop.
    """
    import json as _json
    import subprocess as _subprocess

    if not _OBSIDIAN_RETRIEVE.exists():
        return ""
    try:
        query = f"{ticker} {strategy} {regime} trade setup"
        result = _subprocess.run(
            ["python3", str(_OBSIDIAN_RETRIEVE), query, "--top", "3"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_OBSIDIAN_RETRIEVE.parent.parent),
        )
        if result.returncode != 0:
            return ""
        data = _json.loads(result.stdout)
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        snippets = "\n---\n".join(c.get("snippet", "") for c in candidates if c.get("snippet"))
        if not snippets:
            return ""
        return f"\n\n## Prior trade memory (similar {ticker} setups)\n{snippets}\n"
    except Exception as exc:
        logger.debug("trade memory retrieval failed for %s: %s", ticker, exc)
        return ""

_INTRADAY_STRATEGIES: frozenset[str] = frozenset({"momentum", "orb_equity", "news"})
_OPTIONS_STRATEGIES: frozenset[str] = frozenset({"orb_options", "vol_options"})
_THESIS_STRATEGIES: frozenset[str] = frozenset({"thesis", "recovery", "gap"})
_SWING_STRATEGIES: frozenset[str] = frozenset({"swing"})


class AlphaRuntime:
    """Runs in a dedicated process. Scans universe, runs LLM consensus, sizes
    positions, and writes approved BUY intents to the order_intents DB table.
    Never calls broker.submit_order() for equity entries — that's Protection's job.

    Options entries (ORB options, vol short) still submit directly here because
    they require specific contract structures computed during the scan.
    """

    def __init__(
        self,
        settings: Settings,
        data_client: OpenBBDataClient,
        broker: AlpacaBroker,
        intraday_breaker: CircuitBreaker,
        options_breaker: CircuitBreaker,
        thesis_breaker: CircuitBreaker,
        swing_breaker: CircuitBreaker,
        state_store: StateStore,
        anthropic_client: Anthropic,
        watchlist: list[str],
        halt_callback: Callable[[], None] | None = None,
        wash_sale_guard: WashSaleGuard | None = None,
    ) -> None:
        self._settings = settings
        self._data_client = data_client
        self._broker = broker
        self._intraday_breaker = intraday_breaker
        self._options_breaker = options_breaker
        self._thesis_breaker = thesis_breaker
        self._swing_breaker = swing_breaker
        self._breaker = intraday_breaker
        self._state_store = state_store
        self._anthropic = anthropic_client
        self._watchlist = watchlist
        self._halt_callback = halt_callback
        self._wash_sale_guard = wash_sale_guard or WashSaleGuard(state_store)
        self._pending_payloads: dict[str, ConsensusPayload] = {}
        self._pending_regimes: dict[str, str] = {}
        self._pending_strategies: dict[str, str] = {}

        existing_tickers = {p["ticker"] for p in state_store.get_positions() if p["quantity"] > 0}
        self._scanned_tickers_today: set[str] = set(existing_tickers)
        self._executed_tickers_today: set[str] = set(existing_tickers)

        today_str = date.today().isoformat()
        for run in state_store.get_run_history(limit=100):
            if run.get("created_at", "")[:10] != today_str:
                continue
            if not run.get("is_executable"):
                continue
            try:
                payload = ConsensusPayload.model_validate(run["payload"])
            except Exception:  # noqa: BLE001
                continue
            ticker = payload.proposal.ticker
            if ticker not in self._executed_tickers_today:
                self._pending_payloads[ticker] = payload
                self._pending_regimes[ticker] = "unknown"
                self._pending_strategies[ticker] = "thesis"

        existing_option_underlyings = {p["underlying_symbol"] for p in state_store.get_option_positions() if p.get("quantity", 0) > 0}
        self._scanned_options_tickers_today: set[str] = set(existing_option_underlyings)
        self._scanned_vol_tickers_today: set[str] = set()
        existing_swing_tickers = {
            p["ticker"] for p in state_store.get_positions()
            if p.get("strategy") == "swing" and p["quantity"] > 0
        }
        self._scanned_swing_tickers_today: set[str] = set(existing_swing_tickers)

        self._vol_universe: list[str] = list(watchlist)
        self._daily_regime: DailyRegime | None = None
        self._macro_snapshot: MacroSnapshot = MacroSnapshot()
        self._price_cache: dict[str, list[float]] = {}
        self._thesis_session_buys: int = 0

        vw_model_path = settings.state_db_path.parent / "vw_bandit.model"
        self._vw_bandit = VWSignalBandit(model_path=vw_model_path)
        if not vw_model_path.exists():
            historical_logs = state_store.get_scored_signal_logs(limit=2000)
            if historical_logs:
                self._vw_bandit.warm_start(historical_logs)

    # ── IPC: write approved BUY intents to DB for Protection to execute ───────

    def _queue_pending_as_intents(self, today: date) -> None:
        """Replace direct broker submission — write each approved payload to the
        order_intents table. Protection Plane reads and executes them on its next tick.

        Breaker checks: reads BOTH in-memory state (circuit breaker objects) AND the
        DB-persisted state (set by Protection when it trips a breaker). This way,
        a Protection-tripped breaker is visible to Alpha even without a restart.
        """
        equity = self._broker.get_equity()
        for ticker, payload in self._pending_payloads.items():
            if ticker in self._executed_tickers_today:
                continue
            strategy = self._pending_strategies.get(ticker, "momentum")
            breaker = (
                self._thesis_breaker if strategy in _THESIS_STRATEGIES
                else self._swing_breaker if strategy in _SWING_STRATEGIES
                else self._intraday_breaker
            )
            # Halted state gates ENTRIES only (invariant #1) — a discretionary SELL
            # (e.g. "thesis broken, exit early") must never be blocked by a
            # halted/tripped/profit-locked/globally-halted breaker.
            db_halted = self._state_store.is_breaker_halted(breaker.name)
            globally_halted, global_reason = GlobalRiskState.is_halted_in_db(self._state_store)
            if payload.proposal.action == Action.BUY and (breaker.is_stock_halted or db_halted or globally_halted):
                logger.info("[%s] %s: breaker halted (in_memory=%s db=%s global=%s %s), skipping.",
                            breaker.name, ticker, breaker.is_stock_halted, db_halted, globally_halted, global_reason)
                continue

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
                if proposal.action == Action.BUY:
                    breaker.assert_not_tripped()
                breaker.validate_position_size(proposal, equity)
            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked intent for %s: %s", ticker, exc)
                continue

            # Write intent for Protection to execute
            from execution_layer.broker import _order_id
            client_order_id = _order_id(today.isoformat(), ticker, proposal.action.value, str(proposal.quantity))
            self._state_store.write_order_intent(
                client_order_id=client_order_id,
                strategy=strategy,
                ticker=ticker,
                action=proposal.action.value,
                quantity=proposal.quantity,
                limit_price=proposal.limit_price,
                stop_price=None,  # Protection computes stop from settings
            )
            logger.info("%s: queued intent → Protection will submit (strategy=%s, qty=%d @ %.2f)",
                        ticker, strategy, proposal.quantity, proposal.limit_price)

            if proposal.action == Action.BUY and strategy in _THESIS_STRATEGIES:
                self._thesis_session_buys += 1

    def _assert_not_globally_halted(self) -> None:
        """ORB equity/options and vol_options entries submit directly (this
        file's own docstring: "still submit directly here") and so never pass
        through _queue_pending_as_intents' breaker checks. Call this right
        before each of those direct submissions so a global weekly/trailing
        halt (set by Protection, read via the shared breaker_state table)
        blocks them too, not just the order-intent-queued equity/swing paths.
        """
        halted, reason = GlobalRiskState.is_halted_in_db(self._state_store)
        if halted:
            raise CircuitBreakerTripped(f"globally halted: {reason}")

    # ── Scanning entry points ─────────────────────────────────────────────────

    def pre_market_scan(self) -> None:
        equity = self._broker.get_equity()
        today = date.today()
        for breaker in (self._intraday_breaker, self._options_breaker, self._thesis_breaker, self._swing_breaker):
            breaker.start_trading_day(
                equity=equity, today=today,
                profit_target_pct=self._settings.daily_profit_target_pct,
            )
        self._state_store.record_event(event_type="day_start_equity", detail=f"{equity:.2f}")
        self._pending_payloads.clear()
        self._pending_regimes.clear()
        self._pending_strategies.clear()
        self._scanned_tickers_today.clear()
        self._executed_tickers_today.clear()
        self._scanned_options_tickers_today.clear()
        self._scanned_vol_tickers_today.clear()
        self._scanned_swing_tickers_today.clear()
        self._price_cache.clear()
        self._daily_regime = self._assess_market_regime(today)

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
            logger.info("=== PRE-MARKET: dynamic universe mode — momentum scan runs intraday instead ===")

    def momentum_scan_and_trade(self) -> None:
        # ORB equity: deterministic screen, submits directly (same-day, no overnight protection needed)
        if not self._settings.orb_equity_enabled:
            return
        orb_armed = self._daily_regime.arm_orb_equity if self._daily_regime else True
        if not orb_armed or self._intraday_breaker.is_stock_halted:
            if not orb_armed:
                logger.info("ORB equity scan skipped — regime disarmed")
            return
        today = date.today()
        equity = self._broker.get_equity()
        self._intraday_breaker.ensure_day_started(
            equity=equity, today=today,
            profit_target_pct=self._settings.daily_profit_target_pct,
        )
        self._scan_and_trade_orb_equities(today, equity)

    def thesis_scan_and_trade(self) -> None:
        thesis_armed = self._daily_regime.arm_thesis if self._daily_regime else True
        any_enabled = self._settings.thesis_track_enabled or self._settings.recovery_track_enabled
        if not any_enabled or not thesis_armed or self._thesis_breaker.is_stock_halted:
            if any_enabled and not thesis_armed:
                logger.info("Thesis/recovery scan skipped — regime disarmed (VIX > 30, fear environment)")
            return
        self._thesis_session_buys = 0
        today = date.today()
        equity = self._broker.get_equity()
        self._thesis_breaker.ensure_day_started(
            equity=equity, today=today,
            profit_target_pct=self._settings.daily_profit_target_pct,
        )

        try:
            universe_result = self._data_client.get_thesis_universe()
        except DataLayerError as exc:
            logger.error("Thesis/recovery universe fetch failed: %s", exc)
            return

        if self._settings.thesis_track_enabled:
            thesis_candidates = self._build_thesis_candidates(today, universe=universe_result)
            new_thesis = [c for c in thesis_candidates if c not in self._scanned_tickers_today]
            if new_thesis:
                logger.info("=== THESIS SCAN: %d new candidate(s) ===", len(new_thesis))
                self._scan_and_run_consensus(new_thesis, today, equity, strategy="thesis")

        if self._settings.recovery_track_enabled:
            recovery_candidates = self._build_recovery_candidates(today, universe=universe_result)
            new_recovery = [c for c in recovery_candidates if c not in self._scanned_tickers_today]
            if new_recovery:
                logger.info("=== RECOVERY SCAN: %d new candidate(s) ===", len(new_recovery))
                self._scan_and_run_consensus(new_recovery, today, equity, strategy="recovery")

        if self._daily_regime and self._daily_regime.news_ticker_signals:
            news_bullish = [
                s["ticker"] for s in self._daily_regime.news_ticker_signals
                if s.get("direction") == "bullish" and s["ticker"] not in self._scanned_tickers_today
            ]
            if news_bullish:
                logger.info("=== NEWS-DRIVEN SCAN: %d ticker(s) with bullish catalysts ===", len(news_bullish))
                for sig in self._daily_regime.news_ticker_signals:
                    if sig.get("direction") == "bullish" and sig["ticker"] in news_bullish:
                        logger.info("  %s: %s", sig["ticker"], sig.get("catalyst", ""))
                self._scan_and_run_consensus(news_bullish, today, equity, strategy="news")

        # Write to order_intents instead of direct broker submission
        self._queue_pending_as_intents(today)

        self._state_store.record_scan_session(today, "thesis", buys_placed=self._thesis_session_buys)
        streak = self._state_store.get_zero_buy_streak("thesis")
        if streak >= 3:
            logger.warning("DEAD-MAN ALERT: thesis scan placed 0 BUYs for %d consecutive sessions", streak)
            try:
                alerting.alert_zero_buy_streak("thesis", streak)
            except Exception:  # noqa: BLE001
                pass

    def gap_scan_and_queue(self) -> None:
        if not self._settings.thesis_track_enabled or self._thesis_breaker.is_stock_halted:
            return
        today = date.today()
        equity = self._broker.get_equity()
        self._thesis_breaker.ensure_day_started(
            equity=equity, today=today,
            profit_target_pct=self._settings.daily_profit_target_pct,
        )

        from analyst_layer.gap_scanner import scan_premarket_gaps
        candidates = scan_premarket_gaps(
            watchlist=self._watchlist,
            min_gap_pct=getattr(self._settings, "gap_scan_min_pct", 0.05),
            gap_up_only=True,
            max_candidates=getattr(self._settings, "gap_scan_max_candidates", 5),
        )
        if not candidates:
            logger.info("Gap scan: no stocks gapping ≥%.0f%% pre-market",
                        getattr(self._settings, "gap_scan_min_pct", 0.05) * 100)
            return

        logger.info("=== GAP SCAN: %d candidate(s) for 9:30 execution ===", len(candidates))
        for c in candidates:
            logger.info("  %s: %+.1f%% pre-market (prev_close=%.2f → pre=%.2f)",
                        c.symbol, c.gap_pct * 100, c.prev_close, c.premarket_price)

        symbols = [c.symbol for c in candidates if c.symbol not in self._scanned_tickers_today]
        if symbols:
            self._scan_and_run_consensus(symbols, today, equity, strategy="gap")

    def market_open_execution(self) -> None:
        logger.info("=== MARKET-OPEN EXECUTION ===")
        if self._intraday_breaker.is_stock_halted and self._thesis_breaker.is_stock_halted:
            logger.warning(
                "All equity agents halted — skipping. "
                "intraday: tripped=%s/locked=%s, thesis: tripped=%s/locked=%s",
                self._intraday_breaker.is_tripped, self._intraday_breaker.is_profit_locked,
                self._thesis_breaker.is_tripped, self._thesis_breaker.is_profit_locked,
            )
            return
        self._queue_pending_as_intents(date.today())

    def swing_scan_and_trade(self) -> None:
        if not self._settings.swing_track_enabled or self._swing_breaker.is_stock_halted:
            return
        today = date.today()
        equity = self._broker.get_equity()
        self._swing_breaker.ensure_day_started(
            equity=equity, today=today,
            profit_target_pct=self._settings.daily_profit_target_pct,
        )

        existing_swing = {
            p["ticker"] for p in self._state_store.get_positions()
            if p.get("strategy") == "swing" and p["quantity"] > 0
        }
        remaining_slots = self._settings.swing_max_open_positions - len(existing_swing)
        if remaining_slots <= 0:
            logger.info("Swing scan: %d/%d positions open — at cap, skipping scan",
                        len(existing_swing), self._settings.swing_max_open_positions)
            return

        bearish_news: set[str] = set()
        if self._daily_regime and self._daily_regime.news_ticker_signals:
            bearish_news = {
                s["ticker"] for s in self._daily_regime.news_ticker_signals
                if s.get("direction") == "bearish"
            }

        try:
            raw_universe = self._data_client.get_thesis_universe()
            universe_symbols = [c.symbol for c in raw_universe]
        except DataLayerError as exc:
            logger.warning("Swing scan: thesis universe fetch failed, falling back to watchlist: %s", exc)
            universe_symbols = list(self._watchlist)

        all_symbols = list(dict.fromkeys(universe_symbols + list(self._watchlist)))
        passed: list[tuple[str, float]] = []
        for symbol in all_symbols:
            if symbol in existing_swing or symbol in self._scanned_swing_tickers_today:
                continue
            if symbol in bearish_news:
                logger.debug("%s: swing scan skipped — bearish news catalyst today", symbol)
                continue
            self._scanned_swing_tickers_today.add(symbol)

            try:
                series = self._data_client.get_price_history(
                    symbol, start_date=today - timedelta(days=65), end_date=today
                )
            except DataLayerError as exc:
                logger.debug("%s: swing scan price fetch failed — %s", symbol, exc)
                continue

            signal = swing_scanner.evaluate_swing_candidate(series)
            if signal.passed:
                logger.info("%s: swing scan PASSED (%s)", symbol, "; ".join(signal.reasons))
                self._price_cache[symbol] = [b.close for b in series.bars]
                passed.append((symbol, signal.score))
            else:
                logger.debug("%s: swing scan failed — %s", symbol, signal.reasons[0] if signal.reasons else "")

        passed.sort(key=lambda pair: pair[1], reverse=True)
        candidates = [ticker for ticker, _ in passed[:remaining_slots]]

        self._state_store.record_event(
            event_type="swing_scan_summary",
            detail=(
                f"passed={len(passed)}/{len(all_symbols)} candidates, "
                f"capped to {len(candidates)} (slots remaining={remaining_slots})"
            ),
        )

        if not candidates:
            return

        logger.info("=== SWING SCAN: %d new candidate(s) ===", len(candidates))
        self._scan_and_run_consensus(candidates, today, equity, strategy="swing")
        self._queue_pending_as_intents(today)

    def options_scan_and_trade(self) -> None:
        # Options submit directly (specific contracts chosen during scan, intraday positions)
        if not self._settings.options_track_enabled or self._options_breaker.is_options_halted:
            return
        orb_armed = self._daily_regime.arm_orb_options if self._daily_regime else True
        if not orb_armed:
            logger.info("ORB options scan skipped — regime disarmed")
            return
        today = date.today()
        equity = self._broker.get_equity()
        self._options_breaker.ensure_day_started(
            equity=equity, today=today,
            profit_target_pct=self._settings.daily_profit_target_pct,
        )
        try:
            movers = self._data_client.get_market_movers()
        except DataLayerError as exc:
            logger.error("ORB options scan: market movers fetch failed: %s", exc)
            return
        candidates = [m.symbol for m in movers]
        new_candidates = [c for c in candidates if c not in self._scanned_options_tickers_today]
        if not new_candidates:
            return
        logger.info("=== OPTIONS SCAN (ORB signal): %d new candidate(s) ===", len(new_candidates))
        self._scan_and_trade_options_orb(new_candidates, today, equity)

    def vol_options_scan_and_trade(self) -> None:
        # Vol options: premium-selling structures — submit directly (specific contracts)
        vol_armed = self._daily_regime.arm_vol if self._daily_regime else False
        if not self._settings.vol_options_track_enabled or not vol_armed or self._options_breaker.is_options_halted:
            if self._settings.vol_options_track_enabled and not vol_armed:
                logger.info("Vol options scan skipped — regime disarmed (VIX conditions unfavorable)")
            return
        today = date.today()
        candidates = list(self._vol_universe)
        logger.info("=== VOL OPTIONS SCAN: %d candidate(s) ===", len(candidates))
        equity = self._broker.get_equity()
        self._options_breaker.ensure_day_started(
            equity=equity, today=today,
            profit_target_pct=self._settings.daily_profit_target_pct,
        )
        vix_context = self._fetch_vix_context(today)
        portfolio = self._build_portfolio_greeks(equity)

        existing_vol_tickers = {
            p["underlying_symbol"]
            for p in self._state_store.get_option_positions()
            if p.get("strategy") == "vol_short" and p["quantity"] != 0
        }
        for order in self._broker.get_open_orders():
            if order.get("legs"):
                for leg in order["legs"]:
                    parsed = parse_occ_symbol(leg["symbol"]) if leg.get("symbol") else None
                    if parsed:
                        existing_vol_tickers.add(parsed.underlying_symbol)

        for ticker in candidates:
            if ticker in existing_vol_tickers or ticker in self._scanned_vol_tickers_today:
                continue
            self._scanned_vol_tickers_today.add(ticker)

            try:
                vol_snapshot = self._data_client.get_volatility_snapshot(ticker)
                chain = self._data_client.get_option_chain(ticker)
            except DataLayerError as exc:
                logger.error("%s: vol scan data fetch failed — skipping: %s", ticker, exc)
                continue

            # Hard gate: never sell premium through an earnings event. The LLM
            # agents see earnings_within_dte but this is enforced here too so a
            # miscalibrated agent can't override a fundamental risk constraint.
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

            # GARCH(1,1) realized vol forecast — best-effort, enriches the
            # IVSurfaceAgent's prompt with a forward-looking VRP signal rather
            # than the backward-looking HV. Failure is silent: HV-based VRP
            # in the snapshot is the fallback.
            try:
                price_hist = self._data_client.get_price_history(
                    ticker, start_date=today - timedelta(days=90), end_date=today
                )
                garch_rv = vol_analytics.estimate_garch_rv(
                    price_hist, forecast_horizon=self._settings.vol_options_target_dte
                )
                if garch_rv is not None:
                    vol_snapshot = vol_snapshot.model_copy(update={"garch_rv_forecast": garch_rv})
            except DataLayerError as exc:
                logger.debug("%s: GARCH price fetch failed — using HV-based VRP only: %s", ticker, exc)

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
            if not payload.is_executable:
                logger.info("%s: vol consensus not executable (verdict=%s)", ticker, payload.risk_review.verdict.value)
                continue

            self._open_vol_options_position(ticker, payload, chain, equity, today)

    def post_market_logging(self) -> None:
        logger.info("=== POST-MARKET LOGGING ===")
        positions = self._state_store.get_positions()
        history = self._state_store.get_run_history(limit=len(self._watchlist) or 50)
        cost_summary = self._state_store.get_cost_summary(since=datetime.utcnow().date())
        self._state_store.record_event(
            event_type="post_market_summary",
            detail=(
                f"positions={len(positions)} runs_logged={len(history)} "
                f"intraday_breaker(tripped={self._intraday_breaker.is_tripped},locked={self._intraday_breaker.is_profit_locked}) "
                f"thesis_breaker(tripped={self._thesis_breaker.is_tripped},locked={self._thesis_breaker.is_profit_locked}) "
                f"claude_cost_usd={cost_summary['total_cost_usd']:.4f}"
            ),
        )
        logger.info("Post-market: %d open positions, %d runs logged today.", len(positions), len(history))
        logger.info(
            "Claude spend today: $%.4f across %d calls (%d input, %d output, %d cache-read tokens)",
            cost_summary["total_cost_usd"], cost_summary["total_calls"],
            cost_summary["total_input_tokens"], cost_summary["total_output_tokens"],
            cost_summary["total_cache_read_input_tokens"],
        )
        for row in cost_summary["by_agent"]:
            logger.info("  %-30s %d calls  $%.4f", row["agent_name"], row["calls"], row["cost_usd"])

        try:
            today_str = date.today().isoformat()
            equity_pnl = sum(
                r["realized_pnl"] for r in self._state_store.get_all_realized_sales(limit=100)
                if r["sale_date"] == today_str
            )
            options_pnl = sum(
                r["realized_pnl"] for r in self._state_store.get_all_realized_option_sales(limit=100)
                if r["sale_date"] == today_str
            )
            alerting.alert_daily_summary(
                equity=self._broker.get_equity(),
                realized_pnl=equity_pnl + options_pnl,
                open_positions=len(positions),
            )
        except Exception:  # noqa: BLE001
            pass

        self._backfill_candidate_forward_returns()

        if date.today().weekday() == 4:  # Friday — piggyback on this job rather than a new APScheduler entry
            try:
                from scripts.signal_uplift import compute_uplift
                results = compute_uplift(self._state_store)
                lines = []
                for r in results:
                    header = f"{r['signal_name']}/{r['signal_version']}/{r['metric_name']} (n={r['n']})"
                    if r["status"] == "INSUFFICIENT SAMPLE":
                        lines.append(f"{header}: INSUFFICIENT SAMPLE")
                    else:
                        lines.append(f"{header}: IC={r['incremental_ic']:+.4f} -> {r['verdict']}")
                alerting.alert_signal_uplift_summary(lines)
            except Exception:  # noqa: BLE001 — weekly report is non-decision-path, same as alert_daily_summary above
                pass

    def check_manual_trigger(self) -> None:
        from execution_layer.manual_trigger import read_and_clear_trigger
        trigger = read_and_clear_trigger()
        if trigger is None:
            return
        scan = trigger["scan"]
        tickers = trigger["tickers"]
        today = date.today()
        equity = self._broker.get_equity()
        logger.info("Manual trigger fired: scan=%r tickers=%s", scan, tickers or "full screen")

        if tickers:
            # Per-ticker path: bypass screen, run consensus directly on the given tickers.
            # Dedup against today's already-scanned set so we don't re-evaluate the same
            # ticker twice in one session if the scheduled scan already caught it.
            new_tickers = [t for t in tickers if t not in self._scanned_tickers_today]
            if not new_tickers:
                logger.info("All requested tickers already scanned today: %s", tickers)
                return
            self._thesis_breaker.ensure_day_started(
                equity=equity, today=today,
                profit_target_pct=self._settings.daily_profit_target_pct,
            )
            logger.info("=== MANUAL TICKER SCAN: %s ===", new_tickers)
            self._scan_and_run_consensus(new_tickers, today, equity, strategy=scan)
            self._queue_pending_as_intents(today)
            return

        if scan == "thesis":
            self.thesis_scan_and_trade()
        elif scan == "swing":
            self.swing_scan_and_trade()
        elif scan == "gap":
            self.gap_scan_and_queue()

    # ── Core consensus loop ───────────────────────────────────────────────────

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

            proposed_closes = [b.close for b in price_series.bars]
            self._price_cache[ticker] = proposed_closes

            setup_tags = lesson_store.derive_setup_tags(price_series, strategy)
            if self._settings.freeze_lesson_injection:
                relevant_lessons = []
            else:
                relevant_lessons = lesson_store.get_relevant_lessons(
                    self._state_store, strategy, setup_tags
                )
                if relevant_lessons:
                    logger.debug("%s: injecting %d past lessons into consensus", ticker, len(relevant_lessons))

            accuracy_rows = self._state_store.get_agent_accuracy(strategy, regime)
            accuracy_context = agent_scorer.format_accuracy_context(accuracy_rows)
            vw_win_prob = self._vw_bandit.predict_context(strategy, regime)
            vw_context = agent_scorer.format_vw_context(vw_win_prob, self._vw_bandit.example_count)
            macro_context = self._macro_snapshot.format_for_prompt()
            trade_memory = _fetch_trade_memory(ticker, strategy, regime)
            lessons_text = accuracy_context + vw_context + macro_context + trade_memory + lesson_store.format_for_prompt(relevant_lessons)

            existing_shares = self._broker.get_position_shares(ticker)
            pnl_history = self._state_store.get_pnl_history()
            kelly_fraction, kelly_reason = kelly_fraction_from_pnl_history(
                pnl_history, self._settings.max_position_size_pct
            )

            breaker_for_strategy = (
                self._intraday_breaker if strategy in _INTRADAY_STRATEGIES
                else self._options_breaker if strategy in _OPTIONS_STRATEGIES
                else self._thesis_breaker if strategy in _THESIS_STRATEGIES
                else self._swing_breaker
            )
            if hasattr(breaker_for_strategy, "get_size_multiplier"):
                size_mult = breaker_for_strategy.get_size_multiplier()
                if size_mult < 1.0:
                    logger.info(
                        "%s: circuit breaker size multiplier %.0f%% applied (Kelly %.4f → %.4f)",
                        ticker, size_mult * 100, kelly_fraction, kelly_fraction * size_mult,
                    )
                kelly_fraction = kelly_fraction * size_mult

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
                            pass

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

            self._state_store.record_agent_signal_log(
                ticker=ticker, track=strategy, regime=regime,
                proposed_action=payload.proposal.action.value,
                signals=[
                    {"agent_name": s.agent_name, "stance": s.stance.value, "confidence": s.confidence.value}
                    for s in payload.signals
                ],
            )
            for lesson in relevant_lessons:
                if "id" in lesson:
                    self._state_store.record_lesson_injection(lesson["id"], ticker, strategy)

            self._state_store.log_candidate(
                candidate_date=today, strategy=strategy, ticker=ticker,
                llm_verdict=payload.proposal.action.value,
                gate_result=payload.risk_review.verdict.value,
                traded=payload.is_executable and payload.proposal.action == Action.BUY,
                features={
                    "regime": regime,
                    "price": price_series.bars[-1].close if price_series.bars else None,
                    "kelly_fraction": kelly_fraction,
                    "max_corr": round(max_corr, 3),
                },
                screen_score=None,
            )

            self._pending_payloads[ticker] = payload
            self._state_store.record_run(payload)
            logger.info("%s: proposal=%s verdict=%s", ticker, payload.proposal.action.value, payload.risk_review.verdict.value)

    # ── Supporting methods (same as runtime.py) ───────────────────────────────

    def _assess_market_regime(self, today: date) -> DailyRegime | None:
        macro_sentiment: str | None = None
        macro_confidence: float = 0.0
        macro_themes: list[str] = []
        news_ticker_signals: list[dict] = []
        if self._settings.macro_news_enabled:
            try:
                macro = assess_macro_sentiment(
                    client=self._anthropic,
                    model=self._settings.anthropic_subagent_model,
                    today=today,
                    finnhub_api_key=self._settings.finnhub_api_key,
                )
                macro_sentiment = macro.sentiment
                macro_confidence = macro.confidence
                macro_themes = list(macro.key_themes)
                news_ticker_signals = [
                    {"ticker": s.ticker, "catalyst": s.catalyst, "direction": s.direction}
                    for s in macro.news_tickers
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Macro news assessment failed — proceeding without it: %s", exc)

        try:
            self._macro_snapshot = get_macro_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Macro snapshot fetch failed: %s", exc)

        try:
            spy_series = self._data_client.get_price_history(
                "SPY", start_date=today - timedelta(days=60), end_date=today
            )
            vix_series = self._data_client.get_price_history(
                "^VIX", start_date=today - timedelta(days=45), end_date=today
            )
            regime = assess_daily_regime(
                [b.close for b in spy_series.bars],
                vix_series.bars,
                macro_sentiment=macro_sentiment,
                macro_confidence=macro_confidence,
                macro_themes=macro_themes,
                macro_vix_adjustment=self._settings.macro_news_vix_adjustment,
                macro_min_confidence=self._settings.macro_news_min_confidence,
                news_ticker_signals=news_ticker_signals,
            )
            logger.info("Daily regime assessed: %s", regime)
            self._state_store.record_event(event_type="daily_regime", detail=str(regime))
            # Separate machine-readable event so Protection Plane (a different
            # process, no in-memory access to this DailyRegime) can pick up
            # per-ticker bearish news catalysts for its adverse-news swing exit.
            self._state_store.record_event(
                event_type="daily_regime_news_signals",
                detail=json.dumps(regime.news_ticker_signals),
            )
            return regime
        except Exception as exc:  # noqa: BLE001
            logger.error("Market regime assessment failed: %s", exc)
            return None

    def refresh_vol_universe(self) -> None:
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
                detail=f"screened={result.screened} passed={len(result.passed)}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("vol universe refresh failed — keeping existing universe: %s", exc)

    def _build_thesis_candidates(self, today: date, universe=None) -> list[str]:
        source = universe or []
        passed: list[tuple[str, float]] = []
        for candidate in source:
            symbol = candidate.symbol
            if symbol in self._scanned_tickers_today:
                continue
            signal = thesis_scanner.evaluate_thesis_candidate(
                candidate,
                self._settings.thesis_min_pullback_pct,
                self._settings.thesis_max_pullback_pct,
            )
            if signal.passed:
                logger.info("%s: thesis scan PASSED (%s)", symbol, "; ".join(signal.reasons))
                passed.append((symbol, signal.score))
        passed.sort(key=lambda pair: pair[1], reverse=True)
        capped = [ticker for ticker, _ in passed[: self._settings.thesis_max_daily_candidates]]
        self._state_store.record_event(
            event_type="thesis_scan_summary",
            detail=f"passed={len(passed)}/{len(source)} thesis-universe candidates, capped to {len(capped)}",
        )
        return capped

    def _build_recovery_candidates(self, today: date, universe=None) -> list[str]:
        source = universe or []
        passed: list[tuple[str, float]] = []
        for candidate in source:
            symbol = candidate.symbol
            if symbol in self._scanned_tickers_today:
                continue
            try:
                series = self._data_client.get_price_history(
                    symbol, start_date=today - timedelta(days=65), end_date=today
                )
            except DataLayerError as exc:
                logger.debug("%s: recovery price fetch failed — %s", symbol, exc)
                continue
            signal = evaluate_recovery_candidate(
                series,
                min_pullback_pct=self._settings.recovery_min_pullback_pct,
                max_pullback_pct=self._settings.recovery_max_pullback_pct,
                volume_pickup_ratio=self._settings.recovery_volume_pickup_ratio,
            )
            if signal.passed:
                logger.info("%s: recovery scan PASSED (%s)", symbol, "; ".join(signal.reasons))
                self._price_cache[symbol] = [b.close for b in series.bars]
                passed.append((symbol, signal.score))
            else:
                logger.debug("%s: recovery scan failed — %s", symbol, signal.reasons[0] if signal.reasons else "")
        passed.sort(key=lambda pair: pair[1], reverse=True)
        capped = [ticker for ticker, _ in passed[: self._settings.recovery_max_daily_candidates]]
        self._state_store.record_event(
            event_type="recovery_scan_summary",
            detail=f"passed={len(passed)}/{len(source)} candidates, capped to {len(capped)}",
        )
        return capped

    def _build_momentum_candidates(self, today: date) -> list[str]:
        try:
            movers = self._data_client.get_market_movers()
            return [m.symbol for m in movers[:self._settings.momentum_max_candidates]]
        except DataLayerError as exc:
            logger.warning("Market movers fetch failed: %s", exc)
            return []

    def _filter_static_watchlist(self, today: date) -> list[str]:
        filtered = []
        for ticker in self._watchlist:
            if ticker in self._scanned_tickers_today:
                continue
            try:
                series = self._data_client.get_price_history(
                    ticker, start_date=today - timedelta(days=60), end_date=today
                )
                closes = [b.close for b in series.bars]
                regime = prefilter.compute_regime(
                    closes, self._settings.filter_sma_short_window, self._settings.filter_sma_long_window
                )
                if regime != "downtrend":
                    filtered.append(ticker)
            except DataLayerError:
                continue
        return filtered

    def _get_daily_closes(self, ticker: str, today: date) -> list[float] | None:
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        try:
            series = self._data_client.get_price_history(
                ticker, start_date=today - timedelta(days=65), end_date=today
            )
            closes = [b.close for b in series.bars]
            self._price_cache[ticker] = closes
            return closes
        except DataLayerError:
            return None

    def _backfill_candidate_forward_returns(self) -> None:
        for days in (1, 5, 21, 63):
            rows = self._state_store.get_candidates_needing_forward_returns(days)
            if not rows:
                continue
            for row in rows:
                try:
                    closes = self._get_daily_closes(row["ticker"], date.today())
                    if closes and len(closes) >= 2:
                        entry_price = closes[-(days + 1)]
                        exit_price = closes[-1]
                        fwd_ret = (exit_price - entry_price) / entry_price
                        self._state_store.update_candidate_forward_return(row["id"], days, fwd_ret)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Forward return backfill failed for %s (+%dd): %s", row["ticker"], days, exc)
        logger.info("Candidate ledger forward-return backfill complete.")

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

    def _get_deployed_notional(self, ticker: str) -> float:
        detail = self._broker.get_position_detail(ticker)
        if detail is None:
            return 0.0
        return detail["qty"] * detail["current_price"]

    def _get_total_deployed_notional(self) -> float:
        return sum(p["qty"] * p["current_price"] for p in self._broker.get_all_positions())

    # ── Options methods (submit directly — intraday, no overnight protection) ─

    def _scan_and_trade_orb_equities(self, today: date, equity: float) -> None:
        spy_positive: bool = True
        if self._settings.orb_require_spy_positive:
            try:
                spy_intraday = self._data_client.get_price_history("SPY", start_date=today, end_date=today, interval="5m")
                if spy_intraday.bars:
                    spy_positive = spy_intraday.bars[-1].close >= spy_intraday.bars[0].open
                    if not spy_positive:
                        bearish_mode = self._settings.orb_spy_red_puts_enabled
                        logger.info("ORB scan: SPY red today — %s",
                                    "routing short breakdowns to puts" if bearish_mode else "skipping all entries")
            except DataLayerError:
                pass

        long_signals = 0
        short_signals = 0
        max_equity = self._settings.max_open_equity_positions
        max_opts = self._settings.max_open_options_positions

        try:
            movers = self._data_client.get_market_movers()
        except DataLayerError as exc:
            logger.error("ORB equity scan: market movers fetch failed: %s", exc)
            return
        candidates = [m.symbol for m in movers]
        movers_by_symbol = {m.symbol: m for m in movers}
        for ticker in candidates:
            if spy_positive:
                open_count = len([p for p in self._state_store.get_positions() if p["quantity"] > 0])
                if open_count >= max_equity:
                    logger.info("ORB scan: %d/%d equity positions open — at cap", open_count, max_equity)
                    break
            elif self._settings.orb_spy_red_puts_enabled:
                open_opts = len([p for p in self._state_store.get_option_positions() if p.get("quantity", 0) > 0])
                if open_opts >= max_opts:
                    logger.info("ORB bearish scan: %d/%d option positions open — at cap", open_opts, max_opts)
                    break
            else:
                break

            if spy_positive and movers_by_symbol:
                mover = movers_by_symbol.get(ticker)
                if mover is not None and mover.percent_change < self._settings.orb_min_gap_pct * 100:
                    self._scanned_tickers_today.add(ticker)
                    continue

            self._scanned_tickers_today.add(ticker)

            if spy_positive and self._settings.orb_max_float_shares > 0:
                try:
                    shares_float = self._data_client.get_shares_float(ticker)
                    if shares_float > self._settings.orb_max_float_shares:
                        continue
                except DataLayerError:
                    pass

            try:
                intraday = self._data_client.get_price_history(ticker, start_date=today, end_date=today, interval="5m")
            except DataLayerError as exc:
                logger.debug("%s: skipped for ORB scan (%s)", ticker, exc)
                continue

            prior_close: float | None = None
            if spy_positive:
                try:
                    daily_closes = self._get_daily_closes(ticker, today)
                    if daily_closes and len(daily_closes) >= 2:
                        prior_close = daily_closes[-2]
                except Exception:
                    pass

            signal = orb_scanner.evaluate_orb(
                intraday,
                opening_range_minutes=15,
                volume_confirmation_multiple=self._settings.orb_volume_confirmation_multiple,
                prior_close=prior_close,
                min_gap_pct=self._settings.orb_min_gap_pct if spy_positive else None,
            )

            if spy_positive:
                if signal.direction != "long":
                    continue
                long_signals += 1
                logger.info("%s: ORB long signal — gap=%.1f%% (%s)",
                            ticker, (signal.gap_pct or 0) * 100, "; ".join(signal.reasons))
                self._open_orb_equity_position(ticker, signal, equity, today)
            else:
                if signal.direction != "short":
                    continue
                if ticker in self._scanned_options_tickers_today:
                    continue
                short_signals += 1
                logger.info("%s: ORB short breakdown on red SPY — buying puts", ticker)
                self._scanned_options_tickers_today.add(ticker)
                self._open_option_position(ticker, Action.SELL, equity, today)

        self._state_store.record_event(
            event_type="momentum_orb_scan_summary",
            detail=(
                f"{long_signals} long equity signal(s), {short_signals} bearish put signal(s) "
                f"across {len(candidates)} candidates (SPY={'green' if spy_positive else 'red'})"
            ),
        )

    def _open_orb_equity_position(self, ticker: str, signal, equity: float, today: date) -> None:
        current_price = signal.opening_range_high
        stop_price = signal.opening_range_low
        risk_per_share = current_price - stop_price
        if risk_per_share <= 0:
            logger.info("%s: ORB signal has non-positive risk per share — skipping", ticker)
            return
        target_price = current_price + 2 * risk_per_share
        max_notional = equity * self._settings.max_position_size_pct
        quantity = math.floor(max_notional / current_price)
        if quantity <= 0:
            return

        proposal = TradeProposal(ticker=ticker, action=Action.BUY, quantity=quantity, limit_price=current_price)
        try:
            self._assert_not_globally_halted()
            self._breaker.assert_not_tripped()
            self._breaker.validate_position_size(proposal, equity)
            result = self._broker.submit_order(proposal)
            logger.info("%s: ORB equity order result=%s", ticker, result)
            self._state_store.record_event(
                event_type=f"order_{Action.BUY.value.lower()}",
                detail=f"{ticker}: BUY x{quantity} @ {current_price:.2f} -> {result.get('order_status','unknown')}",
            )
            detail = self._broker.get_position_detail(ticker)
            shares = int(detail["qty"]) if detail else quantity
            avg_price = detail["avg_entry_price"] if detail else current_price
            self._state_store.upsert_position(
                ticker, shares, avg_price,
                last_buy_at=today.isoformat(), entry_regime="bullish_crossover",
                strategy="orb", stop_price=stop_price, target_price=target_price,
            )
            alerting.alert_buy(ticker=ticker, shares=shares, price=avg_price, strategy="orb")
            new_equity = self._broker.get_equity()
            if self._breaker.check_profit_target(new_equity):
                logger.info("PROFIT LOCK after ORB trade on %s, equity=%.2f", ticker, new_equity)
        except CircuitBreakerTripped as exc:
            logger.error("Circuit breaker blocked ORB equity order for %s: %s", ticker, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: ORB equity order failed: %s", ticker, exc)

    def _scan_and_trade_options_orb(self, candidates: list[str], today: date, equity: float) -> None:
        signals_found = 0
        max_opts = self._settings.max_open_options_positions
        for ticker in candidates:
            open_opts = len([p for p in self._state_store.get_option_positions() if p.get("quantity", 0) > 0])
            if open_opts >= max_opts:
                logger.info("Options scan: %d/%d option positions open — at cap", open_opts, max_opts)
                break
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
        wash_warning = self._wash_sale_guard.check_option_buy_for_wash(ticker, today)
        if wash_warning:
            self._state_store.record_event(event_type="wash_sale_option_warning", detail=wash_warning)
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
        max_risk_dollars = equity * self._settings.options_max_risk_pct
        contracts = math.floor(max_risk_dollars / (contract.ask * 100)) if contract.ask > 0 else 0
        if contracts <= 0:
            return

        try:
            self._assert_not_globally_halted()
            self._options_breaker.assert_options_trading_allowed()
            result = self._broker.submit_option_order(
                contract.contract_symbol, side=Action.BUY, contracts=contracts, limit_price=contract.ask
            )
            logger.info("%s: options order result=%s", contract.contract_symbol, result)
            self._state_store.record_event(
                event_type="option_order_buy",
                detail=f"{contract.contract_symbol}: BUY x{contracts} @ {contract.ask:.2f} -> {result.get('order_status','unknown')}",
            )
            detail = self._broker.get_position_detail(contract.contract_symbol)
            qty = int(detail["qty"]) if detail else 0
            avg_price = detail["avg_entry_price"] if detail else contract.ask
            self._state_store.upsert_option_position(
                contract.contract_symbol, contract.underlying_symbol, contract.option_type.value,
                contract.strike, contract.expiration.isoformat(), qty, avg_price,
                opened_at=today.isoformat(),
            )
            alerting.alert_option_buy(
                contract_symbol=contract.contract_symbol, underlying=contract.underlying_symbol,
                contracts=qty, premium=avg_price, strategy="orb_options",
            )
            new_equity = self._broker.get_equity()
            if self._options_breaker.check_profit_target(new_equity):
                logger.info("PROFIT LOCK after options trade on %s, equity=%.2f", ticker, new_equity)
        except CircuitBreakerTripped as exc:
            logger.error("Circuit breaker blocked options order for %s: %s", ticker, exc)

    def _fetch_vix_context(self, today: date) -> VixContext:
        try:
            vix_series = self._data_client.get_price_history("^VIX", start_date=today - timedelta(days=45), end_date=today)
            bars = vix_series.bars
            if not bars:
                return VixContext(vix_current=18.0)
            vix_current = bars[-1].close
            vix_1w_ago = bars[-5].close if len(bars) >= 5 else None
            vix_1m_ago = bars[-21].close if len(bars) >= 21 else None
            vix3m_current: float | None = None
            try:
                vix3m_series = self._data_client.get_price_history("^VIX3M", start_date=today - timedelta(days=5), end_date=today)
                if vix3m_series.bars:
                    vix3m_current = vix3m_series.bars[-1].close
            except DataLayerError:
                pass
            return VixContext(vix_current=vix_current, vix_1w_ago=vix_1w_ago, vix_1m_ago=vix_1m_ago, vix3m_current=vix3m_current)
        except DataLayerError as exc:
            logger.warning("VIX fetch failed — defaulting to stable VIX context: %s", exc)
            return VixContext(vix_current=18.0)

    def _build_portfolio_greeks(self, equity: float) -> PortfolioGreeks:
        """Real per-leg Black-Scholes Greeks, signed by each leg's stored
        quantity (Alpaca reports qty negative for a short leg, so summing
        quantity * per-contract Greeks already nets shorts against longs
        correctly — no separate short/long branch needed).
        """
        option_positions = self._state_store.get_option_positions()
        vol_short_positions = [p for p in option_positions if p.get("strategy") == "vol_short" and p["quantity"] != 0]
        n = len(vol_short_positions)

        net_delta = net_vega = net_theta = 0.0
        chains_by_underlying: dict[str, list] = {}
        for position in vol_short_positions:
            underlying = position["underlying_symbol"]
            if underlying not in chains_by_underlying:
                try:
                    chains_by_underlying[underlying] = self._data_client.get_option_chain(underlying)
                except DataLayerError as exc:
                    logger.warning("%s: option chain fetch failed for Greeks calc — %s", underlying, exc)
                    chains_by_underlying[underlying] = []

            option_type = OptionType.CALL if position["option_type"] == "call" else OptionType.PUT
            contract = self._find_option_contract(
                chains_by_underlying[underlying], option_type,
                date.fromisoformat(position["expiration"]), position["strike"],
            )
            if contract is None or contract.implied_volatility is None:
                logger.debug("%s: no live quote for Greeks calc — excluded from portfolio Greeks", position["contract_symbol"])
                continue

            delta, vega, theta = vol_analytics.black_scholes_greeks(
                contract.underlying_price, contract.strike, contract.dte,
                contract.implied_volatility, option_type,
            )
            qty = position["quantity"] * 100  # already signed: negative for a short leg
            net_delta += delta * qty
            net_vega += vega * qty
            net_theta += theta * qty

        return PortfolioGreeks(net_delta=net_delta, net_vega=net_vega, net_theta=net_theta, portfolio_value=equity, num_open_positions=n)

    def _find_option_contract(self, chain, option_type: OptionType, expiration: date, strike: float):
        for c in chain:
            if c.option_type == option_type and c.expiration == expiration and abs(c.strike - strike) < 0.01:
                return c
        return None

    def _open_vol_options_position(
        self, ticker: str, payload: VolConsensusPayload, chain, equity: float, today: date
    ) -> None:
        proposal = payload.proposal
        wash_warning = self._wash_sale_guard.check_option_buy_for_wash(ticker, today)
        if wash_warning:
            self._state_store.record_event(event_type="wash_sale_option_warning", detail=wash_warning)

        if proposal.structure != StructureType.IRON_CONDOR:
            logger.warning(
                "%s: vol structure %s requires naked shorts (Level 4) — blocked. "
                "Only iron condors are executed on this Level 3 account.",
                ticker, proposal.structure.value,
            )
            self._state_store.record_event(
                event_type="vol_options_blocked_level4",
                detail=f"{ticker}: {proposal.structure.value} blocked — requires Level 4 approval.",
            )
            return

        self._open_iron_condor(ticker, proposal, chain, today)

    def _open_iron_condor(self, ticker: str, proposal, chain, today: date) -> None:
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
            - _mid(long_call.bid, long_call.ask) - _mid(long_put.bid, long_put.ask), 2,
        )
        natural_credit = round(
            (short_call.bid or 0.0) + (short_put.bid or 0.0)
            - (long_call.ask or 0.0) - (long_put.ask or 0.0), 2,
        )
        net_credit = mid_credit if mid_credit > 0 else natural_credit
        if net_credit <= 0:
            logger.info("%s: iron condor net credit %.2f <= 0 — skipping", ticker, net_credit)
            return
        logger.info("%s: iron condor credit: mid=%.2f natural=%.2f submitting at %.2f",
                    ticker, mid_credit, natural_credit, net_credit)

        spread_legs = [
            (short_call.contract_symbol, Action.SELL),
            (short_put.contract_symbol, Action.SELL),
            (long_call.contract_symbol, Action.BUY),
            (long_put.contract_symbol, Action.BUY),
        ]

        try:
            self._assert_not_globally_halted()
            self._options_breaker.assert_options_trading_allowed()
            result = self._broker.submit_spread_order(
                legs=spread_legs, contracts=proposal.quantity, net_credit=net_credit,
            )
            logger.info("%s: iron condor mleg → %s (net_credit=%.2f)",
                        ticker, result.get("order_status", "unknown"), net_credit)
            self._state_store.record_event(
                event_type="vol_options_opened",
                detail=(
                    f"{ticker}: iron_condor mleg "
                    f"{short_call.strike:.0f}C/{short_put.strike:.0f}P short "
                    f"{long_call.strike:.0f}C/{long_put.strike:.0f}P long "
                    f"exp={proposal.expiration} DTE={proposal.dte} net_credit={net_credit:.2f}"
                ),
            )
            for contract, side in (
                (short_call, Action.SELL), (short_put, Action.SELL),
                (long_call, Action.BUY), (long_put, Action.BUY),
            ):
                detail = self._broker.get_position_detail(contract.contract_symbol)
                qty = int(detail["qty"]) if detail else 0
                avg_price = detail["avg_entry_price"] if detail else net_credit
                self._state_store.upsert_option_position(
                    contract.contract_symbol, ticker, contract.option_type.value,
                    contract.strike, contract.expiration.isoformat(),
                    qty, avg_price, opened_at=today.isoformat(), strategy="vol_short",
                )
        except CircuitBreakerTripped as exc:
            logger.error("Circuit breaker blocked iron condor for %s: %s", ticker, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: iron condor mleg rejected: %s", ticker, exc)
            self._state_store.record_event(
                event_type="vol_options_broker_rejected",
                detail=f"{ticker}: iron condor mleg — {exc}",
            )
