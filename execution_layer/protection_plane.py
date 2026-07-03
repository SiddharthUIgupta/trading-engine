"""Protection Plane — the half of the trading engine that survives an Alpha crash.

Owns: exits, reconciliation, order-intent consumption, circuit-breaker enforcement.
Does NOT own: scanning, LLM consensus, sizing, candidate ledger writes.

Process isolation: this module runs as a separate systemd unit (trading-engine-protection).
It communicates with the Alpha Plane through the shared SQLite DB:
  - order_intents table: Alpha writes BUY intents → Protection reads + executes them
  - breaker_state table: Protection writes halted/profit_locked state → Alpha reads before queuing

Both planes share the same state_store (SQLite WAL mode allows concurrent readers/one writer).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from anthropic import Anthropic

from analyst_layer import lesson_store, prefilter
from analyst_layer.reflection_agent import ReflectionAgent
from analyst_layer.agents.intraday_exit_agent import IntradayExitAgent
from analyst_layer.vw_bandit import VWSignalBandit
from config.settings import Settings
from data_layer.exceptions import DataLayerError
from data_layer.occ_symbol import parse_occ_symbol
from execution_layer import alerting, exit_rules
from execution_layer.broker import AlpacaBroker
from execution_layer.guardrails import (
    CircuitBreaker,
    CircuitBreakerTripped,
    execute_global_shutdown,
)
from execution_layer.state_store import StateStore
from execution_layer.tax_compliance import WashSaleGuard
from analyst_layer.schemas import Action, TradeProposal

logger = logging.getLogger(__name__)

_INTRADAY_STRATEGIES: frozenset[str] = frozenset({"momentum", "orb_equity", "news"})
_OPTIONS_STRATEGIES: frozenset[str] = frozenset({"orb_options", "vol_options"})
_THESIS_STRATEGIES: frozenset[str] = frozenset({"thesis", "recovery", "gap"})
_SWING_STRATEGIES: frozenset[str] = frozenset({"swing"})


class ProtectionRuntime:
    """Runs in a dedicated process. Only job: keep positions protected and consume
    order intents written by the Alpha Plane.

    Surviving an Alpha crash is the whole point — this class has NO imports from
    analyst_layer scanners, data_layer OpenBB client, or any LLM consensus logic.
    """

    def __init__(
        self,
        settings: Settings,
        broker: AlpacaBroker,
        state_store: StateStore,
        anthropic_client: Anthropic,
        data_client,  # OpenBBDataClient — used only for regime reversal check in LLM escalation
        intraday_breaker: CircuitBreaker,
        options_breaker: CircuitBreaker,
        thesis_breaker: CircuitBreaker,
        swing_breaker: CircuitBreaker,
        wash_sale_guard: WashSaleGuard | None = None,
    ) -> None:
        self._settings = settings
        self._broker = broker
        self._state_store = state_store
        self._anthropic = anthropic_client
        self._data_client = data_client
        self._intraday_breaker = intraday_breaker
        self._options_breaker = options_breaker
        self._thesis_breaker = thesis_breaker
        self._swing_breaker = swing_breaker
        self._breaker = intraday_breaker  # backward compat alias
        self._wash_sale_guard = wash_sale_guard or WashSaleGuard(state_store)
        self._exit_agent = IntradayExitAgent(
            anthropic_client, settings.anthropic_subagent_model, usage_callback=self._record_usage
        )
        self._reflection_agent = ReflectionAgent(
            anthropic_client, settings.anthropic_subagent_model, usage_callback=self._record_usage
        )
        vw_model_path = settings.state_db_path.parent / "vw_bandit.model"
        self._vw_bandit = VWSignalBandit(model_path=vw_model_path)
        if not vw_model_path.exists():
            historical_logs = state_store.get_scored_signal_logs(limit=2000)
            if historical_logs:
                self._vw_bandit.warm_start(historical_logs)

        # Daily regime snapshot — shared via DB event on each intraday_monitoring tick.
        # Needed only for adverse-news swing exits; loaded from state_store events.
        self._daily_news_ticker_signals: list[dict] = []

    # ── Entry consumption: reads order_intents written by Alpha Plane ─────────

    def consume_order_intents(self, today: date, equity: float) -> None:
        """Read pending BUY intents written by Alpha and submit them to the broker.

        Alpha writes to order_intents when a consensus is executable. Protection
        reads here (on every intraday_monitoring tick) and submits the actual bracket
        orders. This keeps order execution in the plane that manages exits — so a
        bracket entry + its stop-loss order are both placed by the same process.
        """
        intents = self._state_store.get_pending_order_intents()
        if not intents:
            return

        for intent in intents:
            ticker = intent["ticker"]
            strategy = intent["strategy"]
            action = Action(intent["action"])

            breaker = (
                self._thesis_breaker if strategy in _THESIS_STRATEGIES
                else self._swing_breaker if strategy in _SWING_STRATEGIES
                else self._intraday_breaker
            )

            # Mark processed first — prevents double-submission on crash/restart
            self._state_store.mark_order_intent_processed(intent["client_order_id"], "submitted")

            if breaker.is_stock_halted:
                logger.info("[%s] %s: breaker halted, skipping intent", breaker.name, ticker)
                self._state_store.mark_order_intent_processed(intent["client_order_id"], "skipped_halted")
                continue

            proposal = TradeProposal(
                ticker=ticker,
                action=action,
                quantity=intent["quantity"],
                limit_price=intent["limit_price"],
            )

            if action == Action.BUY:
                violation = self._wash_sale_guard.check_before_buy(ticker, today)
                if violation is not None:
                    logger.warning("Blocking BUY for %s — wash sale: %s", ticker, violation.reason)
                    self._state_store.record_event(event_type="wash_sale_blocked", detail=violation.reason)
                    self._state_store.mark_order_intent_processed(intent["client_order_id"], "skipped_wash_sale")
                    continue
            elif action == Action.SELL:
                self._wash_sale_guard.warn_before_sell(ticker, proposal.limit_price, today)

            try:
                breaker.assert_not_tripped()
                breaker.validate_position_size(proposal, equity)

                use_bracket = (
                    action == Action.BUY
                    and strategy in (_THESIS_STRATEGIES | _SWING_STRATEGIES)
                )
                bracket_stop_order_id = None
                stop_price = None
                if use_bracket:
                    stop_loss_pct = (
                        self._settings.thesis_stop_loss_pct if strategy in _THESIS_STRATEGIES
                        else self._settings.swing_stop_loss_pct
                    )
                    stop_price = proposal.limit_price * (1 - stop_loss_pct)
                    result = self._broker.submit_bracket_order(proposal, stop_price=stop_price)
                    bracket_stop_order_id = result.get("stop_order_id")
                else:
                    result = self._broker.submit_order(proposal)

                logger.info("%s: consumed intent, order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(
                    ticker, proposal, today,
                    strategy=strategy,
                    bracket_stop_order_id=bracket_stop_order_id,
                    stop_price=stop_price,
                )

                equity = self._broker.get_equity()
                if breaker.check_profit_target(equity):
                    self._lock_in_profit(
                        reason=f"daily profit target reached after trading {ticker}, equity={equity:.2f}",
                        breaker=breaker,
                    )
                    break

            except CircuitBreakerTripped as exc:
                logger.error("Circuit breaker blocked order for %s: %s", ticker, exc)
                self._trip_agent(breaker, reason=str(exc))

    # ── Main protection loop ───────────────────────────────────────────────────

    def intraday_monitoring(self) -> None:
        """Runs every 15 minutes. Reconcile → ensure day started → check breakers
        → consume order intents → run all exit checks. Order intentionally places
        exit checks LAST so a new entry doesn't count toward the daily drawdown
        check that runs on the same tick.
        """
        self._reconcile_positions()
        if self._settings.options_track_enabled or self._settings.vol_options_track_enabled:
            self._reconcile_option_positions()

        today = date.today()
        equity = self._broker.get_equity()
        for breaker in (self._intraday_breaker, self._options_breaker, self._thesis_breaker, self._swing_breaker):
            breaker.ensure_day_started(
                equity=equity, today=today,
                profit_target_pct=self._settings.daily_profit_target_pct,
            )
        logger.info("Protection tick: equity=%.2f", equity)

        # Drawdown + profit-lock checks per agent
        if not self._intraday_breaker.is_halted:
            intraday_pnl = self._compute_today_pnl(_INTRADAY_STRATEGIES, include_options=False)
            if self._intraday_breaker.check_drawdown(intraday_pnl):
                self._trip_agent(self._intraday_breaker, reason=f"intraday drawdown breach (today_pnl={intraday_pnl:.2f})")
            elif self._intraday_breaker.check_profit_target(equity):
                self._lock_in_profit(reason=f"daily profit target reached, equity={equity:.2f}", breaker=self._intraday_breaker)

        if not self._thesis_breaker.is_halted:
            thesis_pnl = self._compute_today_pnl(_THESIS_STRATEGIES, include_options=False)
            if self._thesis_breaker.check_drawdown(thesis_pnl):
                self._trip_agent(self._thesis_breaker, reason=f"thesis drawdown breach (today_pnl={thesis_pnl:.2f})")
            elif self._thesis_breaker.check_profit_target(equity):
                self._lock_in_profit(reason=f"daily profit target reached, equity={equity:.2f}", breaker=self._thesis_breaker)

        if not self._options_breaker.is_halted:
            options_pnl = self._compute_today_pnl(_OPTIONS_STRATEGIES, include_options=True)
            if self._options_breaker.check_drawdown(options_pnl):
                self._trip_agent(self._options_breaker, reason=f"options drawdown breach (today_pnl={options_pnl:.2f})")
            elif self._options_breaker.check_profit_target(equity):
                self._lock_in_profit(reason=f"daily profit target reached, equity={equity:.2f}", breaker=self._options_breaker)

        if not self._swing_breaker.is_halted:
            swing_pnl = self._compute_today_pnl(_SWING_STRATEGIES, include_options=False)
            if self._swing_breaker.check_drawdown(swing_pnl):
                self._trip_agent(self._swing_breaker, reason=f"swing drawdown breach (today_pnl={swing_pnl:.2f})")
            elif self._swing_breaker.check_profit_target(equity):
                self._lock_in_profit(reason=f"daily profit target reached, equity={equity:.2f}", breaker=self._swing_breaker)

        # Consume pending order intents from Alpha Plane
        self.consume_order_intents(today, equity)

        # Exit checks — unconditional, breakers gate entries only
        self._check_intraday_exits(equity)
        self._check_orb_exits(equity)
        if self._settings.options_track_enabled:
            self._check_options_exits(equity)
        if self._settings.vol_options_track_enabled:
            self._check_vol_options_exits(equity)
        if self._settings.swing_track_enabled:
            self._check_swing_exits(equity)

    # ── Exit checks ───────────────────────────────────────────────────────────

    def _check_intraday_exits(self, equity: float) -> None:
        today = date.today()
        for position in self._state_store.get_positions():
            ticker = position["ticker"]
            if position["quantity"] <= 0 or position["strategy"] in ("orb", "swing"):
                continue

            detail = self._broker.get_position_detail(ticker)
            if detail is None or detail["qty"] <= 0:
                continue

            self._state_store.update_high_water_mark(ticker, detail["current_price"])
            high_water_mark = max(position["high_water_mark"], detail["current_price"])

            bracket_stop_id = position.get("bracket_stop_order_id")
            if bracket_stop_id:
                exit_params = self._exit_params_for(position["strategy"])
                trailing_activation = exit_params.get("trailing_stop_activation_pct", 0.0)
                trailing_pct = exit_params.get("trailing_stop_pct", 0.0)
                gain_pct = (high_water_mark - detail["avg_entry_price"]) / detail["avg_entry_price"]
                if trailing_pct > 0 and gain_pct >= trailing_activation:
                    new_trail_stop = high_water_mark * (1 - trailing_pct)
                    current_stop = position.get("stop_price") or 0.0
                    if new_trail_stop > current_stop:
                        self._broker.amend_stop_order(bracket_stop_id, new_trail_stop)
                        self._state_store.update_broker_stop(ticker, new_trail_stop)

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
                result = self._broker.submit_order(proposal)
                logger.info("%s: intraday exit order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(ticker, proposal, today)
                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after intraday exit of {ticker}, equity={equity:.2f}")
            except Exception as exc:  # noqa: BLE001 — one position's failed exit must not abort the others
                logger.error("%s: intraday exit order FAILED: %s", ticker, exc)

    def _check_orb_exits(self, equity: float) -> None:
        today = date.today()
        open_orders = self._broker.get_open_orders()
        pending_sell_tickers = {
            o["symbol"] for o in open_orders
            if o.get("side") == "sell" and o.get("status") in ("new", "partially_filled", "accepted")
        }
        for position in self._state_store.get_positions():
            ticker = position["ticker"]
            if position["quantity"] <= 0 or position["strategy"] != "orb":
                continue
            if ticker in pending_sell_tickers:
                logger.debug("%s: exit order already pending — skipping", ticker)
                continue

            detail = self._broker.get_position_detail(ticker)
            if detail is None or detail["qty"] <= 0:
                continue
            current_price = detail["current_price"]

            should_exit = False
            reason = ""
            entry_price = position["avg_entry_price"] or 0.0
            if position["last_buy_at"] != today.isoformat():
                should_exit = True
                reason = f"ORB position held past its entry day ({position['last_buy_at']}) — force-closing, day-trade only"
            elif position["stop_price"] is not None and current_price <= position["stop_price"]:
                should_exit = True
                reason = f"stop hit: {current_price:.2f} <= {position['stop_price']:.2f}"
            elif position["target_price"] is not None and current_price >= position["target_price"]:
                should_exit = True
                reason = f"target hit: {current_price:.2f} >= {position['target_price']:.2f}"
            else:
                et_now = datetime.now(ZoneInfo("America/New_York"))
                if et_now.hour >= 14 and entry_price > 0 and current_price <= entry_price:
                    should_exit = True
                    reason = (
                        f"ORB failed-breakout cut: past 2pm ET, "
                        f"price {current_price:.2f} at/below entry {entry_price:.2f}"
                    )

            if not should_exit:
                continue
            logger.info("%s: ORB exit — %s", ticker, reason)

            proposal = TradeProposal(ticker=ticker, action=Action.SELL, quantity=int(detail["qty"]), limit_price=current_price)
            self._wash_sale_guard.warn_before_sell(ticker, proposal.limit_price, today)
            try:
                result = self._broker.submit_order(proposal)
                logger.info("%s: ORB exit order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(ticker, proposal, today)
                equity = self._broker.get_equity()
                if self._breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after ORB exit of {ticker}, equity={equity:.2f}")
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: ORB exit order FAILED: %s", ticker, exc)

    def _check_options_exits(self, equity: float) -> None:
        today = date.today()
        for position in self._state_store.get_option_positions():
            contract_symbol = position["contract_symbol"]
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
            else:
                et_now = datetime.now(ZoneInfo("America/New_York"))
                opened_today = (position.get("opened_at") or "")[:10] == today.isoformat()
                if (
                    opened_today
                    and et_now.hour >= 15
                    and premium_drawdown_pct >= self._settings.options_intraday_stop_pct
                ):
                    should_exit = True
                    reason = (
                        f"ORB intraday exit: breakout stalled, premium down "
                        f"{premium_drawdown_pct:.1%} after 3pm ET"
                    )

            if not should_exit:
                continue
            logger.info("%s: options exit — %s", contract_symbol, reason)

            try:
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
                if self._options_breaker.check_profit_target(equity):
                    self._lock_in_profit(reason=f"daily profit target reached after options exit of {contract_symbol}, equity={equity:.2f}", breaker=self._options_breaker)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: options exit order FAILED: %s", contract_symbol, exc)

    def _check_vol_options_exits(self, equity: float) -> None:
        today = date.today()
        for position in self._state_store.get_option_positions():
            if position.get("strategy") != "vol_short":
                continue
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
            qty = abs(int(detail["qty"]))

            try:
                result = self._broker.submit_option_order(
                    contract_symbol, side=Action.BUY, contracts=qty, limit_price=cost_to_close
                )
                logger.info("%s: vol options close order result=%s", contract_symbol, result)
                self._record_option_order_event(contract_symbol, Action.BUY, qty, cost_to_close, result)

                prior = self._state_store.get_option_position(contract_symbol)
                if prior is not None:
                    self._state_store.record_realized_option_sale(
                        contract_symbol=contract_symbol,
                        underlying_symbol=position["underlying_symbol"],
                        sale_date=today,
                        contracts=qty,
                        sale_price=prior["avg_entry_price"],
                        cost_basis=cost_to_close,
                    )

                closed_detail = self._broker.get_position_detail(contract_symbol)
                closed_qty = int(closed_detail["qty"]) if closed_detail else 0
                self._state_store.upsert_option_position(
                    contract_symbol, position["underlying_symbol"], position["option_type"],
                    position["strike"], position["expiration"], closed_qty, position["avg_entry_price"],
                    strategy=position.get("strategy", "vol_short"),
                )
                self._state_store.record_event(
                    event_type="vol_options_closed",
                    detail=f"{contract_symbol}: {reason}",
                )

                equity = self._broker.get_equity()
                if self._options_breaker.check_profit_target(equity):
                    self._lock_in_profit(
                        reason=f"daily profit target reached after vol options exit of {contract_symbol}, equity={equity:.2f}",
                        breaker=self._options_breaker,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: vol options exit order FAILED: %s", contract_symbol, exc)

    def _check_swing_exits(self, equity: float) -> None:
        today = date.today()
        for position in self._state_store.get_positions():
            ticker = position["ticker"]
            if position["quantity"] <= 0 or position.get("strategy") != "swing":
                continue

            detail = self._broker.get_position_detail(ticker)
            if detail is None or detail["qty"] <= 0:
                continue

            current_price = detail["current_price"]
            avg_entry = position["avg_entry_price"]
            if avg_entry <= 0:
                continue

            should_exit = False
            reason = ""

            loss_pct = (avg_entry - current_price) / avg_entry
            if loss_pct >= self._settings.swing_stop_loss_pct:
                should_exit = True
                reason = (
                    f"swing stop hit: entry {avg_entry:.2f}, current {current_price:.2f} "
                    f"({loss_pct:.1%} loss >= {self._settings.swing_stop_loss_pct:.0%} limit)"
                )

            if not should_exit and position.get("last_buy_at"):
                try:
                    entry_date = date.fromisoformat(position["last_buy_at"][:10])
                    hold_days = (today - entry_date).days
                    if hold_days > self._settings.swing_max_hold_days:
                        should_exit = True
                        reason = f"swing max hold exceeded: {hold_days}d > {self._settings.swing_max_hold_days}d limit"
                except ValueError:
                    pass

            if not should_exit:
                for sig in self._daily_news_ticker_signals:
                    if sig.get("ticker") == ticker and sig.get("direction") == "bearish":
                        should_exit = True
                        reason = f"adverse news catalyst: {sig.get('catalyst', 'bearish news signal')}"
                        break

            if not should_exit:
                self._state_store.update_high_water_mark(ticker, current_price)
                high_water_mark = max(position.get("high_water_mark") or avg_entry, current_price)

                bracket_stop_id = position.get("bracket_stop_order_id")
                if bracket_stop_id:
                    gain_pct = (high_water_mark - avg_entry) / avg_entry
                    if gain_pct >= self._settings.swing_trailing_stop_activation_pct:
                        new_trail_stop = high_water_mark * (1 - self._settings.swing_trailing_stop_pct)
                        current_stop = position.get("stop_price") or 0.0
                        if new_trail_stop > current_stop:
                            self._broker.amend_stop_order(bracket_stop_id, new_trail_stop)
                            self._state_store.update_broker_stop(ticker, new_trail_stop)

                decision = exit_rules.evaluate_exit(
                    avg_entry_price=avg_entry,
                    current_price=current_price,
                    high_water_mark=high_water_mark,
                    stop_loss_pct=self._settings.swing_stop_loss_pct,
                    take_profit_pct=None,
                    trailing_stop_pct=self._settings.swing_trailing_stop_pct,
                    trailing_stop_activation_pct=self._settings.swing_trailing_stop_activation_pct,
                )
                if decision.should_exit:
                    should_exit = True
                    reason = f"swing trailing stop: {decision.reason}"

            if not should_exit:
                continue

            logger.info("%s: swing exit — %s", ticker, reason)
            proposal = TradeProposal(
                ticker=ticker, action=Action.SELL,
                quantity=int(detail["qty"]), limit_price=current_price,
            )
            self._wash_sale_guard.warn_before_sell(ticker, proposal.limit_price, today)
            try:
                result = self._broker.submit_order(proposal)
                logger.info("%s: swing exit order result=%s", ticker, result)
                self._record_order_event(ticker, proposal, result)
                self._record_fill(ticker, proposal, today)
                equity = self._broker.get_equity()
                if self._swing_breaker.check_profit_target(equity):
                    self._lock_in_profit(
                        reason=f"daily profit target reached after swing exit of {ticker}, equity={equity:.2f}",
                        breaker=self._swing_breaker,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: swing exit order FAILED: %s", ticker, exc)

    # ── Pre-close ─────────────────────────────────────────────────────────────

    def pre_close_orb_exit(self) -> None:
        today = date.today()
        for position in self._state_store.get_positions():
            if position.get("strategy") != "orb":
                continue
            if position.get("quantity", 0) <= 0:
                continue
            ticker = position["ticker"]
            avg_entry = position.get("avg_entry_price", 0)
            if avg_entry <= 0:
                continue

            detail = self._broker.get_position_detail(ticker)
            if detail is None or detail.get("qty", 0) <= 0:
                continue

            current_price = detail.get("current_price", 0)
            if current_price <= 0:
                continue

            gain_pct = (current_price - avg_entry) / avg_entry
            if gain_pct >= self._settings.orb_swing_convert_pct:
                spy_green = True
                try:
                    spy_intraday = self._data_client.get_price_history(
                        "SPY", start_date=today, end_date=today, interval="5m"
                    )
                    if spy_intraday.bars:
                        spy_green = spy_intraday.bars[-1].close >= spy_intraday.bars[0].open
                except DataLayerError:
                    pass

                if spy_green:
                    swing_stop = avg_entry * (1 - self._settings.swing_stop_loss_pct)
                    self._state_store.upsert_position(
                        ticker, position["quantity"], avg_entry,
                        stop_price=swing_stop, target_price=None, strategy="swing",
                    )
                    logger.info(
                        "%s: ORB → swing conversion — up %.1f%%, SPY green, holding overnight with stop %.2f",
                        ticker, gain_pct * 100, swing_stop,
                    )
                    continue

            elif gain_pct > 0:
                continue

            qty = int(detail["qty"])
            proposal = TradeProposal(ticker=ticker, action=Action.SELL, quantity=qty, limit_price=current_price)
            logger.info("%s: ORB pre-close exit — current %.2f below entry %.2f", ticker, current_price, avg_entry)
            try:
                result = self._broker.submit_order(proposal)
                self._record_order_event(ticker, proposal, result)
                pnl = self._state_store.record_realized_sale(
                    ticker=ticker, sale_date=today, quantity=qty,
                    sale_price=current_price, cost_basis=avg_entry,
                )
                self._trigger_reflection(ticker, "orb", pnl)
                self._state_store.delete_position(ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: pre-close ORB exit failed: %s", ticker, exc)

    # ── Reconciliation ────────────────────────────────────────────────────────

    def _reconcile_positions(self) -> None:
        for position in self._state_store.get_positions():
            ticker = position["ticker"]
            detail = self._broker.get_position_detail(ticker)
            real_qty = int(detail["qty"]) if detail else 0
            if real_qty != position["quantity"]:
                logger.warning(
                    "%s: local quantity %d out of sync with broker's %d — reconciling",
                    ticker, position["quantity"], real_qty,
                )
                if real_qty == 0:
                    self._state_store.delete_position(ticker)
                else:
                    self._state_store.upsert_position(
                        ticker, real_qty, detail["avg_entry_price"] if detail else position["avg_entry_price"]
                    )

    def _reconcile_option_positions(self) -> None:
        for position in self._state_store.get_option_positions():
            contract_symbol = position["contract_symbol"]
            detail = self._broker.get_position_detail(contract_symbol)
            real_qty = int(detail["qty"]) if detail else 0
            if real_qty != position["quantity"]:
                logger.warning(
                    "%s: local options quantity %d out of sync with broker's %d — reconciling",
                    contract_symbol, position["quantity"], real_qty,
                )
                if real_qty == 0:
                    self._state_store.delete_option_position(contract_symbol)
                else:
                    self._state_store.upsert_option_position(
                        contract_symbol, position["underlying_symbol"], position["option_type"],
                        position["strike"], position["expiration"], real_qty,
                        detail["avg_entry_price"] if detail else position["avg_entry_price"],
                    )

    def _close_orphaned_option_legs(self, underlying: str) -> None:
        for pos in self._state_store.get_option_positions():
            if pos["underlying_symbol"] != underlying or pos["quantity"] == 0:
                continue
            detail = self._broker.get_position_detail(pos["contract_symbol"])
            if detail and abs(detail["qty"]) > 0:
                try:
                    self._broker.submit_option_order(
                        pos["contract_symbol"], side=Action.SELL if detail["qty"] > 0 else Action.BUY,
                        contracts=abs(int(detail["qty"])), limit_price=detail["current_price"],
                    )
                    logger.warning("Closed orphaned option leg %s", pos["contract_symbol"])
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to close orphaned leg %s: %s", pos["contract_symbol"], exc)

    # ── Circuit breaker management ────────────────────────────────────────────

    def _trip_agent(self, breaker: CircuitBreaker, reason: str) -> None:
        logger.error("[%s] CIRCUIT BREAKER TRIPPED (new entries halted): %s", breaker.name, reason)
        self._state_store.record_event(event_type="circuit_breaker_tripped", detail=f"[{breaker.name}] {reason}")
        breaker._tripped = True
        # Persist to DB so the Alpha Plane reads the tripped state without needing a restart
        self._state_store.set_breaker_state(breaker.name, "halted", "true")
        alerting.alert_circuit_breaker(reason=f"[{breaker.name}] {reason}")

    def _lock_in_profit(self, reason: str, breaker: CircuitBreaker | None = None) -> None:
        logger.info("PROFIT LOCK (new stock entries halted): %s", reason)
        self._state_store.record_event(event_type="daily_profit_target_reached", detail=reason)
        if breaker:
            self._state_store.set_breaker_state(breaker.name, "profit_locked", "true")
        try:
            alerting.alert_profit_locked(equity=self._broker.get_equity(), gain=0.0)
        except Exception:  # noqa: BLE001
            pass

    def _close_stocks_and_reconcile(self, reason: str) -> None:
        today = date.today()
        equity_positions = [p for p in self._state_store.get_positions() if p["quantity"] > 0]
        logger.info("STOCK-ONLY PROFIT LOCK: %s", reason)
        self._state_store.record_event(event_type="daily_profit_target_reached", detail=reason)
        for order in self._broker.get_open_orders():
            sym = order["symbol"]
            if sym is not None and parse_occ_symbol(sym) is None:
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

    def _wait_until_flat(self, ticker: str, timeout_seconds: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._broker.get_position_detail(ticker) is None:
                return
            time.sleep(0.5)
        logger.warning("%s: close_position did not confirm flat within %.1fs", ticker, timeout_seconds)

    def _close_all_and_reconcile(self, reason: str, event_type: str = "circuit_breaker_shutdown") -> None:
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
            raw_fill = self._broker.get_last_fill_price(contract_symbol)
            fill_price = max(raw_fill, 0.0) if raw_fill is not None else position["avg_entry_price"]
            self._state_store.record_realized_option_sale(
                contract_symbol=contract_symbol, underlying_symbol=position["underlying_symbol"],
                sale_date=today, contracts=position["quantity"], sale_price=fill_price,
                cost_basis=position["avg_entry_price"],
            )
            self._state_store.upsert_option_position(
                contract_symbol, position["underlying_symbol"], position["option_type"],
                position["strike"], position["expiration"], 0, position["avg_entry_price"],
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _exit_params_for(self, strategy: str) -> dict:
        if strategy in ("thesis", "recovery"):
            return {
                "stop_loss_pct": self._settings.thesis_stop_loss_pct,
                "take_profit_pct": None,
                "trailing_stop_pct": self._settings.thesis_trailing_stop_pct,
                "trailing_stop_activation_pct": self._settings.thesis_trailing_stop_activation_pct,
            }
        return {
            "stop_loss_pct": self._settings.exit_stop_loss_pct,
            "take_profit_pct": None,
            "trailing_stop_pct": self._settings.exit_trailing_stop_pct,
            "trailing_stop_activation_pct": self._settings.exit_trailing_stop_activation_pct,
        }

    def _compute_today_pnl(self, strategies: frozenset[str], include_options: bool = False) -> float:
        today_str = date.today().isoformat()
        pnl = 0.0
        for s in self._state_store.get_all_realized_sales(limit=200):
            if s["sale_date"] == today_str:
                pnl += s["realized_pnl"]
        if include_options:
            for s in self._state_store.get_all_realized_option_sales(limit=200):
                if s["sale_date"] == today_str:
                    pnl += s["realized_pnl"]
        try:
            broker_by_symbol = {p["symbol"]: p for p in self._broker.get_all_positions()}
            if not include_options:
                for pos in self._state_store.get_positions():
                    if (
                        pos.get("strategy", "momentum") in strategies
                        and pos["last_buy_at"]
                        and pos["last_buy_at"][:10] == today_str
                        and pos["ticker"] in broker_by_symbol
                    ):
                        bp = broker_by_symbol[pos["ticker"]]
                        pnl += (bp["current_price"] - bp["avg_entry_price"]) * bp["qty"]
            else:
                for pos in self._state_store.get_option_positions():
                    if (
                        pos.get("strategy", "orb_options") in strategies
                        and pos.get("opened_at")
                        and pos["opened_at"][:10] == today_str
                        and pos["contract_symbol"] in broker_by_symbol
                    ):
                        bp = broker_by_symbol[pos["contract_symbol"]]
                        pnl += (bp["current_price"] - bp["avg_entry_price"]) * bp["qty"] * 100
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch live positions for P&L computation: %s", exc)
        return pnl

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
        bracket_stop_order_id: str | None = None,
        stop_price: float | None = None,
    ) -> None:
        detail = self._broker.get_position_detail(ticker)
        shares = detail["qty"] if detail else 0.0
        avg_price = detail["avg_entry_price"] if detail else proposal.limit_price
        if proposal.action == Action.BUY:
            prior_position = self._state_store.get_position(ticker)
            prior_hwm = (prior_position or {}).get("high_water_mark") or avg_price
            self._state_store.upsert_position(
                ticker, int(shares), avg_price,
                last_buy_at=today.isoformat(),
                entry_regime=entry_regime,
                high_water_mark=max(prior_hwm, proposal.limit_price),
                strategy=strategy,
                stop_price=stop_price,
                bracket_stop_order_id=bracket_stop_order_id,
            )
            alerting.alert_buy(ticker=ticker, shares=int(shares), price=avg_price, strategy=strategy or "unknown")
        else:
            prior_position = self._state_store.get_position(ticker)
            if prior_position is not None:
                pnl = self._state_store.record_realized_sale(
                    ticker=ticker, sale_date=today, quantity=proposal.quantity,
                    sale_price=proposal.limit_price, cost_basis=prior_position["avg_entry_price"],
                )
                self._trigger_reflection(ticker, strategy, pnl)
            self._state_store.upsert_position(ticker, int(shares), avg_price)

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
        detail = self._broker.get_position_detail(contract_symbol)
        qty = detail["qty"] if detail else 0.0
        avg_price = detail["avg_entry_price"] if detail else (sale_price or 0.0)
        if action == Action.BUY:
            self._state_store.upsert_option_position(
                contract_symbol, underlying_symbol, option_type, strike, expiration,
                int(qty), avg_price, opened_at=today.isoformat(), strategy=strategy,
            )
            alerting.alert_option_buy(
                contract_symbol=contract_symbol, underlying=underlying_symbol,
                contracts=int(qty), premium=avg_price, strategy=strategy,
            )
        else:
            prior = self._state_store.get_option_position(contract_symbol)
            if prior is not None and sale_price is not None:
                self._state_store.record_realized_option_sale(
                    contract_symbol=contract_symbol, underlying_symbol=underlying_symbol,
                    sale_date=today, contracts=contracts, sale_price=sale_price,
                    cost_basis=prior["avg_entry_price"],
                )
            self._state_store.upsert_option_position(
                contract_symbol, underlying_symbol, option_type, strike, expiration,
                int(qty), avg_price, strategy=strategy,
            )

    def _trigger_reflection(self, ticker: str, strategy: str, pnl: float) -> None:
        threading.Thread(
            target=self._run_reflection,
            args=(ticker, strategy, pnl),
            daemon=True,
            name=f"reflect-{ticker}",
        ).start()

    def _run_reflection(self, ticker: str, strategy: str, pnl: float) -> None:
        try:
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

            buy_payload = next(
                (r["payload"] for r in runs if r["payload"].get("proposal", {}).get("action") == "BUY"),
                {},
            )
            regime_at_entry = self._state_store.get_entry_regime(ticker) or "unknown"
            market_context = {
                "strategy": strategy,
                "regime_at_entry": regime_at_entry,
                "entry_price": buy_payload.get("proposal", {}).get("limit_price"),
                "shares": buy_payload.get("proposal", {}).get("quantity"),
                "risk_verdict": buy_payload.get("risk_review", {}).get("verdict"),
                "risk_reasons": "; ".join(buy_payload.get("risk_review", {}).get("reasons", [])),
                "entry_date": next(
                    (r["created_at"] for r in runs if r["payload"].get("proposal", {}).get("action") == "BUY"),
                    "unknown",
                ),
                "realized_pnl": f"${pnl:+.2f}",
                "outcome": "win" if pnl > 0 else "loss",
            }

            reflection = self._reflection_agent.reflect(
                strategy=strategy, agent_signals=agent_signals,
                outcome_pnl=pnl, outcome_win=pnl > 0, market_context=market_context,
            )
            if reflection is None:
                return

            self._state_store.record_reflection(
                strategy=strategy, outcome_pnl=pnl, outcome_win=pnl > 0,
                what_happened=reflection.what_happened, root_cause=reflection.root_cause,
                outcome_was_noise=reflection.outcome_was_noise,
            )
            self._state_store.score_agent_signals(ticker, pnl)
            self._state_store.score_lesson_injections(ticker, pnl)
            if agent_signals:
                self._vw_bandit.learn(
                    track=strategy, regime=regime_at_entry, signals=agent_signals, pnl=pnl,
                )

            strategy_breaker = (
                self._intraday_breaker if strategy in _INTRADAY_STRATEGIES
                else self._options_breaker if strategy in _OPTIONS_STRATEGIES
                else self._thesis_breaker if strategy in _THESIS_STRATEGIES
                else self._swing_breaker
            )
            if hasattr(strategy_breaker, "record_trade_outcome"):
                strategy_breaker.record_trade_outcome(pnl > 0)

            if not reflection.outcome_was_noise:
                for lesson_out in reflection.lessons:
                    self._state_store.record_lesson(
                        lesson=lesson_out.lesson, setup_tags=lesson_out.setup_tags,
                        strategy=strategy, outcome_was_win=pnl > 0, source_pnl=pnl,
                    )
                logger.info(
                    "Reflection complete for %s (P&L $%+.2f): %d lesson(s) stored",
                    ticker, pnl, len(reflection.lessons),
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Post-trade reflection failed for %s: %s", ticker, exc)

    def _record_usage(self, agent_name: str, model: str, usage) -> None:
        self._state_store.record_token_usage(
            agent_name=agent_name, model=model,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        )
