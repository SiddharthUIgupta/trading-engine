# Engineering Suggestions

Findings from a full read of `execution_layer/runtime.py` and `execution_layer/scheduler.py`.
Each issue is rated by blast radius: **Critical** = money at risk or silent failure during market hours. **High** = operational gap. **Medium** = correctness or signal quality.

---

## 1. No Process Supervisor — Critical

**Problem:** The system runs as `nohup python main.py &`. If the Pi reboots, OOMs, or the process crashes, nothing restarts it. You find out when you check manually.

**Fix:** Create a systemd service.

```ini
# /etc/systemd/system/trading-engine.service
[Unit]
Description=Trading Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=sidgupta3391
WorkingDirectory=/home/sidgupta3391/trading-engine
ExecStart=/home/sidgupta3391/trading-engine/.venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/sidgupta3391/trading-engine/logs/trading_engine.log
StandardError=append:/home/sidgupta3391/trading-engine/logs/trading_engine.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable trading-engine
sudo systemctl start trading-engine
```

`Restart=always` + `RestartSec=10` means the process is back up in ~10 seconds after any crash.

---

## 2. No Alerting — Critical

**Problem:** Circuit breaker trips, unhandled exceptions, and process crashes are written to a log file on the Pi. There is no external notification. You are blind to real-time failures.

**Fix:** A Telegram bot takes ~30 lines and costs nothing. Add `send_alert(msg)` calls in `_trip_breaker`, `_lock_in_profit`, and a top-level exception handler in `main.py`.

```python
# execution_layer/alerting.py
import os
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_alert(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=5,
        )
    except Exception:
        pass  # alerting must never crash the trading process
```

Wire it into:
- `_trip_breaker()` — drawdown breach
- `_lock_in_profit()` — profit target hit
- `post_market_logging()` — daily P&L summary
- `main.py` top-level `except` — unhandled crash

Also add a daily 4:30pm heartbeat so silence itself is an alert.

---

## 3. Thesis Pending Payloads Lost on Crash — Critical

**Problem:** `_pending_payloads` is an in-memory dict. If the process crashes between the thesis scan at 8:15am and market open execution at 9:30am, all consensus-approved BUY decisions are lost. `market_open_execution` at 9:30 finds an empty dict and does nothing. The `run_history` table has the data but nothing reads it back.

**Current flow:**
```
8:15am  thesis_scan_and_trade → populates _pending_payloads → tries to execute (pre-market)
9:30am  market_open_execution → re-executes anything not yet filled
CRASH between 8:15-9:30 → _pending_payloads = {} on restart → 9:30 executes nothing
```

**Fix:** On `TradingRuntime.__init__`, re-hydrate `_pending_payloads` from today's `run_history` rows that have `action=BUY` and `verdict=APPROVED/AMENDED`, filtering to tickers not already in `_executed_tickers_today`. This is safe because `_executed_tickers_today` is already seeded from open positions, so tickers already filled won't be double-executed.

```python
# In __init__, after seeding _executed_tickers_today:
today_str = date.today().isoformat()
for run in state_store.get_run_history(limit=100):
    created = run.get("created_at", "")[:10]
    if created != today_str:
        continue
    payload = ConsensusPayload(**run["payload"])  # deserialize
    ticker = payload.proposal.ticker
    if payload.is_executable and ticker not in self._executed_tickers_today:
        self._pending_payloads[ticker] = payload
        self._pending_regimes[ticker] = run.get("regime", "unknown")
        self._pending_strategies[ticker] = run.get("strategy", "thesis")
```

---

## 4. Thread Safety on Shared State — High

**Problem:** `_pending_payloads`, `_scanned_tickers_today`, `_price_cache`, and `_vol_universe` are plain dicts/sets accessed from multiple APScheduler threads. No `max_instances=1` is set on any job, and no locks exist. If `thesis_scan_and_trade` runs long and overlaps with `momentum_scan_and_trade`, both write to `_scanned_tickers_today` concurrently.

**Fix — option A (minimal):** Add `max_instances=1` and `coalesce=True` to every job in `scheduler.py`. This makes APScheduler skip a new fire if the previous one is still running.

```python
scheduler.add_job(
    runtime.intraday_monitoring,
    trigger=CronTrigger(...),
    id="intraday_monitoring",
    misfire_grace_time=60,
    max_instances=1,   # add this
    coalesce=True,     # add this
)
```

**Fix — option B (correct):** Wrap shared state mutations in a `threading.Lock`. Necessary if any two jobs must legitimately run concurrently.

---

## 5. ORB Reflection Loop Is a Silent No-Op — High

**Problem:** `pre_close_orb_exit` calls `_trigger_reflection(ticker, "orb", pnl)` after each ORB close. But `_run_reflection` returns immediately at the `if not agent_signals` guard (line 1887) because ORB never goes through consensus and has no `run_history` BUY entry. Every ORB reflection is a confirmed no-op. The learning loop for your second-most-active track produces zero lessons.

**Fix:** Record a minimal ORB entry log when an ORB position is opened — enough for the reflection agent to work with. Either:

- Write a lightweight `run_history` entry at ORB open time (action=BUY, no agents, just the signal reasons), OR
- Modify `_run_reflection` to handle the ORB case without agent signals (just entry price, exit price, P&L, signal reasons stored in the position row).

The second is simpler: pass `signal_reasons` into `upsert_position` as a `notes` field, then surface them in the market_context dict for reflection even when `agent_signals` is empty.

---

## 6. Pre-Close ORB Exit Fragile at 3:30pm — High

**Problem:** `pre_close_orb_exit` has `misfire_grace_time=60`. If the Pi is restarting or slow at 3:30pm and takes more than 60 seconds, the job is silently skipped. Failing ORB positions then survive overnight. `_check_orb_exits` catches them next morning as "held past entry day" — but that's overnight gap risk on positions designed to be same-day closes.

**Fix:** Two changes:

1. Increase `misfire_grace_time` to 300 on `pre_close_orb_exit` — consistent with the other high-stakes jobs.
2. Add a startup check in `intraday_monitoring` that force-closes any ORB position with `last_buy_at != today`. This catches the overnight-survivor case during the first monitoring tick of the next day rather than waiting for `_check_orb_exits`.

---

## 7. `intraday_monitoring` Grace Time Too Short — Medium

**Problem:** `misfire_grace_time=60` on a 15-minute cron means a crash + restart taking more than 60 seconds silently skips that monitoring tick. No stop checks, no trailing stop updates, no drawdown check for that 15-minute window. During a fast-moving market this is material.

**Fix:** Increase to 300 seconds. A monitoring tick running 5 minutes late is better than a silent skip.

```python
scheduler.add_job(
    runtime.intraday_monitoring,
    trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone=EXCHANGE_TZ),
    id="intraday_monitoring",
    misfire_grace_time=300,  # was 60
    max_instances=1,
    coalesce=True,
)
```

---

## 8. Daily Profit Target Too Small — Medium

**Problem:** `DAILY_PROFIT_TARGET_USD=50` halts all new stock entries after $50 of gain, regardless of account size. On a $10K+ account, this is a 0.5% ceiling. The thesis track's average winner is +6.27% — the profit lock fires before most thesis trades have time to develop, cutting the upside the strategy was backtested to capture.

**Options:**
- Switch to `DAILY_PROFIT_TARGET_PCT` (percentage of equity) instead of a fixed dollar amount — e.g., 2% of equity.
- Or raise the absolute value to something proportional to your account size.

The current config is a leftover from early small-account testing. It's now a drag on the thesis strategy.

---

## 9. Startup Health Check Missing — Medium

**Problem:** `main.py` boots and starts the scheduler with no verification that the broker, data client, or database are actually reachable. A silent startup failure (wrong API key, network down, SQLite locked) causes the scheduler to start but every job to fail — nothing in the log makes it obvious the system was broken from the start.

**Fix:** Add a `startup_health_check()` call in `main.py` before `scheduler.start()` that:
1. Calls `broker.get_equity()` — confirms Alpaca connectivity and API keys
2. Calls `data_client.get_price_history("SPY", ...)` — confirms data layer
3. Calls `state_store.get_positions()` — confirms SQLite is writable
4. Sends a Telegram "Engine started" alert with equity snapshot

If any check fails, log and exit immediately rather than starting a broken scheduler.

---

## 10. `_wait_until_flat` Timeout Too Short — Low

**Problem:** `_wait_until_flat` polls for 5 seconds (`timeout_seconds=5.0`) to confirm a position closed. Used in `_close_stocks_and_reconcile` (profit lock path). Alpaca market orders typically fill in <1s, but limit orders near a wide spread can take longer. A 5-second timeout during a volatility spike could log a warning while the position is still open, leading to stale state.

**Fix:** Raise to 30 seconds, or poll until the close order status is `filled`/`canceled` via `broker.get_order_status()` rather than just checking if the position disappeared.

---

## Summary Table

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1 | No systemd service | Critical | 15 min |
| 2 | No alerting | Critical | 1 hour |
| 3 | Thesis payloads lost on crash | Critical | 2 hours |
| 4 | Thread safety on shared state | High | 30 min |
| 5 | ORB reflection no-op | High | 2 hours |
| 6 | Pre-close ORB exit fragile | High | 15 min |
| 7 | intraday_monitoring grace time | Medium | 5 min |
| 8 | Profit target too small | Medium | 5 min |
| 9 | No startup health check | Medium | 1 hour |
| 10 | _wait_until_flat timeout | Low | 15 min |
