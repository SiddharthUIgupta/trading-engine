"""SQLite-backed state persistence.

Tracks three things the runtime needs across restarts: open positions
(mirrors broker state for fast local reads), agent run history (every
ConsensusPayload, for audit/replay), and per-call token/cost logs (so
LLM spend is visible without going to the Anthropic dashboard).

Plain stdlib sqlite3, no ORM — this is a single-writer process and the
schema is small enough that an ORM would be pure overhead.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

from analyst_layer.schemas import ConsensusPayload

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL,
    avg_entry_price REAL NOT NULL,
    last_buy_at TEXT,
    entry_regime TEXT,
    high_water_mark REAL,
    strategy TEXT NOT NULL DEFAULT 'momentum',
    stop_price REAL,
    target_price REAL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS realized_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    sale_date TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    sale_price REAL NOT NULL,
    cost_basis REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    is_executable INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS option_positions (
    contract_symbol TEXT PRIMARY KEY,
    underlying_symbol TEXT NOT NULL,
    option_type TEXT NOT NULL,
    strike REAL NOT NULL,
    expiration TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_entry_price REAL NOT NULL,
    opened_at TEXT,
    strategy TEXT NOT NULL DEFAULT 'orb_options',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS realized_option_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_symbol TEXT NOT NULL,
    underlying_symbol TEXT NOT NULL,
    sale_date TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    sale_price REAL NOT NULL,
    cost_basis REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson TEXT NOT NULL,
    setup_tags_json TEXT NOT NULL DEFAULT '[]',
    strategy TEXT NOT NULL,
    outcome_was_win INTEGER NOT NULL,
    source_pnl REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    outcome_pnl REAL NOT NULL,
    outcome_win INTEGER NOT NULL,
    what_happened TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    outcome_was_noise INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_signal_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    track TEXT NOT NULL,
    regime TEXT NOT NULL DEFAULT 'neutral',
    proposed_action TEXT NOT NULL,
    outcome TEXT,
    outcome_pnl REAL,
    scored_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_signal_detail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id INTEGER NOT NULL REFERENCES agent_signal_log(id),
    agent_name TEXT NOT NULL,
    stance TEXT NOT NULL,
    confidence TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lesson_injection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL REFERENCES agent_lessons(id),
    ticker TEXT NOT NULL,
    track TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    buys_placed INTEGER NOT NULL DEFAULT 0,
    candidates_screened INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(session_date, strategy)
);

CREATE TABLE IF NOT EXISTS order_intents (
    client_order_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    limit_price REAL NOT NULL,
    stop_price REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS breaker_state (
    breaker_name TEXT NOT NULL,
    state_key TEXT NOT NULL,
    state_value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (breaker_name, state_key)
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    features_json TEXT NOT NULL DEFAULT '{}',
    screen_score REAL,
    llm_verdict TEXT,
    llm_mode TEXT NOT NULL DEFAULT 'gating',
    gate_result TEXT,
    traded INTEGER NOT NULL DEFAULT 0,
    fill_ref TEXT,
    fwd_ret_1d REAL,
    fwd_ret_5d REAL,
    fwd_ret_21d REAL,
    fwd_ret_63d REAL,
    hit_stop_within_21d INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_date, strategy, ticker)
);
"""


def _row_to_position(row) -> dict:
    return {
        "ticker": row[0],
        "quantity": row[1],
        "avg_entry_price": row[2],
        "last_buy_at": row[3],
        "entry_regime": row[4],
        "high_water_mark": row[5] if row[5] is not None else row[2],
        "strategy": row[6],
        "stop_price": row[7],
        "target_price": row[8],
        "updated_at": row[9],
        "bracket_stop_order_id": row[10] if len(row) > 10 else None,
    }


def _row_to_option_position(row) -> dict:
    return {
        "contract_symbol": row[0],
        "underlying_symbol": row[1],
        "option_type": row[2],
        "strike": row[3],
        "expiration": row[4],
        "quantity": row[5],
        "avg_entry_price": row[6],
        "opened_at": row[7],
        "strategy": row[8],
        "updated_at": row[9],
    }


class StateStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            # CREATE TABLE IF NOT EXISTS doesn't add columns to an already-
            # existing table — needed for any database created before
            # stop_price/target_price (the ORB equity track) existed.
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
            for column in ("stop_price", "target_price"):
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE positions ADD COLUMN {column} REAL")
            if "bracket_stop_order_id" not in existing_columns:
                conn.execute("ALTER TABLE positions ADD COLUMN bracket_stop_order_id TEXT")
            option_cols = {row[1] for row in conn.execute("PRAGMA table_info(option_positions)")}
            if "strategy" not in option_cols:
                conn.execute("ALTER TABLE option_positions ADD COLUMN strategy TEXT NOT NULL DEFAULT 'orb_options'")
            lesson_cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_lessons)")}
            if "score" not in lesson_cols:
                conn.execute("ALTER TABLE agent_lessons ADD COLUMN score REAL NOT NULL DEFAULT 1.0")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """timeout=30.0 + WAL: Alpha and Protection are separate processes
        writing the same file. The default rollback journal has a 0-second
        busy timeout, so any write-write overlap between the two raises
        `database is locked` immediately rather than waiting. WAL allows one
        writer concurrent with readers and, combined with the timeout, makes
        a same-tick collision wait and retry instead of throwing.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def upsert_position(
        self,
        ticker: str,
        quantity: int,
        avg_entry_price: float,
        last_buy_at: str | None = None,
        entry_regime: str | None = None,
        high_water_mark: float | None = None,
        strategy: str | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
        bracket_stop_order_id: str | None = None,
    ) -> None:
        """`last_buy_at`/`entry_regime`/`high_water_mark`/`strategy`/
        `stop_price`/`target_price` should only be passed when this update
        follows a BUY fill (resetting the trailing-stop baseline, wash-sale
        lookback, which track's exit rules apply, and — ORB only — the
        fixed stop/target levels from the opening range). A SELL-driven
        update passes None for all of these, which preserves whatever was
        already on file. `strategy` defaults to 'momentum' on a brand-new
        position if never specified. stop_price/target_price are NULL for
        momentum/thesis positions — only the ORB track uses them.
        `bracket_stop_order_id` is the Alpaca order ID of the resting stop leg;
        stored so we can amend it as the trailing stop moves up.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO positions
                    (ticker, quantity, avg_entry_price, last_buy_at, entry_regime, high_water_mark, strategy, stop_price, target_price, bracket_stop_order_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 'momentum'), ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_entry_price = excluded.avg_entry_price,
                    last_buy_at = COALESCE(excluded.last_buy_at, positions.last_buy_at),
                    entry_regime = COALESCE(excluded.entry_regime, positions.entry_regime),
                    high_water_mark = COALESCE(excluded.high_water_mark, positions.high_water_mark),
                    strategy = COALESCE(?, positions.strategy),
                    stop_price = COALESCE(excluded.stop_price, positions.stop_price),
                    target_price = COALESCE(excluded.target_price, positions.target_price),
                    bracket_stop_order_id = COALESCE(excluded.bracket_stop_order_id, positions.bracket_stop_order_id),
                    updated_at = excluded.updated_at
                """,
                (
                    ticker,
                    quantity,
                    avg_entry_price,
                    last_buy_at,
                    entry_regime,
                    high_water_mark,
                    strategy,
                    stop_price,
                    target_price,
                    bracket_stop_order_id,
                    datetime.utcnow().isoformat(),
                    strategy,
                ),
            )
            conn.commit()

    def update_broker_stop(self, ticker: str, stop_price: float) -> None:
        """Update the local stop_price record after we amend the resting broker stop.
        Called each time the trailing stop moves up so the DB reflects the current
        broker stop level.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE positions SET stop_price=?, updated_at=? WHERE ticker=?",
                (stop_price, datetime.utcnow().isoformat(), ticker),
            )
            conn.commit()

    def clear_bracket_stop(self, ticker: str) -> None:
        """Clears the resting-stop reference after the stop leg has been
        cancelled — e.g. right before a software-driven exit sells the
        position outright. Uses a plain UPDATE (not upsert_position's
        COALESCE) because COALESCE can never write a column back to NULL.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE positions SET bracket_stop_order_id=NULL, updated_at=? WHERE ticker=?",
                (datetime.utcnow().isoformat(), ticker),
            )
            conn.commit()

    def update_high_water_mark(self, ticker: str, current_price: float) -> None:
        """Bumps the trailing-stop peak. Self-healing against a NULL/unset
        high_water_mark (e.g. a position opened before this column existed)
        by falling back to avg_entry_price as the floor.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE positions SET high_water_mark = MAX(COALESCE(high_water_mark, avg_entry_price), ?) "
                "WHERE ticker = ?",
                (current_price, ticker),
            )
            conn.commit()

    def get_positions(self) -> list[dict]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT ticker, quantity, avg_entry_price, last_buy_at, entry_regime, high_water_mark, strategy, stop_price, target_price, updated_at, bracket_stop_order_id "
                "FROM positions"
            )
            rows = cursor.fetchall()
        return [_row_to_position(r) for r in rows]

    def get_position(self, ticker: str) -> dict | None:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT ticker, quantity, avg_entry_price, last_buy_at, entry_regime, high_water_mark, strategy, stop_price, target_price, updated_at "
                "FROM positions WHERE ticker = ?",
                (ticker,),
            )
            row = cursor.fetchone()
        return None if row is None else _row_to_position(row)

    def delete_position(self, ticker: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
            conn.commit()

    def delete_option_position(self, contract_symbol: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM option_positions WHERE contract_symbol = ?", (contract_symbol,))
            conn.commit()

    def upsert_option_position(
        self,
        contract_symbol: str,
        underlying_symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        quantity: int,
        avg_entry_price: float,
        opened_at: str | None = None,
        strategy: str = "orb_options",
    ) -> None:
        """`opened_at` and `strategy` should only be passed on the fill that first
        opens this contract — COALESCE preserves both values across any later
        fills (e.g. partial closes) the same way equity positions preserve
        `last_buy_at`.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO option_positions
                    (contract_symbol, underlying_symbol, option_type, strike, expiration, quantity, avg_entry_price, opened_at, strategy, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(contract_symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_entry_price = excluded.avg_entry_price,
                    opened_at = COALESCE(option_positions.opened_at, excluded.opened_at),
                    strategy = COALESCE(option_positions.strategy, excluded.strategy),
                    updated_at = excluded.updated_at
                """,
                (
                    contract_symbol, underlying_symbol, option_type, strike, expiration,
                    quantity, avg_entry_price, opened_at, strategy, datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_option_positions(self) -> list[dict]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT contract_symbol, underlying_symbol, option_type, strike, expiration, "
                "quantity, avg_entry_price, opened_at, strategy, updated_at FROM option_positions"
            )
            rows = cursor.fetchall()
        return [_row_to_option_position(r) for r in rows]

    def get_option_position(self, contract_symbol: str) -> dict | None:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT contract_symbol, underlying_symbol, option_type, strike, expiration, "
                "quantity, avg_entry_price, opened_at, strategy, updated_at FROM option_positions WHERE contract_symbol = ?",
                (contract_symbol,),
            )
            row = cursor.fetchone()
        return None if row is None else _row_to_option_position(row)

    def record_realized_option_sale(
        self, contract_symbol: str, underlying_symbol: str, sale_date: date, contracts: int, sale_price: float, cost_basis: float
    ) -> float:
        """Mirrors record_realized_sale, but each contract controls 100
        shares — realized P&L must reflect that multiplier to be a real
        dollar figure, not just a per-share-equivalent difference.
        """
        realized_pnl = (sale_price - cost_basis) * contracts * 100
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO realized_option_sales "
                "(contract_symbol, underlying_symbol, sale_date, contracts, sale_price, cost_basis, realized_pnl, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    contract_symbol, underlying_symbol, sale_date.isoformat(), contracts,
                    sale_price, cost_basis, realized_pnl, datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return realized_pnl

    def get_all_realized_option_sales(self, limit: int = 200) -> list[dict]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT contract_symbol, underlying_symbol, sale_date, contracts, sale_price, cost_basis, realized_pnl, created_at "
                "FROM realized_option_sales ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {
                "contract_symbol": r[0], "underlying_symbol": r[1], "sale_date": r[2], "contracts": r[3],
                "sale_price": r[4], "cost_basis": r[5], "realized_pnl": r[6], "created_at": r[7],
            }
            for r in rows
        ]

    def record_realized_sale(
        self, ticker: str, sale_date: date, quantity: int, sale_price: float, cost_basis: float
    ) -> float:
        """Records a closing trade and returns the realized P&L. This is the
        ledger the wash-sale guard reads to decide whether a future BUY of
        the same ticker would disallow a loss just taken.
        """
        realized_pnl = (sale_price - cost_basis) * quantity
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO realized_sales (ticker, sale_date, quantity, sale_price, cost_basis, realized_pnl, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticker, sale_date.isoformat(), quantity, sale_price, cost_basis, realized_pnl, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return realized_pnl

    def get_recent_loss_sales(self, ticker: str, since: date) -> list[dict]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT ticker, sale_date, quantity, sale_price, cost_basis, realized_pnl FROM realized_sales "
                "WHERE ticker = ? AND sale_date >= ? AND realized_pnl < 0 ORDER BY sale_date DESC",
                (ticker, since.isoformat()),
            )
            rows = cursor.fetchall()
        return [
            {
                "ticker": r[0],
                "sale_date": r[1],
                "quantity": r[2],
                "sale_price": r[3],
                "cost_basis": r[4],
                "realized_pnl": r[5],
            }
            for r in rows
        ]

    def get_pnl_history(self, limit: int = 500) -> list[float]:
        """Return realized P&L values (most recent first) for Kelly sizing.

        Portfolio-wide, no strategy filter — more data gives a more reliable
        Kelly estimate than splitting by strategy with few trades each.
        """
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT realized_pnl FROM realized_sales ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_all_realized_sales(self, limit: int = 200) -> list[dict]:
        """Full trade history (wins and losses) — `get_recent_loss_sales`
        only returns losses (it's the wash-sale guard's input), this is
        for display/reporting.
        """
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT ticker, sale_date, quantity, sale_price, cost_basis, realized_pnl, created_at "
                "FROM realized_sales ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {
                "ticker": r[0],
                "sale_date": r[1],
                "quantity": r[2],
                "sale_price": r[3],
                "cost_basis": r[4],
                "realized_pnl": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    def get_events(self, event_type_like: str | None = None, limit: int = 200) -> list[dict]:
        with closing(self._connect()) as conn:
            if event_type_like:
                cursor = conn.execute(
                    "SELECT event_type, detail, created_at FROM events WHERE event_type LIKE ? "
                    "ORDER BY id DESC LIMIT ?",
                    (event_type_like, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT event_type, detail, created_at FROM events ORDER BY id DESC LIMIT ?", (limit,)
                )
            rows = cursor.fetchall()
        return [{"event_type": r[0], "detail": r[1], "created_at": r[2]} for r in rows]

    def record_run(self, payload: ConsensusPayload) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO run_history (ticker, payload_json, is_executable, created_at) VALUES (?, ?, ?, ?)",
                (
                    payload.ticker,
                    payload.model_dump_json(),
                    int(payload.is_executable),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def record_token_usage(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO token_usage (agent_name, model, input_tokens, output_tokens, "
                "cache_creation_input_tokens, cache_read_input_tokens, estimated_cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_name,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_creation_input_tokens,
                    cache_read_input_tokens,
                    estimated_cost_usd,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_cost_summary(self, since: date | None = None) -> dict:
        """Aggregate spend for visibility into "where did my Claude money go" —
        total cost plus a per-agent breakdown, optionally scoped to a date.
        """
        with closing(self._connect()) as conn:
            where = "WHERE created_at >= ?" if since else ""
            params = (since.isoformat(),) if since else ()

            total_row = conn.execute(
                f"SELECT COALESCE(SUM(estimated_cost_usd), 0), COALESCE(SUM(input_tokens), 0), "
                f"COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cache_creation_input_tokens), 0), "
                f"COALESCE(SUM(cache_read_input_tokens), 0), COUNT(*) FROM token_usage {where}",
                params,
            ).fetchone()

            by_agent_rows = conn.execute(
                f"SELECT agent_name, COUNT(*), SUM(estimated_cost_usd), SUM(input_tokens), SUM(output_tokens) "
                f"FROM token_usage {where} GROUP BY agent_name ORDER BY SUM(estimated_cost_usd) DESC",
                params,
            ).fetchall()

        return {
            "total_cost_usd": total_row[0],
            "total_input_tokens": total_row[1],
            "total_output_tokens": total_row[2],
            "total_cache_creation_input_tokens": total_row[3],
            "total_cache_read_input_tokens": total_row[4],
            "total_calls": total_row[5],
            "by_agent": [
                {
                    "agent_name": r[0],
                    "calls": r[1],
                    "cost_usd": r[2],
                    "input_tokens": r[3],
                    "output_tokens": r[4],
                }
                for r in by_agent_rows
            ],
        }

    def has_intraday_escalation_today(self, ticker: str, today: date) -> bool:
        """Rate limit for the LLM exit-review escalation path — at most one
        per position per day, regardless of how many intraday ticks fire.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE event_type = ? AND created_at >= ? LIMIT 1",
                (f"intraday_llm_exit_escalation:{ticker}", today.isoformat()),
            ).fetchone()
        return row is not None

    def record_event(self, event_type: str, detail: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO events (event_type, detail, created_at) VALUES (?, ?, ?)",
                (event_type, detail, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_run_history(self, ticker: str | None = None, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            if ticker:
                cursor = conn.execute(
                    "SELECT payload_json, is_executable, created_at FROM run_history "
                    "WHERE ticker = ? ORDER BY id DESC LIMIT ?",
                    (ticker, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT payload_json, is_executable, created_at FROM run_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
        return [
            {"payload": json.loads(r[0]), "payload_json": r[0], "is_executable": bool(r[1]), "created_at": r[2]} for r in rows
        ]

    # ── Agent learning: lessons + reflections ─────────────────────────────────

    def record_lesson(
        self,
        lesson: str,
        setup_tags: list[str],
        strategy: str,
        outcome_was_win: bool,
        source_pnl: float,
    ) -> int:
        """Returns the inserted lesson id for injection tracking."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO agent_lessons "
                "(lesson, setup_tags_json, strategy, outcome_was_win, source_pnl, score, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1.0, ?)",
                (
                    lesson,
                    json.dumps(setup_tags),
                    strategy,
                    int(outcome_was_win),
                    source_pnl,
                    datetime.utcnow().isoformat(),
                ),
            )
            lesson_id = cursor.lastrowid
            conn.commit()
        return lesson_id

    def get_lessons(self, strategy: str | None = None, limit: int = 200) -> list[dict]:
        with closing(self._connect()) as conn:
            if strategy:
                cursor = conn.execute(
                    "SELECT id, lesson, setup_tags_json, strategy, outcome_was_win, source_pnl, "
                    "COALESCE(score, 1.0), created_at "
                    "FROM agent_lessons WHERE strategy = ? ORDER BY id DESC LIMIT ?",
                    (strategy, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT id, lesson, setup_tags_json, strategy, outcome_was_win, source_pnl, "
                    "COALESCE(score, 1.0), created_at "
                    "FROM agent_lessons ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "lesson": r[1],
                "setup_tags_json": r[2],
                "strategy": r[3],
                "outcome_was_win": bool(r[4]),
                "source_pnl": r[5],
                "score": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]

    def record_reflection(
        self,
        strategy: str,
        outcome_pnl: float,
        outcome_win: bool,
        what_happened: str,
        root_cause: str,
        outcome_was_noise: bool,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO trade_reflections "
                "(strategy, outcome_pnl, outcome_win, what_happened, root_cause, outcome_was_noise, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    strategy,
                    outcome_pnl,
                    int(outcome_win),
                    what_happened,
                    root_cause,
                    int(outcome_was_noise),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    # ── Agent performance tracking ────────────────────────────────────────────

    def record_agent_signal_log(
        self, ticker: str, track: str, regime: str, proposed_action: str, signals: list[dict]
    ) -> int:
        """Persist signals at consensus time for later scoring. Returns log_id."""
        now = datetime.utcnow().isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO agent_signal_log (ticker, track, regime, proposed_action, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, track, regime, proposed_action, now),
            )
            log_id = cursor.lastrowid
            for s in signals:
                conn.execute(
                    "INSERT INTO agent_signal_detail (log_id, agent_name, stance, confidence) "
                    "VALUES (?, ?, ?, ?)",
                    (log_id, s.get("agent_name", "unknown"), s.get("stance", "HOLD"), s.get("confidence", "low")),
                )
            conn.commit()
        return log_id

    def get_entry_regime(self, ticker: str) -> str | None:
        """Return the regime recorded at the most recent BUY signal log for this ticker."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT regime FROM agent_signal_log WHERE ticker=? AND proposed_action='BUY' "
                "ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        return row[0] if row else None

    def score_agent_signals(self, ticker: str, pnl: float) -> None:
        """Mark all unscored signal logs for this ticker with the realized outcome."""
        outcome = "win" if pnl > 0 else "loss"
        now = datetime.utcnow().isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE agent_signal_log SET outcome=?, outcome_pnl=?, scored_at=? "
                "WHERE ticker=? AND outcome IS NULL",
                (outcome, pnl, now, ticker),
            )
            conn.commit()

    def get_agent_accuracy(self, track: str, regime: str) -> list[tuple[str, int, int]]:
        """Return (agent_name, total_scored, wins) for the given track + regime."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT d.agent_name,
                       COUNT(*) AS total,
                       SUM(CASE WHEN l.outcome='win' THEN 1 ELSE 0 END) AS wins
                FROM agent_signal_detail d
                JOIN agent_signal_log l ON l.id = d.log_id
                WHERE l.track=? AND l.regime=? AND l.outcome IS NOT NULL
                GROUP BY d.agent_name
                """,
                (track, regime),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_scored_signal_logs(self, limit: int = 2000) -> list[dict]:
        """Return all scored signal logs with per-agent details, for VW warm-start.

        Each dict has: track, regime, outcome_pnl, signals (list of agent dicts).
        Only returns logs that have been scored (have a non-null outcome_pnl).
        """
        with closing(self._connect()) as conn:
            log_rows = conn.execute(
                "SELECT id, track, regime, outcome_pnl FROM agent_signal_log "
                "WHERE outcome IS NOT NULL AND outcome_pnl IS NOT NULL "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
            if not log_rows:
                return []
            log_by_id = {
                row[0]: {"track": row[1], "regime": row[2], "outcome_pnl": row[3], "signals": []}
                for row in log_rows
            }
            placeholders = ",".join("?" * len(log_by_id))
            detail_rows = conn.execute(
                f"SELECT log_id, agent_name, stance, confidence FROM agent_signal_detail "
                f"WHERE log_id IN ({placeholders})",
                list(log_by_id.keys()),
            ).fetchall()
            for log_id, agent_name, stance, confidence in detail_rows:
                if log_id in log_by_id:
                    log_by_id[log_id]["signals"].append(
                        {"agent_name": agent_name, "stance": stance, "confidence": confidence}
                    )
        return list(log_by_id.values())

    # ── Lesson validation ─────────────────────────────────────────────────────

    def record_lesson_injection(self, lesson_id: int, ticker: str, track: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO lesson_injection_log (lesson_id, ticker, track, created_at) VALUES (?, ?, ?, ?)",
                (lesson_id, ticker, track, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def score_lesson_injections(self, ticker: str, pnl: float) -> None:
        """Update scores of lessons recently injected for this ticker.

        Win: +0.1 (lesson appears predictive)
        Loss: -0.05 (asymmetric — good lessons don't guarantee every trade wins)
        Scores are clamped to [0.0, 2.0].
        """
        delta = 0.1 if pnl > 0 else -0.05
        with closing(self._connect()) as conn:
            lesson_ids = [
                row[0] for row in conn.execute(
                    "SELECT DISTINCT lesson_id FROM lesson_injection_log "
                    "WHERE ticker=? ORDER BY id DESC LIMIT 50",
                    (ticker,),
                ).fetchall()
            ]
            for lid in lesson_ids:
                conn.execute(
                    "UPDATE agent_lessons SET score = MAX(0.0, MIN(2.0, COALESCE(score, 1.0) + ?)) WHERE id=?",
                    (delta, lid),
                )
            conn.commit()

    def update_lesson_score(self, lesson_id: int, delta: float) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE agent_lessons SET score = MAX(0.0, MIN(2.0, COALESCE(score, 1.0) + ?)) WHERE id=?",
                (delta, lesson_id),
            )
            conn.commit()

    def write_order_intent(
        self,
        client_order_id: str,
        strategy: str,
        ticker: str,
        action: str,
        quantity: int,
        limit_price: float,
        stop_price: float | None = None,
    ) -> None:
        """Alpha Plane writes a BUY/SELL intent. Protection Plane consumes it."""
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO order_intents
                       (client_order_id, strategy, ticker, action, quantity, limit_price, stop_price, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (client_order_id, strategy, ticker, action, quantity, limit_price, stop_price, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def expire_stale_order_intents(self, max_age_hours: float = 4.0) -> int:
        """Marks pending order_intents older than max_age_hours as 'expired'
        instead of letting them execute a stale decision — e.g. an Alpha BUY
        intent written Friday morning that Protection doesn't get to until
        Monday because both processes were down over the weekend. Returns the
        number of rows expired. Called at the top of consume_order_intents,
        before get_pending_order_intents(), so expired rows are never even
        read as candidates for submission.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "UPDATE order_intents SET status='expired', processed_at=? "
                "WHERE status='pending' AND created_at < ?",
                (datetime.utcnow().isoformat(), cutoff),
            )
            conn.commit()
            return cursor.rowcount

    def get_pending_order_intents(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT client_order_id, strategy, ticker, action, quantity, limit_price, stop_price, created_at "
                "FROM order_intents WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        return [
            {"client_order_id": r[0], "strategy": r[1], "ticker": r[2], "action": r[3],
             "quantity": r[4], "limit_price": r[5], "stop_price": r[6], "created_at": r[7]}
            for r in rows
        ]

    def mark_order_intent_processed(self, client_order_id: str, status: str = "submitted") -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE order_intents SET status=?, processed_at=? WHERE client_order_id=?",
                (status, datetime.utcnow().isoformat(), client_order_id),
            )
            conn.commit()

    def set_breaker_state(self, breaker_name: str, key: str, value: str) -> None:
        """Protection Plane writes breaker state. Alpha Plane reads it before queuing intents."""
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO breaker_state (breaker_name, state_key, state_value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(breaker_name, state_key) DO UPDATE SET
                       state_value=excluded.state_value,
                       updated_at=excluded.updated_at""",
                (breaker_name, key, value, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_breaker_state(self, breaker_name: str, key: str, default: str = "false") -> str:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state_value FROM breaker_state WHERE breaker_name=? AND state_key=?",
                (breaker_name, key),
            ).fetchone()
        return row[0] if row else default

    def is_breaker_halted(self, breaker_name: str) -> bool:
        return self.get_breaker_state(breaker_name, "halted", "false") == "true"

    def log_candidate(
        self,
        candidate_date: date,
        strategy: str,
        ticker: str,
        llm_verdict: str,
        gate_result: str,
        traded: bool,
        features: dict | None = None,
        screen_score: float | None = None,
        fill_ref: str | None = None,
    ) -> None:
        """Log every screened candidate, whether traded or not.
        Upserts on (date, strategy, ticker) — re-running a scan on the same day
        updates rather than duplicates.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO candidates
                       (candidate_date, strategy, ticker, features_json, screen_score,
                        llm_verdict, gate_result, traded, fill_ref, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(candidate_date, strategy, ticker) DO UPDATE SET
                       llm_verdict=excluded.llm_verdict,
                       gate_result=excluded.gate_result,
                       traded=excluded.traded,
                       fill_ref=excluded.fill_ref,
                       features_json=excluded.features_json,
                       screen_score=excluded.screen_score""",
                (
                    candidate_date.isoformat(),
                    strategy,
                    ticker,
                    json.dumps(features or {}),
                    screen_score,
                    llm_verdict,
                    gate_result,
                    1 if traded else 0,
                    fill_ref,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_candidates_needing_forward_returns(self, days_ago: int) -> list[dict]:
        """Returns candidates from exactly `days_ago` trading days ago that
        don't yet have the corresponding forward return filled in.
        Used by the nightly forward-return backfill job.
        """
        col = {1: "fwd_ret_1d", 5: "fwd_ret_5d", 21: "fwd_ret_21d", 63: "fwd_ret_63d"}.get(days_ago)
        if col is None:
            return []
        cutoff = date.today() - __import__("datetime").timedelta(days=days_ago)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT id, ticker FROM candidates WHERE candidate_date=? AND {col} IS NULL",
                (cutoff.isoformat(),),
            ).fetchall()
        return [{"id": r[0], "ticker": r[1]} for r in rows]

    def update_candidate_forward_return(self, candidate_id: int, days: int, ret: float) -> None:
        col = {1: "fwd_ret_1d", 5: "fwd_ret_5d", 21: "fwd_ret_21d", 63: "fwd_ret_63d"}.get(days)
        if col is None:
            return
        with closing(self._connect()) as conn:
            conn.execute(f"UPDATE candidates SET {col}=? WHERE id=?", (ret, candidate_id))
            conn.commit()

    def record_scan_session(
        self,
        session_date: date,
        strategy: str,
        buys_placed: int,
        candidates_screened: int = 0,
    ) -> None:
        """Records one thesis/swing/intraday scan session outcome.
        Upserts so a re-run on the same day overwrites rather than duplicates.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO scan_sessions (session_date, strategy, buys_placed, candidates_screened, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_date, strategy) DO UPDATE SET
                       buys_placed=excluded.buys_placed,
                       candidates_screened=excluded.candidates_screened""",
                (session_date.isoformat(), strategy, buys_placed, candidates_screened, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_zero_buy_streak(self, strategy: str, lookback_days: int = 10) -> int:
        """Returns the number of consecutive most-recent sessions with 0 buys placed.
        Looks at the last `lookback_days` recorded sessions for this strategy.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT buys_placed FROM scan_sessions
                   WHERE strategy=?
                   ORDER BY session_date DESC
                   LIMIT ?""",
                (strategy, lookback_days),
            ).fetchall()
        streak = 0
        for (buys,) in rows:
            if buys == 0:
                streak += 1
            else:
                break
        return streak
