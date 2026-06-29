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
from datetime import date, datetime
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
            option_cols = {row[1] for row in conn.execute("PRAGMA table_info(option_positions)")}
            if "strategy" not in option_cols:
                conn.execute("ALTER TABLE option_positions ADD COLUMN strategy TEXT NOT NULL DEFAULT 'orb_options'")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

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
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO positions
                    (ticker, quantity, avg_entry_price, last_buy_at, entry_regime, high_water_mark, strategy, stop_price, target_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 'momentum'), ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_entry_price = excluded.avg_entry_price,
                    last_buy_at = COALESCE(excluded.last_buy_at, positions.last_buy_at),
                    entry_regime = COALESCE(excluded.entry_regime, positions.entry_regime),
                    high_water_mark = COALESCE(excluded.high_water_mark, positions.high_water_mark),
                    strategy = COALESCE(?, positions.strategy),
                    stop_price = COALESCE(excluded.stop_price, positions.stop_price),
                    target_price = COALESCE(excluded.target_price, positions.target_price),
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
                    datetime.utcnow().isoformat(),
                    strategy,
                ),
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
                "SELECT ticker, quantity, avg_entry_price, last_buy_at, entry_regime, high_water_mark, strategy, stop_price, target_price, updated_at "
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
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO agent_lessons "
                "(lesson, setup_tags_json, strategy, outcome_was_win, source_pnl, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    lesson,
                    json.dumps(setup_tags),
                    strategy,
                    int(outcome_was_win),
                    source_pnl,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_lessons(self, strategy: str | None = None, limit: int = 200) -> list[dict]:
        with closing(self._connect()) as conn:
            if strategy:
                cursor = conn.execute(
                    "SELECT lesson, setup_tags_json, strategy, outcome_was_win, source_pnl, created_at "
                    "FROM agent_lessons WHERE strategy = ? ORDER BY id DESC LIMIT ?",
                    (strategy, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT lesson, setup_tags_json, strategy, outcome_was_win, source_pnl, created_at "
                    "FROM agent_lessons ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
        return [
            {
                "lesson": r[0],
                "setup_tags_json": r[1],
                "strategy": r[2],
                "outcome_was_win": bool(r[3]),
                "source_pnl": r[4],
                "created_at": r[5],
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
