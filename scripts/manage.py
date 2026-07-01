#!/usr/bin/env python3
"""Trading engine management CLI using zcmd.

Usage (from repo root, venv active):
    python scripts/manage.py status        # service + process status
    python scripts/manage.py start         # start via systemd
    python scripts/manage.py stop          # stop via systemd
    python scripts/manage.py restart       # restart via systemd
    python scripts/manage.py logs [N]      # tail last N lines (default 50)
    python scripts/manage.py follow        # live log tail (Ctrl-C to stop)
    python scripts/manage.py ps            # show trading-engine process info
    python scripts/manage.py equity        # quick equity / PnL summary from DB
    python scripts/manage.py positions     # open positions
    python scripts/manage.py backtest      # run swing backtest (2 years)
    python scripts/manage.py warmup        # re-run VW bandit warmup
    python scripts/manage.py kill          # kill stale background instances
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
VENV   = ROOT / ".venv" / "bin" / "python"
LOG    = ROOT / "logs" / "trading_engine.log"
DB     = ROOT / "state" / "trading_engine.sqlite3"
SVC    = "trading-engine"

try:
    from zcmd import zcmd
    _ZCMD_OK = True
except ImportError:
    _ZCMD_OK = False


def _run(cmd: str, print_output: bool = True) -> list[str]:
    """Run a shell command via zcmd (or os.system fallback)."""
    if _ZCMD_OK:
        out = zcmd.run(cmd, b_output=True, b_print=print_output)
        return out or []
    else:
        os.system(cmd)
        return []


def _run_quiet(cmd: str) -> list[str]:
    if _ZCMD_OK:
        return zcmd.run(cmd, b_output=True, b_print=False) or []
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.splitlines()


# ─────────────────────────────────────────────────────────────────────────────

def cmd_status() -> None:
    print("=== Service status ===")
    _run(f"systemctl status {SVC} --no-pager -l 2>/dev/null || echo 'systemd not available'")
    print("\n=== Process check ===")
    lines = _run_quiet("pgrep -af 'python.*main.py'")
    if lines:
        for l in lines:
            if l.strip():
                print("  RUNNING:", l.strip())
    else:
        print("  No trading-engine process found")


def cmd_start() -> None:
    print(f"Starting {SVC}...")
    _run(f"sudo systemctl start {SVC}")
    _run(f"systemctl is-active {SVC}")


def cmd_stop() -> None:
    print(f"Stopping {SVC}...")
    _run(f"sudo systemctl stop {SVC}")


def cmd_restart() -> None:
    print(f"Restarting {SVC}...")
    _run(f"sudo systemctl restart {SVC}")
    import time; time.sleep(2)
    _run(f"systemctl is-active {SVC}")


def cmd_logs(n: int = 50) -> None:
    if LOG.exists():
        _run(f"tail -n {n} {LOG}")
    else:
        print(f"Log file not found: {LOG}")


def cmd_follow() -> None:
    if LOG.exists():
        _run(f"tail -f {LOG}")
    else:
        print(f"Log file not found: {LOG}")


def cmd_ps() -> None:
    lines = _run_quiet("pgrep -af 'python.*main.py'")
    if not lines:
        print("No trading-engine process running")
        return
    for pid_line in lines:
        if not pid_line.strip():
            continue
        pid = pid_line.strip().split()[0]
        _run(f"ps -p {pid} -o pid,ppid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null")


def cmd_kill() -> None:
    lines = _run_quiet("pgrep -f 'python.*main.py'")
    pids = [l.strip() for l in lines if l.strip().isdigit()]
    if not pids:
        print("No stale processes found")
        return
    print(f"Found PIDs: {pids}")
    for pid in pids:
        _run(f"kill {pid}")
        print(f"  Killed {pid}")


def cmd_equity() -> None:
    if not DB.exists():
        print(f"DB not found: {DB}")
        return
    conn = sqlite3.connect(DB)
    try:
        # Realized PnL
        r = conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(realized_pnl), 0)
            FROM realized_sales
        """).fetchone()
        trade_count, total_pnl = r[0], r[1]

        # Win rate
        wins = conn.execute(
            "SELECT COUNT(*) FROM realized_sales WHERE realized_pnl > 0"
        ).fetchone()[0]
        losses = conn.execute(
            "SELECT COUNT(*) FROM realized_sales WHERE realized_pnl < 0"
        ).fetchone()[0]

        # Open positions
        positions = conn.execute(
            "SELECT ticker, quantity, avg_entry_price, strategy FROM positions WHERE quantity > 0"
        ).fetchall()
        options = conn.execute(
            "SELECT contract_symbol, quantity, avg_entry_price FROM option_positions WHERE quantity > 0"
        ).fetchall()

        # Recent trades
        recent = conn.execute("""
            SELECT ticker, sale_price, cost_basis, realized_pnl, created_at
            FROM realized_sales
            ORDER BY created_at DESC LIMIT 10
        """).fetchall()

        print("=== EQUITY SUMMARY ===")
        print(f"  Closed trades : {trade_count}")
        print(f"  Total realized: ${total_pnl:+,.2f}")
        print(f"  Win/Loss      : {wins}W / {losses}L", end="")
        if trade_count > 0:
            print(f"  ({wins/trade_count:.0%} win rate)")
        else:
            print()

        print(f"\n=== OPEN POSITIONS ({len(positions)} equity, {len(options)} options) ===")
        if positions:
            for tkr, qty, price, strat in positions:
                notional = qty * price
                print(f"  {tkr:8s} {qty:5.0f} shares @ ${price:.2f}  = ${notional:,.0f}  [{strat}]")
        if options:
            for sym, qty, price in options:
                notional = qty * price * 100
                print(f"  {sym}  {qty} contract(s) @ ${price:.2f}  = ${notional:,.0f}")

        if recent:
            print("\n=== LAST 10 CLOSED TRADES ===")
            for tkr, sale, cost, pnl, ts in recent:
                print(f"  {ts[:16]}  {tkr:6s}  ${pnl:+8.2f}  (sold @{sale:.2f} cost {cost:.2f})")
    finally:
        conn.close()


def cmd_positions() -> None:
    cmd_equity()


def cmd_backtest() -> None:
    _run(f"{VENV} {ROOT}/scripts/backtest.py --strategy swing --years 2", print_output=True)


def cmd_warmup() -> None:
    _run(f"{VENV} {ROOT}/scripts/vw_warmup.py", print_output=True)


# ─────────────────────────────────────────────────────────────────────────────

_COMMANDS = {
    "status":   (cmd_status,   []),
    "start":    (cmd_start,    []),
    "stop":     (cmd_stop,     []),
    "restart":  (cmd_restart,  []),
    "logs":     (cmd_logs,     ["n"]),
    "follow":   (cmd_follow,   []),
    "ps":       (cmd_ps,       []),
    "kill":     (cmd_kill,     []),
    "equity":   (cmd_equity,   []),
    "positions":(cmd_positions,[]),
    "backtest": (cmd_backtest, []),
    "warmup":   (cmd_warmup,   []),
}

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print("Usage: python scripts/manage.py <command>")
        print("Commands:", ", ".join(_COMMANDS))
        sys.exit(1)

    cmd_name = sys.argv[1]
    fn, param_names = _COMMANDS[cmd_name]

    if param_names and len(sys.argv) > 2:
        try:
            arg = int(sys.argv[2])
            fn(arg)
        except ValueError:
            fn()
    else:
        fn()


if __name__ == "__main__":
    main()
