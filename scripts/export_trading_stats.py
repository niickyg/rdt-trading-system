#!/usr/bin/env python3
"""Export RDT trading stats to JSON for GitHub Pages display."""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, date

DB_PATH = "/home/user0/rdt-trading-system/data/rdt_trading.db"
OUTPUT_PATH = "/home/user0/niickyg.github.io/trading-data.json"
REPO_PATH = "/home/user0/niickyg.github.io"


def get_market_status():
    """Check if US stock market is currently open (9:30 AM - 4:00 PM ET)."""
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    # Weekday check (Mon=0, Fri=4)
    if now_et.weekday() > 4:
        return "closed"
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if market_open <= now_et <= market_close:
        return "open"
    return "closed"


def query_db():
    """Query all trading data from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "market_status": get_market_status(),
    }

    # --- Account stats from trades ---
    try:
        cur = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losers,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as gross_profit,
                COALESCE(ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 0) as gross_loss
            FROM trades
            WHERE status = 'CLOSED' AND pnl IS NOT NULL
              AND (exit_reason IS NULL OR exit_reason NOT LIKE 'CLEANUP%')
        """)
        row = cur.fetchone()
    except sqlite3.OperationalError as e:
        print(f"Warning: trades table issue: {e}", file=sys.stderr)
        row = None

    if row:
        total_trades = row["total_trades"] or 0
        winners = row["winners"] or 0
        gross_profit = float(row["gross_profit"])
        gross_loss = float(row["gross_loss"])
    else:
        total_trades = winners = 0
        gross_profit = gross_loss = 0.0

    # Also count open trades as part of total activity
    try:
        cur2 = conn.execute("SELECT COUNT(*) as c FROM trades WHERE status = 'OPEN'")
        open_trade_count = cur2.fetchone()["c"]
    except sqlite3.OperationalError:
        open_trade_count = 0

    win_rate = round(winners / total_trades, 2) if total_trades > 0 else 0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
    total_pnl = round(float(row["total_pnl"]), 2) if row else 0.0

    data["account"] = {
        "total_trades": total_trades + open_trade_count,
        "closed_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
    }

    # --- Open stock positions ---
    positions = []
    try:
        cur = conn.execute("""
            SELECT symbol, direction, entry_price, current_price, shares,
                   unrealized_pnl, updated_at, stop_loss, take_profit
            FROM positions
            ORDER BY entry_price * shares DESC
        """)
        for p in cur.fetchall():
            entry = float(p["entry_price"])
            current = float(p["current_price"]) if p["current_price"] else None
            unrealized = float(p["unrealized_pnl"]) if p["unrealized_pnl"] else None
            pnl_pct = None
            if current and entry:
                if p["direction"] == "LONG":
                    pnl_pct = round((current - entry) / entry * 100, 2)
                else:
                    pnl_pct = round((entry - current) / entry * 100, 2)

            positions.append({
                "symbol": p["symbol"],
                "direction": p["direction"],
                "entry_price": round(entry, 2),
                "current_price": round(current, 2) if current else None,
                "shares": p["shares"],
                "unrealized_pnl": round(unrealized, 2) if unrealized else None,
                "pnl_pct": pnl_pct,
                "stop_loss": round(float(p["stop_loss"]), 2) if p["stop_loss"] else None,
                "take_profit": round(float(p["take_profit"]), 2) if p["take_profit"] else None,
            })
    except sqlite3.OperationalError as e:
        print(f"Warning: positions table issue: {e}", file=sys.stderr)
    data["open_positions"] = positions

    # --- Options positions ---
    options = []
    try:
        cur = conn.execute("""
            SELECT symbol, strategy_name, direction, contracts,
                   entry_premium, total_premium, legs_json, entry_time
            FROM options_positions
            ORDER BY entry_time DESC
        """)
        for o in cur.fetchall():
            legs_raw = o["legs_json"]
            legs_summary = []
            if legs_raw:
                try:
                    legs = json.loads(legs_raw)
                    for leg in legs:
                        strike = leg.get("strike", "")
                        right = "Call" if leg.get("right") == "C" else "Put"
                        action = leg.get("action", "")
                        expiry = leg.get("expiry", "")
                        legs_summary.append(f"{action} {strike} {right} {expiry}")
                except (json.JSONDecodeError, TypeError):
                    pass

            options.append({
                "symbol": o["symbol"],
                "strategy": o["strategy_name"],
                "direction": o["direction"],
                "contracts": o["contracts"],
                "entry_premium": round(float(o["entry_premium"]), 2) if o["entry_premium"] else None,
                "total_premium": round(float(o["total_premium"]), 2) if o["total_premium"] else None,
                "legs": legs_summary,
                "entry_date": o["entry_time"][:10] if o["entry_time"] else None,
            })
    except sqlite3.OperationalError as e:
        print(f"Warning: options_positions table issue: {e}", file=sys.stderr)
    data["options_positions"] = options

    # --- Recent closed trades (last 30, excluding cleanup entries) ---
    recent = []
    try:
        cur = conn.execute("""
            SELECT symbol, direction, entry_price, exit_price, shares,
                   pnl, pnl_percent, exit_reason, exit_time
            FROM trades
            WHERE status = 'CLOSED'
              AND (exit_reason IS NULL OR exit_reason NOT LIKE 'CLEANUP%')
            ORDER BY exit_time DESC
            LIMIT 30
        """)
        for t in cur.fetchall():
            recent.append({
                "symbol": t["symbol"],
                "direction": t["direction"],
                "entry_price": round(float(t["entry_price"]), 2),
                "exit_price": round(float(t["exit_price"]), 2) if t["exit_price"] else None,
                "shares": t["shares"],
                "pnl": round(float(t["pnl"]), 2) if t["pnl"] else 0,
                "pnl_pct": round(float(t["pnl_percent"]), 2) if t["pnl_percent"] else 0,
                "exit_reason": t["exit_reason"],
                "date": t["exit_time"][:10] if t["exit_time"] else None,
            })
    except sqlite3.OperationalError as e:
        print(f"Warning: trades table issue (recent): {e}", file=sys.stderr)
    data["recent_trades"] = recent

    # --- Daily stats (last 30 days) ---
    daily = []
    try:
        cur = conn.execute("""
            SELECT date, num_trades, pnl, win_rate, starting_balance, ending_balance
            FROM daily_stats
            ORDER BY date DESC
            LIMIT 30
        """)
        for d in cur.fetchall():
            daily.append({
                "date": d["date"],
                "trades": d["num_trades"] or 0,
                "pnl": round(float(d["pnl"]), 2) if d["pnl"] else 0,
                "win_rate": round(float(d["win_rate"]), 2) if d["win_rate"] else 0,
                "balance": round(float(d["ending_balance"]), 2) if d["ending_balance"] else None,
            })
    except sqlite3.OperationalError as e:
        print(f"Warning: daily_stats table issue: {e}", file=sys.stderr)
    data["daily_stats"] = list(reversed(daily))  # chronological order

    # --- Equity curve from snapshots (one per day, last 30 days) ---
    equity = []
    try:
        cur = conn.execute("""
            SELECT DATE(timestamp) as snap_date,
                   MAX(equity_value) as equity,
                   MAX(drawdown_pct) as drawdown
            FROM equity_snapshots
            GROUP BY DATE(timestamp)
            ORDER BY snap_date DESC
            LIMIT 30
        """)
        for e in cur.fetchall():
            equity.append({
                "date": e["snap_date"],
                "equity": round(float(e["equity"]), 2),
                "drawdown": round(float(e["drawdown"]), 4) if e["drawdown"] else 0,
            })
    except sqlite3.OperationalError as e:
        print(f"Warning: equity_snapshots table issue: {e}", file=sys.stderr)
    data["equity_curve"] = list(reversed(equity))

    # --- Today's signal count ---
    try:
        today_str = date.today().isoformat()
        cur = conn.execute(
            "SELECT COUNT(*) as c FROM signals WHERE DATE(timestamp) = ?",
            (today_str,)
        )
        data["signals_today"] = cur.fetchone()["c"]
    except sqlite3.OperationalError as e:
        print(f"Warning: signals table issue: {e}", file=sys.stderr)
        data["signals_today"] = 0

    conn.close()
    return data


def git_commit_and_push(json_str):
    """Commit and push if the JSON has changed."""
    # Write file
    with open(OUTPUT_PATH, "w") as f:
        f.write(json_str)

    # Check for changes
    result = subprocess.run(
        ["git", "-C", REPO_PATH, "diff", "--quiet", "trading-data.json"],
        capture_output=True
    )
    if result.returncode == 0:
        # Also check if file is untracked
        status = subprocess.run(
            ["git", "-C", REPO_PATH, "status", "--porcelain", "trading-data.json"],
            capture_output=True, text=True
        )
        if not status.stdout.strip():
            print(f"[{datetime.now()}] No changes, skipping commit.")
            return

    subprocess.run(
        ["git", "-C", REPO_PATH, "add", "trading-data.json"],
        check=True
    )
    subprocess.run(
        ["git", "-C", REPO_PATH, "commit", "-m", "Update trading stats"],
        check=True
    )
    subprocess.run(
        ["git", "-C", REPO_PATH, "push"],
        check=True
    )
    print(f"[{datetime.now()}] Stats exported and pushed.")


def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    data = query_db()
    json_str = json.dumps(data, indent=2)
    git_commit_and_push(json_str)


if __name__ == "__main__":
    main()
