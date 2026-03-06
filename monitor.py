#!/usr/bin/env python3
"""
Live position monitor for RDT Trading System.
Reads from the local database + fetches current prices via yfinance.
No IBKR connection needed — won't conflict with the bot.

Usage: python monitor.py [--watch]
  --watch: refresh every 30 seconds
"""

import sqlite3
import sys
import time
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "rdt_trading.db")


def get_current_prices(symbols):
    """Fetch current prices via yfinance (no IBKR needed)."""
    try:
        import yfinance as yf
        prices = {}
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                info = t.fast_info
                prices[sym] = {
                    "price": info.last_price,
                    "prev_close": info.previous_close,
                    "day_pct": ((info.last_price - info.previous_close) / info.previous_close) * 100
                }
            except Exception as e:
                prices[sym] = {"price": None, "error": str(e)}
        return prices
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return {}


def display_positions():
    """Query DB and display positions with live prices."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get open positions
    cur.execute("""
        SELECT symbol, direction, entry_price, shares, entry_time,
               stop_loss, take_profit, rrs_at_entry
        FROM positions
        WHERE exit_time IS NULL OR status = 'OPEN'
        ORDER BY entry_time
    """)
    positions = cur.fetchall()

    if not positions:
        print("\n  No open positions.\n")
        conn.close()
        return

    symbols = list(set(row["symbol"] for row in positions))
    prices = get_current_prices(symbols)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 80}")
    print(f"  RDT Position Monitor — {now}")
    print(f"{'=' * 80}\n")

    total_pnl = 0.0

    for pos in positions:
        sym = pos["symbol"]
        direction = pos["direction"]
        entry = pos["entry_price"]
        shares = pos["shares"]
        stop = pos["stop_loss"]
        target = pos["take_profit"]
        rrs = pos["rrs_at_entry"]
        entry_time = pos["entry_time"][:16] if pos["entry_time"] else "?"

        price_data = prices.get(sym, {})
        current = price_data.get("price")
        day_pct = price_data.get("day_pct", 0)

        if current is None:
            print(f"  {sym} {direction} — price unavailable")
            continue

        # Calculate P&L
        if direction == "LONG":
            pnl_per_share = current - entry
            risk_per_share = entry - stop if stop else 0
        else:  # SHORT
            pnl_per_share = entry - current
            risk_per_share = stop - entry if stop else 0

        total_pnl_pos = pnl_per_share * abs(shares)
        pnl_pct = (pnl_per_share / entry) * 100
        r_multiple = pnl_per_share / risk_per_share if risk_per_share else 0
        total_pnl += total_pnl_pos

        # Distance to stop/target
        if direction == "LONG":
            stop_dist = ((current - stop) / current * 100) if stop else 0
            target_dist = ((target - current) / current * 100) if target else 0
        else:
            stop_dist = ((stop - current) / current * 100) if stop else 0
            target_dist = ((current - target) / current * 100) if target else 0

        # Color indicators
        pnl_symbol = "+" if total_pnl_pos >= 0 else ""
        r_symbol = "+" if r_multiple >= 0 else ""
        arrow = "LONG" if direction == "LONG" else "SHORT"

        print(f"  {sym} {arrow} | {abs(shares)} shares @ ${entry:.2f} | Entered {entry_time}")
        print(f"    Current: ${current:.2f} ({day_pct:+.2f}% today)")
        print(f"    P&L:     {pnl_symbol}${total_pnl_pos:.2f} ({pnl_pct:+.2f}%) | {r_symbol}{r_multiple:.2f}R")
        print(f"    Stop:    ${stop:.2f} ({stop_dist:.1f}% away) | Target: ${target:.2f} ({target_dist:.1f}% away)")
        print(f"    RRS at entry: {rrs:.2f}")
        print()

    print(f"  {'─' * 40}")
    pnl_sym = "+" if total_pnl >= 0 else ""
    print(f"  Net P&L: {pnl_sym}${total_pnl:.2f}")
    print()

    # Recent closed trades
    cur.execute("""
        SELECT symbol, direction, entry_price, exit_price, shares, pnl, exit_reason, exit_time
        FROM trades
        WHERE exit_time IS NOT NULL AND status != 'OPEN'
        ORDER BY exit_time DESC
        LIMIT 5
    """)
    closed = cur.fetchall()
    if closed:
        print(f"  Recent Closed Trades:")
        print(f"  {'─' * 40}")
        for t in closed:
            pnl_s = f"+${t['pnl']:.2f}" if t['pnl'] and t['pnl'] >= 0 else f"-${abs(t['pnl']):.2f}" if t['pnl'] else "N/A"
            print(f"    {t['symbol']} {t['direction']} | {pnl_s} | {t['exit_reason'] or 'unknown'}")
        print()

    conn.close()


def main():
    watch = "--watch" in sys.argv or "-w" in sys.argv

    if watch:
        print("  Monitoring positions (Ctrl+C to stop)...")
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                display_positions()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    else:
        display_positions()


if __name__ == "__main__":
    main()
