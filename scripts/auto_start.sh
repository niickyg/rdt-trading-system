#!/usr/bin/env bash
# RDT Trading Bot - Pre-market auto-start
# Called by cron at 6:10 AM PT (9:10 AM ET), Mon-Fri
#
# 1. Ensures IB Gateway is running
# 2. Waits for API port to be ready
# 3. Starts the trading bot with sp500 watchlist
#
# Logs to /home/user0/rdt-trading-system/data/logs/auto_start.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$SCRIPT_DIR/data/logs/auto_start.log"
PID_FILE="$SCRIPT_DIR/data/trading_bot.pid"
WATCHLIST="sp500"
LOG_LEVEL="INFO"

mkdir -p "$SCRIPT_DIR/data/logs"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log "=== Auto-start triggered ==="

# Skip weekends
DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
    log "Weekend (day=$DOW) — skipping"
    exit 0
fi

# Check if bot is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        log "Bot already running (PID $PID) — skipping"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# Also check by process name
if pgrep -f "main.py bot" > /dev/null 2>&1; then
    log "Bot process found — skipping"
    exit 0
fi

# Step 1: Ensure IB Gateway is running
if ! pgrep -f "ibgateway" > /dev/null 2>&1; then
    log "Starting IB Gateway..."
    nohup "$SCRIPT_DIR/scripts/start_ibgateway.sh" >> "$LOG_FILE" 2>&1 &

    # Wait up to 90s for IB Gateway to start (port 4000)
    for i in $(seq 1 45); do
        if ss -tlnp 2>/dev/null | grep -q ":4000 "; then
            log "IB Gateway API port ready after ${i}x2s"
            break
        fi
        sleep 2
    done

    if ! ss -tlnp 2>/dev/null | grep -q ":4000 "; then
        log "ERROR: IB Gateway API port (4000) not ready after 90s — aborting"
        exit 1
    fi
else
    log "IB Gateway already running"
fi

# Step 2: Wait for API port to be connectable
sleep 5

# Step 3: Start the trading bot
log "Starting trading bot (watchlist=$WATCHLIST)..."
cd "$SCRIPT_DIR"
WATCHLIST="$WATCHLIST" LOG_LEVEL="$LOG_LEVEL" ./start_trading.sh >> "$LOG_FILE" 2>&1

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    log "Bot started (PID $PID)"
else
    log "ERROR: Bot failed to start (no PID file)"
    exit 1
fi

log "=== Auto-start complete ==="
