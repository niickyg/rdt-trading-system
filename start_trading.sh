#!/bin/bash
# RDT Trading System - Auto Paper Trading Startup Script
#
# Usage:
#   ./start_trading.sh              # Start bot in background
#   ./start_trading.sh --foreground # Start in foreground (for debugging)
#   ./start_trading.sh --stop       # Stop the running bot
#   ./start_trading.sh --status     # Check if bot is running
#   ./start_trading.sh --logs       # Tail the log file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/data/trading_bot.pid"
LOG_FILE="$SCRIPT_DIR/data/logs/rdt-trading.log"
WATCHLIST="${WATCHLIST:-core}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

cd "$SCRIPT_DIR"

stop_bot() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping trading bot (PID $PID)..."
            kill "$PID"
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                echo "Force killing..."
                kill -9 "$PID"
            fi
            echo "Bot stopped."
        else
            echo "Bot not running (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found. Bot may not be running."
    fi
}

check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Trading bot is RUNNING (PID $PID)"
            echo "Watchlist: $WATCHLIST"
            echo "Log: $LOG_FILE"
            return 0
        else
            echo "Trading bot is NOT running (stale PID file)."
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo "Trading bot is NOT running."
        return 1
    fi
}

case "${1:-}" in
    --stop)
        stop_bot
        exit 0
        ;;
    --status)
        check_status
        exit $?
        ;;
    --logs)
        tail -f "$LOG_FILE"
        exit 0
        ;;
    --foreground)
        echo "Starting RDT Trading Bot (foreground mode)"
        echo "  Watchlist: $WATCHLIST"
        echo "  Log level: $LOG_LEVEL"
        echo "  Press Ctrl+C to stop"
        echo ""
        exec python main.py bot --auto --watchlist "$WATCHLIST" --log-level "$LOG_LEVEL"
        ;;
    "")
        # Background mode (default)
        ;;
    *)
        echo "Usage: $0 [--foreground|--stop|--status|--logs]"
        exit 1
        ;;
esac

# Ensure log directory exists
mkdir -p "$SCRIPT_DIR/data/logs"

# Stop any existing instance
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping existing bot (PID $PID)..."
        kill "$PID"
        # Wait up to 10s for clean shutdown (IBKR needs time to release client_id)
        for i in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing..."
            kill -9 "$PID"
            sleep 2
        fi
    fi
    rm -f "$PID_FILE"
fi

echo "Starting RDT Trading Bot..."
echo "  Mode: PAPER / AUTO-TRADE"
echo "  Watchlist: $WATCHLIST ($( python -c "from config.watchlists import get_watchlist_by_name; print(len(get_watchlist_by_name('$WATCHLIST')))" 2>/dev/null || echo '?') symbols)"
echo "  Log level: $LOG_LEVEL"
echo "  Log file: $LOG_FILE"

nohup python main.py bot --auto --watchlist "$WATCHLIST" --log-level "$LOG_LEVEL" \
    > /dev/null 2>&1 &

echo $! > "$PID_FILE"
echo "  PID: $(cat "$PID_FILE")"
echo ""
echo "Bot started! Use these commands:"
echo "  ./start_trading.sh --status  # Check status"
echo "  ./start_trading.sh --logs    # View logs"
echo "  ./start_trading.sh --stop    # Stop bot"
