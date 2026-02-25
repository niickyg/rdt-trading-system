#!/usr/bin/env bash
# RDT Trading System Health Check
# Usage: ./scripts/health_check.sh
set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    local result="$2"  # 0=pass, 1=fail, 2=warn
    local detail="$3"

    if [ "$result" -eq 0 ]; then
        echo -e "  ${GREEN}[OK]${NC}   $name — $detail"
        ((PASS++))
    elif [ "$result" -eq 2 ]; then
        echo -e "  ${YELLOW}[WARN]${NC} $name — $detail"
        ((WARN++))
    else
        echo -e "  ${RED}[FAIL]${NC} $name — $detail"
        ((FAIL++))
    fi
}

echo "=== RDT Trading System Health Check ==="
echo ""

# 1. Xvfb
if pgrep -f "Xvfb :99" > /dev/null 2>&1; then
    check "Xvfb" 0 "running on display :99"
else
    check "Xvfb" 1 "not running"
fi

# 2. IB Gateway
if pgrep -f "ibgateway" > /dev/null 2>&1; then
    check "IB Gateway" 0 "process running"
else
    check "IB Gateway" 1 "not running"
fi

# 3. Port 4000
if ss -tlnp 2>/dev/null | grep -q ":4000 "; then
    check "API Port 4000" 0 "listening"
else
    check "API Port 4000" 1 "not listening"
fi

# 4. Trading bot
if pgrep -f "main.py bot" > /dev/null 2>&1; then
    check "Trading Bot" 0 "running"
else
    check "Trading Bot" 1 "not running"
fi

# 5. Recent log activity
LOG_DIR="/home/user0/rdt-trading-system/data/logs"
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_LOG" ]; then
        AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$LATEST_LOG")) / 60 ))
        if [ "$AGE_MIN" -le 10 ]; then
            check "Log Activity" 0 "last write ${AGE_MIN}m ago ($LATEST_LOG)"
        elif [ "$AGE_MIN" -le 60 ]; then
            check "Log Activity" 2 "last write ${AGE_MIN}m ago ($LATEST_LOG)"
        else
            check "Log Activity" 1 "stale — last write ${AGE_MIN}m ago"
        fi
    else
        check "Log Activity" 2 "no log files found"
    fi
else
    check "Log Activity" 2 "log directory does not exist"
fi

# 6. Systemd timers
START_TIMER=$(systemctl --user is-active rdt-trading-start.timer 2>/dev/null)
STOP_TIMER=$(systemctl --user is-active rdt-trading-stop.timer 2>/dev/null)
if [ "$START_TIMER" = "active" ] && [ "$STOP_TIMER" = "active" ]; then
    check "Systemd Timers" 0 "start + stop timers active"
else
    check "Systemd Timers" 1 "start=$START_TIMER stop=$STOP_TIMER"
fi

echo ""
echo "=== Summary: ${GREEN}${PASS} passed${NC}, ${YELLOW}${WARN} warnings${NC}, ${RED}${FAIL} failed${NC} ==="
exit $FAIL
