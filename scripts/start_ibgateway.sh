#!/usr/bin/env bash
# Start IB Gateway via IBC in headless mode (Xvfb)
# Usage: ./scripts/start_ibgateway.sh
set -euo pipefail

export DISPLAY=:99

# Start Xvfb if not already running
if ! pgrep -f "Xvfb :99" > /dev/null 2>&1; then
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1024x768x24 -ac &
    sleep 2
fi

# Create IBC log directory
mkdir -p /home/user0/ibc/logs

echo "Starting IB Gateway v1037 via IBC (inline mode)..."
# gatewaystart.sh reads config from its own variables (already set).
# -inline avoids xterm dependency and runs in the current terminal.
exec /home/user0/ibc/gatewaystart.sh -inline
