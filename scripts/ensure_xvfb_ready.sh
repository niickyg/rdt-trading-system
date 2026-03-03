#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-99}"
XVFB_DISPLAY=":${DISPLAY_NUM}"
LOCK_FILE="/tmp/.X${DISPLAY_NUM}-lock"
SOCKET_FILE="/tmp/.X11-unix/X${DISPLAY_NUM}"
LOG_FILE="/home/user0/rdt-trading-system/logs/xvfb-bootstrap.log"

mkdir -p "$(dirname "${LOG_FILE}")"

if pgrep -f "Xvfb ${XVFB_DISPLAY}" >/dev/null 2>&1; then
    echo "Xvfb already running on ${XVFB_DISPLAY}"
    exit 0
fi

if [ -f "${LOCK_FILE}" ] && [ ! -S "${SOCKET_FILE}" ]; then
    rm -f "${LOCK_FILE}"
fi

nohup /usr/bin/Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 -ac >> "${LOG_FILE}" 2>&1 &

for _ in $(seq 1 20); do
    if pgrep -f "Xvfb ${XVFB_DISPLAY}" >/dev/null 2>&1 && [ -S "${SOCKET_FILE}" ]; then
        echo "Xvfb ready on ${XVFB_DISPLAY}"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Xvfb failed to become ready on ${XVFB_DISPLAY}"
exit 1
