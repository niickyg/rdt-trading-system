#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-${IBKR_HOST:-127.0.0.1}}"
PORT="${2:-${IBKR_PORT:-4000}}"
TIMEOUT_SECONDS="${3:-${IBKR_SOCKET_WAIT_TIMEOUT:-120}}"
SLEEP_SECONDS="${IBKR_SOCKET_WAIT_INTERVAL:-1}"

start_ts="$(date +%s)"
echo "Waiting for IBKR socket ${HOST}:${PORT} (timeout=${TIMEOUT_SECONDS}s)"

while true; do
    if timeout 1 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
        elapsed="$(( $(date +%s) - start_ts ))"
        echo "IBKR socket ready at ${HOST}:${PORT} after ${elapsed}s"
        exit 0
    fi

    elapsed="$(( $(date +%s) - start_ts ))"
    if [ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]; then
        echo "ERROR: IBKR socket ${HOST}:${PORT} not ready after ${TIMEOUT_SECONDS}s"
        exit 1
    fi

    sleep "${SLEEP_SECONDS}"
done
