#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user0/rdt-trading-system"
GATEWAY_CMD="/home/user0/ibc/gatewaystart.sh -inline"
LOG_FILE="${ROOT_DIR}/logs/ibgateway-bootstrap.log"

HOST="${IBKR_HOST:-127.0.0.1}"
PORT="${IBKR_PORT:-4000}"
TIMEOUT_SECONDS="${IBKR_SOCKET_WAIT_TIMEOUT:-120}"

mkdir -p "$(dirname "${LOG_FILE}")"

if "${ROOT_DIR}/scripts/wait_for_ibkr_socket.sh" "${HOST}" "${PORT}" 2 >/dev/null; then
    echo "IB Gateway socket already available on ${HOST}:${PORT}; startup skipped"
    exit 0
fi

echo "Starting IB Gateway via IBC..."
nohup bash -lc "${GATEWAY_CMD}" >> "${LOG_FILE}" 2>&1 &

"${ROOT_DIR}/scripts/wait_for_ibkr_socket.sh" "${HOST}" "${PORT}" "${TIMEOUT_SECONDS}"
