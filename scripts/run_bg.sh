#!/usr/bin/env bash
# Usage: run_bg.sh <log_name> <python_script> [args...]
set -euo pipefail

LOG_NAME="$1"; shift
SCRIPT="$1"; shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DATE_STAMP="$(date +%Y%m%d)"
LOG_FILE="${LOG_DIR}/${LOG_NAME}_${DATE_STAMP}.log"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

nohup python "${SCRIPT}" "$@" > "${LOG_FILE}" 2>&1 &
PID=$!

sleep 3
echo "PID=${PID}"
ps -p "${PID}" -o pid,etime,cmd 2>&1 || echo "Process ${PID} not running"
echo "--- last 5 log lines ---"
tail -5 "${LOG_FILE}" 2>&1
echo "--- log: ${LOG_FILE}"