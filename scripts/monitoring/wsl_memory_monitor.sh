#!/usr/bin/env bash
# ArqueoGal WSL2 memory monitor
# Usage: bash scripts/monitoring/wsl_memory_monitor.sh
# Logs /proc/meminfo summary every 10 seconds with timestamp.
# Captures the inference process RSS/VMS if its PID can be detected.

set -euo pipefail

readonly INTERVAL_SECONDS=10
readonly LOG_PREFIX="[wsl_mem_monitor]"

echo "${LOG_PREFIX} Starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap_total=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
    swap_free=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    echo "${LOG_PREFIX} ${ts} MemTotal_kB=${mem_total} MemAvail_kB=${mem_available} SwapTotal_kB=${swap_total} SwapFree_kB=${swap_free}"

    # Per-process for python (Stream 3 inference) — best-effort
    py_pids=$(pgrep -f "python.*pipeline1_inference\|python.*build_pipeline1" || echo "")
    if [ -n "${py_pids}" ]; then
        for pid in ${py_pids}; do
            if [ -r "/proc/${pid}/status" ]; then
                rss=$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status")
                vms=$(awk '/^VmSize:/ {print $2}' "/proc/${pid}/status")
                echo "${LOG_PREFIX} ${ts} pid=${pid} VmRSS_kB=${rss} VmSize_kB=${vms}"
            fi
        done
    fi

    sleep "${INTERVAL_SECONDS}"
done
