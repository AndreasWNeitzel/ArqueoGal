#!/usr/bin/env bash
# Re-run every gallery script under sections C-Z and Y after a model retrain.
# A and B are skipped (raw data + preprocessing — independent of the model).
#
# Per-script: continue on failure, log status. Final summary at end.
#
# Usage:
#   bash scripts/regen_gallery_post_train.sh
# Optional env:
#   ARQUEOGAL_PRED_S1=path/to/predictions.parquet  # override the canonical
#                                                    Stream-1 predictions parquet
#                                                    that the Y eval figures
#                                                    consume (see _y_holdout.py)

set -uo pipefail   # NOTE: no -e — one bad script must not stop the rest
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/regen_gallery_${TS}.log"

cd "${REPO}"

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "${LOG}"; }

scripts=(scripts/gallery/[CDEFGHYZ]*.py)
n_total=${#scripts[@]}
n_ok=0
n_fail=0
failed_list=()

log "regen: ${n_total} scripts under sections C-Z + Y"
log "log file = ${LOG}"
[[ -n "${ARQUEOGAL_PRED_S1:-}" ]] && log "ARQUEOGAL_PRED_S1 = ${ARQUEOGAL_PRED_S1}"

for s in "${scripts[@]}"; do
    name="$(basename "$s")"
    log "  ${name}: starting"
    if .venv/bin/python "$s" >> "${LOG}" 2>&1; then
        n_ok=$((n_ok + 1))
        log "  ${name}: OK"
    else
        n_fail=$((n_fail + 1))
        failed_list+=("$name")
        log "  ${name}: FAIL"
    fi
done

log "==== SUMMARY ===="
log "ok    = ${n_ok} / ${n_total}"
log "fail  = ${n_fail} / ${n_total}"
if (( n_fail > 0 )); then
    log "failed scripts:"
    for f in "${failed_list[@]}"; do
        log "  - ${f}"
    done
fi
