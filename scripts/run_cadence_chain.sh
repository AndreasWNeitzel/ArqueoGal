#!/usr/bin/env bash
# Chain pretraining → finetuning → per-epoch prediction emission so the
# whole cadence-animation data pipeline runs unattended.
#
# Outputs: <repo>/models/main/xp_abundances/<DATE_SHA_HASH>{,_finetune}/
# Logs:    <repo>/logs/cadence_chain_<TIMESTAMP>.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/cadence_chain_${TS}.log"

cd "${REPO}"

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S')] $*" | tee -a "${LOG}"; }

trap 'log "ABORTED at line $LINENO"; exit 1' ERR

log "=== STAGE 1/3: contrastive pretrain (200 ep, patience 20, ckpt every 1) ==="
.venv/bin/python scripts/run_contrastive_pretrain.py 2>&1 | tee -a "${LOG}"

# Discover the just-finished pretrain run dir by mtime on its best.pt.
PRETRAIN_BEST="$(ls -1t models/main/xp_abundances/*/xp_abundances_main_contrastive_seed0_best.pt 2>/dev/null | head -1)"
if [[ -z "${PRETRAIN_BEST}" || ! -f "${PRETRAIN_BEST}" ]]; then
    log "ERROR: could not locate a pretrain best.pt after stage 1"
    exit 2
fi
log "pretrain best ckpt = ${PRETRAIN_BEST}"

log "=== STAGE 2/3: supervised finetune (5-label, 20 ep, patience 6, ckpt every 1) ==="
.venv/bin/python scripts/run_supervised_finetune.py \
    --pretrained "${PRETRAIN_BEST}" \
    --label-set 5 2>&1 | tee -a "${LOG}"

# Discover the just-finished finetune run dir.
FINETUNE_DIR="$(ls -1td models/main/xp_abundances/*finetune*/ 2>/dev/null | head -1)"
if [[ -z "${FINETUNE_DIR}" || ! -d "${FINETUNE_DIR}" ]]; then
    log "ERROR: could not locate a finetune run dir after stage 2"
    exit 3
fi
log "finetune run dir = ${FINETUNE_DIR}"

log "=== STAGE 3/3: emit per-epoch predictions ==="
.venv/bin/python scripts/emit_cadence_predictions.py \
    --run-dir "${FINETUNE_DIR%/}" 2>&1 | tee -a "${LOG}"

log "=== ALL STAGES COMPLETE ==="
log "log = ${LOG}"
log "predictions root = data/processed/cadence_predictions/"
