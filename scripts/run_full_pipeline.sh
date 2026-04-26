#!/usr/bin/env bash
# Reproduce the kNN+strong-contrastive-v2 hybrid release end-to-end.
#
# Stages:
#   1. (skipped if checkpoint exists) train strong-contrastive-v2 ensemble
#   2. inference on Stream 3 with the strong-contrastive-v2 ensemble
#   3. latent-kNN rescue parquet
#   4. hybrid release pipeline (regressor + kNN composer)
#   5. diagnostic gallery stages 18, 19, 20
#   6. unit + integration tests
#
# Outputs go to ``release/D-Cat-b/hybrid_pipeline_run`` and
# ``reports/figures/real_data_plots/comprehensive``.
#
# This script assumes the rapids/PyTorch env is already activated (see
# CLAUDE.md). It does NOT install dependencies.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-${REPO_ROOT}/.venv/bin/python}"

ENSEMBLE_DIR="${REPO_ROOT}/models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label"
TRAIN_PARQUET="${REPO_ROOT}/data/processed/pipeline1_features_stream1.parquet"
S3_FEATURES="${REPO_ROOT}/data/processed/pipeline1_features_stream3.parquet"
FROZEN_STATS="${REPO_ROOT}/data/processed/pipeline1_features_stream1.provenance.json"
PRED_PARQUET="${REPO_ROOT}/data/processed/pipeline1_predictions_stream3.parquet"
KNN_PARQUET="${REPO_ROOT}/data/processed/pipeline1_knn_rescue.parquet"
HYBRID_OUT_DIR="${REPO_ROOT}/release/D-Cat-b/hybrid_pipeline_run"
GALLERY_OUT="${REPO_ROOT}/reports/gallery"

echo "===> Stage 1: ensemble checkpoint check"
if [[ ! -d "$ENSEMBLE_DIR/member_seed0" ]]; then
    echo "FATAL: strong-contrastive-v2 ensemble missing at $ENSEMBLE_DIR"
    echo "Train via: $PY scripts/run_ensemble.py (defaults are now SupCon=1.0, Barlow=0.5)"
    exit 1
fi
echo "ensemble OK: $(ls "$ENSEMBLE_DIR"/member_seed*/*.pt | wc -l) member checkpoints"

echo "===> Stage 2: Stream-3 inference (regressor)"
if [[ -f "$PRED_PARQUET" ]]; then
    echo "predictions parquet exists; skipping stage 2"
else
    "$PY" scripts/run_pipeline1_inference.py \
        --ensemble-dir "$ENSEMBLE_DIR" \
        --features "$S3_FEATURES" \
        --output "$PRED_PARQUET"
fi

echo "===> Stage 3: latent-kNN rescue"
"$PY" scripts/run_knn_rescue.py \
    --ensemble-dir "$ENSEMBLE_DIR" \
    --train-parquet "$TRAIN_PARQUET" \
    --infer-parquet "$S3_FEATURES" \
    --frozen-stats "$FROZEN_STATS" \
    --output "$KNN_PARQUET"

echo "===> Stage 4: hybrid release pipeline (regressor + kNN composer)"
"$PY" scripts/build_hybrid_release.py \
    --predictions "$PRED_PARQUET" \
    --features "$S3_FEATURES" \
    --knn-rescue "$KNN_PARQUET" \
    --output-dir "$HYBRID_OUT_DIR"

echo "===> Stage 5: diagnostic gallery (stages 18-20 of the full 20-stage gallery)"
PYTHONPATH=src "$PY" scripts/gallery/run_batch7.py
echo "  (for the full 20-stage rebuild from scratch use: bash scripts/gallery/build_all.sh)"

echo "===> Stage 6: unit tests"
"$PY" -m pytest tests/xp_abundances/main/test_release.py \
                tests/xp_abundances/main/test_knn_rescue.py \
                tests/data/test_release_pipeline.py -q

echo
echo "===> END-TO-END PIPELINE COMPLETE"
echo "  hybrid release manifest: $HYBRID_OUT_DIR/release_pipeline_manifest.json"
echo "  diagnostic gallery:      $GALLERY_OUT/{17_pipeline1_regime_diagnostics,18_pred_vs_truth_splits,19_gmm_cluster_tracking,20_contamination_analysis}/"
echo "  for the full 20-stage gallery rebuild, run:"
echo "    bash scripts/gallery/build_all.sh"
echo "  for the heavy stress battery, run:"
echo "    $PY -m pytest tests/integration --run-stress -v"
