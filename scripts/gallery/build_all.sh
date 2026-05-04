#!/bin/bash
set -euo pipefail

# Build script for the redesigned gallery
# Executes plots in prefix order (A -> B -> C ... -> Z) with grouping per prefix
# Each plot produces PDF + PNG outputs to reports/gallery/<PREFIX>_<TOPIC>/

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

# Flags:
#   --synthetic     use synthetic fixtures (no real data required)
#   --n-stars N     synthetic-fixture sample size (default: per-stream realistic — see below)
USE_SYNTHETIC=""
N_STARS=""
for arg in "$@"; do
    case "$arg" in
        --synthetic) USE_SYNTHETIC="--synthetic" ;;
        --n-stars=*) N_STARS="--n-stars ${arg#--n-stars=}" ;;
    esac
done

if [[ -n "$USE_SYNTHETIC" ]]; then
    echo "Building gallery with SYNTHETIC fixtures"
else
    echo "Building gallery with REAL DATA (production mode)"
fi

# Per-stream realistic sample sizes (used when caller does not pass --n-stars=N).
# Stream 1 (APOGEE × Gaia DR3 train pool): ~50k. Stream 2 (TESS Mon+2021 giants):
# ~2k. Stream 3 (Gaia BP/RP broader cohort): ~100k. We size synthetic figures
# to the largest stream by default so multi-stream plots are not under-sampled.
DEFAULT_N_STARS="--n-stars 50000"
if [[ -z "$N_STARS" ]]; then
    N_STARS="$DEFAULT_N_STARS"
fi

# Helper: run a plot script with optional synthetic + n-stars flags
run_plot() {
    local script="$1"
    if [[ ! -f "scripts/gallery/$script" ]]; then
        echo "  SKIP: $script (not found)"
        return 0
    fi
    echo "  → $script ($USE_SYNTHETIC $N_STARS)"
    python "scripts/gallery/$script" $USE_SYNTHETIC $N_STARS 2>&1 | tail -n 5 || true
}

echo "======================================================================="
echo "GALLERY BUILD: Redesigned prefix-ordered structure"
echo "======================================================================="
echo ""

# PREFIX A: Raw data per stream
echo "PREFIX A: Raw data coverage"
echo "---"
run_plot "A1_source_coverage.py"
run_plot "A2_raw_gaia_distribution.py"
echo ""

# PREFIX B: Preprocessing transforms
echo "PREFIX B: Preprocessing transforms"
echo "---"
run_plot "B1_gaia_corrections.py"
run_plot "B2_parallax_comparison.py"
run_plot "B3_gmag_correction.py"
run_plot "B4_dereddened_broadbands.py"
run_plot "B5_ye2024_flux_correction.py"
run_plot "B6_gaia_xp_raw.py"
run_plot "B7_ye_correction.py"
run_plot "B8_hermite_zscore.py"
run_plot "B9_distance_extinction.py"
run_plot "B10_ir_photometry.py"
echo ""

# PREFIX C: Model architecture, training, losses
echo "PREFIX C: Model training"
echo "---"
run_plot "C1_training.py"
echo ""

# PREFIX D: Model outputs per stream
echo "PREFIX D: Model outputs & predictions"
echo "---"
run_plot "D1_regressor_inference.py"
run_plot "D2_kiel_chem_truth_pred.py"
run_plot "D3_apogee_labels.py"
run_plot "D4_knn_rescue_diagnostics.py"
run_plot "D5_evolutionary_classifier.py"
echo ""

# PREFIX E: Validation + cross-catalogue
echo "PREFIX E: Validation & cross-catalogue"
echo "---"
run_plot "E1_pred_vs_truth_splits.py"
run_plot "E2_cross_catalogue_demo.py"
run_plot "E3_gmm_cluster_tracking.py"
run_plot "E4_contamination.py"
echo ""

# PREFIX F: Kinematics + galpy
echo "PREFIX F: Kinematics & dynamics"
echo "---"
run_plot "F1_kinematics.py"
run_plot "F2_geometry.py"
run_plot "F3_selection_function.py"
echo ""

# PREFIX G: Extinction + reddening
echo "PREFIX G: Extinction & reddening"
echo "---"
run_plot "G1_extinction_corrections.py"
run_plot "G2_extinction_ablation_demo.py"
echo ""

# PREFIX H: kNN rescue + hybrid inference
echo "PREFIX H: kNN rescue & hybrid inference"
echo "---"
run_plot "H1_knn_rescue.py"
run_plot "H2_ood_gates.py"
run_plot "H3_hybrid_composer.py"
run_plot "H4_hybrid_inference_planes.py"
run_plot "H5_release_tier_regime.py"
echo ""

# PREFIX Z: Miscellaneous / debugging
echo "PREFIX Z: Miscellaneous & diagnostics"
echo "---"
run_plot "Z1_stream1_join.py"
run_plot "Z2_feature_matrix.py"
run_plot "Z3_stress_battery.py"
run_plot "Z4_final_model_summary.py"
run_plot "Z5_stream2_inference_summary.py"
echo ""

echo "======================================================================="
echo "Gallery build complete. Outputs at reports/gallery/<PREFIX>_<topic>/"
echo "======================================================================="
