#!/usr/bin/env bash
# Regenerate the full 25-stage hybrid-deployment gallery.
#
# Each plot script reads only the canonical artefact produced by the
# corresponding deploy step. A wrong plot means a wrong artefact, which
# means a wrong deploy step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-${REPO_ROOT}/.venv/bin/python}"

STAGES=(
    plot_00_source_coverage
    plot_01_gaia_corrections
    plot_02_gaia_xp_raw
    plot_03_ye_correction
    plot_04_hermite_zscore
    plot_05_distance_extinction
    plot_06_ir_photometry
    plot_07_apogee_labels
    plot_08_stream1_join
    plot_09_selection_function
    plot_10_kinematics
    plot_11_geometry
    plot_12_feature_matrix
    plot_13_training
    plot_14_pred_vs_truth_splits
    plot_15_kiel_chem_truth_pred
    plot_16_regressor_inference
    plot_16b_kin_ood_detector
    plot_17_knn_rescue
    plot_18_ood_gates
    plot_19_hybrid_composer
    plot_20_hybrid_inference_planes
    plot_21_release_tier_regime
    plot_21b_flag_coloured_chemistry
    plot_22_gmm_cluster_tracking
    plot_23_contamination
    plot_24_stress_battery
    plot_25_final_model_summary
    plot_26_stream2_inference_summary
)

for s in "${STAGES[@]}"; do
    echo "===> $s"
    if [[ -f "scripts/gallery/${s}.py" ]]; then
        PYTHONPATH=src "$PY" "scripts/gallery/${s}.py" || echo "  [WARN] ${s} failed; continuing"
    else
        echo "  [SKIP] scripts/gallery/${s}.py not present"
    fi
done

echo
echo "===> GALLERY COMPLETE (27 stages)"
echo "  walk reports/gallery/00..26 top-to-bottom"
