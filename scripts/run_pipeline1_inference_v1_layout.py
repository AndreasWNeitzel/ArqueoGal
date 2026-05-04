"""Run the v1-tagged 5-member ensemble at the v1 layout (139-D, no av_los).

Hypothesis (verified post-augmentation): the v1 ensemble was trained with
DEFAULT_AUX_COLS minus the fused-A_V column ``av_los``. Adding ``av_los`` to
the layout post-tag pushed input_dim from 139 to 140, breaking checkpoint
loading. This wrapper instantiates a FeatureLayout that excludes ``av_los``
(everything else identical to current default) and calls
``scripts.run_pipeline1_inference.run_inference``.

Usage:
  python scripts/run_pipeline1_inference_v1_layout.py STREAM
where STREAM in {1, 2, 3}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from arqueogal.xp_abundances.main.data import (
    DEFAULT_AUX_COLS,
    DEFAULT_RESIDUAL_COLS,
    DEFAULT_XP_COEF_INDICES,
    DEFAULT_XP_SCALAR_COLS,
    FeatureLayout,
)

import run_pipeline1_inference as drv  # type: ignore

V1_AUX_COLS = tuple(c for c in DEFAULT_AUX_COLS if c != "av_los")
assert len(V1_AUX_COLS) == len(DEFAULT_AUX_COLS) - 1


def make_v1_layout() -> FeatureLayout:
    layout = FeatureLayout(
        xp_bp_indices=DEFAULT_XP_COEF_INDICES,
        xp_rp_indices=DEFAULT_XP_COEF_INDICES,
        xp_scalar_cols=DEFAULT_XP_SCALAR_COLS,
        residual_cols=DEFAULT_RESIDUAL_COLS,
        aux_cols=V1_AUX_COLS,
    )
    assert layout.input_dim == 139, f"expected 139, got {layout.input_dim}"
    return layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=REPO / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    in_parquet = REPO / f"data/processed/pipeline1_features_stream{args.stream}.parquet"
    out_parquet = REPO / f"data/processed/pipeline1_predictions_stream{args.stream}_v1.parquet"
    frozen_stats = REPO / "data/processed/pipeline1_features_stream1.provenance.json"
    ood_training_parquet = REPO / "data/processed/pipeline1_features_stream1.parquet"
    mode_ambiguous_grid = REPO / "data/processed/mode_ambiguous_grid.npz"

    layout = make_v1_layout()
    print(f"[v1-layout] input_dim={layout.input_dim} (av_los excluded)")
    print(f"[v1-layout] in:  {in_parquet}")
    print(f"[v1-layout] out: {out_parquet}")

    drv.run_inference(
        ensemble_dir=args.ensemble_dir,
        input_parquet=in_parquet,
        frozen_stats_path=frozen_stats,
        output_parquet=out_parquet,
        batch_size=args.batch_size,
        device=args.device,
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=ood_training_parquet,
        mode_ambiguous_grid_path=mode_ambiguous_grid,
        selection_artifact_path=None,
        layout=layout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
