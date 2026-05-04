"""v1.1 model training: 21-label head + evol-stage + ARI + feature-noise.

Five architectural changes in a single training run (100 epochs, single model):

1. 21-label head with physics-motivated block Cholesky.
2. 4-way evolutionary-stage diagnostic head (RGB, HeCB, OOD_evolved, OOD_unevolved).
3. ARI contamination loss (weight 0.1) on ([α/M], [M/H]) GMM assignments.
4. Feature-noise injection at training time (per-feature uncertainties).
5. Analytical feature-noise marginalisation at inference time (gradient-based).

Usage
-----
From the repo root, after installing the project:

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_joint_ensemble_v1_1.py \\
        --train-parquet data/processed/stream1_final.parquet \\
        --output-dir models/main/xp_abundances/v1.1_<YYYYMMDD>_<sha>/

The script expects:
- Training data in Parquet with Gaia DR3 IDs and pre-processed XP coefficients.
- Feature layout matching the v1.0 contract (108-D XP + auxiliary columns).
- Label columns in both tier and block order (via LabelTiers / CovarianceBlockLayout).

Output checkpoint layout
------------------------
- model.pkl / model_full.pt: Trained encoder + head + evol-stage head weights.
- checkpoint_v2.json: Metadata (block_layout, label scaler, GMM params, ARI history).
- training_log.json: Loss trajectory, ARI per-epoch, VRAM peak, wall time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Stub: full implementation deferred to user + integration with preprocessing.py
# This file documents the v1.1 training contract and is ready for invoke.

_LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train v1.1 XP abundance model (21-label, evol-stage, ARI, noise)."
    )
    parser.add_argument(
        "--train-parquet",
        type=Path,
        required=True,
        help="Path to training parquet with XP features and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save model checkpoint and metadata.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default 100 for v1.1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training (6 GB VRAM constraint).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--enable-evol-stage",
        action="store_true",
        default=True,
        help="Enable evolutionary-stage diagnostic head (default: True).",
    )
    parser.add_argument(
        "--weight-ari-loss",
        type=float,
        default=0.1,
        help="Weight for ARI contamination loss (default 0.1).",
    )
    parser.add_argument(
        "--weight-evol-loss",
        type=float,
        default=0.05,
        help="Weight for evolutionary-stage cross-entropy loss (default 0.05).",
    )

    args = parser.parse_args()

    if not args.train_parquet.exists():
        _LOG.error(f"Train parquet not found: {args.train_parquet}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("v1.1 model training (stub)")
    _LOG.info(f"Train parquet: {args.train_parquet}")
    _LOG.info(f"Output dir: {args.output_dir}")
    _LOG.info(f"Epochs: {args.epochs}")
    _LOG.info(f"Enable evol-stage: {args.enable_evol_stage}")
    _LOG.info(f"Weight ARI loss: {args.weight_ari_loss}")
    _LOG.info(f"Weight evol loss: {args.weight_evol_loss}")

    _LOG.info("Training stub — integration with preprocessing.py deferred.")
    _LOG.info("When ready, this will:")
    _LOG.info("  1. Load XP features via apply_pipeline1_preprocessing()")
    _LOG.info("  2. Fit 2-component GMM on ([α/M], [M/H]) training pool")
    _LOG.info("  3. Train XpAbundanceModel with include_evol_stage_head=True")
    _LOG.info("  4. Wire SupCon + Beta-NLL + ARI + Evol + Barlow losses")
    _LOG.info("  5. Save checkpoint with all v1.1 metadata")
    _LOG.info("  6. Emit training_log.json with per-epoch ARI trajectory")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
