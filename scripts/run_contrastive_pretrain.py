"""Run A — contrastive pretraining driver (#131).

Runs :func:`arqueogal.xp_abundances.main.training.train_model` with the
contrastive-pretrain configuration the project's locked architectural
decisions call for:

- ``use_c0_scalars=False`` — XP c0 scalars zeroed via the #130 adapter so
  positive pairs in the same (Teff, logg, [M/H]) cell do not align on
  luminosity/extinction confounders.
- ``LossWeights(supcon=1.0, beta_nll=0.0, supcon_label_n_first=None)`` — SupCon
  Gaussian-kernel pair weighting on **all** production labels (Tier-1
  atmospherics + [α/M] + [Mg/H] for the 5-label head). The pre-v2 run used
  ``supcon_label_n_first=3`` (atmospherics only), which trained the encoder
  to treat stars with identical Teff/logg/[M/H] as maximally positive
  regardless of chemistry → encoder became α/M-blind → supervised head
  developed a prototype attractor at `[α/M]=+0.11, [M/H]=-1`. See
  ``docs/decisions/0014_contrastive_alpham_blind_catastrophe.md``.
- Relative min-delta early stopping (``early_stop_min_delta=0.01,
  relative_min_delta=True``) — SupCon loss magnitudes drift over training so
  absolute deltas are misleading; 1 % relative improvement is the patience
  trigger.
- ``checkpoint_every_n_epochs=10`` cadence rollback points, plus the
  best-val checkpoint returned by ``train_model``.
- ``output_prefix="xp_abundances_main_contrastive"`` keeps the phase's
  checkpoints from clobbering the supervised/ensemble runs.

Outputs:

- ``models/main/xp_abundances/<date>_<sha>_<cfg-hash>/xp_abundances_main_contrastive_seed<N>_best.pt``
- ``models/main/xp_abundances/<date>_<sha>_<cfg-hash>/cadence/…epoch####.pt`` (every 10 epochs)
- ``reports/pipeline1/run_a/contrastive_history.json`` — JSON-canonical log.
- ``reports/pipeline1/run_a/contrastive_config.json`` — flattened TrainingConfig.

Run: ``PYTHONPATH=src python scripts/run_contrastive_pretrain.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.training import save_checkpoint, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_contrastive_pretrain")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_MODEL_DIR = REPO_ROOT / "models/main/xp_abundances"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/run_a"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _cfg_hash(cfg: TrainingConfig) -> str:
    """Stable short hash of the config — pins the run directory name."""
    payload = json.dumps(asdict(cfg), default=str, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:7]


def build_contrastive_config(
    *,
    parquet: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
) -> TrainingConfig:
    return TrainingConfig(
        train_parquet=parquet,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=2,
        amp_dtype="bfloat16",
        max_lr=2e-3,
        pct_start=0.15,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        early_stop_patience=5,
        early_stop_min_delta=0.01,
        relative_min_delta=True,
        use_c0_scalars=False,
        encoder_lr_ratio=1.0,
        checkpoint_every_n_epochs=10,
        output_prefix="xp_abundances_main_contrastive",
        loss_weights=LossWeights(
            supcon=1.0,
            beta_nll=0.0,
            beta=0.5,
            supcon_sigma=0.10,
            supcon_label_n_first=None,  # All 5 labels — fixes α/M-blind encoder (2026-04-21).
        ),
        temperature_init=0.10,
        ensemble_seeds=(0,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Hard cap per Run A directive; patience stops earlier.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label-set",
        choices=("21", "5"),
        default="5",
        help="21 = default LabelTiers (21 columns, several with 1-5%% NaN). "
        "5 = LabelTiers.five_label() {Teff, logg, [M/H], [α/M], [Mg/H]} — "
        "matches the 5-label production head's label space, so the "
        "SupCon kernel trains the encoder to respect the same geometry "
        "the supervised head will fine-tune on. Default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build cfg + model scaffold, skip training (smoke test).",
    )
    args = parser.parse_args()

    date_tag = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    sha = _git_sha()
    sha7 = sha[:7] if sha != "nogit" else "nogit"

    args.report_dir.mkdir(parents=True, exist_ok=True)

    # Config scaffold — output_dir is a per-run subdir so checkpoints group.
    tmp_cfg = build_contrastive_config(
        parquet=args.parquet,
        output_dir=args.model_dir / "pending",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    cfg_hash = _cfg_hash(tmp_cfg)
    run_dir = args.model_dir / f"{date_tag}_{sha7}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_contrastive_config(
        parquet=args.parquet,
        output_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    layout = FeatureLayout()
    tiers = LabelTiers.five_label() if args.label_set == "5" else LabelTiers()

    with (args.report_dir / "contrastive_config.json").open("w") as f:
        payload = asdict(cfg)
        payload["train_parquet"] = str(payload["train_parquet"])
        payload["output_dir"] = str(payload["output_dir"])
        payload["git_sha"] = sha
        payload["cfg_hash"] = cfg_hash
        json.dump(payload, f, indent=2, default=str)
    _LOG.info("run_dir=%s cfg_hash=%s sha=%s", run_dir, cfg_hash, sha7)

    if args.dry_run:
        _LOG.info("dry run — skipping train_model")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)
    result = train_model(cfg, layout, tiers, seed=args.seed, device=device)

    # Write the best-val checkpoint with DESIGN-compliant filename.
    best_path = save_checkpoint(
        run_dir / f"{cfg.output_prefix}_seed{args.seed}_best.pt",
        model=result["model"],
        log_temp=result["log_temp"],
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=result["label_scaler"],
        seed=args.seed,
        training_metrics={
            "best_val_loss": float(result["best_val_loss"]),
            "best_epoch": int(result["best_epoch"]),
            "history": result["history"],
            "cadence_checkpoints": [str(p) for p in result["cadence_checkpoints"]],
        },
        git_sha=sha,
    )
    with (args.report_dir / "contrastive_history.json").open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "best_path": str(best_path),
                "best_val_loss": float(result["best_val_loss"]),
                "best_epoch": int(result["best_epoch"]),
                "device": str(device),
                "cfg_hash": cfg_hash,
                "git_sha": sha,
                "history": result["history"],
                "cadence_checkpoints": [str(p) for p in result["cadence_checkpoints"]],
            },
            f,
            indent=2,
            default=str,
        )
    _LOG.info(
        "done: best_val_loss=%.4f at epoch %d, saved to %s",
        result["best_val_loss"],
        result["best_epoch"],
        best_path,
    )


if __name__ == "__main__":
    main()
