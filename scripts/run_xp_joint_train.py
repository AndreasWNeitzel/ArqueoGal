"""TESS_ML-style joint training driver — XP-only, 5-label, single stage.

Motivation
----------
ArqueoGal's production recipe is two-stage: SupCon-only contrastive pretrain
(``run_contrastive_pretrain.py``) followed by β-NLL supervised fine-tune
with the encoder quasi-frozen (``encoder_lr_ratio=0.1``). The TESS_ML
prototype (``~/projects/TESS_ML/src/contrastive/training.py``) uses a single
joint-loss training loop

    loss = supcon(z, zk, yk, queue_keys) + gnll(μ, log σ², y) + λ·barlow(h)

with a momentum queue of 8192 keys and all encoder parameters trainable. On
the same XP-coefficient input it successfully separates low-α and high-α
disc stars in chemical space — the property ArqueoGal's v1 → v2 runs have
not recovered.

This script is the minimum-changes port of that recipe into ArqueoGal's
production code paths. No new training loop; ``train_model`` has been
extended (#ref losses.ContrastiveQueue / losses.barlow_twins_loss) to accept
``cfg.queue_size > 0`` and ``cfg.loss_weights.barlow > 0``, at which point
it composes the three losses exactly as TESS_ML does.

Architectural decisions (departures from ``run_ensemble.py``)
-------------------------------------------------------------
- ``FeatureLayout(aux_cols=(), residual_cols=())`` — 110-D XP-only input
  (54 BP + 54 RP + 2 c0 scalars). The 27 aux features carry Gaia photometry,
  astrometry, BJ21 distances, and four dust maps; hypothesis (ADR-0014 Bug
  A root cause #2) is that they swamp the weak XP α-signal at intermediate
  [M/H] and flatten the conditional posterior.
- ``use_c0_scalars=False`` — drop ``bp_c0_z`` / ``rp_c0_z`` entirely (XP
  *shape* coefficients only, 108-D effective). Matches TESS_ML's
  ``normalize_by_first_coeff: true`` choice: the prototype lets the shape
  coefficients carry abundance information without the luminosity/extinction
  confounder of absolute c0.
- ``LabelTiers.five_label()`` — {Teff, log g, [M/H], [α/M], [Mg/H]}. SupCon
  kernel weights on all five (ADR-0014 D1: not just Tier-1 atmospherics).
- ``LossWeights(supcon=1.0, beta_nll=1.0, beta=0.0, barlow=1.0,
  barlow_lam=0.005, supcon_label_n_first=None, supcon_sigma=0.10)`` — the
  TESS_ML joint triple. β=0 is plain MVN-NLL (no Seitzer σ-weighting);
  barlow prevents latent collapse; supcon uses all 5 labels.
- ``queue_size=8192, queue_warm_start=True`` — raises SupCon effective key
  count to ``batch_size + 8192``. The prototype attributes its chemistry
  separation to this.
- ``encoder_lr_ratio=1.0`` — encoder trains jointly with head. No quasi-
  freezing. (ADR-0014 Bug B root cause.)
- ``epochs=200`` — TESS_ML used 1000 on 2 labels; 200 on our 5-label
  training set (~290k stars dedup) converges the smoke without an
  overnight wall-clock.
- ``ensemble_seeds=(0,)`` — single-seed smoke. Once chemical-space
  separation is confirmed on Stream-1 val, the recipe gets rolled into
  the ensemble with more seeds.

Outputs
-------
- ``models/main/xp_abundances/<date>_<sha>_<cfg>_xp_joint/xp_abundances_main_xp_joint_seed0_best.pt``
- ``reports/pipeline1/run_a/xp_joint_history.json`` — JSON training log.
- ``reports/pipeline1/run_a/xp_joint_config.json`` — flattened TrainingConfig.

Run: ``PYTHONPATH=src python scripts/run_xp_joint_train.py``.
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
_LOG = logging.getLogger("run_xp_joint_train")

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
    payload = json.dumps(asdict(cfg), default=str, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:7]


def build_joint_config(
    *,
    parquet: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    queue_size: int,
    barlow_weight: float,
    barlow_lam: float,
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
        early_stop_patience=20,
        early_stop_min_delta=1e-4,
        use_c0_scalars=False,
        encoder_lr_ratio=1.0,
        checkpoint_every_n_epochs=20,
        output_prefix="xp_abundances_main_xp_joint",
        loss_weights=LossWeights(
            supcon=1.0,
            beta_nll=1.0,
            beta=0.0,
            barlow=barlow_weight,
            barlow_lam=barlow_lam,
            supcon_sigma=0.10,
            supcon_label_n_first=None,
        ),
        temperature_init=0.10,
        queue_size=queue_size,
        queue_warm_start=True,
        grad_norm_abort_threshold=500.0,
        ensemble_seeds=(0,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--queue-size", type=int, default=8192)
    parser.add_argument("--barlow-weight", type=float, default=1.0)
    parser.add_argument("--barlow-lam", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    date_tag = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    sha = _git_sha()
    sha7 = sha[:7] if sha != "nogit" else "nogit"

    args.report_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = build_joint_config(
        parquet=args.parquet,
        output_dir=args.model_dir / "pending",
        epochs=args.epochs,
        batch_size=args.batch_size,
        queue_size=args.queue_size,
        barlow_weight=args.barlow_weight,
        barlow_lam=args.barlow_lam,
    )
    cfg_hash = _cfg_hash(tmp_cfg)
    run_dir = args.model_dir / f"{date_tag}_{sha7}_{cfg_hash}_xp_joint"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_joint_config(
        parquet=args.parquet,
        output_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        queue_size=args.queue_size,
        barlow_weight=args.barlow_weight,
        barlow_lam=args.barlow_lam,
    )

    # XP-only 110-D layout. 108-D effective after use_c0_scalars=False zeroes c0.
    layout = FeatureLayout(aux_cols=(), residual_cols=())
    tiers = LabelTiers.five_label()

    with (args.report_dir / "xp_joint_config.json").open("w") as f:
        payload = asdict(cfg)
        payload["train_parquet"] = str(payload["train_parquet"])
        payload["output_dir"] = str(payload["output_dir"])
        payload["git_sha"] = sha
        payload["cfg_hash"] = cfg_hash
        payload["feature_layout"] = {
            "input_dim": layout.input_dim,
            "aux_cols": list(layout.aux_cols),
            "residual_cols": list(layout.residual_cols),
        }
        payload["tiers"] = list(tiers.all_labels)
        json.dump(payload, f, indent=2, default=str)
    _LOG.info(
        "run_dir=%s cfg_hash=%s sha=%s input_dim=%d labels=%s queue=%d barlow=%s",
        run_dir,
        cfg_hash,
        sha7,
        layout.input_dim,
        tiers.all_labels,
        args.queue_size,
        args.barlow_weight,
    )

    if args.dry_run:
        _LOG.info("dry run — skipping train_model")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)
    result = train_model(cfg, layout, tiers, seed=args.seed, device=device)

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
    with (args.report_dir / "xp_joint_history.json").open("w") as f:
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
