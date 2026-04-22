"""Joint-loss training — production feature layout, 5-label, ensemble.

Supersedes the two-stage pretrain→finetune path (``run_contrastive_pretrain``
→ ``run_ensemble``) for Pipeline 1 v3+. Single-stage training with

    loss = λ_supcon · supcon(z, zk, yk, queue) + β_nll(μ, L, y) + λ_bt · BT(h)

over the production 140-D feature layout: 54 BP shape coefs + 54 RP shape
coefs + 2 c0_z scalars + 3 Hermite-reprojection residuals + 27 aux features
(Gaia photometry + astrometry + BJ21 distances + four dust maps +
GSP-Phot Av). Same layout as ``run_ensemble.py`` — the data contract is
preserved; only the architecture changes.

Motivation
----------
The two-stage recipe's chemistry-space failure (ADR-0014) was diagnosed as
the compound effect of four bugs: SupCon kernel weighted on Tier-1 only
(``supcon_label_n_first=3``), encoder quasi-frozen at
``encoder_lr_ratio=0.1``, no latent-collapse prevention, and no momentum
queue. The XP-only smoke (``run_xp_joint_train.py``, 2026-04-22) confirmed
that the TESS_ML joint-loss recipe — single-stage, SupCon on all labels,
momentum queue, Barlow Twins, encoder trains jointly — restores the
low-α / high-α disc bifurcation on Stream-1 val (attractor-stripe fraction
5.15 %, per-[M/H]-bin median residuals ±0.01 dex).

This driver ports that recipe onto the production feature set so the full
five-label head (Teff, log g, [M/H], [α/M], [Mg/H]) benefits from it,
including the extinction-corrected aux features.

Architectural choices
---------------------
- ``FeatureLayout()`` — production default, 140-D. Not the XP-only
  110-D smoke variant.
- ``use_c0_scalars=True`` — the aux dust stack disentangles the
  luminosity / extinction degree of freedom, so c0_z is useful here.
  (The XP-only smoke turned this off because without aux features the
  network couldn't tell c0_z variation from distance variation.)
- ``encoder_lr_ratio=1.0`` — encoder trains jointly with the head.
  No quasi-freezing. (ADR-0014 Bug B resolution.)
- ``loss_weights = LossWeights(supcon=1.0, beta_nll=1.0, beta=0.0,
  barlow=1.0, barlow_lam=0.005, supcon_sigma=0.10,
  supcon_label_n_first=None)``. β=0 is plain MVN-NLL (no Seitzer
  σ-weighting — that was the σ-absorbs-bias Bug B mechanism). All five
  labels participate in the SupCon kernel (ADR-0014 Bug A resolution).
- ``queue_size=8192, queue_warm_start=True`` — momentum queue gives SupCon
  an effective key count of ``batch_size + 8192`` per step.
- ``grad_norm_abort_threshold=500.0`` — β=0 canary.
- ``LabelTiers.five_label()`` — {Teff, log g, [M/H], [α/M], [Mg/H]} with a
  single 5×5 Cholesky block.

Outputs
-------
- ``models/main/xp_abundances/<date>_<sha>_<cfg>_joint/member_seed<N>/``
  with one best-val checkpoint per seed.
- ``reports/pipeline1/run_a/joint_ensemble_config.json`` — flattened cfg.
- ``reports/pipeline1/run_a/joint_ensemble_history.json`` — per-seed
  best-val losses, spread, paths.

Run: ``PYTHONPATH=src python scripts/run_joint_ensemble.py --seeds 0``.
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
_LOG = logging.getLogger("run_joint_ensemble")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_MODEL_DIR = REPO_ROOT / "models/main/xp_abundances"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/run_a"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _cfg_hash(cfg: TrainingConfig) -> str:
    payload = json.dumps(asdict(cfg), default=str, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:7]


def build_joint_config(
    *, parquet: Path, output_dir: Path, epochs: int, batch_size: int,
    queue_size: int, barlow_weight: float, barlow_lam: float,
    seeds: tuple[int, ...], max_lr: float, pct_start: float,
    grad_abort_threshold: float,
) -> TrainingConfig:
    return TrainingConfig(
        train_parquet=parquet,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=2,
        amp_dtype="bfloat16",
        max_lr=max_lr,
        pct_start=pct_start,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        early_stop_patience=20,
        early_stop_min_delta=1e-4,
        use_c0_scalars=True,
        encoder_lr_ratio=1.0,
        checkpoint_every_n_epochs=20,
        stage_dataset_on_gpu=True,
        grad_norm_abort_threshold=grad_abort_threshold,
        output_prefix="xp_abundances_main_joint",
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
        ensemble_seeds=seeds,
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
    parser.add_argument("--max-lr", type=float, default=1e-3,
                        help="OneCycleLR peak. Default 1e-3 for production-139D. "
                             "The XP-only 110-D smoke used 2e-3 with no grad blow-up; "
                             "the extra residuals + aux features destabilize SupCon "
                             "under 2e-3 within the first 5 epochs.")
    parser.add_argument("--pct-start", type=float, default=0.3,
                        help="OneCycleLR warmup fraction. Default 0.3 for 139-D "
                             "(vs 0.15 in the XP-only smoke) — longer warmup lets "
                             "the queue fill before the LR peaks.")
    parser.add_argument("--grad-abort-threshold", type=float, default=5000.0,
                        help="Pre-clip grad-norm ceiling. SupCon's log-sum-exp "
                             "over queue+batch at τ~0.1 is inherently "
                             "high-dynamic-range; the 500.0 canary used in the "
                             "β=0 fine-tune is miscalibrated for joint-loss "
                             "training — parameter updates are bounded by "
                             "grad_clip_norm=1.0 regardless. Pass `inf` to disable.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--tier-set", choices=["five", "two"], default="five",
                        help="Label-tier set to train on. 'five' is the production "
                             "{Teff, logg, [M/H], [α/M], [Mg/H]} head; 'two' is the "
                             "TESS_ML-matched {[M/H], [α/M]} capacity-dilution test.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    date_tag = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    sha = _git_sha()
    sha7 = sha[:7] if sha != "nogit" else "nogit"

    args.report_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = build_joint_config(
        parquet=args.parquet, output_dir=args.model_dir / "pending",
        epochs=args.epochs, batch_size=args.batch_size,
        queue_size=args.queue_size,
        barlow_weight=args.barlow_weight, barlow_lam=args.barlow_lam,
        seeds=seeds, max_lr=args.max_lr, pct_start=args.pct_start,
        grad_abort_threshold=args.grad_abort_threshold,
    )
    cfg_hash = _cfg_hash(tmp_cfg)
    ensemble_dir = args.model_dir / f"{date_tag}_{sha7}_{cfg_hash}_joint"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_joint_config(
        parquet=args.parquet, output_dir=ensemble_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        queue_size=args.queue_size,
        barlow_weight=args.barlow_weight, barlow_lam=args.barlow_lam,
        seeds=seeds, max_lr=args.max_lr, pct_start=args.pct_start,
        grad_abort_threshold=args.grad_abort_threshold,
    )

    # Production 140-D layout.
    layout = FeatureLayout()
    if args.tier_set == "five":
        tiers = LabelTiers.five_label()
    elif args.tier_set == "two":
        tiers = LabelTiers.two_label()
    else:  # pragma: no cover — argparse `choices=` prevents this
        raise ValueError(f"unknown tier set: {args.tier_set}")

    with (args.report_dir / "joint_ensemble_config.json").open("w") as f:
        payload = asdict(cfg)
        payload["train_parquet"] = str(payload["train_parquet"])
        payload["output_dir"] = str(payload["output_dir"])
        payload["git_sha"] = sha
        payload["cfg_hash"] = cfg_hash
        payload["feature_layout"] = {
            "input_dim": layout.input_dim,
            "aux_cols": list(layout.aux_cols),
            "residual_cols": list(layout.residual_cols),
            "xp_scalar_cols": list(layout.xp_scalar_cols),
        }
        payload["tiers"] = list(tiers.all_labels)
        json.dump(payload, f, indent=2, default=str)
    _LOG.info(
        "ensemble_dir=%s cfg_hash=%s sha=%s input_dim=%d labels=%s seeds=%s queue=%d barlow=%s",
        ensemble_dir, cfg_hash, sha7, layout.input_dim, tiers.all_labels,
        seeds, args.queue_size, args.barlow_weight,
    )

    if args.dry_run:
        _LOG.info("dry run — skipping training")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    members: list[dict] = []
    for seed in seeds:
        _LOG.info("=== training joint-ensemble member seed=%d ===", seed)
        result = train_model(cfg, layout, tiers, seed=seed, device=device)
        member_dir = ensemble_dir / f"member_seed{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_checkpoint(
            member_dir / f"{cfg.output_prefix}_seed{seed}_best.pt",
            model=result["model"], log_temp=result["log_temp"],
            cfg=cfg, layout=layout, tiers=tiers,
            label_scaler=result["label_scaler"],
            seed=seed,
            training_metrics={
                "best_val_loss": float(result["best_val_loss"]),
                "best_epoch": int(result["best_epoch"]),
                "history": result["history"],
                "cadence_checkpoints": [str(p) for p in result["cadence_checkpoints"]],
            },
            git_sha=sha,
        )
        members.append({
            "seed": seed,
            "ckpt": str(best_path),
            "best_val_loss": float(result["best_val_loss"]),
            "best_epoch": int(result["best_epoch"]),
            "history": result["history"],
        })
        _LOG.info(
            "member seed=%d: best_val_loss=%.4f at epoch %d",
            seed, result["best_val_loss"], result["best_epoch"],
        )

    summary = {
        "ensemble_dir": str(ensemble_dir),
        "cfg_hash": cfg_hash,
        "git_sha": sha,
        "device": str(device),
        "members": members,
        "val_loss_mean": float(sum(m["best_val_loss"] for m in members) / len(members)),
        "val_loss_spread": float(
            max(m["best_val_loss"] for m in members)
            - min(m["best_val_loss"] for m in members),
        ) if len(members) > 1 else 0.0,
    }
    with (args.report_dir / "joint_ensemble_history.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _LOG.info(
        "joint ensemble done: mean val loss=%.4f, spread=%.4f, %d members",
        summary["val_loss_mean"], summary["val_loss_spread"], len(members),
    )


if __name__ == "__main__":
    main()
