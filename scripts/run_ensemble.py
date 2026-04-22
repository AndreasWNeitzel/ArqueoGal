"""Run A — ensemble driver (#134).

Trains 5 supervised-fine-tune members sequentially, each starting from the
same pretrained encoder checkpoint and varying the random seed. Head weights
reinit per seed (``reload_head_from_pretrained=False`` — the DESIGN-mandated
default). Sequential training keeps VRAM usage bounded on the RTX 3060.

Architectural decisions inherited from :mod:`run_supervised_finetune`:
- ``use_c0_scalars=True``
- ``encoder_lr_ratio=0.1``
- ``pretrained_encoder_ckpt=<contrastive pretrain ckpt>``
- ``LossWeights(supcon=0.1, beta_nll=1.0, beta=0.0)`` — β=0 exposes per-cell
  μ bias as explicit mean error instead of letting β-NLL (β=0.5) absorb it
  into inflated σ. A small SupCon auxiliary term (supcon=0.1) keeps the
  latent structure from collapsing while the head re-learns with the
  honest NLL. See 2026-04-21 catastrophe diagnosis (prototype attractor
  at [α/M]=+0.11, [M/H]=-1).
- ``grad_norm_abort_threshold=500.0`` — β=0 canary. Pure Gaussian NLL can
  explode on high-σ samples; abort early if grads diverge.
- ``early_stop_patience=3``, ``epochs=10``
- ``output_prefix="xp_abundances_main_ensemble"``

Each member checkpoint lands in its own run-directory alongside a shared
``ensemble_history.json`` that records per-seed best-val losses, β-NLL
spread, and paths — consumed by the #135 calibration harness.

Run: ``PYTHONPATH=src python scripts/run_ensemble.py --pretrained <path>``.
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


def _tiers_for_label_set(label_set: str) -> LabelTiers:
    if label_set == "5":
        return LabelTiers.five_label()
    return LabelTiers()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_ensemble")

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


def build_ensemble_config(
    *, parquet: Path, output_dir: Path, pretrained_ckpt: Path,
    epochs: int, batch_size: int, seeds: tuple[int, ...],
    output_prefix: str = "xp_abundances_main_ensemble",
    inverse_freq_weighting: bool = False,
    inverse_freq_clip: float = 5.0,
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
        early_stop_patience=3,
        early_stop_min_delta=1e-4,
        relative_min_delta=False,
        use_c0_scalars=True,
        encoder_lr_ratio=0.1,
        pretrained_encoder_ckpt=pretrained_ckpt,
        reload_head_from_pretrained=False,
        checkpoint_every_n_epochs=0,  # per-member only keeps best-val
        output_prefix=output_prefix,
        loss_weights=LossWeights(
            supcon=0.1, beta_nll=1.0,
            beta=0.0, supcon_sigma=0.10, supcon_label_n_first=None,
        ),
        grad_norm_abort_threshold=500.0,
        temperature_init=0.10,
        ensemble_seeds=seeds,
        inverse_freq_weighting=inverse_freq_weighting,
        inverse_freq_clip=inverse_freq_clip,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--label-set", choices=("21", "5"), default="21",
        help="21 = default LabelTiers (production, 4-block Cholesky). "
             "5 = LabelTiers.five_label() {Teff, logg, [M/H], [α/M], [Mg/H]} "
             "with a single 5×5 full Cholesky block (production as of #143).",
    )
    parser.add_argument(
        "--inverse-freq", action="store_true",
        help="Enable inverse-frequency [M/H]-bin weighting in the NLL. "
             "v1.1 fix for the metal-poor [α/M] regression-to-mean "
             "surfaced downstream of v1 (#198).",
    )
    parser.add_argument(
        "--inverse-freq-clip", type=float, default=5.0,
        help="Max w = 1/p(bin) before mean-1 normalisation. Default 5.0.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    tiers = _tiers_for_label_set(args.label_set)
    label_tag = f"_{args.label_set}label" if args.label_set != "21" else ""
    output_prefix = f"xp_abundances_main_ensemble{label_tag}"
    ensemble_suffix = f"_ensemble{label_tag}"

    date_tag = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    sha = _git_sha()
    sha7 = sha[:7] if sha != "nogit" else "nogit"
    args.report_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = build_ensemble_config(
        parquet=args.parquet, output_dir=args.model_dir / "pending",
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs, batch_size=args.batch_size, seeds=seeds,
        output_prefix=output_prefix,
        inverse_freq_weighting=args.inverse_freq,
        inverse_freq_clip=args.inverse_freq_clip,
    )
    cfg_hash = _cfg_hash(tmp_cfg)
    ensemble_dir = args.model_dir / f"{date_tag}_{sha7}_{cfg_hash}{ensemble_suffix}"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_ensemble_config(
        parquet=args.parquet, output_dir=ensemble_dir,
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs, batch_size=args.batch_size, seeds=seeds,
        output_prefix=output_prefix,
        inverse_freq_weighting=args.inverse_freq,
        inverse_freq_clip=args.inverse_freq_clip,
    )
    layout = FeatureLayout()

    with (args.report_dir / "ensemble_config.json").open("w") as f:
        payload = asdict(cfg)
        payload["train_parquet"] = str(payload["train_parquet"])
        payload["output_dir"] = str(payload["output_dir"])
        payload["pretrained_encoder_ckpt"] = str(payload["pretrained_encoder_ckpt"])
        payload["git_sha"] = sha
        payload["cfg_hash"] = cfg_hash
        json.dump(payload, f, indent=2, default=str)
    _LOG.info("ensemble_dir=%s cfg_hash=%s sha=%s seeds=%s",
              ensemble_dir, cfg_hash, sha7, seeds)

    if args.dry_run:
        _LOG.info("dry run — skipping training")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    members: list[dict] = []
    for seed in seeds:
        _LOG.info("=== training ensemble member seed=%d ===", seed)
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
        "pretrained_ckpt": str(args.pretrained),
        "cfg_hash": cfg_hash,
        "git_sha": sha,
        "device": str(device),
        "members": members,
        "val_loss_mean": float(sum(m["best_val_loss"] for m in members) / len(members)),
        "val_loss_spread": float(
            max(m["best_val_loss"] for m in members)
            - min(m["best_val_loss"] for m in members),
        ),
    }
    with (args.report_dir / "ensemble_history.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _LOG.info(
        "ensemble done: mean val loss=%.4f, spread=%.4f, %d members",
        summary["val_loss_mean"], summary["val_loss_spread"], len(members),
    )


if __name__ == "__main__":
    main()
