"""Run A — supervised fine-tune driver (#133).

Loads the pretrained encoder from a contrastive-pretrain checkpoint and
trains the block-Cholesky head against the 21 APOGEE labels under β-NLL.
Architectural decisions locked by the DESIGN + Run A directive:

- ``use_c0_scalars=True`` — supervised head gets a real luminosity/extinction
  degree of freedom via c0 scalars.
- ``encoder_lr_ratio=0.1`` — pretrained encoder trains at 10× lower LR than
  the head (two AdamW groups) per DESIGN.
- ``LossWeights(supcon=0.0, beta_nll=1.0)`` — no contrastive term at fine-tune.
- ``early_stop_patience=3`` epochs on absolute min_delta — supervised losses
  have stable magnitudes so absolute thresholds are appropriate.
- Hard cap ``epochs=10``.
- ``checkpoint_every_n_epochs=1`` — fine-tune is short, every epoch matters.
- ``output_prefix="xp_abundances_main_finetune"``.

Run: ``PYTHONPATH=src python scripts/run_supervised_finetune.py --pretrained <path>``.
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
    """Resolve the ``--label-set`` CLI value to a concrete :class:`LabelTiers`."""
    if label_set == "5":
        return LabelTiers.five_label()
    return LabelTiers()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_supervised_finetune")

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


def build_finetune_config(
    *,
    parquet: Path,
    output_dir: Path,
    pretrained_ckpt: Path,
    epochs: int,
    batch_size: int,
    beta: float = 0.5,
    grad_norm_abort_threshold: float = float("inf"),
    output_prefix: str = "xp_abundances_main_finetune",
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
        early_stop_patience=20,  # cadence run: 200 ep / patience 20 to match pretrain
        early_stop_min_delta=1e-4,
        relative_min_delta=False,
        use_c0_scalars=True,
        encoder_lr_ratio=0.1,
        pretrained_encoder_ckpt=pretrained_ckpt,
        reload_head_from_pretrained=False,  # head trains from scratch per seed
        checkpoint_every_n_epochs=1,
        output_prefix=output_prefix,
        loss_weights=LossWeights(
            supcon=1.0,  # canonical
            beta_nll=1.0,  # canonical
            barlow=0.5,  # canonical
            ari=0.0,  # OFF — sigmoid-hard split at [α/M]=0.15 was
            # creating a "barbell" prediction pattern at
            # any weight; user disabled 2026-05-03.
            beta=beta,
            supcon_sigma=0.10,
            supcon_label_n_first=None,
        ),
        temperature_init=0.10,
        ensemble_seeds=(0,),
        grad_norm_abort_threshold=grad_norm_abort_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=200)  # cadence run: 200 ep / patience 20
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Seitzer β in β-NLL. 0.5 is the production default; 0.0 is the "
        "pure-Gaussian-NLL canary for the #135 escalation.",
    )
    parser.add_argument(
        "--grad-norm-abort",
        type=float,
        default=None,
        help="Abort training if any batch exceeds this pre-clip grad norm. "
        "Defaults to 500.0 when --beta=0 (canary), inf otherwise.",
    )
    parser.add_argument(
        "--label-set",
        choices=("21", "5"),
        default="21",
        help="21 = default LabelTiers (production, 4-block Cholesky). "
        "5 = LabelTiers.five_label() {Teff, logg, [M/H], [α/M], [Mg/H]} "
        "with a single 5×5 full Cholesky block (#143).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # β=0 canary gets a sensible grad-norm abort by default; 500 is well above
    # typical β=0.5 grads (0.5–5) so it only fires on the pathological case.
    if args.grad_norm_abort is None:
        grad_abort = 500.0 if args.beta == 0.0 else float("inf")
    else:
        grad_abort = args.grad_norm_abort

    tiers = _tiers_for_label_set(args.label_set)
    label_tag = f"_{args.label_set}label" if args.label_set != "21" else ""

    if args.beta == 0.0:
        output_prefix = f"xp_abundances_main_finetune_beta0{label_tag}"
    else:
        output_prefix = f"xp_abundances_main_finetune{label_tag}"

    date_tag = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    sha = _git_sha()
    sha7 = sha[:7] if sha != "nogit" else "nogit"
    args.report_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = build_finetune_config(
        parquet=args.parquet,
        output_dir=args.model_dir / "pending",
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        grad_norm_abort_threshold=grad_abort,
        output_prefix=output_prefix,
    )
    cfg_hash = _cfg_hash(tmp_cfg)
    if args.beta == 0.0:
        run_suffix = f"_finetune_beta0{label_tag}"
    else:
        run_suffix = f"_finetune{label_tag}"
    run_dir = args.model_dir / f"{date_tag}_{sha7}_{cfg_hash}{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_finetune_config(
        parquet=args.parquet,
        output_dir=run_dir,
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        grad_norm_abort_threshold=grad_abort,
        output_prefix=output_prefix,
    )
    layout = FeatureLayout()

    config_name = f"finetune_config{label_tag}.json"
    history_name = f"finetune_history{label_tag}.json"
    with (args.report_dir / config_name).open("w") as f:
        payload = asdict(cfg)
        payload["train_parquet"] = str(payload["train_parquet"])
        payload["output_dir"] = str(payload["output_dir"])
        payload["pretrained_encoder_ckpt"] = str(payload["pretrained_encoder_ckpt"])
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

    best_path = save_checkpoint(
        run_dir / f"{cfg.output_prefix}_seed{args.seed}_best.pt",
        model=result["model"],
        log_temp=result["log_temp"],
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=result["label_scaler"],
        feature_scaler=result.get("feature_scaler"),  # required for inference reload
        seed=args.seed,
        training_metrics={
            "best_val_loss": float(result["best_val_loss"]),
            "best_epoch": int(result["best_epoch"]),
            "history": result["history"],
            "cadence_checkpoints": [str(p) for p in result["cadence_checkpoints"]],
        },
        git_sha=sha,
    )
    with (args.report_dir / history_name).open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "best_path": str(best_path),
                "pretrained_ckpt": str(args.pretrained),
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
