"""Per-label Stream-1 val diagnostics for a joint-trained Pipeline 1 checkpoint.

Companion to ``plot_xp_joint_chemistry.py``. That script answers one question
(does chemistry bifurcate?); this one answers the rest — does the single-stage
joint-loss recipe produce good per-star predictions for every label in
``LabelTiers.five_label()``:

    {Teff, log g, [M/H], [α/M], [Mg/H]}

For each label, the figure shows:

- Row 1: hexbin pred-vs-truth with 1:1 line and per-label axis range.
- Row 2: residual (pred − truth) histogram with Gaussian fit;
  legend reports mean, MAE, RMSE, σ_res.

Summary JSON at ``<out-dir>/<prefix>_validation_summary.json`` carries the
sample size and per-label mean_residual / MAE / RMSE / median / p16 / p84.

Auto-detects the feature layout from the checkpoint's ``input_dim``:
110 → XP-only smoke (``FeatureLayout(aux_cols=(), residual_cols=())``),
139 → production (``FeatureLayout()``, XP + c0_z + 3 residuals + 26 aux).

Run: ``PYTHONPATH=src python scripts/plot_joint_validation.py \\
        --ckpt models/main/xp_abundances/<run-dir>/<ckpt>.pt``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelTiers,
)
from arqueogal.xp_abundances.main.model import (
    ModelConfig,
    XpAbundanceModel,
    five_label_block_layout,
)
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("plot_joint_validation")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "reports/gallery/12_pipeline1_validation"

LABEL_ORDER = ("teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee")
LABEL_PRETTY = {
    "teff_apogee": r"$T_{\rm eff}$ [K]",
    "logg_apogee": r"$\log g$ [dex]",
    "mh_apogee": r"$[{\rm M/H}]$ [dex]",
    "alpha_m_apogee": r"$[\alpha/{\rm M}]$ [dex]",
    "mg_h_apogee": r"$[{\rm Mg/H}]$ [dex]",
}
LABEL_RANGE = {
    "teff_apogee": (3500, 5800),
    "logg_apogee": (0.7, 3.9),
    "mh_apogee": (-2.5, 0.6),
    "alpha_m_apogee": (-0.1, 0.45),
    "mg_h_apogee": (-2.0, 0.6),
}
RESIDUAL_RANGE = {
    "teff_apogee": (-400, 400),
    "logg_apogee": (-0.6, 0.6),
    "mh_apogee": (-0.5, 0.5),
    "alpha_m_apogee": (-0.2, 0.2),
    "mg_h_apogee": (-0.5, 0.5),
}


def _rebuild_cfg_from_checkpoint(blob: dict) -> TrainingConfig:
    cfg_dict = json.loads(blob["config_yaml"])
    cfg_dict.pop("git_sha", None)
    cfg_dict.pop("cfg_hash", None)
    cfg_dict.pop("feature_layout", None)
    cfg_dict.pop("tiers", None)
    lw = cfg_dict.pop("loss_weights")
    cfg_dict["loss_weights"] = LossWeights(**lw)
    cfg_dict["train_parquet"] = Path(cfg_dict["train_parquet"])
    cfg_dict["output_dir"] = Path(cfg_dict["output_dir"])
    if cfg_dict.get("pretrained_encoder_ckpt"):
        cfg_dict["pretrained_encoder_ckpt"] = Path(cfg_dict["pretrained_encoder_ckpt"])
    cfg_dict["trunk_hidden"] = tuple(cfg_dict["trunk_hidden"])
    cfg_dict["fracs"] = tuple(cfg_dict["fracs"])
    cfg_dict["ensemble_seeds"] = tuple(cfg_dict["ensemble_seeds"])
    cfg_dict["temperature_bounds"] = tuple(cfg_dict["temperature_bounds"])
    cfg_dict["inverse_freq_bin_edges"] = tuple(cfg_dict["inverse_freq_bin_edges"])
    return TrainingConfig(**cfg_dict)


def _layout_from_input_dim(input_dim: int) -> FeatureLayout:
    if input_dim == 110:
        return FeatureLayout(aux_cols=(), residual_cols=())
    if input_dim == 139:
        return FeatureLayout()
    raise ValueError(
        f"input_dim={input_dim} does not match a known joint layout "
        "(110 = XP-only, 139 = production)",
    )


def _predict_val(
    ckpt_path: Path,
    parquet: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, LabelTiers]:
    """Return ``(mu_raw, y_raw, tiers)`` on the val partition, human-order."""
    blob = load_checkpoint(ckpt_path, map_location=device)
    cfg = _rebuild_cfg_from_checkpoint(blob)
    cfg = replace(cfg, train_parquet=parquet)

    layout = _layout_from_input_dim(int(blob["input_dim"]))
    tiers = LabelTiers.five_label()

    train_loader, val_loader, _split_ids, label_scaler, _ = build_dataloaders(
        cfg,
        layout,
        tiers,
        seed=blob.get("seed", 0),
    )
    del train_loader

    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=five_label_block_layout(),
            latent_dim=cfg.latent_dim,
            trunk_hidden=cfg.trunk_hidden,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
        ),
    ).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["regressor"])
    adapter = XpFeatureAdapter(layout, use_c0_scalars=cfg.use_c0_scalars).to(device)

    model.eval()
    mu_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]
            mu_block, _L, _h, _z = model(adapter(x))
            mu_human = model.block_layout.reorder_block_to_human(mu_block)
            mu_chunks.append(mu_human.detach().cpu().float().numpy())
            y_chunks.append(y.detach().cpu().float().numpy())

    mu_scaled = np.concatenate(mu_chunks, axis=0)
    y_scaled = np.concatenate(y_chunks, axis=0)
    mu_raw = label_scaler.inverse_mean(mu_scaled)
    y_raw = label_scaler.inverse_mean(y_scaled)
    return mu_raw, y_raw, tiers


def _validation_figure(
    mu_raw: np.ndarray,
    y_raw: np.ndarray,
    tiers: LabelTiers,
) -> tuple[plt.Figure, dict]:
    """2-row × 5-col: hexbin pred-vs-truth (row 1) + residual hist (row 2)."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150)

    per_label: dict[str, dict] = {}
    n_total = mu_raw.shape[0]

    for col, lbl in enumerate(LABEL_ORDER):
        idx = tiers.all_labels.index(lbl)
        truth = y_raw[:, idx]
        pred = mu_raw[:, idx]
        mask = np.isfinite(truth) & np.isfinite(pred)
        truth, pred = truth[mask], pred[mask]
        residual = pred - truth

        lo, hi = LABEL_RANGE[lbl]
        ax = axes[0, col]
        ax.hexbin(truth, pred, gridsize=50, cmap="viridis", mincnt=1, extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{LABEL_PRETTY[lbl]} — APOGEE truth")
        ax.set_ylabel(f"{LABEL_PRETTY[lbl]} — predicted")
        ax.set_title(f"{lbl.replace('_apogee', '')}  (n={len(truth):,})")

        ax = axes[1, col]
        r_lo, r_hi = RESIDUAL_RANGE[lbl]
        ax.hist(residual, bins=80, range=(r_lo, r_hi), density=True, color="steelblue", alpha=0.7)
        ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
        mean_r = float(np.mean(residual))
        median_r = float(np.median(residual))
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        sigma_r = float(np.std(residual))
        p16 = float(np.percentile(residual, 16))
        p84 = float(np.percentile(residual, 84))
        txt = (
            f"mean={mean_r:+.3g}\n"
            f"median={median_r:+.3g}\n"
            f"MAE={mae:.3g}\n"
            f"RMSE={rmse:.3g}\n"
            f"σ={sigma_r:.3g}"
        )
        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
        ax.set_xlim(r_lo, r_hi)
        ax.set_xlabel(f"Δ{LABEL_PRETTY[lbl]}  (pred − truth)")
        ax.set_ylabel("density")
        ax.set_title(f"Residuals — {lbl.replace('_apogee', '')}")

        per_label[lbl] = {
            "n": int(len(truth)),
            "mean_residual": mean_r,
            "median_residual": median_r,
            "mae": mae,
            "rmse": rmse,
            "sigma_residual": sigma_r,
            "p16_residual": p16,
            "p84_residual": p84,
        }

    fig.tight_layout()

    summary = {"n_val_rows": int(n_total), "per_label": per_label}
    return fig, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Joint best-val checkpoint.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", type=str, default="joint")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("predicting val partition via %s on %s", args.ckpt, device)
    mu_raw, y_raw, tiers = _predict_val(args.ckpt, args.parquet, device)

    fig, summary = _validation_figure(mu_raw, y_raw, tiers)
    png_path = args.out_dir / f"{args.prefix}_validation.png"
    fig.savefig(png_path)
    plt.close(fig)

    summary["ckpt"] = str(args.ckpt)
    summary["parquet"] = str(args.parquet)
    summary["device"] = str(device)
    summary_path = args.out_dir / f"{args.prefix}_validation_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    per = summary["per_label"]
    _LOG.info("wrote %s (n=%d)", png_path, summary["n_val_rows"])
    for lbl in LABEL_ORDER:
        s = per[lbl]
        _LOG.info(
            "  %-15s  mean=%+.3g  MAE=%.3g  RMSE=%.3g  σ=%.3g",
            lbl,
            s["mean_residual"],
            s["mae"],
            s["rmse"],
            s["sigma_residual"],
        )


if __name__ == "__main__":
    main()
