"""Stream-1 val [α/M] vs [M/H] diagnostic for the XP-joint checkpoint.

This is the figure that answers the user's top-priority question: does the
TESS_ML-style joint training (SupCon + Gaussian-NLL + Barlow Twins with a
momentum queue) restore chemical-space structural fidelity — specifically,
the low-α vs high-α disc bifurcation at intermediate [M/H]?

The plot is a 3-panel density:

- Left: APOGEE truth [α/M] vs [M/H] on the Stream-1 val partition. The
  reference: Hayden+2015 α-bimodality is visible as the thick α-enhanced
  branch at [α/M] ~ 0.2 overlapping the thin disc at [α/M] ~ 0.0 across
  [-0.5, 0] [M/H].
- Middle: XP-joint predicted [α/M] vs [M/H] on the same stars. This is
  the output under test.
- Right: 2-D residual density (pred − truth) vs truth [M/H]. A conditional-
  mean collapse would show a band at the mean [α/M] ~ +0.08.

Outputs
-------
- ``reports/gallery/12_pipeline1_validation/xp_joint_alpham_vs_mh.png``
- ``reports/gallery/12_pipeline1_validation/xp_joint_chemistry_summary.json``
  — sample size, attractor-stripe fraction, per-[M/H]-bin median residuals.

Run: ``PYTHONPATH=src python scripts/plot_xp_joint_chemistry.py \\
        --ckpt models/main/xp_abundances/<run-dir>/xp_abundances_main_xp_joint_seed0_best.pt``.
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
from arqueogal.xp_abundances.main.config import TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelScaler,
    LabelTiers,
)
from arqueogal.xp_abundances.main.model import (
    ModelConfig,
    XpAbundanceModel,
    five_label_block_layout,
)
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("plot_xp_joint_chemistry")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "reports/gallery/12_pipeline1_validation"


def _rebuild_cfg_from_checkpoint(blob: dict) -> TrainingConfig:
    """Reconstruct the TrainingConfig used to train the checkpoint."""
    cfg_dict = json.loads(blob["config_yaml"])
    cfg_dict.pop("git_sha", None)
    cfg_dict.pop("cfg_hash", None)
    cfg_dict.pop("feature_layout", None)
    cfg_dict.pop("tiers", None)
    # Re-wrap nested dataclasses.
    from arqueogal.xp_abundances.main.config import LossWeights

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


def _predict_val(
    ckpt_path: Path, parquet: Path, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, LabelTiers]:
    """Return ``(mu_raw, y_raw, tiers)`` on the val partition, in human (tier) order."""
    blob = load_checkpoint(ckpt_path, map_location=device)
    cfg = _rebuild_cfg_from_checkpoint(blob)
    cfg = replace(cfg, train_parquet=parquet)

    # Layout is inferred from the checkpoint's recorded input_dim:
    # 110 = XP-only joint smoke (54 BP + 54 RP + 2 c0_z, no residuals, no aux).
    # 139 = production joint (XP + c0_z + 3 residuals + 26 aux).
    input_dim = int(blob["input_dim"])
    if input_dim == 110:
        layout = FeatureLayout(aux_cols=(), residual_cols=())
    elif input_dim == 139:
        layout = FeatureLayout()
    else:
        raise ValueError(
            f"checkpoint input_dim={input_dim} does not match a known joint layout "
            "(110 = XP-only, 139 = production)",
        )
    tiers = LabelTiers.five_label()

    train_loader, val_loader, _split_ids, label_scaler = build_dataloaders(
        cfg, layout, tiers, seed=blob.get("seed", 0),
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


def _chemistry_figure(
    mu_raw: np.ndarray, y_raw: np.ndarray, tiers: LabelTiers,
) -> tuple[plt.Figure, dict]:
    """3-panel: APOGEE truth, XP-joint pred, residual vs [M/H]."""
    i_mh = tiers.all_labels.index("mh_apogee")
    i_alpha = tiers.all_labels.index("alpha_m_apogee")
    mh_t, alpha_t = y_raw[:, i_mh], y_raw[:, i_alpha]
    mh_p, alpha_p = mu_raw[:, i_mh], mu_raw[:, i_alpha]
    finite = np.isfinite(mh_t) & np.isfinite(alpha_t) & np.isfinite(mh_p) & np.isfinite(alpha_p)
    mh_t, alpha_t, mh_p, alpha_p = mh_t[finite], alpha_t[finite], mh_p[finite], alpha_p[finite]

    mh_range = (-2.5, 0.6)
    alpha_range = (-0.1, 0.45)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=150)

    axes[0].hist2d(mh_t, alpha_t, bins=120, range=(mh_range, alpha_range),
                   cmap="magma", cmin=1)
    axes[0].set_xlabel("[M/H] (APOGEE truth)")
    axes[0].set_ylabel(r"[$\alpha$/M] (APOGEE truth)")
    axes[0].set_title(f"APOGEE truth — val ({len(mh_t):,} stars)")

    axes[1].hist2d(mh_p, alpha_p, bins=120, range=(mh_range, alpha_range),
                   cmap="magma", cmin=1)
    axes[1].set_xlabel("[M/H] (predicted)")
    axes[1].set_ylabel(r"[$\alpha$/M] (predicted)")
    axes[1].set_title("XP-joint prediction — same stars")

    residual = alpha_p - alpha_t
    axes[2].hist2d(mh_t, residual, bins=120,
                   range=(mh_range, (-0.3, 0.3)), cmap="coolwarm", cmin=1)
    axes[2].axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
    axes[2].set_xlabel("[M/H] (APOGEE truth)")
    axes[2].set_ylabel(r"[$\alpha$/M]$_\mathrm{pred} -$ [$\alpha$/M]$_\mathrm{truth}$")
    axes[2].set_title("Residual vs truth [M/H]")

    fig.tight_layout()

    # Diagnostic stats: attractor-stripe fraction and per-bin residual medians.
    stripe_mask = (alpha_p >= 0.09) & (alpha_p <= 0.13) & (mh_p >= -1.5) & (mh_p <= -0.3)
    n_stripe = int(stripe_mask.sum())
    stripe_frac = n_stripe / len(mh_p) if len(mh_p) else 0.0

    bin_edges = np.array([-2.5, -1.5, -1.0, -0.5, 0.0, 0.6])
    bin_idx = np.digitize(mh_t, bin_edges) - 1
    per_bin = []
    for b in range(len(bin_edges) - 1):
        m = bin_idx == b
        if m.sum() < 10:
            per_bin.append({"edge_lo": float(bin_edges[b]), "edge_hi": float(bin_edges[b + 1]),
                            "n": int(m.sum()), "median_resid": None})
            continue
        per_bin.append({
            "edge_lo": float(bin_edges[b]), "edge_hi": float(bin_edges[b + 1]),
            "n": int(m.sum()),
            "median_resid": float(np.median(residual[m])),
            "p16_resid": float(np.percentile(residual[m], 16)),
            "p84_resid": float(np.percentile(residual[m], 84)),
        })

    summary = {
        "n_val": len(mh_t),
        "attractor_stripe_pred": {
            "definition": "[α/M]_pred ∈ [+0.09, +0.13] ∧ [M/H]_pred ∈ [-1.5, -0.3]",
            "n": n_stripe,
            "fraction": stripe_frac,
        },
        "per_mh_bin_residuals": per_bin,
    }
    return fig, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="XP-joint best-val checkpoint.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", type=str, default="xp_joint")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("predicting val partition via %s on %s", args.ckpt, device)
    mu_raw, y_raw, tiers = _predict_val(args.ckpt, args.parquet, device)

    fig, summary = _chemistry_figure(mu_raw, y_raw, tiers)
    png_path = args.out_dir / f"{args.prefix}_alpham_vs_mh.png"
    fig.savefig(png_path)
    plt.close(fig)

    summary["ckpt"] = str(args.ckpt)
    summary["parquet"] = str(args.parquet)
    summary["device"] = str(device)
    summary_path = args.out_dir / f"{args.prefix}_chemistry_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _LOG.info("wrote figure to %s (n=%d, stripe frac=%.2f%%)",
              png_path, summary["n_val"], summary["attractor_stripe_pred"]["fraction"] * 100)


if __name__ == "__main__":
    main()
