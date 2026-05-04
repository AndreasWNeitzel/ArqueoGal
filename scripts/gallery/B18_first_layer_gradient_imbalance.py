"""B18: Encoder first-layer per-feature gradient magnitudes, pre vs post FeatureScaler.

The FeatureScaler does NOT change mutual information (strict monotone
transforms preserve MI by construction; B16 and the deleted MI-post
plot were mathematically identical). What it DOES change is the
gradient the encoder receives along each input dimension.

For a Linear layer ``y = x W^T + b`` with loss ``L``, the gradient of
the loss w.r.t. the i-th column of W is ``dL/dW_i = x_i * dL/dy``,
i.e. proportional to the per-star input magnitude on that dimension.
If feature i has typical magnitude 10^3 and feature j has typical
magnitude 10^-18, the optimiser sees a 10^21 imbalance in weight-
update step sizes between the two columns — even when both features
carry equally relevant information about the label. The first feature's
weights move erratically; the second feature's weights are functionally
frozen at their initialisation.

This plot demonstrates the imbalance directly. For each encoder input
column, we compute the typical magnitude that ``|x_i|`` takes on the
training pool, with and without the FeatureScaler applied. We do NOT
need to actually run a backward pass — by the chain rule above, the
per-column scale of ``|x_i|`` IS the per-column scale of the gradient
on that input dimension.

Layout (3 rows x 1 col):
  row 0 = pre-scaling: log10(median(|x_i|)) per feature, all 140 cols
  row 1 = post-scaling: log10(median(|x_i|)) per feature
  row 2 = ratio: pre / post (gradient-imbalance reduction factor per
          feature). Shows which columns saw the most equalisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig
from arqueogal.xp_abundances.main.data import FeatureLayout, FeatureScaler

OUT = REPO / "reports/gallery/B_preprocessing"
CKPT_PATH = REPO / (
    "models/main/xp_abundances/20260501_1d71682_26312a4_ensemble_5label/"
    "member_seed0/xp_abundances_main_ensemble_5label_seed0_best.pt"
)


def main() -> int:
    apply_style()
    layout = FeatureLayout()
    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists() or not CKPT_PATH.exists():
        print(f"Error: missing input.\n  parquet: {parquet}\n  ckpt: {CKPT_PATH}")
        return 1

    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    fs_blob = blob.get("feature_scaler")
    if fs_blob is None:
        print("Error: no feature_scaler in checkpoint")
        return 1
    fs = FeatureScaler(
        mean=np.asarray(fs_blob["mean"], dtype=np.float32),
        scale=np.asarray(fs_blob["scale"], dtype=np.float32),
        feature_names=tuple(fs_blob["feature_names"]),
        log10_mask=np.asarray(fs_blob["log10_mask"], dtype=bool),
        apply_mask=np.asarray(fs_blob["apply_mask"], dtype=bool),
    )

    feature_names = list(layout.all_required_columns)
    df = pd.read_parquet(parquet, columns=feature_names)
    X_raw = np.column_stack([df[c].to_numpy(dtype=np.float32) for c in feature_names])
    X_scaled = fs.transform(X_raw)
    print(f"[B18] cohort n={len(df):,}")

    bp_cols = set(layout.bp_coef_cols)
    rp_cols = set(layout.rp_coef_cols)
    scalar_cols = set(layout.xp_scalar_cols)
    res_cols = set(layout.residual_cols)
    family = []
    for name in feature_names:
        if name in bp_cols:
            family.append("BP")
        elif name in rp_cols:
            family.append("RP")
        elif name in scalar_cols:
            family.append("c0")
        elif name in res_cols:
            family.append("residual")
        else:
            family.append("aux")
    family = np.array(family)
    family_color = {
        "BP": "#1f77b4", "RP": "#d62728", "c0": "#2ca02c",
        "residual": "#7f7f7f", "aux": "#9467bd",
    }
    colors = [family_color[f] for f in family]

    # Per-column median-|x| magnitudes. Use the median rather than mean to
    # avoid heavy-tail domination (residual RMS has outliers up to 1e36).
    abs_pre = np.abs(X_raw)
    abs_post = np.abs(X_scaled)
    abs_pre = np.where(np.isfinite(abs_pre), abs_pre, np.nan)
    abs_post = np.where(np.isfinite(abs_post), abs_post, np.nan)
    med_pre = np.array([
        float(np.nanmedian(abs_pre[:, j])) if np.isfinite(abs_pre[:, j]).any() else np.nan
        for j in range(abs_pre.shape[1])
    ])
    med_post = np.array([
        float(np.nanmedian(abs_post[:, j])) if np.isfinite(abs_post[:, j]).any() else np.nan
        for j in range(abs_post.shape[1])
    ])

    # Take log10 so we can plot 50+ orders of magnitude on a sensible axis.
    # log10(0) → -inf; clamp at -25 to keep axes finite.
    log_pre = np.where(med_pre > 0, np.log10(med_pre), np.nan)
    log_post = np.where(med_post > 0, np.log10(med_post), np.nan)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in family_color.values()]

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 1, hspace=0.5)

    # Row 0: pre-scaling log10(|x|).
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(len(feature_names)), log_pre, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Feature index (encoder input order)")
    ax.set_ylabel(r"$\log_{10}$ median$|x_i|$")
    ax.set_title("Pre-scaling: per-column log10 median |x| - "
                 "directly proportional to the gradient on that input "
                 "dimension at the encoder's first Linear layer")
    ax.legend(handles, list(family_color.keys()), fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # Annotate the most extreme pre-scaling features.
    extremes = np.argsort(np.where(np.isnan(log_pre), 0, log_pre))
    for j in list(extremes[:3]) + list(extremes[-3:]):
        if not np.isnan(log_pre[j]):
            ax.annotate(feature_names[j],
                        (j, log_pre[j]),
                        textcoords="offset points", xytext=(2, 4),
                        fontsize=7, rotation=30, ha="left")

    # Row 1: post-scaling.
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(np.arange(len(feature_names)), log_post, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Feature index (encoder input order)")
    ax.set_ylabel(r"$\log_{10}$ median$|x_i|$")
    ax.set_title("Post-scaling: per-column log10 median |x| - "
                 "FeatureScaler equalises gradient magnitudes across "
                 "all 140 inputs")
    ax.grid(axis="y", alpha=0.25)

    # Row 2: ratio (= pre/post in log = log_pre - log_post). Shows which
    # columns saw the most equalisation.
    ax = fig.add_subplot(gs[2, 0])
    ratio_log = log_pre - log_post
    ax.bar(np.arange(len(feature_names)), ratio_log, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Feature index (encoder input order)")
    ax.set_ylabel(r"$\log_{10}\,(\mathrm{pre}/\mathrm{post})$ "
                 r"$=$ gradient-imbalance reduction factor")
    ax.set_title("Imbalance reduction factor per column "
                 "(positive = scaler shrunk this column's gradient; "
                 "negative = scaler grew it; 0 = passthrough or already O(1))")
    ax.grid(axis="y", alpha=0.25)

    # Annotate biggest reductions.
    big = np.argsort(np.where(np.isnan(ratio_log), 0, np.abs(ratio_log)))[::-1][:5]
    for j in big:
        if not np.isnan(ratio_log[j]):
            ax.annotate(feature_names[j],
                        (j, ratio_log[j]),
                        textcoords="offset points", xytext=(2, 4),
                        fontsize=7, rotation=30, ha="left")

    fig.suptitle(
        f"B18 - Encoder first-layer gradient imbalance, pre vs post "
        f"FeatureScaler  (n={len(df):,}).  "
        f"By chain rule on a Linear layer, "
        r"$|\partial L / \partial W_i| \propto |x_i|$" + ", "
        f"so per-column |x| sets per-column gradient scale.\n"
        f"Pre-scaling, columns range over ~50 orders of magnitude; "
        f"post-scaling, all O(1).",
        fontsize=11, fontweight="semibold", y=0.995,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B18_first_layer_gradient_imbalance",
             formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
