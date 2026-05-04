"""B17: Encoder feature overview AFTER FeatureScaler is applied.

B11 shows the raw feature distributions; this plot shows what the
encoder actually receives at training and inference time, after the
FeatureScaler that the canonical checkpoint persists. Same panel layout
as B11 so they can be diffed visually.

The XP block (108 Hermite shape coefs + 2 c0_z scalars) is unchanged
(passthrough). The 30 scaled columns (27 aux + 3 residual RMS) appear
as zero-mean unit-std distributions; the residual RMS is log10
+ z-scored.

Loads the FeatureScaler from the canonical checkpoint and applies it
to the same Stream-1 Kiel-bounded pool the model was trained on.
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

    print("[B17] loading checkpoint FeatureScaler")
    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    fs_blob = blob.get("feature_scaler")
    if fs_blob is None:
        print("Error: canonical checkpoint has no feature_scaler block")
        return 1
    fs = FeatureScaler(
        mean=np.asarray(fs_blob["mean"], dtype=np.float32),
        scale=np.asarray(fs_blob["scale"], dtype=np.float32),
        feature_names=tuple(fs_blob["feature_names"]),
        log10_mask=np.asarray(fs_blob["log10_mask"], dtype=bool),
        apply_mask=np.asarray(fs_blob["apply_mask"], dtype=bool),
    )
    print(
        f"[B17] FeatureScaler: {int(fs.apply_mask.sum())} scaled columns "
        f"({int(fs.log10_mask.sum())} log10), "
        f"{int((~fs.apply_mask).sum())} passthrough"
    )

    cols = list(layout.all_required_columns)
    df = pd.read_parquet(parquet, columns=cols)
    X = np.column_stack([df[c].to_numpy(dtype=np.float32) for c in cols])
    X_scaled = fs.transform(X)
    print(f"[B17] cohort n={len(df):,}, encoder input_dim={layout.input_dim}")

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.35)

    bp_idx = list(layout.xp_bp_indices)
    rp_idx = list(layout.xp_rp_indices)
    bp_cols = list(layout.bp_coef_cols)
    rp_cols = list(layout.rp_coef_cols)

    # --- Row 0: XP block (passthrough — same as B11) ---
    bp_arr = X_scaled[:, : len(bp_cols)].astype(np.float64)
    rp_arr = X_scaled[:, len(bp_cols) : len(bp_cols) + len(rp_cols)].astype(np.float64)

    ax = fig.add_subplot(gs[0, 0])
    bp_med = np.nanmedian(bp_arr, axis=0)
    bp_p16 = np.nanpercentile(bp_arr, 16, axis=0)
    bp_p84 = np.nanpercentile(bp_arr, 84, axis=0)
    ax.fill_between(bp_idx, bp_p16, bp_p84, color="#1f77b4", alpha=0.25, label="16-84 percentile")
    ax.plot(bp_idx, bp_med, "-", color="#1f77b4", lw=1.4, label="median")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Hermite index i (BP)")
    ax.set_ylabel(r"$bp\_coef\_norm_i$ (post-scaling)")
    ax.set_title(f"BP shape coefs (passthrough)\nn={len(df):,}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    rp_med = np.nanmedian(rp_arr, axis=0)
    rp_p16 = np.nanpercentile(rp_arr, 16, axis=0)
    rp_p84 = np.nanpercentile(rp_arr, 84, axis=0)
    ax.fill_between(rp_idx, rp_p16, rp_p84, color="#d62728", alpha=0.25, label="16-84 percentile")
    ax.plot(rp_idx, rp_med, "-", color="#d62728", lw=1.4, label="median")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Hermite index i (RP)")
    ax.set_ylabel(r"$rp\_coef\_norm_i$ (post-scaling)")
    ax.set_title(f"RP shape coefs (passthrough)\nn={len(df):,}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    bp_c0z_idx = len(bp_cols) + len(rp_cols)
    ax = fig.add_subplot(gs[0, 2])
    v = X_scaled[:, bp_c0z_idx].astype(np.float64)
    finite = np.isfinite(v)
    ax.hist(v[finite], bins=60, color="#1f77b4", alpha=0.8, edgecolor="#1f77b4", lw=0.4)
    ax.set_xlabel("bp_c0_z (post-scaling, passthrough)")
    ax.set_ylabel("count")
    ax.set_title(f"BP c0 (z) passthrough\nmed {np.nanmedian(v):+.2f}, sd {np.nanstd(v):.2f}")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[0, 3])
    v = X_scaled[:, bp_c0z_idx + 1].astype(np.float64)
    finite = np.isfinite(v)
    ax.hist(v[finite], bins=60, color="#d62728", alpha=0.8, edgecolor="#d62728", lw=0.4)
    ax.set_xlabel("rp_c0_z (post-scaling, passthrough)")
    ax.set_ylabel("count")
    ax.set_title(f"RP c0 (z) passthrough\nmed {np.nanmedian(v):+.2f}, sd {np.nanstd(v):.2f}")
    ax.grid(axis="y", alpha=0.25)

    # Per-feature sanity bar: post-scaling mean and std for the SCALED columns.
    ax = fig.add_subplot(gs[0, 4])
    apply_idx = np.where(fs.apply_mask)[0]
    scaled_means = np.nanmean(X_scaled[:, apply_idx], axis=0)
    scaled_stds = np.nanstd(X_scaled[:, apply_idx], axis=0)
    x = np.arange(len(apply_idx))
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.axhline(1, color="0.5", lw=0.5, ls=":", alpha=0.5)
    ax.plot(x, scaled_means, "o-", color="#1f77b4", ms=3, label="post-scaling mean")
    ax.plot(x, scaled_stds, "s-", color="#d62728", ms=3, label="post-scaling std")
    ax.set_xlabel("scaled feature index (residual + aux)")
    ax.set_ylabel("statistic")
    ax.set_title("Sanity: post-scaling mean (~0)\nand std (~1) per scaled column")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # --- Row 1: residuals + aux NaN-spike check + scaled-vs-unscaled aux corr ---
    res_titles = ["combined RMS", "BP RMS", "RP RMS"]
    res_colors = ["#444444", "#1f77b4", "#d62728"]
    res_offset = bp_c0z_idx + 2  # residuals start right after the c0_z scalars
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        v = X_scaled[:, res_offset + i].astype(np.float64)
        # log10 was applied at FeatureScaler.transform time; the column
        # may carry NaN from non-positive raw residuals — drop them.
        finite = np.isfinite(v)
        if int(finite.sum()) > 0:
            ax.hist(
                v[finite], bins=60, color=res_colors[i], alpha=0.8, edgecolor=res_colors[i], lw=0.4
            )
            ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5, label="z=0 (mean)")
            ax.legend(fontsize=8)
        ax.set_xlabel(f"log10({res_titles[i]}) z-scored")
        ax.set_ylabel("count")
        ax.set_title(f"{res_titles[i]} (post log10+z)\nn={int(finite.sum()):,}")
        ax.grid(axis="y", alpha=0.25)

    aux_cols = list(layout.aux_cols)
    ax = fig.add_subplot(gs[1, 3])
    aux_offset = res_offset + 3  # aux block starts after the 3 residual cols
    np.array(
        [
            int((X_scaled[:, aux_offset + j] == 0.0).sum())
            - int((np.abs(X_scaled[:, aux_offset + j] - 0.0) < 1e-9).sum() == 0)
            for j in range(len(aux_cols))
        ]
    )
    # The clearer thing to plot: NaN fraction in the *raw* columns vs the
    # post-scaling distribution at exactly z=0. nan_to_num maps every NaN
    # to 0 *before* training but in B17 we transform without imputing,
    # so NaN survive and we can show the fraction transparently.
    raw_nan_frac = np.array([float(df[c].isna().mean()) for c in aux_cols])
    ax.bar(np.arange(len(aux_cols)), raw_nan_frac, color="#9467bd", width=0.85)
    ax.set_xticks(np.arange(len(aux_cols)))
    ax.set_xticklabels(aux_cols, rotation=90, fontsize=6)
    ax.set_ylabel("raw NaN fraction")
    ax.set_title("Raw aux NaN fraction\n(unchanged by scaling)")
    ax.grid(axis="y", alpha=0.25)

    # Pairwise aux correlation post-scaling, NaN-pairwise.
    ax = fig.add_subplot(gs[1, 4])
    aux_arr = X_scaled[:, aux_offset : aux_offset + len(aux_cols)].astype(np.float64)
    n_aux = len(aux_cols)
    corr = np.full((n_aux, n_aux), np.nan)
    for ii in range(n_aux):
        for jj in range(ii, n_aux):
            x_ii, x_jj = aux_arr[:, ii], aux_arr[:, jj]
            m = np.isfinite(x_ii) & np.isfinite(x_jj)
            if int(m.sum()) >= 100:
                xi, xj = x_ii[m], x_jj[m]
                vi, vj = xi.std(), xj.std()
                if vi > 0 and vj > 0:
                    corr[ii, jj] = float(np.corrcoef(xi, xj)[0, 1])
                    corr[jj, ii] = corr[ii, jj]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(np.arange(n_aux))
    ax.set_xticklabels(aux_cols, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n_aux))
    ax.set_yticklabels(aux_cols, fontsize=5)
    ax.set_title("Aux Pearson r (post-scaling,\nNaN-pairwise)")

    # --- Row 2: scaled photometry distributions ---
    photom_pos = ["g_mag", "bp_mag", "rp_mag", "bp_rp", "bp_g"]
    photom_colors = ["#1f77b4", "#1f77b4", "#d62728", "#9467bd", "#9467bd"]
    for i, col in enumerate(photom_pos):
        col_idx = aux_offset + aux_cols.index(col)
        ax = fig.add_subplot(gs[2, i])
        v = X_scaled[:, col_idx].astype(np.float64)
        finite = np.isfinite(v)
        if finite.any():
            ax.hist(
                v[finite],
                bins=60,
                color=photom_colors[i],
                alpha=0.75,
                edgecolor=photom_colors[i],
                lw=0.4,
            )
            ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5)
        ax.set_xlabel(f"{col} (z-scored)")
        ax.set_ylabel("count")
        ax.set_title(f"{col}\nmed {np.nanmedian(v):+.2f}, sd {np.nanstd(v):.2f}")
        ax.grid(axis="y", alpha=0.25)

    # --- Row 3: astrometry + extinction post-scaling ---
    diag_pos = ["parallax", "ruwe", "r_med_photogeo", "av_los", "ag_gspphot"]
    diag_titles = [
        "parallax",
        "RUWE",
        "BJ21 distance",
        r"$A_V^\mathrm{LOS}$",
        r"$A_G^\mathrm{GSP-Phot}$",
    ]
    diag_colors = ["#2ca02c", "#444444", "#1f77b4", "#9467bd", "#9467bd"]
    for i, col in enumerate(diag_pos):
        col_idx = aux_offset + aux_cols.index(col)
        ax = fig.add_subplot(gs[3, i])
        v = X_scaled[:, col_idx].astype(np.float64)
        finite = np.isfinite(v)
        if finite.any():
            ax.hist(
                v[finite],
                bins=60,
                color=diag_colors[i],
                alpha=0.8,
                edgecolor=diag_colors[i],
                lw=0.4,
            )
            ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5)
        ax.set_xlabel(f"{col} (z-scored)")
        ax.set_ylabel("count")
        ax.set_title(f"{diag_titles[i]} post-scaling\nn={int(finite.sum()):,}")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"B17 - Encoder input AFTER FeatureScaler "
        f"(n={len(df):,} stars, {layout.input_dim}-D vector).  "
        f"110 XP cols passthrough, 30 aux+residual cols z-scored "
        f"(3 with log10 first).\n"
        f"Stats loaded from canonical checkpoint cfg 26312a4.",
        fontsize=11,
        fontweight="semibold",
        y=0.995,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B17_feature_overview_post_scaling", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
