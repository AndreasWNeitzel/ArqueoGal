"""Stage 09: Pipeline-1 NN input feature matrix (110 XP + aux).

Outputs:
  - reports/gallery/09_feature_matrix/feature_layout_schema.png
  - reports/gallery/09_feature_matrix/feature_distributions_xp.png
  - reports/gallery/09_feature_matrix/feature_distributions_aux.png
  - reports/gallery/09_feature_matrix/feature_correlation_heatmap.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_PROCESSED,
    GALLERY,
    apply_style,
    save_fig,
)

OUT = GALLERY / "09_feature_matrix"


def _have(schema, name: str) -> bool:
    return name in {f.name for f in schema}


AUX_COLS = [
    # c0 scalars
    "bp_c0_z",
    "rp_c0_z",
    # residuals
    "reprojection_residual_rms",
    "reprojection_residual_rms_bp",
    "reprojection_residual_rms_rp",
    # astrometry + quality
    "parallax",
    "parallax_error",
    "parallax_corr",
    "ruwe",
    # photometry
    "g_mag",
    "bp_mag",
    "rp_mag",
    "bp_rp",
    "bp_g",
    "g_rp",
    # IR + errors
    "j_mag",
    "h_mag",
    "k_mag",
    "w1_mag",
    "w2_mag",
    "e_j_mag",
    "e_h_mag",
    "e_k_mag",
    "e_w1_mag",
    "e_w2_mag",
    # distance
    "r_med_photogeo",
    "r_lo_photogeo",
    "r_hi_photogeo",
    # extinction block
    "av_edenhofer",
    "av_sfd",
    "av_lallement",
    "av_nbhd_median",
    "av_nbhd_std",
    "n_neighbors_75pc",
    "ag_gspphot",
    "ebv_edenhofer_2023",
    "ebv_sfd",
    # flags
    "ye2024_flag",
    "xp_fit_flag_residual_high",
    "xp_fit_flag_residual_high_global",
    "ir_missing_flag",
    "extinction_missing_flag",
]


def feature_layout_schema() -> None:
    """Schematic of the 153-D input tensor layout."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 153)
    ax.set_ylim(-0.5, 2.0)
    ax.set_yticks([])
    ax.set_xticks([0, 54, 108, 110, 113, 153])
    ax.set_xticklabels(["0", "54", "108", "110", "113", "153"])

    segments = [
        (0, 54, "BP coef_norm 1..54", "#1f77b4"),
        (54, 108, "RP coef_norm 1..54", "#1f77b4"),
        (108, 110, "c0 z\n(BP, RP)", "#ff7f0e"),
        (110, 113, "reproj\nresiduals", "#2ca02c"),
        (113, 153, "aux: Gaia + IR + extinction + flags", "#9467bd"),
    ]
    for x0, x1, lbl, color in segments:
        ax.add_patch(
            Rectangle((x0, 0), x1 - x0, 1.0, facecolor=color, edgecolor="#222", lw=0.7, alpha=0.85)
        )
        ax.text(
            (x0 + x1) / 2,
            0.5,
            lbl,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="semibold",
        )
        ax.text(
            (x0 + x1) / 2,
            1.15,
            f"{x1 - x0} dims",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333",
        )

    ax.set_xlabel("input-tensor column index")
    ax.set_title(
        r"Pipeline-1 NN input layout  —  153 dims  =  108 XP shape  +  2 $c_0$ scalars  "
        r"+  3 residuals  +  40 aux",
        fontsize=12,
        fontweight="bold",
    )
    # spine cleanup
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)

    # footer note
    ax.text(
        0.5,
        -0.3,
        "Contract frozen in src/arqueogal/xp_abundances/main/DESIGN.md  —  any column "
        "change requires a same-commit DESIGN update.",
        ha="center",
        va="top",
        fontsize=9,
        color="#555",
        transform=ax.transAxes,
    )

    save_fig(fig, OUT / "feature_layout_schema.png")


def feature_distributions_xp() -> None:
    path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    schema = pq.read_schema(path)
    cols = [f"bp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"bp_coef_norm_{k}")] + [
        f"rp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"rp_coef_norm_{k}")
    ]
    df = pq.read_table(path, columns=cols).to_pandas()
    if len(df) > 120_000:
        df = df.sample(120_000, random_state=11)

    fig, axes = plt.subplots(10, 11, figsize=(22, 18), sharex=False, sharey=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(cols):
            ax.set_visible(False)
            continue
        col = cols[i]
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            ax.set_visible(False)
            continue
        # post-z-score column — should be ~N(0, 1); clip tails at [-5, 5] for display
        vals_c = np.clip(vals, -5, 5)
        ax.hist(
            vals_c,
            bins=np.linspace(-5, 5, 40),
            color="#1f77b4",
            alpha=0.8,
            density=True,
            edgecolor="#333",
            linewidth=0.3,
        )
        # reference Gaussian
        xs = np.linspace(-5, 5, 200)
        ax.plot(xs, np.exp(-(xs**2) / 2) / np.sqrt(2 * np.pi), color="#d62728", lw=0.7)
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        short = col.replace("_coef_norm_", "_")
        ax.set_title(rf"{short}  $\mu$={mu:.2f} $\sigma$={sd:.2f}", fontsize=7, pad=1)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    fig.suptitle(
        r"Pipeline-1 XP feature distributions  —  110 post-z-score Hermite coefs "
        r"(target $\mathcal{N}(0,1)$, red curve)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_fig(fig, OUT / "feature_distributions_xp.png", tight=False)


def feature_distributions_aux() -> None:
    path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    schema = pq.read_schema(path)
    cols = [c for c in AUX_COLS if _have(schema, c)]
    df = pq.read_table(path, columns=cols).to_pandas()
    if len(df) > 120_000:
        df = df.sample(120_000, random_state=11)

    # 7x7 panel = 49 slots
    fig, axes = plt.subplots(7, 7, figsize=(20, 18))
    for i, ax in enumerate(axes.flat):
        if i >= len(cols):
            ax.set_visible(False)
            continue
        col = cols[i]
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 20:
            ax.set_visible(False)
            continue
        lo, hi = np.percentile(v, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = v.min(), v.max() if v.max() > v.min() else (v.min() - 1, v.min() + 1)
        # flags → stick histogram
        is_flag = col.endswith("_flag") or col == "n_neighbors_75pc"
        if is_flag and col != "n_neighbors_75pc":
            uniq, counts = np.unique(v.astype(int), return_counts=True)
            ax.bar(uniq, counts, color="#2ca02c", edgecolor="#333", alpha=0.85)
            ax.set_xticks(uniq)
        else:
            ax.hist(
                np.clip(v, lo, hi),
                bins=np.linspace(lo, hi, 36),
                color="#9467bd",
                alpha=0.8,
                edgecolor="#333",
                linewidth=0.3,
            )
        pct_nan = 100.0 * df[col].isna().mean()
        ax.set_title(f"{col}\nn={len(v):,}  NaN={pct_nan:.1f}%", fontsize=8, pad=2)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle(
        "Pipeline-1 aux feature distributions  —  Gaia + IR + extinction + flags",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_fig(fig, OUT / "feature_distributions_aux.png", tight=False)


def feature_correlation_heatmap() -> None:
    path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    schema = pq.read_schema(path)
    xp = [f"bp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"bp_coef_norm_{k}")] + [
        f"rp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"rp_coef_norm_{k}")
    ]
    aux = [c for c in AUX_COLS if _have(schema, c)]
    cols = xp + aux
    df = pq.read_table(path, columns=cols).to_pandas()
    if len(df) > 80_000:
        df = df.sample(80_000, random_state=11)
    # use pandas corr which is NaN-safe by pairwise
    mat = df[cols].corr().to_numpy()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.01, label="Pearson corr")
    # block lines
    n_xp = len(xp)
    n_bp = 54 if all(_have(schema, f"bp_coef_norm_{k}") for k in range(1, 55)) else n_xp // 2
    boundaries = [n_bp, n_xp, len(cols)]
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="#111", lw=0.8)
        ax.axvline(b - 0.5, color="#111", lw=0.8)

    # block labels
    def _mid(a, b):
        return (a + b - 1) / 2

    ticks = [_mid(0, n_bp), _mid(n_bp, n_xp), _mid(n_xp, len(cols))]
    labels = ["BP 1..54", "RP 1..54", "aux"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=10, rotation=90, va="center")
    ax.set_title(
        f"Pipeline-1 {len(cols)}×{len(cols)} input-feature correlation matrix "
        "(BP-block, RP-block, aux-block)",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, OUT / "feature_correlation_heatmap.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    feature_layout_schema()
    feature_distributions_xp()
    feature_distributions_aux()
    feature_correlation_heatmap()


if __name__ == "__main__":
    main()
