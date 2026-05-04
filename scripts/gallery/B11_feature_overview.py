"""B11: Encoder input feature overview - what the model actually sees.

Renders a single-figure dashboard documenting the 140-D feature vector
the encoder consumes. Useful for sanity-checking that no feature has
a degenerate distribution (NaN-saturated, near-constant, extreme tails)
that would silently degrade the contrastive geometry.

Layout (4 rows x 5 cols):
  row 0 (XP block, 108 D + 2):
    col 0 = BP Hermite shape coefs heatmap (per-coef median/p16/p84 vs index)
    col 1 = RP Hermite shape coefs heatmap (per-coef median/p16/p84 vs index)
    col 2 = bp_c0_z histogram
    col 3 = rp_c0_z histogram
    col 4 = NaN fraction per XP feature (54+54+2 = 110 features) bar chart
  row 1 (residual RMS):
    col 0..2 = reprojection_residual_rms / _bp / _rp histograms
    col 3 = NaN fraction across the 27 aux columns
    col 4 = pairwise correlation heatmap of the 27 aux features
  row 2 (aux photometry, 6 D):
    histograms for g_mag, bp_mag, rp_mag, bp_rp, bp_g, g_rp
  row 3 (aux astrometry + IR + extinction, 21 D, summarised):
    col 0 = parallax / parallax_error histogram (clipped log)
    col 1 = ruwe histogram
    col 2 = r_med_photogeo histogram (kpc)
    col 3 = av_los histogram (clipped 99.5th)
    col 4 = ag_gspphot histogram

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (the 324k-star
  Kiel-bounded RGB training pool).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig

from arqueogal.xp_abundances.main.data import FeatureLayout

OUT = REPO / "reports/gallery/B_preprocessing"


def main() -> int:
    apply_style()
    layout = FeatureLayout()
    print("[B11] Loading Kiel-bounded Stream-1 features")
    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        print(f"Error: {parquet} not found")
        return 1

    cols = list(layout.all_required_columns)
    df = pd.read_parquet(parquet, columns=cols)
    print(f"[B11] cohort n={len(df):,}, encoder input_dim={layout.input_dim}")

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.35)

    bp_idx = list(layout.xp_bp_indices)
    rp_idx = list(layout.xp_rp_indices)
    bp_cols = list(layout.bp_coef_cols)
    rp_cols = list(layout.rp_coef_cols)

    # --- Row 0: XP block ---
    ax = fig.add_subplot(gs[0, 0])
    bp_arr = df[bp_cols].to_numpy(dtype=np.float64)
    bp_med = np.nanmedian(bp_arr, axis=0)
    bp_p16 = np.nanpercentile(bp_arr, 16, axis=0)
    bp_p84 = np.nanpercentile(bp_arr, 84, axis=0)
    ax.fill_between(bp_idx, bp_p16, bp_p84, color="#1f77b4", alpha=0.25, label="16-84 percentile")
    ax.plot(bp_idx, bp_med, "-", color="#1f77b4", lw=1.4, label="median")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Hermite index i (BP)")
    ax.set_ylabel(r"$bp\_coef\_norm_i$  (z-score)")
    ax.set_title(
        f"BP shape coefs (54-D)\nn={int(np.isfinite(bp_arr).all(axis=1).sum()):,} finite-row"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    rp_arr = df[rp_cols].to_numpy(dtype=np.float64)
    rp_med = np.nanmedian(rp_arr, axis=0)
    rp_p16 = np.nanpercentile(rp_arr, 16, axis=0)
    rp_p84 = np.nanpercentile(rp_arr, 84, axis=0)
    ax.fill_between(rp_idx, rp_p16, rp_p84, color="#d62728", alpha=0.25, label="16-84 percentile")
    ax.plot(rp_idx, rp_med, "-", color="#d62728", lw=1.4, label="median")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Hermite index i (RP)")
    ax.set_ylabel(r"$rp\_coef\_norm_i$  (z-score)")
    ax.set_title(
        f"RP shape coefs (54-D)\nn={int(np.isfinite(rp_arr).all(axis=1).sum()):,} finite-row"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    bp_c0z = df["bp_c0_z"].to_numpy(dtype=np.float64)
    finite = np.isfinite(bp_c0z)
    ax.hist(bp_c0z[finite], bins=60, color="#1f77b4", alpha=0.8, edgecolor="#1f77b4", lw=0.4)
    ax.set_xlabel("bp_c0_z (z-scored log10 c0)")
    ax.set_ylabel("count")
    ax.set_title(
        f"BP c0 (z)\nn={int(finite.sum()):,}, "
        f"med {np.nanmedian(bp_c0z):+.2f}, sd {np.nanstd(bp_c0z):.2f}"
    )
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[0, 3])
    rp_c0z = df["rp_c0_z"].to_numpy(dtype=np.float64)
    finite = np.isfinite(rp_c0z)
    ax.hist(rp_c0z[finite], bins=60, color="#d62728", alpha=0.8, edgecolor="#d62728", lw=0.4)
    ax.set_xlabel("rp_c0_z (z-scored log10 c0)")
    ax.set_ylabel("count")
    ax.set_title(
        f"RP c0 (z)\nn={int(finite.sum()):,}, "
        f"med {np.nanmedian(rp_c0z):+.2f}, sd {np.nanstd(rp_c0z):.2f}"
    )
    ax.grid(axis="y", alpha=0.25)

    # NaN fraction per XP feature (108 + 2 = 110)
    ax = fig.add_subplot(gs[0, 4])
    xp_cols = bp_cols + rp_cols + list(layout.xp_scalar_cols)
    nan_frac_xp = np.array([float(df[c].isna().mean()) for c in xp_cols])
    ax.bar(
        np.arange(len(xp_cols)),
        nan_frac_xp,
        color=["#1f77b4"] * 54 + ["#d62728"] * 54 + ["#2ca02c"] * 2,
        width=1.0,
    )
    ax.set_xlabel("XP feature index (BP1..BP54, RP1..RP54, bp_c0_z, rp_c0_z)")
    ax.set_ylabel("NaN fraction")
    ax.set_title(f"XP NaN coverage\n(max {nan_frac_xp.max() * 100:.2f}%)")
    ax.set_ylim(0, max(0.05, float(nan_frac_xp.max()) * 1.1 + 1e-3))
    ax.grid(axis="y", alpha=0.25)

    # --- Row 1: residuals + aux NaN fraction + aux corr ---
    res_cols = list(layout.residual_cols)
    res_titles = ["combined RMS", "BP RMS", "RP RMS"]
    res_colors = ["#444444", "#1f77b4", "#d62728"]
    for i, col in enumerate(res_cols):
        ax = fig.add_subplot(gs[1, i])
        v = df[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(v) & (v > 0)
        # Residual RMS spans 50+ orders of magnitude (1e-20 to 1e+36) —
        # plot log10 of the value so the bulk distribution is readable.
        if finite.any():
            logv = np.log10(v[finite])
            bins = np.linspace(logv.min(), logv.max(), 60)
            ax.hist(
                logv, bins=bins, color=res_colors[i], alpha=0.8, edgecolor=res_colors[i], lw=0.4
            )
            med_log = float(np.median(logv))
            ax.axvline(med_log, color="k", lw=0.8, ls="--", label=f"log10 median = {med_log:.2f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(f"log10({col})")
        ax.set_ylabel("count")
        ax.set_title(f"{res_titles[i]}\nn={int(finite.sum()):,}")
        ax.grid(axis="y", alpha=0.25)

    aux_cols = list(layout.aux_cols)
    ax = fig.add_subplot(gs[1, 3])
    nan_frac_aux = np.array([float(df[c].isna().mean()) for c in aux_cols])
    bar_colors = []
    for c in aux_cols:
        if c.startswith("av_") or c.startswith("ag_"):
            bar_colors.append("#9467bd")
        elif c.endswith("_dered") or c.endswith("_mag"):
            bar_colors.append("#ff7f0e")
        elif c.startswith("parallax") or c == "ruwe" or c.startswith("r_"):
            bar_colors.append("#2ca02c")
        else:
            bar_colors.append("#777777")
    ax.bar(np.arange(len(aux_cols)), nan_frac_aux, color=bar_colors, width=0.85)
    ax.set_xticks(np.arange(len(aux_cols)))
    ax.set_xticklabels(aux_cols, rotation=90, fontsize=6)
    ax.set_ylabel("NaN fraction")
    ax.set_ylim(0, max(0.05, float(nan_frac_aux.max()) * 1.1 + 1e-3))
    ax.set_title(f"Aux NaN coverage\n(max {nan_frac_aux.max() * 100:.2f}%)")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, 4])
    aux_arr = df[aux_cols].to_numpy(dtype=np.float64)
    # NaN-aware pairwise Pearson correlation (computed on each pair's
    # complete-case mask). No imputation. This avoids the Pearson
    # correlation artefact that median-fill introduces by inflating
    # dependence between any two columns that share NaN rows.
    n_aux = len(aux_cols)
    corr = np.full((n_aux, n_aux), np.nan)
    for ii in range(n_aux):
        x = aux_arr[:, ii]
        for jj in range(ii, n_aux):
            y = aux_arr[:, jj]
            m = np.isfinite(x) & np.isfinite(y)
            if int(m.sum()) >= 100:
                xm, ym = x[m], y[m]
                vx, vy = xm.std(), ym.std()
                if vx > 0 and vy > 0:
                    corr[ii, jj] = float(np.corrcoef(xm, ym)[0, 1])
                    corr[jj, ii] = corr[ii, jj]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r (NaN-pairwise)")
    ax.set_xticks(np.arange(n_aux))
    ax.set_xticklabels(aux_cols, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n_aux))
    ax.set_yticklabels(aux_cols, fontsize=5)
    ax.set_title("Aux Pearson r (NaN-pairwise,\nno imputation)")

    # --- Row 2: aux photometry distributions ---
    photom = ["g_mag", "bp_mag", "rp_mag", "bp_rp", "bp_g", "g_rp"]
    photom_colors = ["#1f77b4", "#1f77b4", "#d62728", "#9467bd", "#9467bd", "#9467bd"]
    for i, col in enumerate(photom[:5]):
        ax = fig.add_subplot(gs[2, i])
        v = df[col].to_numpy(dtype=np.float64)
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
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(f"{col} - n={int(finite.sum()):,}")
        ax.grid(axis="y", alpha=0.25)

    # --- Row 3: astrometry + IR + extinction summaries ---
    plx = df["parallax"].to_numpy(dtype=np.float64)
    plx_err = df["parallax_error"].to_numpy(dtype=np.float64)
    plx_snr = np.where(np.isfinite(plx_err) & (plx_err > 0), np.abs(plx) / plx_err, np.nan)
    ax = fig.add_subplot(gs[3, 0])
    finite = np.isfinite(plx_snr) & (plx_snr > 0)
    if finite.any():
        ax.hist(
            np.log10(plx_snr[finite]),
            bins=60,
            color="#2ca02c",
            alpha=0.8,
            edgecolor="#2ca02c",
            lw=0.4,
        )
        ax.axvline(np.log10(5), color="r", lw=0.8, ls="--", label="log10 SNR = log10(5)")
        ax.legend(fontsize=8)
    ax.set_xlabel(r"$\log_{10}\, |\varpi|/\sigma_\varpi$")
    ax.set_ylabel("count")
    ax.set_title(f"parallax SNR (n={int(finite.sum()):,})")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[3, 1])
    ruwe = df["ruwe"].to_numpy(dtype=np.float64)
    finite = np.isfinite(ruwe) & (ruwe > 0)
    if finite.any():
        p995 = float(np.nanpercentile(ruwe, 99.5))
        ax.hist(
            np.clip(ruwe[finite], 0, p995),
            bins=60,
            color="#444444",
            alpha=0.8,
            edgecolor="#444444",
            lw=0.4,
        )
        ax.axvline(1.4, color="r", lw=0.8, ls="--", label="ruwe=1.4")
        ax.legend(fontsize=8)
    ax.set_xlabel("ruwe")
    ax.set_ylabel("count")
    ax.set_title(f"RUWE (n={int(finite.sum()):,})")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[3, 2])
    rmed = df["r_med_photogeo"].to_numpy(dtype=np.float64) / 1000.0
    finite = np.isfinite(rmed)
    if finite.any():
        p995 = float(np.nanpercentile(rmed, 99.5))
        ax.hist(
            rmed[finite],
            bins=np.linspace(0, max(p995, 0.5), 60),
            color="#1f77b4",
            alpha=0.8,
            edgecolor="#1f77b4",
            lw=0.4,
        )
    ax.set_xlabel(r"$r_\mathrm{med, photogeo}$ [kpc]")
    ax.set_ylabel("count")
    ax.set_title(f"BJ21 distance (n={int(finite.sum()):,})")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[3, 3])
    av = df["av_los"].to_numpy(dtype=np.float64)
    finite = np.isfinite(av)
    if finite.any():
        p995 = float(np.nanpercentile(av, 99.5))
        ax.hist(
            np.clip(av[finite], 0, p995),
            bins=60,
            color="#9467bd",
            alpha=0.8,
            edgecolor="#9467bd",
            lw=0.4,
        )
        ax.axvline(
            np.nanmedian(av), color="k", lw=0.8, ls="--", label=f"median {np.nanmedian(av):.3f}"
        )
        ax.legend(fontsize=8)
    ax.set_xlabel(r"$A_V^\mathrm{LOS}$ [mag]")
    ax.set_ylabel("count")
    ax.set_title(f"Fused $A_V$ (n={int(finite.sum()):,})")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[3, 4])
    ag = df["ag_gspphot"].to_numpy(dtype=np.float64)
    finite = np.isfinite(ag)
    if finite.any():
        p995 = float(np.nanpercentile(ag, 99.5))
        ax.hist(
            np.clip(ag[finite], 0, p995),
            bins=60,
            color="#9467bd",
            alpha=0.8,
            edgecolor="#9467bd",
            lw=0.4,
        )
    ax.set_xlabel(r"$A_G^\mathrm{GSP-Phot}$ [mag]")
    ax.set_ylabel("count")
    ax.set_title(f"GSP-Phot $A_G$ (n={int(finite.sum()):,})")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"B11 - Encoder input feature overview "
        f"(n={len(df):,} stars, {layout.input_dim}-D vector: "
        f"{len(bp_cols)} BP + {len(rp_cols)} RP Hermite shape coefs + "
        f"2 c0 scalars + 3 residual RMS + {len(aux_cols)} aux features). "
        f"Source: pipeline1_features_stream1_kiel.parquet.",
        fontsize=11,
        fontweight="semibold",
        y=0.995,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B11_feature_overview", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
