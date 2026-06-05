"""F09: dual-Mahalanobis tier gate + Av trend (slide 10)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import load_s1_holdout  # noqa: E402

from arqueogal.style import (  # noqa: E402
    LABELS, TIER, apply_style, colorbar, hexbin_density, save,
)


def main() -> int:
    apply_style()
    df = load_s1_holdout()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5),
                              layout="constrained")

    ax = axes[0]
    px_d = df["ood_mahalanobis_score"].to_numpy()
    py_d = df["label_mahalanobis_score"].to_numpy()
    tiers = df["release_tier"].to_numpy()
    fx_flag = df["ood_joint_flag"].astype(bool).to_numpy()
    fy_flag = df["label_extrapolation_flag"].astype(bool).to_numpy()
    ok = (np.isfinite(px_d) & np.isfinite(py_d)
          & (px_d > 0) & (py_d > 0))
    px_d = px_d[ok]; py_d = py_d[ok]
    tiers = tiers[ok]; fx_flag = fx_flag[ok]; fy_flag = fy_flag[ok]
    # Recover the fitted (training-set) thresholds from the flag boundary.
    # Holdout percentiles are not the right reference: the threshold was
    # fit on the *training* APOGEE-truth distribution, so a holdout p99
    # would draw the line in the wrong place relative to the coloured
    # tier-assignments.
    xcut = float(np.nanmin(px_d[fx_flag])) if fx_flag.any() else float(
        np.nanpercentile(px_d, 99.0))
    ycut = float(np.nanmin(py_d[fy_flag])) if fy_flag.any() else float(
        np.nanpercentile(py_d, 99.0))
    # Colour-code each star by release tier (T1 first so T2/T3 sit on top).
    for tier, z in [(1, 1), (2, 4), (3, 5)]:
        m = tiers == tier
        if m.any():
            ax.scatter(px_d[m], py_d[m], s=1.8,
                       alpha=0.20 if tier == 1 else 0.85,
                       color=TIER[f"T{tier}"], edgecolors="none",
                       rasterized=True, zorder=z)
    ax.axvline(xcut, color=TIER["T3"], lw=1.4, ls="--",
               label=r"XP p99 (Tier 3 cut)")
    ax.axhline(ycut, color=TIER["T2"], lw=1.4, ls="--",
               label=r"label p99 (Tier 2 cut)")
    # Tier-region annotations with white halo so they stand out.
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]
    ax.text(xcut * 0.4, ycut * 0.4, r"Tier 1", fontsize=18,
             color=TIER["T1"], ha="center", va="center",
             fontweight="regular",
             zorder=6, path_effects=halo)
    ax.text(xcut * 0.4, ycut * 1.8, r"Tier 2", fontsize=18,
             color=TIER["T2"], ha="center", va="center",
             fontweight="regular",
             zorder=6, path_effects=halo)
    ax.text(xcut * 1.8, ycut * 0.4, r"Tier 3", fontsize=18,
             color=TIER["T3"], ha="center", va="center",
             fontweight="regular",
             zorder=6, path_effects=halo)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"XP-Mahalanobis distance $d_M^{\rm XP}$")
    ax.set_ylabel(r"label-Mahalanobis distance $d_M^{\rm label}$")
    ax.set_title(r"dual-Mahalanobis percentile cuts")
    leg = ax.legend(loc="upper left", fontsize=9.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_linewidth(0.6)
    leg.set_frame_on(True)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    av = df["av_los"].to_numpy()
    ok = (np.isfinite(av) & np.isfinite(df["ood_mahalanobis_percentile"])
          & np.isfinite(df["label_mahalanobis_percentile"]))
    av = av[ok]
    feat = df["ood_mahalanobis_percentile"].to_numpy()[ok]
    lab = df["label_mahalanobis_percentile"].to_numpy()[ok]
    bins = np.linspace(np.nanpercentile(av, 1),
                        np.nanpercentile(av, 99), 22)
    bin_id = np.digitize(av, bins) - 1
    centres, fm, fl, fh, lm, ll, lh = [], [], [], [], [], [], []
    for k in range(len(bins) - 1):
        m = bin_id == k
        if m.sum() < 30:
            continue
        centres.append(0.5 * (bins[k] + bins[k + 1]))
        fm.append(np.nanmedian(feat[m]))
        fl.append(np.nanpercentile(feat[m], 16))
        fh.append(np.nanpercentile(feat[m], 84))
        lm.append(np.nanmedian(lab[m]))
        ll.append(np.nanpercentile(lab[m], 16))
        lh.append(np.nanpercentile(lab[m], 84))
    centres = np.asarray(centres)
    ax.fill_between(centres, fl, fh, color="#0072B2", alpha=0.18)
    ax.plot(centres, fm, color="#0072B2", lw=1.6, label="XP-Mahalanobis")
    ax.fill_between(centres, ll, lh, color=TIER["T2"], alpha=0.18)
    ax.plot(centres, lm, color=TIER["T2"], lw=1.6, label="label-Mahalanobis")
    ax.axhline(0.99, color="#000000", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel(LABELS["Av"])
    ax.set_ylabel(LABELS["Mahal_pct"])
    ax.set_ylim(0.0, 1.02)
    ax.set_title(r"Mahalanobis percentile vs extinction")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.30)

    save(fig, "F09_tier_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
