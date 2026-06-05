"""F07: calibration check, pull histograms + reliability (slide 8)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import load_s1_holdout  # noqa: E402

from arqueogal.style import (  # noqa: E402
    ACCENT_PRIMARY, CHROME, LABELS, annotate_corner, apply_style, save,
)

LABS = [
    ("teff",    "Teff"),
    ("logg",    "logg"),
    ("mh",      "Mh"),
    ("alpha_m", "alpha_M"),
    ("mg_h",    "MgH"),
]


def _pull(t1, key):
    r = t1[f"{key}_pred"].to_numpy() - t1[f"{key}_apogee"].to_numpy()
    s = t1[f"{key}_sigma"].to_numpy()
    p = r / np.where(s > 0, s, np.nan)
    return p[np.isfinite(p)]


def main() -> int:
    apply_style()
    df = load_s1_holdout()
    t1 = df.loc[df["release_tier"] == 1].reset_index(drop=True)

    fig, axes = plt.subplots(2, 5, figsize=(13.0, 5.5),
                              layout="constrained")

    bins = np.linspace(-4, 4, 51)
    for j, (key, lab) in enumerate(LABS):
        ax = axes[0, j]
        p = _pull(t1, key)
        ax.hist(p, bins=bins, density=True,
                color=ACCENT_PRIMARY, alpha=0.6, edgecolor="#1A2B4C", lw=0.8)
        x = np.linspace(-4, 4, 200)
        ax.plot(x, norm.pdf(x, 0, 1), color="#000000", lw=1.0, ls="--")
        annotate_corner(
            ax, rf"std = {p.std():.2f}" "\n" rf"mean = {p.mean():+.2f}",
            loc="upper right", fontsize=9,
        )
        ax.set_xlim(-4, 4)
        ax.set_xlabel(r"pull")
        ax.set_ylabel(r"density" if j == 0 else "")
        ax.set_title(LABELS[lab])
        ax.grid(True, alpha=0.30)

    for j, (key, lab) in enumerate(LABS):
        ax = axes[1, j]
        sigma = t1[f"{key}_sigma"].to_numpy()
        residual = t1[f"{key}_pred"].to_numpy() - t1[f"{key}_apogee"].to_numpy()
        ok = np.isfinite(sigma) & np.isfinite(residual) & (sigma > 0)
        sigma = sigma[ok]; residual = residual[ok]
        bins_pct = np.percentile(sigma, np.linspace(0, 100, 11))
        bins_pct = np.unique(bins_pct)
        idx = np.digitize(sigma, bins_pct) - 1
        sigma_med, rmse_obs, rmse_lo, rmse_hi = [], [], [], []
        for k in range(len(bins_pct) - 1):
            m = idx == k
            if m.sum() < 30:
                continue
            sigma_med.append(np.median(sigma[m]))
            r = residual[m]
            rmse_obs.append(np.sqrt(np.mean(r * r)))
            boots = []
            rng = np.random.default_rng(0)
            for _ in range(80):
                s = rng.choice(r, size=len(r), replace=True)
                boots.append(np.sqrt(np.mean(s * s)))
            lo, hi = np.percentile(boots, [16, 84])
            rmse_lo.append(lo); rmse_hi.append(hi)
        sigma_med = np.asarray(sigma_med)
        rmse_obs = np.asarray(rmse_obs)
        rmse_lo = np.asarray(rmse_lo); rmse_hi = np.asarray(rmse_hi)
        ax.errorbar(
            sigma_med, rmse_obs,
            yerr=[rmse_obs - rmse_lo, rmse_hi - rmse_obs],
            fmt="o", color=ACCENT_PRIMARY, markersize=4, lw=1.0,
        )
        lo = float(min(sigma_med.min(), rmse_obs.min()))
        hi = float(max(sigma_med.max(), rmse_obs.max()))
        ax.plot([lo, hi], [lo, hi], color="#000000", lw=0.8, ls="--")
        if rmse_obs.size and sigma_med.size:
            ratio = float(np.median(rmse_obs / sigma_med))
            annotate_corner(ax, rf"RMSE / $\sigma$ = {ratio:.2f}",
                            loc="upper left", fontsize=9)
        ax.set_xlabel(r"predicted $\sigma$")
        ax.set_ylabel(r"observed RMSE" if j == 0 else "")
        ax.set_title(LABELS[lab])
        ax.grid(True, alpha=0.30)

    save(fig, "F07_pulls_reliability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
