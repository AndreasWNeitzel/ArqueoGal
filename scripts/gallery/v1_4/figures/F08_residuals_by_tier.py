"""F08: residual histograms by tier (slide 9)."""

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
    ACCENT_PRIMARY, LABELS, TIER, apply_style, save,
)

LABS = [
    ("teff",    "Teff",    400.0),
    ("logg",    "logg",    0.6),
    ("mh",      "Mh",      0.5),
    ("alpha_m", "alpha_M", 0.20),
]


def _stats(r):
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(r)), float(np.sqrt(np.mean(r * r)))


def main() -> int:
    apply_style()
    df = load_s1_holdout()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.0),
                              layout="constrained")

    for ax, (key, lab, half_range) in zip(axes.ravel(), LABS):
        bins = np.linspace(-half_range, half_range, 60)
        res_all = (df[f"{key}_pred"] - df[f"{key}_apogee"]).to_numpy()
        bias_all, rmse_all = _stats(res_all)
        ax.hist(res_all[np.isfinite(res_all)], bins=bins,
                color=ACCENT_PRIMARY, alpha=0.25, edgecolor="none",
                label=rf"all: bias = {bias_all:+.2g}, RMSE = {rmse_all:.2g}")
        for tier in (1, 2, 3):
            sub = df.loc[df["release_tier"] == tier]
            if not len(sub):
                continue
            r = (sub[f"{key}_pred"] - sub[f"{key}_apogee"]).to_numpy()
            bias, rmse = _stats(r)
            ax.hist(r[np.isfinite(r)], bins=bins,
                    histtype="step", color=TIER[f"T{tier}"], lw=1.6,
                    label=rf"T{tier}: bias = {bias:+.2g}, RMSE = {rmse:.2g}")
        ax.set_yscale("log")
        ax.set_xlim(-half_range, half_range)
        residual_xlabels = {
            "teff":    r"$T_\mathrm{eff}$ residual [K]",
            "logg":    r"$\log g$ residual [dex]",
            "mh":      r"$\mathrm{[M/H]}$ residual [dex]",
            "alpha_m": r"$\mathrm{[\alpha/M]}$ residual [dex]",
        }
        ax.set_xlabel(residual_xlabels[key])
        ax.set_ylabel(r"counts")
        ax.set_title(LABELS[lab])
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25, which="both")

    save(fig, "F08_residuals_by_tier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
