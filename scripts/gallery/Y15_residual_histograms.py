"""Y15: Residual (Δ = pred − truth) histograms per label.

Five-panel figure: one histogram per label on Stream-1 Tier-1 held-out.
Each panel annotated with bias, RMSE, robust σ (1.4826·MAD), and a Gaussian
curve of matching σ overlaid as a sanity reference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402
from _y_holdout import LABELS, load_holdout  # noqa: E402


def _xrange(key: str) -> tuple[float, float]:
    return {"teff": (-300, 300), "logg": (-0.6, 0.6),
            "mh": (-0.4, 0.4), "alpha_m": (-0.15, 0.15),
            "mg_h": (-0.4, 0.4)}[key]


def main() -> int:
    apply_style()
    df = load_holdout()
    n = len(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(wspace=0.32, hspace=0.40)
    axes = axes.ravel()
    axes[5].axis("off")

    for ax, spec in zip(axes[:5], LABELS):
        k = spec["key"]
        delta = df[f"{k}_pred"].to_numpy() - df[f"{k}_apogee"].to_numpy()
        delta = delta[np.isfinite(delta)]
        bias = float(np.mean(delta))
        rmse = float(np.sqrt(np.mean(delta ** 2)))
        rsig = float(1.4826 * np.median(np.abs(delta - np.median(delta))))

        lo, hi = _xrange(k)
        ax.hist(delta, bins=80, range=(lo, hi),
                color=PALETTE["navy_light"], edgecolor="white",
                linewidth=0.4, density=True, label="residuals")
        # Gaussian comparison curve at robust σ.
        xs = np.linspace(lo, hi, 400)
        gauss = (1.0 / (rsig * np.sqrt(2 * np.pi))) \
                * np.exp(-0.5 * ((xs - bias) / rsig) ** 2)
        ax.plot(xs, gauss, color=PALETTE["accent"], lw=2.4,
                label=rf"$N$(bias, $\sigma_{{\rm MAD}}$)")
        ax.axvline(0.0, color=PALETTE["ink"], lw=1.0, ls="-", alpha=0.5)
        ax.axvline(bias, color=PALETTE["accent"], lw=1.6, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(rf"$\Delta$ {spec['name']}  ({spec['unit']})")
        ax.set_ylabel("density")
        ax.set_title(spec["name"], color=PALETTE["navy"])
        ax.text(
            0.02, 0.97,
            f"bias = {bias:+.3g}\nRMSE = {rmse:.3g} {spec['rmse_unit']}\n"
            rf"$\sigma_{{\rm MAD}}$ = {rsig:.3g} {spec['rmse_unit']}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["mist"]),
        )
        ax.legend(loc="upper right", fontsize=9.5)

    headline(
        fig,
        "Residuals — Δ = prediction − truth",
        f"Stream 1 Tier 1 held-out, n = {n:,}.  "
        "Centre = no bias.  Robust σ from the median absolute deviation.",
        top=0.89,
    )
    save(fig, "Y15_residual_histograms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
