"""Y16: Pull distributions — z = (pred − truth)/σ_pred per label.

If the model's per-star σ is calibrated, the pull distribution is N(0, 1).
Each panel overlays the empirical pull histogram on a unit-normal reference
and annotates the empirical mean and width. Width > 1 → σ underestimated;
width < 1 → σ overestimated.

Note: this is a per-label diagnostic, not the full covariance check (which
would be a multivariate χ² test). The covariance diagonal is what calibrated
σ promises in the catalog ReadMe; that is what this figure tests.
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
        sig = df[f"{k}_sigma"].to_numpy()
        ok = np.isfinite(delta) & np.isfinite(sig) & (sig > 0)
        z = delta[ok] / sig[ok]
        # Clip outliers for the plot only (keep stats on the full set).
        z_plot = np.clip(z, -5.0, 5.0)
        mu = float(np.mean(z))
        std = float(np.std(z))
        rsig = float(1.4826 * np.median(np.abs(z - np.median(z))))

        ax.hist(
            z_plot,
            bins=80,
            range=(-5, 5),
            color=PALETTE["navy_light"],
            edgecolor="white",
            linewidth=0.4,
            density=True,
            label="pulls",
        )
        xs = np.linspace(-5, 5, 400)
        n01 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs**2)
        ax.plot(xs, n01, color=PALETTE["accent"], lw=2.4, label=r"$N(0, 1)$ ideal")
        ax.axvline(0.0, color=PALETTE["ink"], lw=1.0, ls="-", alpha=0.5)
        ax.axvline(mu, color=PALETTE["accent"], lw=1.4, ls="--")
        ax.set_xlim(-5, 5)
        ax.set_xlabel(rf"$z$ = $\Delta${spec['name']} / $\sigma_{{\rm pred}}$")
        ax.set_ylabel("density")
        ax.set_title(spec["name"], color=PALETTE["navy"])

        # Verdict colour: green if width within 20% of 1, orange otherwise.
        width_color = PALETTE["tier1"] if 0.8 <= rsig <= 1.2 else PALETTE["tier2"]
        ax.text(
            0.02,
            0.97,
            f"mean = {mu:+.2f}\nstd = {std:.2f}\n"
            rf"$\sigma_{{\rm MAD}}$ = {rsig:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=width_color, edgecolor="none"),
        )
        ax.legend(loc="upper right", fontsize=9.5)

    headline(
        fig,
        "Pulls — is the predicted σ honest?",
        f"Stream 1 Tier 1 held-out, n = {n:,}.  "
        r"Pull width $\approx 1$ means $\sigma_{\rm pred}$ is calibrated; "
        "green if within ±20% of unity.",
        top=0.89,
    )
    save(fig, "Y16_pull_distributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
