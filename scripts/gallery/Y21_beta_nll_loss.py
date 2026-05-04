"""Y21: β-NLL loss explained — why the model can be trusted to report σ.

Plain Gaussian NLL has a known failure mode: the optimiser learns to inflate
σ on hard cases, which softens the loss without improving the mean. Seitzer
et al. 2022 propose β-NLL: re-weight every star's NLL by σ²^β (with a
gradient stop on the σ term). At β=0 this is plain NLL; at β=0.5 the gradient
on hard cases is restored.

Three panels:

  (left)   the β-NLL formula in math, with the Seitzer trick highlighted
  (centre) hand-built example: same residual, three σ candidates, two losses.
           Shows where the minimum sits as a function of σ for each loss.
  (right)  empirical effect on Stream-1 Tier-1 held-out: pull width vs β.
           Numbers are the live values from Y16.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402


def _math_panel(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rect = mpatches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6, facecolor=PALETTE["paper"], edgecolor=PALETTE["mist"],
    )
    ax.add_patch(rect)
    ax.text(0.5, 0.92, "The loss in one equation",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=PALETTE["navy"])

    ax.text(0.5, 0.78,
            r"$L_{\beta-NLL} \;=\; (\sigma^{2})^{\beta}_{\rm sg}\;\cdot\;"
            r"[\,\frac{1}{2}\log\sigma^{2} + \frac{(\hat{y}-y)^{2}}{2\sigma^{2}}\,]$",
            ha="center", va="center", fontsize=16, color=PALETTE["ink"])
    ax.text(0.5, 0.69,
            r"reweight ($\sigma^{2})^{\beta}$ has gradient stopped on σ.",
            ha="center", va="center", fontsize=10.5, color=PALETTE["ash"],
            fontstyle="italic")

    ax.text(0.5, 0.62,
            r"$\beta = 0$  $\rightarrow$  plain Gaussian NLL "
            r"(σ-inflation pathology)",
            ha="center", va="center", fontsize=12, color=PALETTE["tier3"])
    ax.text(0.5, 0.54,
            r"$\beta = 0.5$  $\rightarrow$  Seitzer+2022, gradient on hard "
            r"cases restored",
            ha="center", va="center", fontsize=12, color=PALETTE["tier1"])

    ax.text(0.5, 0.40,
            "ArqueoGal v1 trains with β = 0 (plain NLL) plus the\n"
            "Cholesky covariance head; the L matrix shape is what\n"
            "absorbs the σ-inflation pressure rather than the β\n"
            "reweighting. The full multivariate version is\n"
            r"``losses.beta_nll_block_cholesky`` in the codebase.",
            ha="center", va="center", fontsize=11.5, color=PALETTE["ash"],
            fontstyle="italic")

    ax.text(0.5, 0.13,
            "→ pull-width per label is the empirical proof\n  it works  (panel right)",
            ha="center", va="center", fontsize=11, color=PALETTE["accent"],
            fontweight="bold")


def _curves_panel(ax):
    sigma = np.linspace(0.05, 2.0, 400)
    residual = 1.0
    nll = 0.5 * np.log(sigma ** 2) + 0.5 * (residual ** 2) / (sigma ** 2)
    bnll = (sigma ** 2) ** 0.5 * nll  # β=0.5 reweight (no stopgrad here, illustrative)

    sigma_min_nll = sigma[np.argmin(nll)]
    sigma_min_bnll = sigma[np.argmin(bnll)]

    ax.plot(sigma, nll, color=PALETTE["tier3"], lw=2.4,
            label=r"$\beta = 0$ (plain NLL)")
    ax.plot(sigma, bnll - bnll.min() + nll.min(), color=PALETTE["tier1"], lw=2.4,
            label=r"$\beta = 0.5$ (Seitzer+2022)")
    ax.axvline(sigma_min_nll, color=PALETTE["tier3"], ls="--", lw=1.4, alpha=0.7)
    ax.axvline(sigma_min_bnll, color=PALETTE["tier1"], ls="--", lw=1.4, alpha=0.7)
    ax.axvline(residual, color=PALETTE["ink"], ls=":", lw=1.6,
               label=r"$\sigma$ = residual (ideal)")
    ax.set_xlabel(r"predicted $\sigma$  (residual = 1)")
    ax.set_ylabel("loss value")
    ax.set_title("Same residual — where each loss prefers σ to land",
                 color=PALETTE["navy"])
    ax.legend(loc="upper right")
    ax.text(sigma_min_nll, ax.get_ylim()[0] + 0.12,
            f"  argmin = {sigma_min_nll:.2f}",
            color=PALETTE["tier3"], fontsize=10, fontweight="bold")
    ax.text(sigma_min_bnll, ax.get_ylim()[0] + 0.36,
            f"  argmin = {sigma_min_bnll:.2f}",
            color=PALETTE["tier1"], fontsize=10, fontweight="bold")


def _empirical_panel(ax):
    # Live numbers carried over from Y16.
    labels = [r"$T_{\rm eff}$", r"$\log g$", "[M/H]", r"[$\alpha$/M]", "[Mg/H]"]
    pull_sig = [0.85, 0.88, 0.92, 0.96, 0.92]
    x = np.arange(len(labels))
    bars = ax.bar(x, pull_sig, color=PALETTE["navy_light"],
                  edgecolor="white", linewidth=1.4, width=0.65)
    ax.axhspan(0.8, 1.2, color=PALETTE["tier1"], alpha=0.15,
               label=r"calibrated band  ($|\sigma_{\rm pull} - 1| < 0.2$)")
    ax.axhline(1.0, color=PALETTE["tier1"], lw=2.2, ls="--",
               label="ideal = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel(r"empirical pull $\sigma_{\rm MAD}$  (Y16)")
    ax.set_title(r"Loss → calibrated $\sigma$  (Stream-1 held-out)",
                 color=PALETTE["navy"])
    ax.legend(loc="upper right", fontsize=10)
    for bar, v in zip(bars, pull_sig):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=PALETTE["ink"])


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    plt.subplots_adjust(wspace=0.28, left=0.05, right=0.97, bottom=0.12)
    _math_panel(axes[0])
    _curves_panel(axes[1])
    _empirical_panel(axes[2])
    headline(
        fig,
        "The loss — and why it produces honest σ",
        "Gaussian NLL with block-Cholesky head; β-reweighting kept available "
        "(losses.beta_nll_block_cholesky) for ablation but β=0 already calibrates.",
        top=0.85,
    )
    save(fig, "Y21_beta_nll_loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
