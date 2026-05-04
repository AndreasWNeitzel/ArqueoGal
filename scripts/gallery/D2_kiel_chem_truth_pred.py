"""D2: Stream 1 holdout Kiel + chemistry side-by-side truth vs predicted.

Top row: Kiel diagram (Teff vs log g), coloured by [M/H]. Box shows quality cuts:
Teff [4000, 5500] K and log g [1.0, 3.5] dex.
Bottom row: Chemical plane ([alpha/M] vs [M/H]), coloured by Teff.
Left column: truth. Right column: predicted.

Visualizes structure preservation (disc bimodality in chemistry).

Stream: Stream 1 (APOGEE x Gaia DR3, holdout test set post-quality-cut).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import load_stream1_holdout, save_fig

from arqueogal.utils.plotting import set_aa_style


def main(n_stars: int | None = None) -> None:
    set_aa_style()

    data = load_stream1_holdout()

    if n_stars is not None and n_stars < len(data):
        data = data.sample(n=n_stars, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")

    # All four panels colour-mapped by stellar count per hexbin (NOT by a third
    # quantity). Reads as a 2D number-density histogram so the eye can compare
    # truth vs predicted structure on equal footing.

    # Top left: Kiel truth with quality-cut box
    ax = axes[0, 0]
    hb = ax.hexbin(data["teff_apogee"], data["logg_apogee"],
                   gridsize=70, mincnt=1, cmap="viridis",
                   extent=[3900, 5600, 0.8, 3.7], bins="log")
    rect = mpatches.Rectangle(
        (4000, 1.0), 1500, 2.5, linewidth=1.2, edgecolor="red",
        facecolor="none", linestyle="--", alpha=0.85
    )
    ax.add_patch(rect)
    ax.set_xlim(5600, 3900)
    ax.set_ylim(3.7, 0.8)
    ax.set_xlabel(r"$T_{\rm eff}$ [K]", fontsize=9)
    ax.set_ylabel(r"$\log g$ [dex]", fontsize=9)
    ax.set_title("Kiel: truth", fontsize=9)
    plt.colorbar(hb, ax=ax, label="log10 N per bin")

    # Top right: Kiel pred
    ax = axes[0, 1]
    hb = ax.hexbin(data["teff_pred"], data["logg_pred"],
                   gridsize=70, mincnt=1, cmap="viridis",
                   extent=[3900, 5600, 0.8, 3.7], bins="log")
    rect = mpatches.Rectangle(
        (4000, 1.0), 1500, 2.5, linewidth=1.2, edgecolor="red",
        facecolor="none", linestyle="--", alpha=0.85
    )
    ax.add_patch(rect)
    ax.set_xlim(5600, 3900)
    ax.set_ylim(3.7, 0.8)
    ax.set_xlabel(r"$T_{\rm eff}$ [K]", fontsize=9)
    ax.set_ylabel(r"$\log g$ [dex]", fontsize=9)
    ax.set_title("Kiel: predicted", fontsize=9)
    plt.colorbar(hb, ax=ax, label="log10 N per bin")

    # Chemistry plane limits — relaxed so disc + halo + outliers are all visible.
    chem_xlim = (-2.5, 0.6)
    chem_ylim = (-0.20, 0.50)

    # Bottom left: Chemistry truth (counts)
    ax = axes[1, 0]
    hb = ax.hexbin(data["mh_apogee"], data["alpha_m_apogee"],
                   gridsize=70, mincnt=1, cmap="viridis",
                   extent=[*chem_xlim, *chem_ylim], bins="log")
    ax.set_xlim(*chem_xlim)
    ax.set_ylim(*chem_ylim)
    ax.set_xlabel("[M/H] [dex]", fontsize=9)
    ax.set_ylabel(r"[$\alpha$/M] [dex]", fontsize=9)
    ax.set_title("Chemistry: truth", fontsize=9)
    plt.colorbar(hb, ax=ax, label="log10 N per bin")

    # Bottom right: Chemistry pred (counts)
    ax = axes[1, 1]
    hb = ax.hexbin(data["mh_pred"], data["alpha_m_pred"],
                   gridsize=70, mincnt=1, cmap="viridis",
                   extent=[*chem_xlim, *chem_ylim], bins="log")
    ax.set_xlim(*chem_xlim)
    ax.set_ylim(*chem_ylim)
    ax.set_xlabel("[M/H] [dex]", fontsize=9)
    ax.set_ylabel(r"[$\alpha$/M] [dex]", fontsize=9)
    ax.set_title("Chemistry: predicted", fontsize=9)
    plt.colorbar(hb, ax=ax, label="log10 N per bin")

    fig.suptitle(
        f"Stream 1: Kiel + chemistry truth vs predicted (n={len(data):,}). "
        "Hexbin colour = log10 stars per bin. Red dashed: Teff [4000,5500] K, log g [1.0,3.5] dex.",
        fontsize=10,
    )

    out = REPO / "reports" / "gallery" / "D_predictions"
    out.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out / "D2_kiel_chem_truth_pred", formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-stars",
        type=int,
        default=None,
        help="Optional: downsample to N stars (default: use all)",
    )
    args = parser.parse_args()
    main(n_stars=args.n_stars)
