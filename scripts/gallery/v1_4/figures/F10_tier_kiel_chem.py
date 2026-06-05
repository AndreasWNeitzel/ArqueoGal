"""F10: 2x3 Kiel + chemistry per release tier (slide 11)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import load_s1_holdout  # noqa: E402

from arqueogal.style import (  # noqa: E402
    LABELS, apply_style, colorbar, hexbin_density, save,
)

KIEL = (3500, 6500, 0.0, 5.0)
CHEM = (-1.6, 0.55, -0.10, 0.45)


def main() -> int:
    apply_style()
    df = load_s1_holdout()

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.5),
                              layout="constrained")

    # Use the SAME gridsize and mincnt across all three columns so every
    # panel is rendered at uniform resolution. Tier 2 has only ~336
    # stars; we keep mincnt = 1 globally so its panel is not blank.
    GRIDSIZE = 50
    MINCNT = 1
    for j, tier in enumerate((1, 2, 3)):
        sub = df.loc[df["release_tier"] == tier]
        n = len(sub)
        ax = axes[0, j]
        if n:
            hb = hexbin_density(
                ax, sub["teff_pred"].to_numpy(), sub["logg_pred"].to_numpy(),
                gridsize=GRIDSIZE, mincnt=MINCNT, extent=KIEL,
            )
            if j == 2:
                colorbar(ax, hb, LABELS["counts_log"])
        ax.set_xlim(KIEL[1], KIEL[0])
        ax.set_ylim(KIEL[3], KIEL[2])
        ax.set_title(rf"Tier {tier}, $n$ = {n:,}")
        if j == 0:
            ax.set_ylabel(LABELS["logg"])
        if j > 0:
            ax.set_yticklabels([])
        ax.set_xlabel(LABELS["Teff"])
        ax.grid(False)

    for j, tier in enumerate((1, 2, 3)):
        sub = df.loc[df["release_tier"] == tier]
        n = len(sub)
        ax = axes[1, j]
        if n:
            hb = hexbin_density(
                ax, sub["mh_pred"].to_numpy(), sub["alpha_m_pred"].to_numpy(),
                gridsize=GRIDSIZE, mincnt=MINCNT, extent=CHEM,
            )
            if j == 2:
                colorbar(ax, hb, LABELS["counts_log"])
        ax.set_xlim(CHEM[0], CHEM[1])
        ax.set_ylim(CHEM[2], CHEM[3])
        ax.set_title(rf"Tier {tier}, $n$ = {n:,}")
        if j == 0:
            ax.set_ylabel(LABELS["alpha_M"])
        if j > 0:
            ax.set_yticklabels([])
        ax.set_xlabel(LABELS["Mh"])
        ax.grid(False)

    save(fig, "F10_tier_kiel_chem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
