"""F11: three-way Stream-1 comparison, Kiel + chemistry, GSP-Spec / APOGEE / JANUS.

Same layout and aesthetic as F12 (Stream-2 transfer): 2x3 grid with
Kiel on top, chemistry plane on bottom; one column per chemistry source.
Stream-1 Tier-1 holdout cohort.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import load_s1_apogee, load_s1_gspspec, load_s1_holdout  # noqa: E402

from arqueogal.style import (  # noqa: E402
    LABELS, apply_style, colorbar, hexbin_density, save,
)

KIEL = (3500, 6500, 0.0, 5.0)
CHEM = (-1.6, 0.55, -0.10, 0.45)


def main() -> int:
    apply_style()
    df = load_s1_holdout()
    t1 = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    s1_ids = t1["source_id"].to_numpy()
    gsp = load_s1_gspspec(s1_ids)
    apo = load_s1_apogee(s1_ids)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.5),
                              layout="constrained")

    # Kiel.
    panels_kiel = [
        (r"GSP-Spec", gsp, "teff_gspspec", "logg_gspspec"),
        (r"APOGEE DR19", apo, "teff", "logg"),
        (r"JANUS Tier 1", t1, "teff_pred", "logg_pred"),
    ]
    last_top = None
    for j, (name, frame, xcol, ycol) in enumerate(panels_kiel):
        ax = axes[0, j]
        last_top = hexbin_density(
            ax, frame[xcol].to_numpy(), frame[ycol].to_numpy(),
            gridsize=50, mincnt=1, extent=KIEL,
        )
        ax.set_xlim(KIEL[1], KIEL[0]); ax.set_ylim(KIEL[3], KIEL[2])
        ax.set_xlabel(LABELS["Teff"])
        ax.set_ylabel(LABELS["logg"] if j == 0 else "")
        if j > 0:
            ax.set_yticklabels([])
        ax.set_title(rf"{name} ($n$ = {len(frame):,})", fontweight="regular")
        ax.grid(False)

    # Chemistry.
    panels_chem = [
        (r"GSP-Spec", gsp, "mh_gspspec", "alphafe_gspspec", "alpha_Fe"),
        (r"APOGEE DR19", apo, "m_h_atm", "alpha_m_atm", "alpha_M"),
        (r"JANUS Tier 1", t1, "mh_pred", "alpha_m_pred", "alpha_M"),
    ]
    last_bot = None
    for j, (name, frame, mh, am, ylab) in enumerate(panels_chem):
        ax = axes[1, j]
        last_bot = hexbin_density(
            ax, frame[mh].to_numpy(), frame[am].to_numpy(),
            gridsize=50, mincnt=1, extent=CHEM,
        )
        ax.set_xlim(CHEM[0], CHEM[1]); ax.set_ylim(CHEM[2], CHEM[3])
        ax.set_xlabel(LABELS["Mh"])
        ax.set_ylabel(LABELS[ylab] if j == 0 else "")
        if j > 0:
            ax.set_yticklabels([])
        ax.set_title(rf"{name} ($n$ = {len(frame):,})", fontweight="regular")
        ax.grid(False)

    if last_top is not None:
        colorbar(axes[0, -1], last_top, LABELS["counts_log"])
    if last_bot is not None:
        colorbar(axes[1, -1], last_bot, LABELS["counts_log"])

    save(fig, "F11_three_way_chemistry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
