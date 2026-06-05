"""F12: Stream-2 transfer (slide 13)."""

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

from _loaders import (  # noqa: E402
    load_s2_apogee, load_s2_gspspec, load_s2_predictions,
)
from arqueogal.style import (  # noqa: E402
    LABELS, annotate_corner, apply_style, colorbar, hexbin_density, save,
)

KIEL = (3500, 6500, 0.0, 5.0)
CHEM = (-1.6, 0.55, -0.10, 0.45)


def main() -> int:
    apply_style()
    pred = load_s2_predictions()
    t1 = pred.loc[pred["release_tier"] == 1].reset_index(drop=True)
    s2_ids = pred["source_id"].to_numpy()
    gsp = load_s2_gspspec()
    gsp = gsp.loc[gsp["source_id"].isin(set(s2_ids))]
    apo = load_s2_apogee(s2_ids)

    # Match F10 (slide 11) layout: figsize 11 x 5.5, constrained_layout.
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.5),
                              layout="constrained")

    panels_kiel = [
        (r"GSP-Spec", gsp, "teff_gspspec", "logg_gspspec"),
        (r"APOGEE DR19", apo, "teff", "logg"),
        (r"JANUS Tier 1", t1, "teff_pred", "logg_pred"),
    ]
    last_top = None
    last_bot = None
    last = None
    for j, (name, frame, x, y) in enumerate(panels_kiel):
        ax = axes[0, j]
        last_top = hexbin_density(
            ax, frame[x].to_numpy(), frame[y].to_numpy(),
            gridsize=50, mincnt=4, extent=KIEL,
        )
        last = last_top
        ax.set_xlim(KIEL[1], KIEL[0]); ax.set_ylim(KIEL[3], KIEL[2])
        ax.set_xlabel(LABELS["Teff"])
        ax.set_ylabel(LABELS["logg"] if j == 0 else "")
        if j > 0: ax.set_yticklabels([])
        ax.set_title(rf"{name} ($n$ = {len(frame):,})", fontweight="regular")
        ax.grid(False)
        if j == 1:
            annotate_corner(
                ax,
                rf"only {len(apo) / max(len(s2_ids), 1) * 100:.1f}% of S2"
                "\n" r"has APOGEE chemistry",
                loc="lower left", fontsize=9,
            )

    panels_chem = [
        (r"GSP-Spec", gsp, "mh_gspspec", "alphafe_gspspec", "alpha_Fe"),
        (r"APOGEE DR19", apo, "m_h_atm", "alpha_m_atm", "alpha_M"),
        (r"JANUS Tier 1", t1, "mh_pred", "alpha_m_pred", "alpha_M"),
    ]
    for j, (name, frame, mh, am, ylab) in enumerate(panels_chem):
        ax = axes[1, j]
        last_bot = hexbin_density(
            ax, frame[mh].to_numpy(), frame[am].to_numpy(),
            gridsize=50, mincnt=4, extent=CHEM,
        )
        last = last_bot
        ax.set_xlim(CHEM[0], CHEM[1]); ax.set_ylim(CHEM[2], CHEM[3])
        ax.set_xlabel(LABELS["Mh"])
        ax.set_ylabel(LABELS[ylab] if j == 0 else "")
        if j > 0: ax.set_yticklabels([])
        ax.set_title(rf"{name} ($n$ = {len(frame):,})", fontweight="regular")
        ax.grid(False)

    # Track the last hexbin in EACH row separately so both rows get
    # their own colorbar (the v1.4-rev only had one on the top row).
    if last is not None:
        colorbar(axes[0, -1], last_top, LABELS["counts_log"])
        colorbar(axes[1, -1], last_bot, LABELS["counts_log"])
    save(fig, "F12_stream2_transfer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
