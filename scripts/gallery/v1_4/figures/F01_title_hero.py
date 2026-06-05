"""F01: title-slide hero, Tier-1 chemistry plane with APOGEE-truth contour."""

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
    LABELS, apply_style, hexbin_density, save,
)

CHEM = (-1.6, 0.55, -0.10, 0.45)


def main() -> int:
    apply_style()
    df = load_s1_holdout()
    t1 = df.loc[df["release_tier"] == 1]

    fig, ax = plt.subplots(figsize=(6.0, 5.5), layout="constrained")
    # Use np.histogram2d for the prediction surface so the contour and the
    # density share the same binning grid (the v1.3 hexbin / contour
    # mismatch came from contouring on histogram bins while showing
    # hexbins on a different tessellation).
    bins = 60
    H_pred, xe, ye = np.histogram2d(
        t1["mh_pred"].to_numpy(), t1["alpha_m_pred"].to_numpy(),
        bins=bins, range=[[CHEM[0], CHEM[1]], [CHEM[2], CHEM[3]]],
    )
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    from matplotlib.colors import LogNorm
    pcm = ax.pcolormesh(xe, ye, H_pred.T,
                         norm=LogNorm(vmin=1, vmax=H_pred.max()),
                         cmap="viridis", shading="auto")
    H_truth, _, _ = np.histogram2d(
        t1["mh_apogee"].to_numpy(), t1["alpha_m_apogee"].to_numpy(),
        bins=bins, range=[[CHEM[0], CHEM[1]], [CHEM[2], CHEM[3]]],
    )
    Hf = H_truth.T.astype(float)
    flat = Hf.ravel()
    flat = flat[flat > 0]
    if flat.size:
        thr = np.percentile(flat, [50.0, 90.0])
        ax.contour(xc, yc, Hf, levels=thr, colors="#000000",
                   linewidths=1.0, linestyles="--")
    ax.set_xlim(CHEM[0], CHEM[1]); ax.set_ylim(CHEM[2], CHEM[3])
    ax.set_xlabel(LABELS["Mh"])
    ax.set_ylabel(LABELS["alpha_M"])
    ax.set_aspect("auto")
    ax.grid(False)

    save(fig, "F01_title_hero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
