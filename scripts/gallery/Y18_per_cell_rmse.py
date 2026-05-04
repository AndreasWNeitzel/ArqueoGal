"""Y18: Per-cell RMSE in (Teff, log g) Kiel space, per label.

Five-panel heatmap: for each label, the held-out RMSE computed in cells of
APOGEE truth (Teff, log g). Cells with < 30 stars are masked.

The talk message: average RMSE hides a lot. The model works best in the
mid-RGB; the cool tip and the warm-low-gravity corner have larger errors.
This is the kind of plot that sets users' expectations honestly.
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

TEFF_EDGES = np.linspace(4000, 5500, 11)
LOGG_EDGES = np.linspace(1.0, 3.5, 11)
MIN_PER_CELL = 30


def _vmax(key: str) -> float:
    return {"teff": 100.0, "logg": 0.18, "mh": 0.12, "alpha_m": 0.06, "mg_h": 0.12}[key]


def main() -> int:
    apply_style()
    df = load_holdout()
    n = len(df)

    teff = df["teff_apogee"].to_numpy()
    logg = df["logg_apogee"].to_numpy()
    teff_idx = np.clip(np.searchsorted(TEFF_EDGES, teff, side="right") - 1, 0, len(TEFF_EDGES) - 2)
    logg_idx = np.clip(np.searchsorted(LOGG_EDGES, logg, side="right") - 1, 0, len(LOGG_EDGES) - 2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.40, hspace=0.42)
    axes = axes.ravel()
    axes[5].axis("off")

    for ax, spec in zip(axes[:5], LABELS):
        k = spec["key"]
        delta = df[f"{k}_pred"].to_numpy() - df[f"{k}_apogee"].to_numpy()
        nT = len(TEFF_EDGES) - 1
        nG = len(LOGG_EDGES) - 1
        rmse = np.full((nG, nT), np.nan, dtype=np.float64)
        cnt = np.zeros((nG, nT), dtype=np.int64)
        for i in range(nT):
            for j in range(nG):
                m = (teff_idx == i) & (logg_idx == j) & np.isfinite(delta)
                if int(m.sum()) >= MIN_PER_CELL:
                    rmse[j, i] = float(np.sqrt(np.mean(delta[m] ** 2)))
                    cnt[j, i] = int(m.sum())

        im = ax.imshow(
            rmse,
            origin="lower",
            cmap="viridis",
            extent=(TEFF_EDGES[0], TEFF_EDGES[-1], LOGG_EDGES[0], LOGG_EDGES[-1]),
            aspect="auto",
            vmin=0,
            vmax=_vmax(k),
        )
        ax.set_xlim(TEFF_EDGES[-1], TEFF_EDGES[0])
        ax.set_ylim(LOGG_EDGES[-1], LOGG_EDGES[0])
        ax.set_xlabel(r"$T_{\rm eff}$ truth (K)")
        ax.set_ylabel(r"$\log g$ truth (dex)")
        ax.set_title(spec["name"], color=PALETTE["navy"])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(rf"RMSE ({spec['rmse_unit']})", fontsize=10)
        # Overlay sample-count text in each cell.
        for i in range(nT):
            for j in range(nG):
                if cnt[j, i] >= MIN_PER_CELL:
                    cx = 0.5 * (TEFF_EDGES[i] + TEFF_EDGES[i + 1])
                    cy = 0.5 * (LOGG_EDGES[j] + LOGG_EDGES[j + 1])
                    ax.text(
                        cx,
                        cy,
                        f"{cnt[j, i]:,}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        alpha=0.85,
                    )

    headline(
        fig,
        "Where the model works — RMSE in Kiel cells",
        f"Stream 1 Tier 1 held-out, n = {n:,}.  "
        f"Per-cell numbers = stars contributing.  Cells below {MIN_PER_CELL} stars greyed.",
        top=0.90,
    )
    save(fig, "Y18_per_cell_rmse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
