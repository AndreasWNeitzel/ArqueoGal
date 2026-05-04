"""G4: Dereddened IR photometry — quality + Av-corrected magnitudes.

Five panels comparing raw vs dereddened 2MASS/WISE bands, plus Av-vs-Δm
diagnostics so you can see the correction working.

Per-band: raw → dereddened histograms + scatter of (raw − dered) vs Av.
The correction should be approximately Av · (k_λ / R_V) for each band.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig  # noqa: E402

FEAT = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
OUT = REPO / "reports/gallery/G_extinction"
BANDS = (
    ("j", "J", "2MASS J"),
    ("h", "H", "2MASS H"),
    ("k", "K", "2MASS K"),
    ("w1", "W1", "WISE W1"),
    ("w2", "W2", "WISE W2"),
)
# Cardelli+1989 R_V=3.1 extinction coefficients k_λ / A_V.
COEFF = {"j": 0.282, "h": 0.175, "k": 0.112, "w1": 0.057, "w2": 0.038}


def main() -> int:
    apply_style()
    cols = ["source_id", "av_los"]
    for short, _, _ in BANDS:
        cols.append(f"{short}_mag")
        cols.append(f"{short}_mag_dered")
    df = pd.read_parquet(FEAT, columns=cols).drop_duplicates("source_id")
    n = len(df)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(
        2, 5, hspace=0.40, wspace=0.32, top=0.91, bottom=0.06, left=0.05, right=0.97
    )

    for j, (short, name, full) in enumerate(BANDS):
        # Top: raw + dereddened histogram.
        ax = fig.add_subplot(gs[0, j])
        raw = df[f"{short}_mag"].to_numpy()
        dered = df[f"{short}_mag_dered"].to_numpy()
        ok_raw = np.isfinite(raw) & (raw > -10) & (raw < 25)
        ok_dered = np.isfinite(dered) & (dered > -10) & (dered < 25)
        bins = np.linspace(2, 18, 80)
        ax.hist(
            raw[ok_raw],
            bins=bins,
            histtype="step",
            color="#1f77b4",
            lw=2.0,
            label=f"raw   med={float(np.median(raw[ok_raw])):.2f}  n={int(ok_raw.sum()):,}",
        )
        ax.hist(
            dered[ok_dered],
            bins=bins,
            histtype="step",
            color="#2ca02c",
            lw=2.0,
            label=f"dered med={float(np.median(dered[ok_dered])):.2f}  n={int(ok_dered.sum()):,}",
        )
        ax.set_xlim(2, 18)
        ax.set_xlabel(f"{name} (mag)")
        ax.set_ylabel("count")
        ax.set_title(f"({chr(ord('a') + j)}) {full}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)

        # Bottom: (raw - dered) vs Av — should track the C+89 coefficient.
        ax = fig.add_subplot(gs[1, j])
        delta = raw - dered
        av = df["av_los"].to_numpy()
        ok = (
            np.isfinite(delta) & np.isfinite(av) & (av >= 0) & (av < 6) & (delta > -1) & (delta < 3)
        )
        if ok.sum() >= 50:
            ax.hexbin(
                av[ok],
                delta[ok],
                gridsize=60,
                extent=(0, 6, -0.2, 2.0),
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            # Theoretical line.
            x = np.array([0, 6])
            ax.plot(
                x,
                COEFF[short] * x,
                color="#e07b00",
                lw=2.0,
                ls="--",
                label=rf"C+89 $k_\lambda/A_V = {COEFF[short]}$",
            )
            ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, 6)
        ax.set_ylim(-0.2, 2.0)
        ax.set_xlabel(r"$A_V$ (mag)")
        ax.set_ylabel(rf"raw $-$ dered  ({name})")
        ax.set_title(f"({chr(ord('f') + j)}) Δ{name} vs $A_V$")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"G4. Dereddened IR photometry  (Stream 1, n = {n:,})\n"
        "Top row: raw vs dereddened magnitude histograms.  "
        "Bottom row: per-star magnitude correction vs Av (orange = "
        "Cardelli+1989 R_V=3.1 expectation).",
        fontsize=12,
        fontweight="semibold",
        y=0.985,
    )
    save_fig(fig, OUT / "G4_dereddened_photometry", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
