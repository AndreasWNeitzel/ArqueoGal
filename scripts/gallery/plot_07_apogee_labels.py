"""Stage 07: APOGEE DR19 labels + Mészáros+2025 [X/M] correction.

What the deploy did: ``data.apogee_dr19`` ingested DR19 ASPCAP and applied
the Mészáros+2025 Table-3 polynomial corrections per element on RGB-only
stars (logg < 3.8). AGENTS.md hard rule #13 — corrections are mandatory
before training.

What we plot: per-label NaN rates, Teff vs logg pseudo-Kiel from APOGEE
labels (training pool ground truth), [Mg/H] vs [Fe/H] α-bimodality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/07_apogee_labels"
S1 = REPO / "data/processed/pipeline1_features_stream1.parquet"

LABEL_COLS = (
    "teff_apogee",
    "logg_apogee",
    "mh_apogee",
    "alpha_m_apogee",
    "mg_h_apogee",
    "fe_h_apogee",
    "ca_h_apogee",
    "si_h_apogee",
    "al_h_apogee",
    "n_h_apogee",
    "c_h_apogee",
)


def main() -> None:
    apply_style()
    df = pd.read_parquet(S1, columns=list(LABEL_COLS))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # NaN rates
    rates = [(c.replace("_apogee", ""), df[c].isna().mean() * 100) for c in LABEL_COLS]
    names = [r[0] for r in rates]
    pcts = [r[1] for r in rates]
    bars = axes[0].bar(range(len(names)), pcts, color="#1f77b4")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("% NaN in Stream 1")
    axes[0].set_title("APOGEE label coverage (post-Mészáros+25)")
    for b, p in zip(bars, pcts):
        if p > 0.1:
            axes[0].text(
                b.get_x() + b.get_width() / 2, p + 0.1, f"{p:.1f}", ha="center", fontsize=7
            )

    # APOGEE pseudo-Kiel
    teff = df["teff_apogee"].to_numpy()
    logg = df["logg_apogee"].to_numpy()
    m = np.isfinite(teff) & np.isfinite(logg)
    h = axes[1].hexbin(
        teff[m],
        logg[m],
        gridsize=70,
        mincnt=10,
        cmap="viridis",
        bins="log",
        extent=[3500, 6500, 0.5, 4.0],
    )
    plt.colorbar(h, ax=axes[1], label="log10 N")
    axes[1].invert_xaxis()
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"$T_{\rm eff}$ (K), APOGEE")
    axes[1].set_ylabel(r"$\log g$ (dex), APOGEE")
    axes[1].set_title(f"Stream-1 APOGEE Kiel ({int(m.sum()):,})")

    # α/M vs M/H bimodality
    mh = df["mh_apogee"].to_numpy()
    am = df["alpha_m_apogee"].to_numpy()
    m = np.isfinite(mh) & np.isfinite(am)
    h = axes[2].hexbin(
        mh[m],
        am[m],
        gridsize=70,
        mincnt=10,
        cmap="viridis",
        bins="log",
        extent=[-2.5, 0.6, -0.2, 0.5],
    )
    plt.colorbar(h, ax=axes[2], label="log10 N")
    axes[2].set_xlabel("[M/H] (dex), APOGEE")
    axes[2].set_ylabel(r"[$\alpha$/M] (dex), APOGEE")
    axes[2].set_title("Training-pool α-bimodality")

    fig.suptitle(
        "APOGEE DR19 training labels (Mészáros+2025 [X/M] corrections applied)", fontsize=11
    )
    save_fig(fig, OUT / "apogee_labels.png")


if __name__ == "__main__":
    main()
