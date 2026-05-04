"""D3: Stream 1 APOGEE training-label distributions.

3x7 grid of histograms, one per element (21 labels total). Displays:
- Distribution of training labels from APOGEE DR19.
- Median and per-element sigma_train annotated (defines sigma-inflation
  thresholds in release.py).

Stream: Stream 1 (APOGEE DR19 training pool).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import load_real_stream, save_fig

from arqueogal.utils.plotting import set_aa_style


def main() -> None:
    set_aa_style()

    data = load_real_stream(1, columns=[
        "source_id",
        "teff_apogee", "logg_apogee", "mh_apogee",
        "c_h_apogee", "n_h_apogee", "o_h_apogee", "na_h_apogee",
        "mg_h_apogee", "al_h_apogee", "si_h_apogee", "s_h_apogee",
        "k_h_apogee", "ca_h_apogee", "ti_h_apogee", "v_h_apogee",
        "cr_h_apogee", "mn_h_apogee", "fe_h_apogee",
        "ni_h_apogee", "alpha_m_apogee",
    ])

    label_names = [
        "Teff", "log g", "[M/H]", "[C/H]", "[N/H]", "[O/H]", "[Na/H]",
        "[Mg/H]", "[Al/H]", "[Si/H]", "[S/H]", "[K/H]", "[Ca/H]", "[Ti/H]",
        "[V/H]", "[Cr/H]", "[Mn/H]", "[Fe/H]", "[Ni/H]", "[alpha/M]",
    ]

    col_names = [
        "teff_apogee", "logg_apogee", "mh_apogee", "c_h_apogee", "n_h_apogee",
        "o_h_apogee", "na_h_apogee", "mg_h_apogee", "al_h_apogee", "si_h_apogee",
        "s_h_apogee", "k_h_apogee", "ca_h_apogee", "ti_h_apogee", "v_h_apogee",
        "cr_h_apogee", "mn_h_apogee", "fe_h_apogee", "ni_h_apogee",
        "alpha_m_apogee",
    ]

    labels = {name: data[col].dropna().values for name, col in zip(label_names, col_names)}
    sigma_train = {name: data[col].std() for name, col in zip(label_names, col_names)}

    label_names = list(labels.keys())
    # 20 elements -> 4x5 grid fills exactly; avoids an empty last subplot.
    n_rows, n_cols = 4, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 9))

    for _idx, (ax, name) in enumerate(zip(axes.flatten(), label_names)):
        data = labels[name]
        sigma = sigma_train[name]
        median = np.median(data)

        # Histogram
        ax.hist(data, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(median, color="#d62728", lw=1.2, ls="--")

        ax.set_title(name, fontsize=8)
        ax.set_xlabel("value", fontsize=7)
        ax.set_ylabel("N", fontsize=7)
        ax.tick_params(labelsize=6)

        # Annotate sigma_train
        ax.text(
            0.98,
            0.97,
            f"sigma={sigma:.3f}\nmed={median:.2f}",
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.88, pad=1),
        )

    fig.suptitle(
        "Stream 1 (APOGEE DR19): training-label distributions "
        "(defines sigma_train per element for release tier annotation)",
        fontsize=10,
    )

    out = REPO / "reports/gallery/D_predictions"
    save_fig(fig, out / "D3_apogee_labels", formats=("pdf", "png"))


if __name__ == "__main__":
    main()
