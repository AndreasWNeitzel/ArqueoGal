"""F18: literature head-to-head, paper-quoted RMSE per label (slide 19)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.style import LABELS, OKABE_ITO, apply_style, save  # noqa: E402

# (key, latex, unit, APOGEE floor, dict {paper -> rmse}). NaN = not published.
# Guiglion+ 2024 (arXiv:2306.05688) numbers are paper-quoted from their
# CNN-on-XP holdout (giants subset).
TABLE = [
    ("teff",    "Teff",    "K",   80.0, {
        "JANUS":  42.4,
        "Andrae+ 2023":    91.0,
        "Zhang+ 2023":     110.0,
        "Khalatyan+ 2024": 105.0,
        "Guiglion+ 2024":  73.0,
    }),
    ("logg",    "logg",    "dex", 0.05, {
        "JANUS":  0.117,
        "Andrae+ 2023":    0.13,
        "Zhang+ 2023":     0.18,
        "Khalatyan+ 2024": 0.16,
        "Guiglion+ 2024":  0.15,
    }),
    ("mh",      "Mh",      "dex", 0.04, {
        "JANUS":  0.069,
        "Andrae+ 2023":    0.10,
        "Zhang+ 2023":     0.12,
        "Khalatyan+ 2024": 0.11,
        "Guiglion+ 2024":  0.10,
    }),
    ("alpha_m", "alpha_M", "dex", 0.03, {
        "JANUS":  0.039,
        "Andrae+ 2023":    np.nan,
        "Zhang+ 2023":     0.07,
        "Khalatyan+ 2024": 0.06,
        "Guiglion+ 2024":  0.05,
    }),
]

PAPERS = ["JANUS", "Andrae+ 2023", "Zhang+ 2023",
          "Khalatyan+ 2024", "Guiglion+ 2024"]
COLORS = {
    "JANUS":  OKABE_ITO["blue"],
    "Andrae+ 2023":    OKABE_ITO["green"],
    "Zhang+ 2023":     OKABE_ITO["red_purple"],
    "Khalatyan+ 2024": OKABE_ITO["orange"],
    "Guiglion+ 2024":  OKABE_ITO["sky"],
}


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 5.5),
                              layout="constrained")
    width = 0.6
    x_pos = np.arange(len(PAPERS))
    handles, labels_seen = [], []
    for ax_idx, (ax, (_key, lab, unit, floor, vals)) in enumerate(zip(axes, TABLE)):
        rmse = np.asarray([vals.get(p, np.nan) for p in PAPERS])
        for i, p in enumerate(PAPERS):
            v = rmse[i]
            if np.isfinite(v):
                bar = ax.bar(x_pos[i], v, width=width,
                              color=COLORS[p], edgecolor="#1A2B4C", lw=0.6)
                ax.text(x_pos[i], v * 1.03, f"{v:.3g}",
                         ha="center", va="bottom", fontsize=9)
                if p not in labels_seen:
                    handles.append(bar[0]); labels_seen.append(p)
            else:
                ax.bar(x_pos[i], 0.001, width=width,
                        color="white", edgecolor=COLORS[p], lw=1.0)
                ax.text(x_pos[i], 0.05 * np.nanmax(rmse), "n/a",
                         ha="center", va="bottom", fontsize=9,
                         color="#5C6378")
        ax.axhline(floor, color="#000000", lw=1.0, ls="--", alpha=0.7,
                    label="APOGEE floor" if ax_idx == 0 else None)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(PAPERS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"RMSE ({unit})")
        finite = rmse[np.isfinite(rmse)]
        if finite.size:
            ax.set_ylim(0, max(float(finite.max()) * 1.30, floor * 1.4))
        ax.set_title(LABELS[lab].split(" [")[0])
        ax.grid(True, axis="y", alpha=0.30)

    fig.legend(handles, labels_seen, loc="lower center",
                ncol=5, fontsize=10, frameon=False,
                bbox_to_anchor=(0.5, -0.04))

    save(fig, "F18_literature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
