"""Stage 19: hybrid composer source selection across S2 + S3.

Per-element regressor / kNN / regressor_caveat split, plus per-element
hybrid_tier counts, both shown side-by-side for the two inference cohorts.

Layout 2 × 2:
- (0,0) S3 source split per element
- (0,1) S3 hybrid_tier counts per element
- (1,0) S2 source split per element
- (1,1) S2 hybrid_tier counts per element
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

OUT = REPO / "reports/gallery/19_hybrid_composer"
S3_PATH = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"
S2_PATH = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
EL_LBL = {"teff": "Teff", "logg": "log g", "mh": "[M/H]",
          "alpha_m": "[α/M]", "mg_h": "[Mg/H]"}
SRC_COLORS = {"regressor": "#1f77b4", "knn": "#ff7f0e",
              "regressor_caveat": "#d62728"}


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cols = ([f"{e}_hybrid_source" for e in ELEMENTS]
            + [f"{e}_hybrid_tier" for e in ELEMENTS])
    return pd.read_parquet(path, columns=cols)


def _source_panel(ax, df, name):
    n = len(df)
    bottoms = np.zeros(len(ELEMENTS))
    for src in ("regressor", "knn", "regressor_caveat"):
        fracs = np.array([(df[f"{e}_hybrid_source"] == src).sum() / n * 100
                            for e in ELEMENTS])
        ax.bar(range(len(ELEMENTS)), fracs, bottom=bottoms,
                color=SRC_COLORS[src], label=src, edgecolor="white", lw=0.4)
        for i, f in enumerate(fracs):
            if f > 5:
                ax.text(i, bottoms[i] + f / 2, f"{f:.1f}%",
                         ha="center", va="center", fontsize=7, color="white")
        bottoms += fracs
    ax.set_xticks(range(len(ELEMENTS)))
    ax.set_xticklabels([EL_LBL[e] for e in ELEMENTS])
    ax.set_ylabel("% of stars")
    ax.set_title(f"{name} hybrid source per element (n={n:,})")
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")


def _tier_panel(ax, df, name):
    width = 0.35
    x = np.arange(len(ELEMENTS))
    t1 = [int((df[f"{e}_hybrid_tier"] == 1).sum()) for e in ELEMENTS]
    t2 = [int((df[f"{e}_hybrid_tier"] == 2).sum()) for e in ELEMENTS]
    ax.bar(x - width/2, t1, width, color="#2ca02c", label="tier 1")
    ax.bar(x + width/2, t2, width, color="#ff7f0e", label="tier 2")
    ax.set_xticks(x); ax.set_xticklabels([EL_LBL[e] for e in ELEMENTS])
    ax.set_ylabel("count")
    ax.set_title(f"{name} per-element hybrid_tier")
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")


def main() -> None:
    apply_style()
    s3 = _load(S3_PATH); s2 = _load(S2_PATH)
    if s3 is None and s2 is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    if s3 is not None:
        _source_panel(axes[0, 0], s3, "Stream 3")
        _tier_panel(axes[0, 1], s3, "Stream 3")
    else:
        for ax in axes[0]:
            ax.set_axis_off()
    if s2 is not None:
        _source_panel(axes[1, 0], s2, "Stream 2")
        _tier_panel(axes[1, 1], s2, "Stream 2")
    else:
        for ax in axes[1]:
            ax.set_axis_off()

    fig.suptitle(
        "Stage 19 — hybrid composer source selection (regressor / kNN / regressor_caveat) "
        "for both inference streams.",
        fontsize=11,
    )
    save_fig(fig, OUT / "hybrid_composer.png")


if __name__ == "__main__":
    main()
