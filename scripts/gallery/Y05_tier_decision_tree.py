"""Y05: Tier 1 / 2 / 3 decision tree, with live fractions from Stream 1.

A talk-friendly visual of the per-star tier rule:

  ood_joint_flag fired (Mahalanobis OOD or per-element NaN)  →  Tier 3
  any per-element σ over its training threshold OR kin_ood   →  Tier 2
  otherwise                                                  →  Tier 1

Per-element thresholds (release.py):
  σ_Teff > 150 K, σ_logg > 0.30 dex, σ_M/H > 0.20 dex,
  σ_α/M > 0.05 dex, σ_Mg/H > 0.20 dex.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"


def _tier_fractions() -> dict[int, float]:
    p = pd.read_parquet(
        PRED_S1,
        columns=[
            "source_id",
            "teff_sigma",
            "logg_sigma",
            "mh_sigma",
            "alpha_m_sigma",
            "mg_h_sigma",
            "ood_joint_flag",
            "label_extrapolation_flag",
        ],
    ).drop_duplicates("source_id")
    f = pd.read_parquet(
        FEAT_S1, columns=["source_id", "fe_h_apogee", "teff_apogee", "b_deg"]
    ).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    split = stratified_split_ids(df, seed=0)
    ho = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    n = len(df)
    return {
        1: float((df["release_tier"] == 1).sum() / n),
        2: float((df["release_tier"] == 2).sum() / n),
        3: float((df["release_tier"] == 3).sum() / n),
        "n": n,
    }


def _diamond(ax, x, y, w, h, text, color):
    diamond = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        facecolor="white",
        edgecolor=color,
    )
    ax.add_patch(diamond)
    ax.text(
        x, y, text, ha="center", va="center", fontsize=12, color=PALETTE["ink"], fontweight="bold"
    )


def _tier_box(ax, x, y, w, h, name, fraction, n, color, body):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.4,
        facecolor=color,
        edgecolor="white",
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y + h * 0.30,
        name,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="white",
    )
    ax.text(x, y + h * 0.05, body, ha="center", va="center", fontsize=11, color="white", alpha=0.95)
    ax.text(
        x,
        y - h * 0.30,
        f"{fraction * 100:.1f}%  (n={n:,})",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
    )


def _arrow(ax, x0, y0, x1, y1, color, label=None, label_side="right"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=2.0, shrinkA=4, shrinkB=4, mutation_scale=18
        ),
    )
    if label is not None:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        offset = 0.18 if label_side == "right" else -0.18
        ax.text(
            mx + offset,
            my,
            label,
            ha="left" if offset > 0 else "right",
            va="center",
            fontsize=11,
            color=color,
            fontweight="bold",
        )


def main() -> int:
    apply_style()
    frac = _tier_fractions()
    n = frac["n"]
    n1 = int(round(frac[1] * n))
    n2 = int(round(frac[2] * n))
    n3 = int(round(frac[3] * n))

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top: input.
    _diamond(ax, 6.0, 6.4, 4.4, 0.7, "per-star prediction\n(μ, σ for each label)", PALETTE["navy"])

    # Decision 1: ood_joint_flag.
    _diamond(
        ax,
        6.0,
        5.0,
        4.6,
        0.85,
        "ood_joint_flag fired?\n(Mahalanobis on XP block, or any NaN)",
        PALETTE["tier3"],
    )
    # Decision 2: σ-inflation OR kin_ood, split into two lines for legibility.
    _diamond(
        ax,
        6.0,
        3.3,
        5.6,
        1.1,
        "any σ over training threshold?\n"
        r"$\sigma_{T_{\rm eff}}>150,\ \sigma_{\log g}>0.30$" + "\n"
        r"$\sigma_{[M/H]}>0.20,\ \sigma_{[\alpha/M]}>0.05,\ \sigma_{[Mg/H]}>0.20$"
        "\nOR kin_ood_flag",
        PALETTE["tier2"],
    )

    # Tier outcome boxes.
    _tier_box(
        ax,
        1.6,
        1.0,
        2.6,
        1.5,
        "TIER 3",
        frac[3],
        n3,
        PALETTE["tier3"],
        "do-not-trust\n(mask in science cuts)",
    )
    _tier_box(
        ax,
        6.0,
        1.0,
        2.6,
        1.5,
        "TIER 2",
        frac[2],
        n2,
        PALETTE["tier2"],
        "use with caution\n(σ-inflated or kin OOD)",
    )
    _tier_box(
        ax,
        10.4,
        1.0,
        2.6,
        1.5,
        "TIER 1",
        frac[1],
        n1,
        PALETTE["tier1"],
        "science-grade\n(default for analysis)",
    )

    # Arrows.
    _arrow(ax, 6.0, 6.05, 6.0, 5.45, PALETTE["navy"])
    _arrow(ax, 4.6, 4.7, 2.4, 1.85, PALETTE["tier3"], "yes", "left")
    _arrow(ax, 6.0, 4.55, 6.0, 3.85, PALETTE["navy"], "no")
    _arrow(ax, 4.4, 2.95, 5.0, 1.85, PALETTE["tier2"], "yes", "left")
    _arrow(ax, 7.6, 2.95, 9.5, 1.85, PALETTE["tier1"], "no")

    headline(
        fig,
        "Release tiering, three rules, evaluated in this order",
        f"Live fractions on Stream 1 held-out (val+test, seed=0), n = {n:,}.",
        top=0.88,
    )
    save(fig, "Y05_tier_decision_tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
