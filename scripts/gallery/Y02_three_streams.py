"""Y02: The three Gaia DR3 XP cohorts that Pipeline 1 consumes.

Three large cards (one per stream) describing the cohort selection,
the truth labels available (or not), and the cohort size after the
canonical merges. Numbers come from the live parquet files at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

PRED = {sid: REPO / f"data/processed/pipeline1_predictions_stream{sid}.parquet"
        for sid in (1, 2, 3)}


def _count(sid: int) -> int:
    df = pd.read_parquet(PRED[sid], columns=["source_id"])
    return int(df["source_id"].drop_duplicates().shape[0])


def _card(ax, *, name, role, body, color, n):
    rect = mpatches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.4, facecolor="white", edgecolor=color,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    # Stripe accent at the top.
    stripe = mpatches.Rectangle(
        (0.04, 0.85), 0.92, 0.11,
        linewidth=0, facecolor=color, transform=ax.transAxes, alpha=0.92,
    )
    ax.add_patch(stripe)
    ax.text(
        0.5, 0.905, name, ha="center", va="center",
        fontsize=22, fontweight="bold", color="white",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.78, role, ha="center", va="center",
        fontsize=14, fontstyle="italic", color=PALETTE["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.55, body, ha="center", va="center",
        fontsize=13, color=PALETTE["ink"], transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.20, f"n = {n:,}", ha="center", va="center",
        fontsize=26, fontweight="bold", color=color,
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.10, "Gaia DR3 source_ids after dedup",
        ha="center", va="center", fontsize=10, color=PALETTE["ash"],
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.10)

    n1, n2, n3 = (_count(s) for s in (1, 2, 3))

    _card(
        axes[0],
        name="STREAM 1",
        role="Training cohort",
        body="APOGEE DR19 × Gaia DR3 XP\n"
             "Truth labels: Teff, log g, [M/H],\n"
             "[α/M], [Mg/H]\n\n"
             "Kiel mask:\nTeff ∈ [4000, 5500] K\nlog g ∈ [1.0, 3.5]",
        color=PALETTE["navy"], n=n1,
    )
    _card(
        axes[1],
        name="STREAM 2",
        role="Asteroseismic cross-check",
        body="TESS giants × Gaia DR3 XP\n"
             "νmax / Δν → seismic log g\n\n"
             "Used to test the model on\n"
             "an independent gravity scale,\n"
             "not for training.",
        color=PALETTE["accent"], n=n2,
    )
    _card(
        axes[2],
        name="STREAM 3",
        role="Inference at scale",
        body="Andrae+23 RGB stratified\n"
             "Gaia DR3 XP (no truth labels)\n\n"
             "The downstream Starfold\n"
             "consumer; predictions are the\n"
             "deliverable for D-Cat-d.",
        color=PALETTE["tier1"], n=n3,
    )

    headline(
        fig,
        "Three streams, one model",
        "The same XP-only MLP is trained on Stream 1 and applied unchanged to Streams 2 and 3.",
        top=0.82,
    )
    save(fig, "Y02_three_streams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
