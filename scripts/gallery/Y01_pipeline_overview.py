"""Y01: Pipeline overview schematic — boxes, arrows, no plotted data.

A talk-slide opener that names the four stages of Pipeline 1 in order:
data sources → preprocessing → ML model → release tiers. Designed for the
title slide of a methods talk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402


def _box(
    ax,
    x,
    y,
    w,
    h,
    *,
    title,
    body,
    fc,
    tc=PALETTE["ink"],
    edge=PALETTE["ash"],
):
    """Rounded rectangle with a bold title and a body line."""
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        facecolor=fc,
        edgecolor=edge,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=tc,
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=11,
        color=tc,
        alpha=0.9,
    )


def _arrow(ax, x0, y0, x1, y1, color=PALETTE["ash"]):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.0,
            shrinkA=4,
            shrinkB=4,
            mutation_scale=18,
        ),
    )


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # 4 boxes laid out left-to-right at y=2.0, h=1.4, w=2.0, gap=0.4.
    bw, bh, by = 2.0, 1.6, 1.7
    xs = [0.20, 2.60, 5.00, 7.40]

    _box(
        ax,
        xs[0],
        by,
        bw,
        bh,
        title="DATA",
        body="Gaia DR3 XP\n+ APOGEE truth\n+ kinematics",
        fc=PALETTE["paper"],
    )
    _box(
        ax,
        xs[1],
        by,
        bw,
        bh,
        title="PREPROCESS",
        body="Ye+2024 NN\nHermite reproj\nfrozen z-score",
        fc="#dfe7f1",
    )
    _box(
        ax,
        xs[2],
        by,
        bw,
        bh,
        title="MODEL",
        body=r"MLP, $\beta$-NLL" + "\n140-D → 5 labels\n+ per-star σ",
        fc="#e8d5b4",
        tc=PALETTE["ink"],
    )
    _box(
        ax,
        xs[3],
        by,
        bw,
        bh,
        title="RELEASE",
        body="Tier 1 / 2 / 3\n(Mahalanobis + σ\n+ kin_ood gates)",
        fc="#cce3d4",
    )

    # Arrows between boxes.
    for i in range(3):
        _arrow(ax, xs[i] + bw, by + bh / 2, xs[i + 1], by + bh / 2, color=PALETTE["navy"])

    # Stream-fan-out below the model box. Centred horizontally on x=5.0 (the
    # midpoint of the figure) so the row never overflows the right edge.
    sw, sh = 2.2, 1.0
    sgap = 0.30
    n_streams = 3
    total_w = n_streams * sw + (n_streams - 1) * sgap
    sx0 = (10.0 - total_w) / 2.0  # centre on figure midline
    sxs = [sx0 + i * (sw + sgap) for i in range(n_streams)]
    sb_titles = ["Stream 1", "Stream 2", "Stream 3"]
    sb_bodies = ["APOGEE × XP\nn ≈ 293k", "TESS × XP\nn ≈ 72k", "Andrae+23 RGB\nn ≈ 614k"]
    for sx, t, b in zip(sxs, sb_titles, sb_bodies):
        _box(
            ax,
            sx,
            0.10,
            sw,
            sh,
            title=t,
            body=b,
            fc="#fff4e0",
            edge=PALETTE["accent"],
        )
    # Three arrows from the bottom of MODEL box down to each stream-card top.
    src_x = xs[2] + bw / 2
    for sx in sxs:
        target_x = sx + sw / 2
        _arrow(ax, src_x, by, target_x, 0.10 + sh, color=PALETTE["accent"])

    headline(
        fig,
        "ArqueoGal Pipeline 1",
        "From Gaia DR3 XP coefficients to per-star stellar labels with calibrated uncertainty.",
        top=0.88,
    )
    save(fig, "Y01_pipeline_overview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
