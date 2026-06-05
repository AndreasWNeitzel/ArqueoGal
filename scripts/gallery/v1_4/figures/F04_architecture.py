"""F04: MLP architecture, vertically parallel heads, taller layout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.style import CHROME, OKABE_ITO, apply_style, save  # noqa: E402

# Trunk x-positions (axes coords, 0..1).
X_INPUT = 0.06
X_H1    = 0.20
X_H2    = 0.32
X_H3    = 0.44
X_HEAD  = 0.66

# Trunk vertical band.
Y_TRUNK_TOP    = 0.85
Y_TRUNK_BOTTOM = 0.55

# Head vertical positions (parallel, stacked). The heads frame is
# anchored to the same y-extent as the trunk frame so both look the same
# size; the neurons inside the heads are slightly more spread than in
# v1.4-rev, leaving room above and below for the head labels.
Y_HEAD_MU  = 0.79
Y_HEAD_LC  = 0.63
HEAD_BAND  = 0.05


def _draw_layer(ax, x, *, y_top, y_bottom, n_show, n_total, color,
                 label_top=None, sub_below=True):
    ys = np.linspace(y_top, y_bottom, n_show)
    for y in ys:
        ax.add_patch(plt.Circle((x, y), 0.012,
                                  facecolor="white", edgecolor=color,
                                  linewidth=1.0, zorder=4))
    if label_top:
        ax.text(x, y_top + 0.030, label_top,
                ha="center", va="bottom", fontsize=11,
                color=CHROME["body"])
    if sub_below:
        ax.text(x, y_bottom - 0.025, rf"({n_total} units)",
                 ha="center", va="top", fontsize=8.5,
                 color=CHROME["muted"], style="italic")
    return ys


def _connect(ax, x0, ys0, x1, ys1, color, alpha=0.18, lw=0.4):
    for y0 in ys0:
        for y1 in ys1:
            ax.plot([x0, x1], [y0, y1],
                     color=color, alpha=alpha, lw=lw, zorder=2)


def _frame(ax, x0, x1, y0, y1, *, edge, label, label_color, label_y_pad=0.014):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor="none", edgecolor=edge, linewidth=1.0,
        linestyle=(0, (3, 2)),
    ))
    ax.text(0.5 * (x0 + x1), y1 + label_y_pad, label,
             ha="center", va="bottom", fontsize=11,
             color=label_color)


def _loss_box(ax, x, y, w, h, *, edge, label, sub):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        facecolor=CHROME["surface_alt"], edgecolor=edge, linewidth=1.0,
    ))
    ax.text(x, y + 0.030, label,
             ha="center", va="center", fontsize=10.5, color=edge)
    ax.text(x, y - 0.030, sub,
             ha="center", va="center", fontsize=8.5,
             color=CHROME["muted"], style="italic")


def _arrow(ax, x0, y0, x1, y1, color, lw=0.9):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="->", color=color, lw=lw,
        shrinkA=2, shrinkB=2, mutation_scale=10, zorder=5,
    ))


def main() -> int:
    apply_style()
    # Taller figure to give the loss row more room and the trunk/head
    # neurons a less-cramped vertical stretch.
    fig, ax = plt.subplots(figsize=(11.0, 6.6), layout="constrained")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_aspect("auto")
    ax.axis("off")

    # Trunk frame.
    _frame(
        ax,
        x0=X_H1 - 0.05, x1=X_H3 + 0.05,
        y0=Y_TRUNK_BOTTOM - 0.07, y1=Y_TRUNK_TOP + 0.06,
        edge=OKABE_ITO["blue"],
        label=r"trunk (shared encoder)",
        label_color=OKABE_ITO["blue"],
    )

    # Heads frame, same y-extent as the trunk frame so they read as
    # equal-sized boxes.
    HEAD_X0 = X_HEAD - 0.07
    HEAD_X1 = X_HEAD + 0.07
    HEAD_Y0 = Y_TRUNK_BOTTOM - 0.07
    HEAD_Y1 = Y_TRUNK_TOP + 0.06
    _frame(
        ax,
        x0=HEAD_X0, x1=HEAD_X1,
        y0=HEAD_Y0, y1=HEAD_Y1,
        edge=OKABE_ITO["green"],
        label=r"heads",
        label_color=OKABE_ITO["green"],
    )

    # Trunk layers.
    ys_in = _draw_layer(
        ax, X_INPUT, y_top=Y_TRUNK_TOP, y_bottom=Y_TRUNK_BOTTOM,
        n_show=8, n_total=140,
        color=OKABE_ITO["blue"], label_top=r"input",
    )
    ys_h1 = _draw_layer(
        ax, X_H1, y_top=Y_TRUNK_TOP, y_bottom=Y_TRUNK_BOTTOM,
        n_show=10, n_total=256,
        color=OKABE_ITO["blue"], label_top=r"H$_1$",
    )
    ys_h2 = _draw_layer(
        ax, X_H2, y_top=Y_TRUNK_TOP, y_bottom=Y_TRUNK_BOTTOM,
        n_show=10, n_total=256,
        color=OKABE_ITO["blue"], label_top=r"H$_2$",
    )
    ys_h3 = _draw_layer(
        ax, X_H3, y_top=Y_TRUNK_TOP, y_bottom=Y_TRUNK_BOTTOM,
        n_show=10, n_total=256,
        color=OKABE_ITO["blue"], label_top=r"H$_3$",
    )

    # Heads, vertically parallel. Slightly more spread than v1.4-rev,
    # but leaving white space inside the box for the labels above the
    # mu head and below the L_chol head.
    ys_mu = _draw_layer(
        ax, X_HEAD,
        y_top=Y_HEAD_MU + HEAD_BAND, y_bottom=Y_HEAD_MU - HEAD_BAND,
        n_show=5, n_total=5,
        color=OKABE_ITO["green"],
        label_top=r"$\mu \in \mathbb{R}^{5}$",
        sub_below=False,
    )
    ys_lc = _draw_layer(
        ax, X_HEAD,
        y_top=Y_HEAD_LC + HEAD_BAND, y_bottom=Y_HEAD_LC - HEAD_BAND,
        n_show=5, n_total=15,
        color=OKABE_ITO["blue"],
        label_top=None,
        sub_below=False,
    )
    # L_chol label centred between the two heads' boundary so the heads
    # box keeps both ends of the label visible.
    ax.text(X_HEAD, Y_HEAD_LC - HEAD_BAND - 0.025,
             r"$L_\mathrm{chol} \in \mathbb{R}^{5 \times 5}$",
             ha="center", va="top", fontsize=11, color=CHROME["body"])

    # Connections.
    _connect(ax, X_INPUT, ys_in, X_H1, ys_h1, OKABE_ITO["blue"])
    _connect(ax, X_H1, ys_h1, X_H2, ys_h2, OKABE_ITO["blue"])
    _connect(ax, X_H2, ys_h2, X_H3, ys_h3, OKABE_ITO["blue"])
    _connect(ax, X_H3, ys_h3, X_HEAD, ys_mu, OKABE_ITO["green"])
    _connect(ax, X_H3, ys_h3, X_HEAD, ys_lc, OKABE_ITO["blue"])

    # GELU + LN annotation just under the trunk frame's lower edge.
    ax.text(0.5 * (X_H1 + X_H3), Y_TRUNK_BOTTOM - 0.10,
             r"GELU + LayerNorm at every hidden layer",
             ha="center", va="top", fontsize=9.5,
             color=CHROME["muted"], style="italic")

    # Loss row. SupCon + Barlow merged into a single trunk-loss box that
    # sits directly under the trunk's centre. beta-NLL stays under the
    # heads. Both arrows go straight up to the trunk (or heads) box,
    # never to a single neuron column.
    TRUNK_CENTER_X = 0.5 * (X_H1 + X_H3)
    LOSS_Y = 0.18
    LOSS_H = 0.16
    TRUNK_LOSS_W = 0.30
    NLL_W = 0.22

    _loss_box_two_lines = _loss_box   # (alias for clarity)
    # Custom box that fits two-line label content for the merged trunk loss.
    box_x = TRUNK_CENTER_X
    box_y = LOSS_Y
    rect = FancyBboxPatch(
        (box_x - TRUNK_LOSS_W / 2, box_y - LOSS_H / 2),
        TRUNK_LOSS_W, LOSS_H,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        facecolor=CHROME["surface_alt"],
        edgecolor=OKABE_ITO["blue"], linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(box_x, box_y + 0.046,
             r"trunk losses",
             ha="center", va="center", fontsize=11,
             color=OKABE_ITO["blue"])
    ax.text(box_x, box_y + 0.014,
             r"SupCon ($\lambda = 1.0$): label-aware metric",
             ha="center", va="center", fontsize=9.5,
             color=CHROME["body"])
    ax.text(box_x, box_y - 0.024,
             r"Barlow Twins ($\lambda = 0.5$): redundancy reduction",
             ha="center", va="center", fontsize=9.5,
             color=OKABE_ITO["vermillion"])
    ax.text(box_x, box_y - 0.058,
             r"both attach to the trunk output",
             ha="center", va="center", fontsize=9,
             color=CHROME["muted"], style="italic")
    _arrow(ax, TRUNK_CENTER_X, box_y + LOSS_H / 2,
              TRUNK_CENTER_X, Y_TRUNK_BOTTOM - 0.07,
              OKABE_ITO["blue"], lw=1.0)

    _loss_box(
        ax, X_HEAD, LOSS_Y + 0.03, NLL_W, LOSS_H - 0.06,
        edge=OKABE_ITO["green"],
        label=r"$\beta$-NLL ($\lambda = 1.0$, $\beta = 0.5$)",
        sub=r"heteroscedastic Gaussian NLL",
    )
    _arrow(ax, X_HEAD, LOSS_Y + 0.03 + (LOSS_H - 0.06) / 2,
              X_HEAD, HEAD_Y0 - 0.005,
              OKABE_ITO["green"], lw=1.0)

    save(fig, "F04_architecture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
