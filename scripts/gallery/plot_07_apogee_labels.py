"""Stage 07: APOGEE DR19 labels (training targets) — Mészáros+2025 corrected.

Outputs:
  - reports/gallery/07_apogee_labels/label_pairwise_hexbin.png
  - reports/gallery/07_apogee_labels/label_nan_rates.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_PROCESSED,
    GALLERY,
    apply_style,
    save_fig,
)

OUT = GALLERY / "07_apogee_labels"

ELEMENTS = [
    # (column, display, group)
    ("teff_apogee", "T_eff", "atmospheric"),
    ("logg_apogee", "log g", "atmospheric"),
    ("mh_apogee", "[M/H]", "atmospheric"),
    ("alpha_m_apogee", "[alpha/M]", "alpha"),
    ("fe_h_apogee", "[Fe/H]", "Fe-peak"),
    ("mg_h_apogee", "[Mg/H]", "alpha"),
    ("si_h_apogee", "[Si/H]", "alpha"),
    ("ca_h_apogee", "[Ca/H]", "alpha"),
    ("ti_h_apogee", "[Ti/H]", "alpha"),
    ("mn_h_apogee", "[Mn/H]", "Fe-peak"),
    ("ni_h_apogee", "[Ni/H]", "Fe-peak"),
    ("cr_h_apogee", "[Cr/H]", "Fe-peak"),
    ("c_h_apogee", "[C/H]", "light"),
    ("n_h_apogee", "[N/H]", "light"),
    ("o_h_apogee", "[O/H]", "light"),
    ("na_h_apogee", "[Na/H]", "light"),
    ("al_h_apogee", "[Al/H]", "light"),
    ("k_h_apogee", "[K/H]", "light"),
    ("s_h_apogee", "[S/H]", "alpha"),
    ("v_h_apogee", "[V/H]", "Fe-peak"),
    ("ce_h_apogee", "[Ce/H]", "s-process"),
]

GROUP_COLOR = {
    "atmospheric": "#1f77b4",
    "alpha": "#2ca02c",
    "Fe-peak": "#d62728",
    "light": "#9467bd",
    "s-process": "#ff7f0e",
}


def _have(schema, name: str) -> bool:
    return name in {f.name for f in schema}


def label_pairwise_hexbin() -> None:
    path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    cols_needed = ["mh_apogee", "alpha_m_apogee", "fe_h_apogee", "mg_h_apogee"]
    schema = pq.read_schema(path)
    have = [c for c in cols_needed if _have(schema, c)]
    df = pq.read_table(path, columns=have).to_pandas()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        (
            "mh_apogee",
            "alpha_m_apogee",
            r"$[{\rm M}/{\rm H}]$",
            r"$[\alpha/{\rm M}]$",
            (-2.2, 0.6),
            (-0.25, 0.55),
        ),
        (
            "mh_apogee",
            "fe_h_apogee",
            r"$[{\rm M}/{\rm H}]$",
            r"$[{\rm Fe}/{\rm H}]$",
            (-2.2, 0.6),
            (-2.2, 0.6),
        ),
        (
            "alpha_m_apogee",
            "mg_h_apogee",
            r"$[\alpha/{\rm M}]$",
            r"$[{\rm Mg}/{\rm H}]$",
            (-0.25, 0.55),
            (-1.8, 0.6),
        ),
    ]

    for ax, (x, y, xl, yl, xlim, ylim) in zip(axes.flat, pairs):
        if x not in df.columns or y not in df.columns:
            ax.set_visible(False)
            continue
        xv = df[x].to_numpy(dtype=float)
        yv = df[y].to_numpy(dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv)
        hb = ax.hexbin(
            xv[m],
            yv[m],
            gridsize=80,
            cmap="magma",
            bins="log",
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            mincnt=1,
        )
        plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="log N")
        # identity for mh vs feh
        if x == "mh_apogee" and y == "fe_h_apogee":
            ax.plot(xlim, xlim, "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}   n={int(m.sum()):,}")

    fig.suptitle(
        "APOGEE DR19 label pairwise hexbin  —  Mészáros+2025-corrected, Stream 1",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, OUT / "label_pairwise_hexbin.png")


def label_nan_rates() -> None:
    path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    cols = [c for (c, _, _) in ELEMENTS]
    schema = pq.read_schema(path)
    have = [c for c in cols if _have(schema, c)]
    df = pq.read_table(path, columns=have).to_pandas()
    n = len(df)

    rows = []
    for col, name, group in ELEMENTS:
        if col not in df.columns:
            continue
        nan_frac = 100.0 * df[col].isna().mean()
        rows.append((col, name, group, nan_frac))

    rows.sort(key=lambda r: r[3])  # ascending
    names = [r[1] for r in rows]
    fracs = np.array([r[3] for r in rows])
    colors = [GROUP_COLOR[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(rows))
    bars = ax.barh(y, fracs, color=colors, edgecolor="#333", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("NaN rate  [%]")
    ax.set_title(
        rf"APOGEE DR19 per-label NaN rate  —  n={n:,} Stream 1 stars  "
        r"(drives `mask=` path in beta_nll_block_cholesky)",
        fontsize=11,
        fontweight="semibold",
    )
    # annotations
    for b, pct in zip(bars, fracs):
        ax.text(
            b.get_width() + 0.15,
            b.get_y() + b.get_height() / 2,
            f"{pct:.2f}%",
            va="center",
            fontsize=8,
            color="#333",
        )
    ax.set_xlim(0, max(25, 1.1 * fracs.max() if len(fracs) else 10))
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # legend for groups
    from matplotlib.patches import Patch

    legend = [Patch(facecolor=c, edgecolor="#333", label=g) for g, c in GROUP_COLOR.items()]
    ax.legend(handles=legend, loc="lower right", title="group")

    save_fig(fig, OUT / "label_nan_rates.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    label_pairwise_hexbin()
    label_nan_rates()


if __name__ == "__main__":
    main()
