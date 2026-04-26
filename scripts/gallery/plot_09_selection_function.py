"""Stage 09: selection function ω(s) across all three streams.

Layout 2 × 3:
- Row 1: ω total distribution overlay (S1/S2/S3); ω vs G overlay; total
  components per stream as a stacked summary.
- Row 2: per-stream ω sky map (S1 / S2 / S3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import (
    PALETTE,
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = REPO / "reports/gallery/09_selection_function"

STREAMS = [
    ("Stream 1", REPO / "data/processed/pipeline1_features_stream1.parquet", PALETTE["apogee"]),
    ("Stream 2", REPO / "data/processed/pipeline1_features_stream2.parquet", "#9467bd"),
    (
        "Stream 3",
        REPO / "data/processed/pipeline1_features_stream3.parquet",
        PALETTE["andrae_volume"],
    ),
]


def _load(path: Path) -> tuple[pd.DataFrame, str | None] | None:
    if not path.exists():
        return None
    full_cols = pd.read_parquet(path).iloc[:0].columns
    omega_col = "selection_prob" if "selection_prob" in full_cols else None
    cols = [c for c in ["ra_deg", "dec_deg", "g_mag"] if c in full_cols]
    if omega_col:
        cols.append(omega_col)
    df = pd.read_parquet(path, columns=cols)
    return df, omega_col


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)

    rng = np.random.default_rng(0)

    loaded = []
    for name, path, color in STREAMS:
        d = _load(path)
        if d is None:
            loaded.append((name, color, None, None))
        else:
            loaded.append((name, color, d[0], d[1]))

    # Row 1 col 0: ω distribution overlay
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 51)
    for name, color, df, omega_col in loaded:
        if df is None or omega_col is None:
            continue
        v = df[omega_col].dropna().to_numpy()
        ax.hist(
            v,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            lw=1.4,
            label=f"{name} (n={len(v):,}, mean={v.mean():.3f})",
        )
    ax.set_xlabel("selection_prob ω(s)")
    ax.set_ylabel("density")
    ax.set_title("ω(s) distribution overlay")
    ax.legend(
        fontsize=7,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # Row 1 col 1: ω vs G hexbin (each stream)
    ax = fig.add_subplot(gs[0, 1])
    for name, color, df, omega_col in loaded:
        if df is None or omega_col is None:
            continue
        g = df["g_mag"].to_numpy()
        o = df[omega_col].to_numpy()
        m = np.isfinite(g) & np.isfinite(o)
        if m.sum() < 100:
            continue
        # Median curve only — overlay would be cluttered
        bins_g = np.linspace(6, 18, 25)
        centres = 0.5 * (bins_g[1:] + bins_g[:-1])
        bin_idx = np.digitize(g[m], bins_g) - 1
        meds = np.full(len(centres), np.nan)
        for i in range(len(centres)):
            sel = bin_idx == i
            if sel.sum() > 50:
                meds[i] = np.median(o[m][sel])
        ax.plot(
            centres, meds, "o-", color=color, lw=1.2, ms=4, label=f"{name} (n={int(m.sum()):,})"
        )
    ax.set_xlabel("G (mag)")
    ax.set_ylabel("median ω(s)")
    ax.set_title("ω(s) vs G — median per bin")
    ax.legend(
        fontsize=7,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # Row 1 col 2: per-stream summary stats bar
    ax = fig.add_subplot(gs[0, 2])
    stats = ["min", "p10", "median", "p90", "mean"]
    n_streams = sum(1 for _, _, df, oc in loaded if df is not None and oc is not None)
    width = 0.85 / max(n_streams, 1)
    x = np.arange(len(stats))
    for i, (name, color, df, oc) in enumerate(loaded):
        if df is None or oc is None:
            continue
        v = df[oc].dropna().to_numpy()
        bars = [
            v.min(),
            float(np.percentile(v, 10)),
            float(np.median(v)),
            float(np.percentile(v, 90)),
            float(v.mean()),
        ]
        offset = (i - (n_streams - 1) / 2) * width
        ax.bar(x + offset, bars, width=width * 0.95, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ω(s)")
    ax.set_title("Per-stream ω(s) summary")
    ax.legend(
        fontsize=7,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # Row 2: per-stream sky map of ω
    for i, (name, color, df, oc) in enumerate(loaded):
        ax = fig.add_subplot(gs[1, i], projection="mollweide")
        if df is None:
            ax.set_title(f"{name}\n(not built)", fontsize=9)
            continue
        idx = sample_index(len(df), 60_000, rng)
        x, y = radec_to_galactic_mollweide(
            df.ra_deg.iloc[idx].to_numpy(), df.dec_deg.iloc[idx].to_numpy()
        )
        if oc:
            c = df[oc].iloc[idx].to_numpy()
            sc = ax.scatter(
                x, y, s=0.5, c=c, cmap="viridis", vmin=0, vmax=1, alpha=0.5, rasterized=True
            )
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.05, label="ω(s)")
        else:
            ax.scatter(x, y, s=0.4, alpha=0.35, color=color, rasterized=True)
        style_galactic_mollweide(ax)
        ax.set_title(f"{name} ω(s) sky map", fontsize=9)

    fig.suptitle("Stage 09 — selection function ω(s) across S1 / S2 / S3", fontsize=10)
    save_fig(fig, OUT / "selection_function.png")


if __name__ == "__main__":
    main()
