"""Stage 02: raw Gaia XP coefficients across all three streams.

What the deploy did: ``data.ingest_xp`` / ``scripts/fetch_gaia_xp.py`` (and the
S2/S3 delta variants) fetched ``xp_continuous_mean_spectrum`` rows from gaiadr3
via TAP, returning 55 BP + 55 RP coefficients per star (raw, pre-Ye,
pre-normalisation). Stage A and B then produced normalised c_i/c_0 ratios and
a c0 scalar, in a schema that differs by stream:

- Stream 1 (training): ``bp_c0_z`` / ``rp_c0_z`` (frozen-z-scored), no raw c0.
- Stream 2 / 3 (inference): ``bp_c0_log`` / ``rp_c0_log`` (raw log10(c_0)).

What we plot, 2 × 2:
- (0,0) Per-coefficient IQR overlay across S1 / S2 / S3 — confirms shape
  consistency. This is the input to the trunk.
- (0,1) c0 vs G hexbin per stream (3-panel inset) — confirms absolute flux
  scale vs magnitude.
- (1,0) Example-star BP-coef SEDs across BP-RP colour (Stream 1).
- (1,1) Stream-by-stream summary of n / median c0 / IQR width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import PALETTE, apply_style, save_fig

OUT = REPO / "reports/gallery/02_gaia_xp_raw"

S1 = REPO / "data/processed/pipeline1_features_stream1.parquet"
S2 = REPO / "data/processed/pipeline1_features_stream2.parquet"
S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"

STREAMS = [
    ("Stream 1 (APOGEE × Gaia, training)", S1, PALETTE["apogee"]),
    ("Stream 2 (Hon+21 TESS asteroseismic)", S2, "#9467bd"),
    ("Stream 3 (Andrae+23 inference)", S3, PALETTE["andrae_volume"]),
]


def _load_norm_and_c0(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (bp_norm[N,54], rp_norm[N,54], c0_scalar[N], g_mag[N], bp_rp[N])."""
    if not path.exists():
        return None
    schema = pd.read_parquet(path).iloc[:0]
    have_norm = all(f"bp_coef_norm_{i}" in schema.columns for i in range(1, 55))
    if not have_norm:
        return None
    c0_col = (
        "bp_c0_z"
        if "bp_c0_z" in schema.columns
        else ("bp_c0_log" if "bp_c0_log" in schema.columns else None)
    )
    cols = (
        ["g_mag", "bp_rp"]
        + [f"bp_coef_norm_{i}" for i in range(1, 55)]
        + [f"rp_coef_norm_{i}" for i in range(1, 55)]
    )
    if c0_col:
        cols.append(c0_col)
    df = pd.read_parquet(path, columns=cols)
    bp = np.column_stack([df[f"bp_coef_norm_{i}"].to_numpy() for i in range(1, 55)])
    rp = np.column_stack([df[f"rp_coef_norm_{i}"].to_numpy() for i in range(1, 55)])
    c0 = df[c0_col].to_numpy() if c0_col else np.full(len(df), np.nan)
    return bp, rp, c0, df["g_mag"].to_numpy(), df["bp_rp"].to_numpy()


def main() -> None:
    apply_style()

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30, width_ratios=[1, 1, 1])

    # Load the three streams
    streams_data = []
    for name, path, color in STREAMS:
        d = _load_norm_and_c0(path)
        streams_data.append((name, color, d))

    # (0,0) Per-coefficient IQR overlay (all 3 streams, BP only for clarity)
    ax = fig.add_subplot(gs[0, 0])
    idx = np.arange(1, 55)
    for name, color, d in streams_data:
        if d is None:
            continue
        bp, _, _, _, _ = d
        q = np.nanpercentile(bp, [25, 50, 75], axis=0)
        ax.fill_between(idx, q[0], q[2], color=color, alpha=0.20)
        ax.plot(idx, q[1], color=color, lw=1.0, label=f"{name.split(' (')[0]} (n={len(bp):,})")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("BP coef index (1..54)")
    ax.set_ylabel("normalised coef value (median + IQR shade)")
    ax.set_title("Per-coefficient IQR — BP, all streams overlaid")
    ax.legend(
        fontsize=7,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # (0,1) RP IQR
    ax = fig.add_subplot(gs[0, 1])
    for name, color, d in streams_data:
        if d is None:
            continue
        _, rp, _, _, _ = d
        q = np.nanpercentile(rp, [25, 50, 75], axis=0)
        ax.fill_between(idx, q[0], q[2], color=color, alpha=0.20)
        ax.plot(idx, q[1], color=color, lw=1.0, label=f"{name.split(' (')[0]}")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("RP coef index (1..54)")
    ax.set_ylabel("normalised coef value")
    ax.set_title("Per-coefficient IQR — RP")
    ax.legend(
        fontsize=7,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # (0,2) c0 vs G — per-stream histogram (1D) since hexbin × 3 takes too much room
    ax = fig.add_subplot(gs[0, 2])
    for name, color, d in streams_data:
        if d is None:
            continue
        _, _, c0, g, _ = d
        m = np.isfinite(c0) & np.isfinite(g)
        if m.sum() < 100:
            continue
        ax.hist(
            c0[m],
            bins=80,
            density=True,
            histtype="step",
            color=color,
            lw=1.2,
            label=f"{name.split(' (')[0]} (n={int(m.sum()):,})",
        )
    ax.set_xlabel("c0 (z-scored or log10)")
    ax.set_ylabel("density")
    ax.set_title("c0 (absolute flux scale) per stream")
    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # (1,0..2) c0 vs G hexbin per stream
    for i, (name, _color, d) in enumerate(streams_data):
        ax = fig.add_subplot(gs[1, i])
        if d is None:
            ax.text(
                0.5,
                0.5,
                "(stream features not built yet)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_axis_off()
            continue
        _, _, c0, g, _ = d
        m = np.isfinite(g) & np.isfinite(c0)
        h = ax.hexbin(g[m], c0[m], gridsize=60, mincnt=5, cmap="viridis", bins="log")
        plt.colorbar(h, ax=ax, label="log10 N")
        ax.set_xlabel("G (mag)")
        ax.set_ylabel("BP c0 (z or log10)")
        ax.set_title(f"{name.split(' (')[0]}: c0 vs G (n={int(m.sum()):,})", fontsize=9)

    fig.suptitle(
        "Stage 02 — raw Gaia XP coefficient distributions across S1 / S2 / S3.\n"
        "S1 stores z-scored c0 (frozen-v1 stats); S2/S3 store raw log10(c0) — "
        "inference applies the frozen z-score at runtime.",
        fontsize=10,
    )
    save_fig(fig, OUT / "xp_coef_distributions.png")


if __name__ == "__main__":
    main()
