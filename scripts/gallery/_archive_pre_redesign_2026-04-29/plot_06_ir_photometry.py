"""Stage 06: IR photometry join (2MASS JHK + AllWISE W1, W2) across all streams.

Layout 2 × 3:
- Row 1: per-stream IR coverage sky map (S1 / S2 / S3).
- Row 2: IR-missing rate vs G (overlay), J-K vs G-K colour-colour (S2-vs-S3
  overlay), magnitude distributions (J / K / W1 / W2 medians per stream).
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

OUT = REPO / "reports/gallery/06_ir_photometry"

STREAMS = [
    ("Stream 1", REPO / "data/processed/pipeline1_features_stream1.parquet", PALETTE["apogee"]),
    ("Stream 2", REPO / "data/processed/pipeline1_features_stream2.parquet", "#9467bd"),
    (
        "Stream 3",
        REPO / "data/processed/pipeline1_features_stream3.parquet",
        PALETTE["andrae_volume"],
    ),
]


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    schema = pd.read_parquet(path).iloc[:0]
    base_cols = ["ra_deg", "dec_deg", "g_mag", "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
    cols = [c for c in base_cols if c in schema.columns]
    if "ir_missing_flag" in schema.columns:
        cols.append("ir_missing_flag")
    df = pd.read_parquet(path, columns=cols)
    if "ir_missing_flag" not in df.columns:
        # Stream 1 features parquet does not carry the explicit flag — derive
        # it from J/K coverage to match Stream 2/3 semantics.
        df["ir_missing_flag"] = ~(df["j_mag"].notna() & df["k_mag"].notna())
    return df


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30, height_ratios=[1, 1])
    rng = np.random.default_rng(0)

    loaded = []
    for i, (name, path, color) in enumerate(STREAMS):
        ax = fig.add_subplot(gs[0, i], projection="mollweide")
        df = _load(path)
        if df is None:
            ax.set_title(f"{name}\n(not built)", fontsize=9)
            continue
        loaded.append((name, color, df))
        idx = sample_index(len(df), 60_000, rng)
        sub = df.iloc[idx]
        has_ir = sub["k_mag"].notna() & sub["j_mag"].notna()
        x_ok, y_ok = radec_to_galactic_mollweide(
            sub.loc[has_ir, "ra_deg"].to_numpy(), sub.loc[has_ir, "dec_deg"].to_numpy()
        )
        x_no, y_no = radec_to_galactic_mollweide(
            sub.loc[~has_ir, "ra_deg"].to_numpy(), sub.loc[~has_ir, "dec_deg"].to_numpy()
        )
        ax.scatter(
            x_ok,
            y_ok,
            s=0.4,
            alpha=0.35,
            color="#2ca02c",
            rasterized=True,
            label=f"IR ({int(has_ir.sum()):,})",
        )
        if (~has_ir).sum():
            ax.scatter(
                x_no,
                y_no,
                s=0.5,
                alpha=0.6,
                color="#d62728",
                rasterized=True,
                label=f"missing ({int((~has_ir).sum()):,})",
            )
        style_galactic_mollweide(ax)
        full_pct = 100.0 * (df["k_mag"].notna() & df["j_mag"].notna()).sum() / len(df)
        ax.set_title(f"{name} IR coverage ({full_pct:.1f}%)", fontsize=9)
        ax.legend(
            fontsize=6,
            loc="lower left",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.4",
            markerscale=4,
        )

    # Row 2 col 0: IR-missing rate vs G overlay
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(6, 18, 25)
    centres = 0.5 * (bins[1:] + bins[:-1])
    for name, color, df in loaded:
        if "ir_missing_flag" not in df.columns or "g_mag" not in df.columns:
            continue
        g = df["g_mag"].to_numpy()
        flag = df["ir_missing_flag"].to_numpy().astype(bool)
        rate_n, _ = np.histogram(g[flag & np.isfinite(g)], bins=bins)
        denom, _ = np.histogram(g[np.isfinite(g)], bins=bins)
        rate_pct = 100.0 * rate_n / np.maximum(denom, 1)
        ax.plot(centres, rate_pct, "o-", color=color, lw=1.2, ms=4, label=f"{name}")
    ax.set_xlabel("G (mag)")
    ax.set_ylabel("% IR-missing")
    ax.set_title("IR-missing rate vs G (overlay)")
    ax.legend(
        fontsize=7,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # Row 2 col 1: J-K vs G-K colour-colour, S2 vs S3 overlay
    ax = fig.add_subplot(gs[1, 1])
    for name, color, df in loaded:
        if not {"j_mag", "k_mag", "g_mag"} <= set(df.columns):
            continue
        g = df["g_mag"].to_numpy()
        j = df["j_mag"].to_numpy()
        k = df["k_mag"].to_numpy()
        m = np.isfinite(g) & np.isfinite(j) & np.isfinite(k)
        if m.sum() < 100:
            continue
        idx = rng.choice(np.flatnonzero(m), min(20000, m.sum()), replace=False)
        ax.scatter(
            (g - k)[idx],
            (j - k)[idx],
            s=0.6,
            alpha=0.20,
            color=color,
            rasterized=True,
            label=f"{name} ({m.sum():,})",
        )
    ax.set_xlabel("G − K (mag)")
    ax.set_ylabel("J − K (mag)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.5)
    ax.set_title("Colour-colour overlay")
    ax.legend(
        fontsize=7,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
        markerscale=6,
    )

    # Row 2 col 2: median IR magnitudes per stream
    ax = fig.add_subplot(gs[1, 2])
    bands = ["j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
    band_labels = ["J", "H", "K", "W1", "W2"]
    x = np.arange(len(bands))
    n_streams = len(loaded)
    width = 0.85 / max(n_streams, 1)
    for i, (name, color, df) in enumerate(loaded):
        meds = []
        for b in bands:
            v = df[b].dropna().to_numpy() if b in df.columns else np.array([])
            meds.append(float(np.median(v)) if len(v) else np.nan)
        offset = (i - (n_streams - 1) / 2) * width
        ax.bar(x + offset, meds, width=width * 0.95, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("median magnitude")
    ax.set_title("Median IR magnitudes per stream")
    ax.legend(
        fontsize=7,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    fig.suptitle("Stage 06 — IR photometry (2MASS + AllWISE) across S1 / S2 / S3", fontsize=10)
    save_fig(fig, OUT / "ir_photometry.png")


if __name__ == "__main__":
    main()
