"""Stage 05: IR photometry coverage (2MASS J/H/K + WISE W1/W2).

Outputs:
  - reports/gallery/05_ir_photometry/ir_coverage_sky.png
  - reports/gallery/05_ir_photometry/ir_color_color.png
  - reports/gallery/05_ir_photometry/ir_missing_vs_g.png
  - reports/gallery/05_ir_photometry/ir_magnitude_distributions.png
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
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = GALLERY / "05_ir_photometry"


def _load(stream: str = "stream3") -> "pd.DataFrame":
    import pandas as pd  # noqa: F401  (pandas return type)
    cols = ["source_id", "ra_deg", "dec_deg", "g_mag", "bp_rp",
            "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
            "ir_missing_flag"]
    path = (DATA_PROCESSED / f"pipeline1_features_{stream}.parquet")
    schema = pq.read_schema(path)
    have = [c for c in cols if c in set(f.name for f in schema)]
    return pq.read_table(path, columns=have).to_pandas()


def ir_coverage_sky() -> None:
    df = _load("stream3")
    rng = np.random.default_rng(17)
    idx = sample_index(len(df), 80_000, rng)
    sub = df.iloc[idx]
    have_all = np.isfinite(sub[["j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]].to_numpy()).all(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "mollweide"})
    ra, dec = sub["ra_deg"].to_numpy(), sub["dec_deg"].to_numpy()

    for ax, mask, title, color in (
        (axes[0], have_all, f"IR-complete (JHK+W1W2)  n={int(have_all.sum()):,}", "#2ca02c"),
        (axes[1], ~have_all, f"IR-missing (any band)  n={int((~have_all).sum()):,}", "#d62728"),
    ):
        x, y = radec_to_galactic_mollweide(ra[mask], dec[mask])
        ax.scatter(x, y, s=0.4, alpha=0.35, color=color, rasterized=True)
        ax.set_title(title)
        style_galactic_mollweide(ax)

    fig.suptitle(f"Stream 3 IR coverage across the sky  —  Galactic coords  "
                 f"(n={len(sub):,} plotted of {len(df):,})",
                 fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, OUT / "ir_coverage_sky.png")


def ir_color_color() -> None:
    df = _load("stream3").dropna(subset=["j_mag", "k_mag", "w1_mag", "w2_mag"])
    jk = df["j_mag"] - df["k_mag"]
    w12 = df["w1_mag"] - df["w2_mag"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(jk, w12, gridsize=80, cmap="viridis", bins="log",
              extent=(-0.2, 1.8, -0.5, 0.7))
    ax.set_xlabel("J - K  [mag]")
    ax.set_ylabel("W1 - W2  [mag]")
    ax.set_title(f"Stream 3 IR colour-colour (n={len(df):,})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0.6, color="k", lw=0.5, ls="--")
    save_fig(fig, OUT / "ir_color_color.png")


def ir_missing_vs_g() -> None:
    df = _load("stream3")
    g_bins = np.linspace(df["g_mag"].min(), df["g_mag"].max(), 36)
    g_centers = 0.5 * (g_bins[:-1] + g_bins[1:])
    totals, _ = np.histogram(df["g_mag"], bins=g_bins)
    missing, _ = np.histogram(df.loc[df["ir_missing_flag"] == 1, "g_mag"], bins=g_bins)
    rate = np.where(totals > 0, missing / np.maximum(totals, 1), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(g_centers, totals, width=(g_bins[1] - g_bins[0]) * 0.9,
                color="#888", edgecolor="#333", alpha=0.85, label="total")
    axes[0].bar(g_centers, missing, width=(g_bins[1] - g_bins[0]) * 0.9,
                color="#d62728", edgecolor="#333", alpha=0.9, label="IR-missing")
    axes[0].set_xlabel("G [mag]")
    axes[0].set_ylabel("count")
    axes[0].set_title("Stream 3: total vs IR-missing per G-bin")
    axes[0].legend()

    axes[1].plot(g_centers, 100 * rate, "o-", color="#d62728", lw=1.6)
    axes[1].set_xlabel("G [mag]")
    axes[1].set_ylabel("IR-missing rate  [%]")
    axes[1].set_title("Missingness vs G")
    axes[1].axhline(100 * df["ir_missing_flag"].mean(), color="#333", lw=0.8, ls="--",
                    label=f"overall = {100*df['ir_missing_flag'].mean():.1f}%")
    axes[1].legend()

    save_fig(fig, OUT / "ir_missing_vs_g.png")


def ir_mag_distributions() -> None:
    df = _load("stream3")
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(6, 16, 60)
    for col, color in [("j_mag", "#1f77b4"), ("h_mag", "#2ca02c"),
                       ("k_mag", "#9467bd"), ("w1_mag", "#ff7f0e"),
                       ("w2_mag", "#d62728")]:
        vals = df[col].dropna().to_numpy()
        ax.hist(vals, bins=bins, histtype="step", lw=1.5, color=color,
                label=f"{col.upper().replace('_MAG','')}  (n={len(vals):,}, "
                       f"med={np.median(vals):.1f})")
    ax.set_xlabel("magnitude")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("Stream 3 IR magnitude distributions")
    ax.legend()
    save_fig(fig, OUT / "ir_magnitude_distributions.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ir_coverage_sky()
    ir_color_color()
    ir_missing_vs_g()
    ir_mag_distributions()


if __name__ == "__main__":
    main()
