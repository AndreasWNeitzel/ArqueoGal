"""Stage 06: selection function omega = P_ye * P_ir * P_parallax * P_extinction.

Outputs:
  - reports/gallery/06_selection_function/omega_total_sky.png
  - reports/gallery/06_selection_function/omega_components.png
  - reports/gallery/06_selection_function/omega_vs_g.png
  - reports/gallery/06_selection_function/omega_histograms.png

Reads the per-star selection probabilities computed in
src/arqueogal/data/selection_function.py. If the parquet doesn't exist yet,
this script falls back to computing them on the fly from the feature matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

OUT = GALLERY / "06_selection_function"


def _have(schema: "pq.Schema", name: str) -> bool:
    return name in {f.name for f in schema}


def _load_omegas() -> pd.DataFrame:
    path = DATA_PROCESSED / "pipeline1_features_stream3.parquet"
    schema = pq.read_schema(path)

    cols = ["source_id", "ra_deg", "dec_deg", "b_deg", "g_mag", "ir_missing_flag"]
    for c in (
        "p_parallax",
        "p_extinction",
        "p_ye",
        "p_ir",
        "omega_total",
        "selection_omega",
        "parallax_over_error",
    ):
        if _have(schema, c):
            cols.append(c)
    df = pq.read_table(path, columns=cols).to_pandas()

    # Compute components that are missing, using standard assumptions.
    if "p_ir" not in df.columns:
        df["p_ir"] = 1.0 - df["ir_missing_flag"].astype(float)
    if "p_parallax" not in df.columns and "parallax_over_error" in df.columns:
        # soft logistic gate at parallax_over_error >= 5
        plx_snr = df["parallax_over_error"].fillna(0.0).to_numpy()
        df["p_parallax"] = 1.0 / (1.0 + np.exp(-(plx_snr - 5.0)))
    if "p_parallax" not in df.columns:
        df["p_parallax"] = 1.0
    if "p_ye" not in df.columns:
        # Stream 3 is conditioned on Ye==OK → all 1
        df["p_ye"] = 1.0
    if "p_extinction" not in df.columns:
        df["p_extinction"] = 1.0

    if "omega_total" not in df.columns and "selection_omega" in df.columns:
        df["omega_total"] = df["selection_omega"]
    if "omega_total" not in df.columns:
        df["omega_total"] = df["p_ye"] * df["p_ir"] * df["p_parallax"] * df["p_extinction"]
    return df


def omega_total_sky() -> None:
    df = _load_omegas()
    rng = np.random.default_rng(19)
    idx = sample_index(len(df), 80_000, rng)
    sub = df.iloc[idx]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    x, y = radec_to_galactic_mollweide(sub["ra_deg"].to_numpy(), sub["dec_deg"].to_numpy())
    sc = ax.scatter(
        x,
        y,
        c=np.clip(sub["omega_total"].to_numpy(), 0, 1),
        cmap="viridis",
        s=2.5,
        alpha=0.75,
        rasterized=True,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label=r"$\omega_{\rm total}$  (probability)")
    ax.set_title(
        rf"Compound selection function $\omega_{{\rm total}}$  —  Galactic coords  "
        rf"(n={len(sub):,} plotted / {len(df):,})",
        fontsize=12,
        fontweight="bold",
    )
    style_galactic_mollweide(ax)
    save_fig(fig, OUT / "omega_total_sky.png")


def omega_components() -> None:
    df = _load_omegas()
    rng = np.random.default_rng(23)
    idx = sample_index(len(df), 60_000, rng)
    sub = df.iloc[idx]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), subplot_kw={"projection": "mollweide"})
    comps = [
        ("p_ye", r"$P_{\rm Ye}$  (Ye+2024 correction survives)", "Ye+2024 OK prob."),
        ("p_ir", r"$P_{\rm IR}$  (2MASS + WISE coverage)", "IR coverage prob."),
        ("p_parallax", r"$P_{\varpi}$  (parallax SNR $\geq 5$ gate)", "parallax SNR prob."),
        ("p_extinction", r"$P_{A_V}$  ($A_V$ budget met)", r"$A_V$ budget prob."),
    ]
    x, y = radec_to_galactic_mollweide(sub["ra_deg"].to_numpy(), sub["dec_deg"].to_numpy())
    for ax, (col, title, cbar_lbl) in zip(axes.flat, comps):
        vals = np.clip(sub[col].to_numpy(), 0, 1)
        sc = ax.scatter(
            x, y, c=vals, cmap="viridis", s=2.5, alpha=0.75, rasterized=True, vmin=0, vmax=1
        )
        mean_v = float(sub[col].mean())
        p05, p95 = np.nanpercentile(sub[col].to_numpy(), [5, 95])
        if p95 - p05 < 1e-3:
            stat_line = rf"$\mu$={mean_v:.3f}  (near-constant: $P_5 - P_{{95}} < 0.001$)"
        else:
            stat_line = rf"$\mu$={mean_v:.3f}   $P_5$={p05:.2f}   $P_{{95}}$={p95:.2f}"
        ax.set_title(f"{title}\n{stat_line}", fontsize=10)
        style_galactic_mollweide(ax)
        plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label=cbar_lbl)
    fig.suptitle(
        r"$\omega$ decomposed across its four components  —  Galactic coords  "
        r"($P_{\rm Ye}$ and $P_{A_V}$ are conditioned to $\approx 1$ on Stream 3)",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )
    save_fig(fig, OUT / "omega_components.png")


def omega_vs_g() -> None:
    df = _load_omegas()
    g_bins = np.linspace(df["g_mag"].min(), df["g_mag"].max(), 30)
    g_centers = 0.5 * (g_bins[:-1] + g_bins[1:])
    means = []
    for lo, hi in zip(g_bins[:-1], g_bins[1:]):
        m = (df["g_mag"] >= lo) & (df["g_mag"] < hi)
        means.append(df.loc[m, "omega_total"].mean() if m.any() else np.nan)
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(g_centers, means, "o-", color="#1f77b4", lw=1.5)
    ax.axhline(
        df["omega_total"].mean(),
        color="#333",
        lw=0.8,
        ls="--",
        label=f"overall mean = {df['omega_total'].mean():.3f}",
    )
    ax.set_xlabel("G [mag]")
    ax.set_ylabel("mean omega_total")
    ax.set_title("Stream 3 selection function vs apparent magnitude")
    ax.set_ylim(0, 1.05)
    ax.legend()
    save_fig(fig, OUT / "omega_vs_g.png")


def omega_histograms() -> None:
    df = _load_omegas()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(0, 1, 40)
    for col, color in [
        ("p_ye", "#2ca02c"),
        ("p_ir", "#1f77b4"),
        ("p_parallax", "#ff7f0e"),
        ("p_extinction", "#9467bd"),
    ]:
        axes[0].hist(
            df[col].dropna(),
            bins=bins,
            histtype="step",
            lw=1.5,
            color=color,
            label=f"{col}  (mean={df[col].mean():.2f})",
        )
    axes[0].set_xlabel("component probability")
    axes[0].set_ylabel("count")
    axes[0].set_yscale("log")
    axes[0].set_title("Per-component omega distributions")
    axes[0].legend()

    axes[1].hist(
        df["omega_total"].dropna(), bins=bins, color="#d62728", edgecolor="#333", alpha=0.85
    )
    axes[1].axvline(
        df["omega_total"].mean(),
        color="k",
        lw=1,
        ls="--",
        label=f"mean = {df['omega_total'].mean():.3f}",
    )
    axes[1].set_xlabel("omega_total")
    axes[1].set_ylabel("count")
    axes[1].set_title("Compound omega_total distribution")
    axes[1].legend()

    save_fig(fig, OUT / "omega_histograms.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    omega_total_sky()
    omega_components()
    omega_vs_g()
    omega_histograms()


if __name__ == "__main__":
    main()
