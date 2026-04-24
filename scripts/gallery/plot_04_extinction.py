"""Stage 04: extinction stack — Edenhofer / Lallement / SFD / nbhd-median.

Outputs:
  - reports/gallery/04_extinction/av_map_stack_stream3.png
  - reports/gallery/04_extinction/av_source_breakdown_stream3.png
  - reports/gallery/04_extinction/av_histograms_by_source.png
  - reports/gallery/04_extinction/av_scatter_edenhofer_vs_lallement.png
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
    PALETTE,
    apply_style,
    radec_to_galactic_mollweide,
    save_fig,
    style_galactic_mollweide,
)

OUT = GALLERY / "04_extinction"


def _have(schema, name: str) -> bool:
    return name in {f.name for f in schema}


def _load() -> "pd.DataFrame":
    import pandas as pd  # noqa: F401

    path = DATA_PROCESSED / "pipeline1_features_stream3.parquet"
    schema = pq.read_schema(path)
    want = [
        "source_id",
        "ra_deg",
        "dec_deg",
        "b_deg",
        "g_mag",
        "r_med_photogeo",
        "distance_pc",
        "av_edenhofer",
        "av_lallement",
        "av_sfd",
        "av_nbhd_median",
        "av_nbhd_std",
    ]
    have = [c for c in want if _have(schema, c)]
    t = pq.read_table(path, columns=have).to_pandas()
    if "distance_pc" not in t.columns and "r_med_photogeo" in t.columns:
        t["distance_pc"] = t["r_med_photogeo"]
    return t


def av_map_stack() -> None:
    df = _load()
    n_plot = 80_000
    rng = np.random.default_rng(9)
    idx = rng.choice(len(df), size=min(n_plot, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), subplot_kw={"projection": "mollweide"})

    # For each panel, report: distance regime the source is valid over, the LOS
    # integration depth (CRUCIAL context: SFD integrates to infinity, Edenhofer is
    # truncated at 1.25 kpc so values are lower by construction), coverage, median.
    panels = [
        (
            "av_edenhofer",
            r"Edenhofer+2024  —  3D, truncates at $d=1.25$ kpc",
            "depth: 0 to min(d, 1.25) kpc",
        ),
        (
            "av_lallement",
            r"Lallement+2022  —  3D, truncates at $d=3$ kpc",
            "depth: 0 to min(d, 3) kpc",
        ),
        (
            "av_sfd",
            r"SFD 1998  —  2D asymptotic (full LOS to $\infty$)",
            "depth: full LOS  (→ highest values in disc by design)",
        ),
        (
            "av_nbhd_median",
            r"GSP-Phot 75 pc nbhd-median (fallback)",
            "depth: per-star GSP-Phot out to infinity, median over 75 pc sphere",
        ),
    ]
    x_all, y_all = radec_to_galactic_mollweide(sub["ra_deg"].to_numpy(), sub["dec_deg"].to_numpy())

    for ax, (col, title, depth_note) in zip(axes.flat, panels):
        if col not in sub.columns:
            ax.set_title(f"{title}\n(column absent in Stream 3)", fontsize=10)
            style_galactic_mollweide(ax)
            continue
        vals = sub[col].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        n_valid = int(valid.sum())
        frac_valid = 100 * valid.mean() if len(valid) else 0.0

        if n_valid == 0:
            # Honest "no coverage" panel
            ax.text(
                0,
                0,
                "0 stars with valid " + col + "\n(Stream 3 is RGB at d > 1.25 kpc;\n"
                "Edenhofer+2024 doesn't reach these stars)",
                ha="center",
                va="center",
                fontsize=9,
                color="#666",
                transform=ax.transData,
            )
            ax.set_title(f"{title}\n{depth_note}\ncoverage: 0%", fontsize=10)
            style_galactic_mollweide(ax)
            continue

        sc = ax.scatter(
            x_all[valid],
            y_all[valid],
            c=np.clip(vals[valid], 0, 3),
            cmap="viridis",
            s=0.6,
            alpha=0.7,
            rasterized=True,
            vmin=0,
            vmax=3,
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label=r"$A_V$ [mag]")
        med = float(np.nanmedian(vals))
        ax.set_title(
            f"{title}\n{depth_note}\n"
            rf"coverage: {frac_valid:.1f}%   median $A_V$ = {med:.2f} mag   "
            f"n={n_valid:,}",
            fontsize=9,
        )
        style_galactic_mollweide(ax)

    fig.suptitle(
        r"Stream 3 — four-source $A_V$ stack  (Galactic coords)  —  "
        r"the four maps probe different LOS depths and are not directly "
        r"comparable",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )
    save_fig(fig, OUT / "av_map_stack_stream3.png")


def av_source_breakdown() -> None:
    """Pick 'primary' A_V per star using the stream3_av source column if available."""
    try:
        src = pq.read_table(
            Path("data/interim/stream3_av.parquet"),
            columns=["source_id", "av_los", "av_los_source"],
        ).to_pandas()
    except Exception:
        return
    feat = pq.read_table(
        DATA_PROCESSED / "pipeline1_features_stream3.parquet",
        columns=["source_id", "ra_deg", "dec_deg"],
    ).to_pandas()
    merged = feat.merge(src, on="source_id", how="inner")
    counts = merged["av_los_source"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(
        counts.index.astype(str),
        counts.values,
        color=[PALETTE.get(k, "#888") for k in counts.index.astype(str)],
    )
    axes[0].set_ylabel("n stars")
    axes[0].set_title(f"Primary A_V source in Stream 3 (total n={len(merged):,})")
    for i, (k, v) in enumerate(counts.items()):
        axes[0].text(
            i, v, f"{v:,}\n({100 * v / len(merged):.1f}%)", ha="center", va="bottom", fontsize=9
        )

    ax_map = fig.add_subplot(122, projection="mollweide")
    ax_map.set_position(axes[1].get_position())
    axes[1].remove()
    colors_by_src = {
        str(k): PALETTE.get(str(k), f"C{i}") for i, k in enumerate(counts.index.astype(str))
    }
    rng = np.random.default_rng(11)
    idx = rng.choice(len(merged), size=min(60_000, len(merged)), replace=False)
    sub = merged.iloc[idx]
    for k, color in colors_by_src.items():
        m = sub["av_los_source"].astype(str) == k
        if m.sum() == 0:
            continue
        x, y = radec_to_galactic_mollweide(
            sub.loc[m, "ra_deg"].to_numpy(), sub.loc[m, "dec_deg"].to_numpy()
        )
        ax_map.scatter(
            x, y, s=0.5, alpha=0.4, color=color, label=f"{k} (n={m.sum():,})", rasterized=True
        )
    ax_map.set_title(r"Sky map coloured by primary $A_V$ source (Galactic)")
    style_galactic_mollweide(ax_map)
    ax_map.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.02, -0.15))

    save_fig(fig, OUT / "av_source_breakdown_stream3.png", tight=False)


def av_histograms() -> None:
    df = _load()
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 3, 80)
    for col, label, color in [
        ("av_edenhofer", "Edenhofer (d < 1.25 kpc)", PALETTE["edenhofer"]),
        ("av_lallement", "Lallement (1.25 - 3 kpc)", PALETTE["lallement"]),
        ("av_sfd", "SFD (asymptotic, full LOS)", PALETTE["sfd"]),
        ("av_nbhd_median", "GSP-Phot nbhd-median (fallback)", PALETTE["nbhd"]),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(
            np.clip(vals, 0, 3),
            bins=bins,
            histtype="step",
            lw=1.5,
            color=color,
            label=f"{label}  (n={len(vals):,}, med={np.median(vals):.2f})",
        )
    ax.set_xlabel(r"$A_V$ [mag]  (clipped at 3)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(
        r"Stream 3 $A_V$ distributions across the four extinction sources  "
        r"—  depth differences (SFD > Lallement > Edenhofer) are structural, "
        r"not calibration error"
    )
    ax.legend()
    save_fig(fig, OUT / "av_histograms_by_source.png")


def av_edenhofer_vs_lallement() -> None:
    """Cross-check on Stream 1 (Stream 3 has 0% Edenhofer coverage)."""
    df = (
        pq.read_table(
            DATA_PROCESSED / "pipeline1_features_stream1.parquet",
            columns=["av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median", "r_med_photogeo"],
        )
        .to_pandas()
        .dropna(subset=["av_edenhofer", "av_lallement"])
    )
    df["distance_pc"] = df["r_med_photogeo"]
    if len(df) == 0:
        return
    rng = np.random.default_rng(13)
    idx = rng.choice(len(df), size=min(60_000, len(df)), replace=False)
    sub = df.iloc[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    hb0 = axes[0].hexbin(
        sub["av_edenhofer"],
        sub["av_lallement"],
        gridsize=60,
        cmap="viridis",
        bins="log",
        extent=(0, 2, 0, 2),
        mincnt=1,
    )
    axes[0].plot([0, 2], [0, 2], "r-", lw=1.2, label="1:1")
    plt.colorbar(hb0, ax=axes[0], shrink=0.85, pad=0.02, label="log N")
    axes[0].set_xlabel(r"$A_V$ (Edenhofer)  [mag]")
    axes[0].set_ylabel(r"$A_V$ (Lallement)  [mag]")
    axes[0].set_title(f"Edenhofer vs Lallement  (Stream 1, n={len(sub):,})")
    axes[0].legend()
    axes[0].set_xlim(0, 2)
    axes[0].set_ylim(0, 2)

    delta = sub["av_lallement"] - sub["av_edenhofer"]
    hb1 = axes[1].hexbin(
        sub["distance_pc"] / 1000,
        delta,
        gridsize=60,
        cmap="plasma",
        bins="log",
        extent=(0, 4, -1, 1),
        mincnt=1,
    )
    axes[1].axhline(0, color="#222", lw=0.8, ls="--")
    axes[1].axvline(1.25, color="k", lw=0.7, ls="--", label="1.25 kpc (Edenhofer horizon)")
    axes[1].axvline(3.0, color="k", lw=0.7, ls=":", label="3 kpc (Lallement horizon)")
    plt.colorbar(hb1, ax=axes[1], shrink=0.85, pad=0.02, label="log N")
    axes[1].set_xlabel("distance [kpc]")
    axes[1].set_ylabel(r"$A_V$ (Lallement) $-$ $A_V$ (Edenhofer)  [mag]")
    axes[1].set_title("Lallement - Edenhofer vs distance")
    axes[1].legend(loc="lower right")
    axes[1].set_xlim(0, 4)
    axes[1].set_ylim(-1, 1)

    save_fig(fig, OUT / "av_scatter_edenhofer_vs_lallement.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    av_map_stack()
    av_source_breakdown()
    av_histograms()
    av_edenhofer_vs_lallement()


if __name__ == "__main__":
    main()
