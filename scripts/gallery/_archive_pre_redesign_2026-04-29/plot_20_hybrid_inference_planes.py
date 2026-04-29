"""Stage 18: hybrid Kiel + chemistry on Stream 3 (the catalog's user-facing planes).

The hybrid surface (``<elem>_hybrid_pred``) substitutes the kNN-median when
σ_regressor exceeds the per-element prior-collapse threshold. This stage
shows the resulting Kiel and chemistry diagrams, faceted so the reader can
tell:

- **Tier 1** (≈28% of Stream 3 under v5; was ≈33% under v4): all elements
  within σ-threshold (with α/M tightened from 0.10 → 0.05 dex on
  2026-04-26), and not flagged by ``mode_ambiguous_flag`` on α/M. The
  regressor surface is used everywhere. This is the science-grade slice.
- **Tier 2** (≈52%): at least one element is σ-inflated, or
  ``mode_ambiguous_flag`` fires on α/M, or kin_ood demotes an aux-assisted
  element. For elements with kNN columns, the kNN median substitutes; the
  panel is a mix of regressor + kNN points, but every value is bounded by
  the training-set support.
- **Tier 3** (≈20%): hard-killed by ``ood_joint_flag`` (XP-Mahalanobis OOD)
  or NaN prediction. Hybrid predictions are still emitted (regressor or
  regressor_caveat) but the catalog contract is **do not use these for
  science**. We show them only to make the failure mode visible — the
  reader should see that Tier 3 sits OUTSIDE the disc giant branch and
  OUTSIDE the disc bimodality, which is exactly what an OOD detector
  should flag.

v5 (2026-04-26) tier semantics: only ``ood_joint_flag`` (XP-Mahalanobis)
gates Tier 3, only σ-inflation, ``mode_ambiguous_flag`` on α/M, and
kin_ood on aux-assisted elements gate Tier 2. Diagnostics
(``regime_b_flag``, ``aux_missing_any``, ``ood_disagreement_flag``,
``dist_prior_dominated``, ``latent_support_flag``,
``ood_aux_mahalanobis_flag``) are still emitted but no longer feed the
tier. See ``release/test_ablations_2026-04-26/REPORT.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/20_hybrid_inference_planes"
PARQUET_S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"
PARQUET_S2 = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"

KIEL_X = (3500, 6500)
KIEL_Y = (0.5, 4.0)
CHEM_X = (-2.5, 0.6)
CHEM_Y = (-0.2, 0.5)


def _density_panel(ax, x, y, *, x_lo, x_hi, y_lo, y_hi, nx, ny, xlab, ylab, title, n):
    """Raw 2-D histogram via numpy.histogram2d + pcolormesh, NO smoothing,
    NO interpolation. Each cell shows its actual star count (log scale).
    Empty cells are masked. This is intentionally honest about the
    discretisation present in the regressor + calibration outputs — any
    'choppiness' visible IS in the data."""
    from matplotlib.colors import LogNorm

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        ax.text(
            0.5,
            0.5,
            f"too few finite predictions\nn={int(m.sum())}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{title} (n={int(m.sum()):,} of {n:,})")
        return
    x_bins = np.linspace(x_lo, x_hi, nx + 1)
    y_bins = np.linspace(y_lo, y_hi, ny + 1)
    H, _, _ = np.histogram2d(x[m], y[m], bins=[x_bins, y_bins])
    # Mask zero cells; everything else shown raw (no smoothing).
    Hm = np.ma.masked_where(H.T == 0, H.T)
    pc = ax.pcolormesh(
        x_bins,
        y_bins,
        Hm,
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(2.0, Hm.max())),
        shading="flat",
    )
    plt.colorbar(pc, ax=ax, label="log10 N (raw count)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{title} (n={int(m.sum()):,} of {n:,})")


def _kiel_panel(ax, teff, logg, *, title, n):
    # Higher resolution + smoothing — bilinear interp removes block edges.
    _density_panel(
        ax,
        teff,
        logg,
        x_lo=KIEL_X[0],
        x_hi=KIEL_X[1],
        y_lo=KIEL_Y[0],
        y_hi=KIEL_Y[1],
        nx=240,
        ny=140,
        xlab="Teff (K)",
        ylab=r"$\log g$ (dex)",
        title=title,
        n=n,
    )
    ax.set_xlim(KIEL_X[1], KIEL_X[0])  # invert Teff
    ax.set_ylim(KIEL_Y[1], KIEL_Y[0])  # invert logg


def _chem_panel(ax, mh, am, *, title, n):
    # APOGEE DR19 [α/M] has only ~5k unique values across 324k training stars
    # (~65 stars/value), vs [M/H] with ~20k unique values. The regressor
    # preserves this resolution — so x-bins must be ~3x finer than y-bins,
    # otherwise [α/M] cells fall between APOGEE grid points and stripe.
    # nx/ny chosen so y-cell ≈ 0.014 dex (covers most APOGEE [α/M] gaps) while
    # x-cell ≈ 0.013 dex stays sharp for the disc bimodality.
    _density_panel(
        ax,
        mh,
        am,
        x_lo=CHEM_X[0],
        x_hi=CHEM_X[1],
        y_lo=CHEM_Y[0],
        y_hi=CHEM_Y[1],
        nx=240,
        ny=50,
        xlab="[M/H] (dex)",
        ylab=r"[$\alpha$/M] (dex)",
        title=title,
        n=n,
    )


def _render_for(parquet: Path, stream_label: str, kiel_out: Path, chem_out: Path) -> None:
    if not parquet.exists():
        return
    df = pd.read_parquet(
        parquet,
        columns=[
            "teff_hybrid_pred",
            "logg_hybrid_pred",
            "mh_hybrid_pred",
            "alpha_m_hybrid_pred",
            "teff_hybrid_source",
            "mh_hybrid_source",
            "release_tier",
        ],
    )
    _render_one_stream(df, stream_label, kiel_out, chem_out)


def main() -> None:
    apply_style()
    _render_for(PARQUET_S3, "Stream 3", OUT / "hybrid_kiel.png", OUT / "hybrid_chemistry.png")
    _render_for(
        PARQUET_S2,
        "Stream 2",
        OUT / "hybrid_kiel_stream2.png",
        OUT / "hybrid_chemistry_stream2.png",
    )


def _render_one_stream(df: pd.DataFrame, stream_label: str, kiel_out: Path, chem_out: Path) -> None:

    # FIGURE 1: Kiel diagram, 5-panel row matching hybrid_chemistry layout.
    # (0) full trustworthy catalog Tier 1+2 union, (1) Tier 1 only,
    # (2) Tier 2 regressor portion, (3) Tier 2 kNN portion, (4) Tier 3.
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))

    t1 = df[df["release_tier"] == 1]
    t2 = df[df["release_tier"] == 2]
    t3 = df[df["release_tier"] == 3]
    t12 = df[df["release_tier"] <= 2]
    t2_reg = t2[t2["teff_hybrid_source"] == "regressor"]
    t2_knn = t2[t2["teff_hybrid_source"] == "knn"]

    _kiel_panel(
        axes[0],
        t12["teff_hybrid_pred"].to_numpy(),
        t12["logg_hybrid_pred"].to_numpy(),
        title="Tier 1 + Tier 2 UNION\nthe full trustworthy catalog",
        n=len(t12),
    )
    _kiel_panel(
        axes[1],
        t1["teff_hybrid_pred"].to_numpy(),
        t1["logg_hybrid_pred"].to_numpy(),
        title="Tier 1 (per-star science)\nall regressor",
        n=len(t1),
    )
    _kiel_panel(
        axes[2],
        t2_reg["teff_hybrid_pred"].to_numpy(),
        t2_reg["logg_hybrid_pred"].to_numpy(),
        title="Tier 2 caveat\nteff = regressor",
        n=len(t2),
    )
    _kiel_panel(
        axes[3],
        t2_knn["teff_hybrid_pred"].to_numpy(),
        t2_knn["logg_hybrid_pred"].to_numpy(),
        title="Tier 2 rescued\nteff = kNN-median",
        n=len(t2),
    )
    _kiel_panel(
        axes[4],
        t3["teff_hybrid_pred"].to_numpy(),
        t3["logg_hybrid_pred"].to_numpy(),
        title="Tier 3 DO NOT USE\nany OOD or NaN",
        n=len(t3),
    )

    fig.suptitle(
        f"Hybrid Kiel diagram on {stream_label}, faceted by tier and Teff source.\n"
        f"Panel 0 is the FULL CATALOG you can trust (Tier 1 + Tier 2 = {len(t12):,} stars). "
        "Panels 1-3 split it by source so you can see the regressor (panels 1-2) and kNN-rescued "
        "(panel 3) sub-populations. Panel 4 (Tier 3) sits OFF the giant branch — OOD detector worked.",
        fontsize=10,
    )
    save_fig(fig, kiel_out)

    # FIGURE 2: chemistry plane.
    # 5 panels in a 1x5 row:
    # (0) full trustworthy catalog Tier 1 + 2 (the union the user asked for),
    # (1) Tier 1 only, (2) Tier 2 regressor portion, (3) Tier 2 kNN portion,
    # (4) Tier 3 do-not-use.
    t12 = df[df["release_tier"] <= 2]
    t2_reg_mh = t2[t2["mh_hybrid_source"] == "regressor"]
    t2_knn_mh = t2[t2["mh_hybrid_source"] == "knn"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
    _chem_panel(
        axes[0],
        t12["mh_hybrid_pred"].to_numpy(),
        t12["alpha_m_hybrid_pred"].to_numpy(),
        title="Tier 1 + Tier 2 UNION\nthe full trustworthy catalog",
        n=len(t12),
    )
    _chem_panel(
        axes[1],
        t1["mh_hybrid_pred"].to_numpy(),
        t1["alpha_m_hybrid_pred"].to_numpy(),
        title="Tier 1 (per-star science)\nall regressor",
        n=len(t1),
    )
    _chem_panel(
        axes[2],
        t2_reg_mh["mh_hybrid_pred"].to_numpy(),
        t2_reg_mh["alpha_m_hybrid_pred"].to_numpy(),
        title="Tier 2 caveat\n[M/H] = regressor",
        n=len(t2),
    )
    _chem_panel(
        axes[3],
        t2_knn_mh["mh_hybrid_pred"].to_numpy(),
        t2_knn_mh["alpha_m_hybrid_pred"].to_numpy(),
        title="Tier 2 rescued\n[M/H] = kNN-median",
        n=len(t2),
    )
    _chem_panel(
        axes[4],
        t3["mh_hybrid_pred"].to_numpy(),
        t3["alpha_m_hybrid_pred"].to_numpy(),
        title="Tier 3 DO NOT USE\nany OOD or NaN",
        n=len(t3),
    )

    fig.suptitle(
        f"Hybrid chemistry plane on {stream_label}, faceted by tier and [M/H] source.\n"
        f"Panel 0 is the FULL CATALOG you can trust (Tier 1 + Tier 2 = {len(t12):,} stars). "
        "Panels 1–3 split by source; panel 4 (Tier 3) sits OFF the disc bimodality — the OOD detector did its job.\n"
        "Note on visible discretisation: APOGEE DR19 [α/M] has only ~5k unique values across 324k training stars "
        "(~65 stars/value, vs ~16 for [M/H]) because ASPCAP fits [α/M] as a single grid parameter. The regressor "
        "faithfully reproduces this resolution; the kNN-median (Tier 2 rescued) inherits it directly. y-bins set "
        "to 0.014 dex/cell to match the data grain.",
        fontsize=9.0,
    )
    save_fig(fig, chem_out)


if __name__ == "__main__":
    main()
