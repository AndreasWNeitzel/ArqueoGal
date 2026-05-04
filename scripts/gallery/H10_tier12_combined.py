"""H10: Tier 1 + Tier 2 combined per stream.

3 rows (Stream 1 / Stream 2 / Stream 3) × 3 cols (Kiel diagram /
chemistry plane / Galactic Mollweide).  Tier 1 stars drawn in green,
Tier 2 stars overlaid in orange so the *combined* T1 + T2 cohort is
visible together with the *contribution* T2 makes on top of T1.

Tier comes from ``assign_release_tier`` applied per stream.  Stream 2
and Stream 3 pull ``kin_ood_flag`` from the hybrid release dirs when
available; Stream 1 has no kin_ood column so it defaults to False.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import (  # noqa: E402
    apply_style,
    galactic_mollweide,
    radec_to_galactic,
    save_fig,
    style_galactic_mollweide,
)

from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED = {
    sid: REPO / f"data/processed/pipeline1_predictions_stream{sid}.parquet" for sid in (1, 2, 3)
}
FEAT = {
    1: REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
    2: REPO / "data/processed/pipeline1_features_stream2.parquet",
    3: REPO / "data/processed/pipeline1_features_stream3.parquet",
}
HYBRID = {
    2: REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet",
    3: REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet",
}
OUT = REPO / "reports/gallery/H_hybrid_release"

T1_COLOR = "#009E73"  # Okabe-Ito green
T2_COLOR = "#D55E00"  # Okabe-Ito vermilion
T3_COLOR = "#CC79A7"  # Okabe-Ito red-purple
NSIDE = 32


def _load_stream(sid: int) -> pd.DataFrame:
    pcols = [
        "source_id",
        "teff_pred",
        "logg_pred",
        "mh_pred",
        "alpha_m_pred",
        "mg_h_pred",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    # Tolerate older predictions parquets that lack label_extrapolation_flag.
    try:
        p = pd.read_parquet(PRED[sid], columns=pcols)
    except (KeyError, ValueError):
        p = pd.read_parquet(PRED[sid], columns=pcols[:-1])
    p = p.drop_duplicates("source_id")
    f = pd.read_parquet(FEAT[sid], columns=["source_id", "ra_deg", "dec_deg"]).drop_duplicates(
        "source_id"
    )
    df = f.merge(p, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    h = HYBRID.get(sid)
    if h is not None and h.exists():
        hd = pd.read_parquet(h, columns=["source_id", "kin_ood_flag"])
        df = df.merge(hd, on="source_id", how="left", suffixes=("", "_h"))
        if "kin_ood_flag_h" in df.columns:
            df["kin_ood_flag"] = df["kin_ood_flag_h"].fillna(False).astype(bool)
            df = df.drop(columns=["kin_ood_flag_h"])
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    return df


def _kiel_density(ax, df_combined, sid):
    """T1+T2 combined density (hist2d, log-N)."""
    x = df_combined["teff_pred"].to_numpy()
    y = df_combined["logg_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(
        x[ok],
        y[ok],
        gridsize=80,
        extent=(3500, 6500, 0.0, 5.0),
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N", fraction=0.046, pad=0.02)
    ax.set_xlim(6500, 3500)
    ax.set_ylim(5.0, 0.0)
    ax.set_xlabel(r"$T_{\rm eff,\,pred}$ (K)")
    ax.set_ylabel(r"$\log g_{\rm pred}$ (dex)")
    ax.set_title(f"Stream {sid} — Kiel density (T1+T2)", color="#15355f")
    ax.grid(True, alpha=0.20)


def _kiel_scatter(ax, df_t1, df_t2, df_t3, sid):
    ax.scatter(
        df_t1["teff_pred"],
        df_t1["logg_pred"],
        s=1.0,
        alpha=0.18,
        color=T1_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"T1  n={len(df_t1):,}",
    )
    ax.scatter(
        df_t2["teff_pred"],
        df_t2["logg_pred"],
        s=2.5,
        alpha=0.55,
        color=T2_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"T2  n={len(df_t2):,}",
    )
    ax.scatter(
        df_t3["teff_pred"],
        df_t3["logg_pred"],
        s=2.5,
        alpha=0.45,
        color=T3_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"T3  n={len(df_t3):,}",
    )
    ax.set_xlim(6500, 3500)
    ax.set_ylim(5.0, 0.0)
    ax.set_xlabel(r"$T_{\rm eff,\,pred}$ (K)")
    ax.set_ylabel(r"$\log g_{\rm pred}$ (dex)")
    ax.set_title(f"Stream {sid} — Kiel by tier", color="#15355f")
    ax.legend(loc="lower right", fontsize=9, markerscale=4)
    ax.grid(True, alpha=0.20)


def _chem_density(ax, df_combined, sid):
    """T1+T2 combined density (hist2d, log-N)."""
    x = df_combined["mh_pred"].to_numpy()
    y = df_combined["alpha_m_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(
        x[ok],
        y[ok],
        gridsize=80,
        extent=(-1.6, 0.55, -0.10, 0.42),
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N", fraction=0.046, pad=0.02)
    ax.axhline(0.15, color="white", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(-0.10, 0.42)
    ax.set_xlabel("[M/H] pred (dex)")
    ax.set_ylabel(r"[$\alpha$/M] pred (dex)")
    ax.set_title(f"Stream {sid} — chemistry density (T1+T2)", color="#15355f")
    ax.grid(True, alpha=0.20)


def _chem_scatter(ax, df_t1, df_t2, df_t3, sid):
    ax.scatter(
        df_t1["mh_pred"],
        df_t1["alpha_m_pred"],
        s=1.0,
        alpha=0.18,
        color=T1_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"T1  n={len(df_t1):,}",
    )
    ax.scatter(
        df_t2["mh_pred"],
        df_t2["alpha_m_pred"],
        s=2.5,
        alpha=0.55,
        color=T2_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"T2  n={len(df_t2):,}",
    )
    ax.scatter(
        df_t3["mh_pred"],
        df_t3["alpha_m_pred"],
        s=2.5,
        alpha=0.45,
        color=T3_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"T3  n={len(df_t3):,}",
    )
    ax.axhline(0.15, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(-0.10, 0.42)
    ax.set_xlabel("[M/H] pred (dex)")
    ax.set_ylabel(r"[$\alpha$/M] pred (dex)")
    ax.set_title(f"Stream {sid} — chemistry by tier", color="#15355f")
    ax.legend(loc="upper right", fontsize=9, markerscale=4)
    ax.grid(True, alpha=0.20)


def _sky_panel(ax, df_t1, df_t2, df_t3, sid):
    for df, color, label, s, alpha in [
        (df_t1, T1_COLOR, "T1", 0.6, 0.20),
        (df_t2, T2_COLOR, "T2", 1.6, 0.55),
        (df_t3, T3_COLOR, "T3", 1.6, 0.45),
    ]:
        if not len(df):
            continue
        l, b = radec_to_galactic(df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy())
        lon, lat = galactic_mollweide(l, b)
        ax.scatter(
            lon,
            lat,
            s=s,
            color=color,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            label=f"{label}  n={len(df):,}",
        )
    style_galactic_mollweide(ax)
    ax.set_title(f"Stream {sid} — sky", color="#15355f")
    ax.legend(loc="lower left", fontsize=10, markerscale=6)


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    streams: dict[int, pd.DataFrame] = {}
    for sid in (1, 2, 3):
        if not PRED[sid].exists():
            print(f"[H10] Stream {sid}: predictions parquet missing, skipping")
            continue
        df = _load_stream(sid)
        n_t1 = int((df["release_tier"] == 1).sum())
        n_t2 = int((df["release_tier"] == 2).sum())
        n_t3 = int((df["release_tier"] == 3).sum())
        n = len(df)
        print(
            f"[H10] Stream {sid}: T1={n_t1:,} ({n_t1 / n * 100:.1f}%)  "
            f"T2={n_t2:,} ({n_t2 / n * 100:.1f}%)  "
            f"T3={n_t3:,} ({n_t3 / n * 100:.1f}%)  "
            f"total={n:,}"
        )
        streams[sid] = df

    if not streams:
        print("[H10] no streams available")
        return 1

    n_streams = len(streams)
    fig = plt.figure(figsize=(28, 6.5 * n_streams))
    gs = fig.add_gridspec(
        n_streams,
        5,
        hspace=0.40,
        wspace=0.32,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.4],
        top=0.93,
        bottom=0.05,
        left=0.04,
        right=0.98,
    )

    for r, (sid, df) in enumerate(sorted(streams.items())):
        df_t1 = df.loc[df["release_tier"] == 1]
        df_t2 = df.loc[df["release_tier"] == 2]
        df_t3 = df.loc[df["release_tier"] == 3]
        df_combined = df.loc[df["release_tier"].isin([1, 2, 3])]
        _kiel_density(fig.add_subplot(gs[r, 0]), df_combined, sid)
        _kiel_scatter(fig.add_subplot(gs[r, 1]), df_t1, df_t2, df_t3, sid)
        _chem_density(fig.add_subplot(gs[r, 2]), df_combined, sid)
        _chem_scatter(fig.add_subplot(gs[r, 3]), df_t1, df_t2, df_t3, sid)
        _sky_panel(fig.add_subplot(gs[r, 4], projection="mollweide"), df_t1, df_t2, df_t3, sid)

    fig.suptitle(
        "H10. Tier 1 + Tier 2 combined per stream\n"
        "T1 (green) is the science-grade default; T2 (orange) is the "
        "use-with-caution add-on.  T1 + T2 = the maximally-inclusive "
        "released cohort the user can choose to filter.",
        fontsize=13,
        fontweight="semibold",
        y=0.985,
    )
    save_fig(fig, OUT / "H10_tier12_combined", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
