"""Y34: Stream-1 holdout Kiel and chemistry, sliced by release tier.

5 columns x 2 rows (top: Kiel diagram; bottom: [M/H] vs [α/M] chemistry).
Columns from left to right:
  1. all data
  2. Tier 1 only
  3. Tier 2 only
  4. Tier 3 only
  5. all data, color-coded by tier (Okabe-Ito green/vermillion/red-purple).

Slide-friendly 18:8 layout.
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

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

T1_COLOR = PALETTE["tier1"]   # green
T2_COLOR = PALETTE["tier2"]   # vermillion
T3_COLOR = PALETTE["tier3"]   # red-purple

_TITLE_KW = dict(fontsize=10, fontweight="normal", color=PALETTE["ink"], pad=6)

KIEL_TEFF = (3500, 6500)
KIEL_LOGG = (5.0, 0.0)        # inverted on plot
MH_LIM = (-1.6, 0.55)
AM_LIM = (-0.10, 0.45)
HEX_KIEL = 70
HEX_CHEM = 70


def _load_holdout() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "fe_h_apogee", "teff_apogee", "b_deg"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    return df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)


def _kiel_density(ax, sub: pd.DataFrame, title: str) -> None:
    x = sub["teff_pred"].to_numpy()
    y = sub["logg_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 0:
        ax.hexbin(
            x[ok], y[ok],
            gridsize=HEX_KIEL,
            extent=(KIEL_TEFF[0], KIEL_TEFF[1], 0.0, 5.0),
            mincnt=1, bins="log", cmap="viridis",
        )
    ax.set_xlim(KIEL_TEFF[1], KIEL_TEFF[0])
    ax.set_ylim(KIEL_LOGG[0], KIEL_LOGG[1])
    ax.set_xlabel(r"$T_{\rm eff,\,pred}$ (K)")
    ax.set_ylabel(r"$\log g_{\rm pred}$ (dex)")
    ax.set_title(f"{title}  (n = {int(ok.sum()):,})", **_TITLE_KW)


def _chem_density(ax, sub: pd.DataFrame, title: str) -> None:
    x = sub["mh_pred"].to_numpy()
    y = sub["alpha_m_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 0:
        ax.hexbin(
            x[ok], y[ok],
            gridsize=HEX_CHEM,
            extent=(MH_LIM[0], MH_LIM[1], AM_LIM[0], AM_LIM[1]),
            mincnt=1, bins="log", cmap="viridis",
        )
    ax.axhline(0.15, color="white", lw=1.0, ls=":", alpha=0.85)
    ax.set_xlim(MH_LIM)
    ax.set_ylim(AM_LIM)
    ax.set_xlabel("[M/H] pred (dex)")
    ax.set_ylabel(r"[$\alpha$/M] pred (dex)")
    ax.set_title(f"{title}  (n = {int(ok.sum()):,})", **_TITLE_KW)


def _kiel_by_tier(ax, df: pd.DataFrame) -> None:
    for tier, color, label, s, alpha in [
        (1, T1_COLOR, "Tier 1", 1.0, 0.18),
        (2, T2_COLOR, "Tier 2", 2.5, 0.55),
        (3, T3_COLOR, "Tier 3", 2.5, 0.45),
    ]:
        sub = df.loc[df["release_tier"] == tier]
        if not len(sub):
            continue
        ax.scatter(
            sub["teff_pred"], sub["logg_pred"],
            s=s, alpha=alpha, color=color, edgecolors="none",
            rasterized=True, label=f"{label}  n={len(sub):,}",
        )
    ax.set_xlim(KIEL_TEFF[1], KIEL_TEFF[0])
    ax.set_ylim(KIEL_LOGG[0], KIEL_LOGG[1])
    ax.set_xlabel(r"$T_{\rm eff,\,pred}$ (K)")
    ax.set_ylabel(r"$\log g_{\rm pred}$ (dex)")
    ax.set_title("all data, by tier", **_TITLE_KW)
    ax.legend(loc="lower right", fontsize=9, markerscale=4, frameon=False)


def _chem_by_tier(ax, df: pd.DataFrame) -> None:
    for tier, color, label, s, alpha in [
        (1, T1_COLOR, "Tier 1", 1.0, 0.18),
        (2, T2_COLOR, "Tier 2", 2.5, 0.55),
        (3, T3_COLOR, "Tier 3", 2.5, 0.45),
    ]:
        sub = df.loc[df["release_tier"] == tier]
        if not len(sub):
            continue
        ax.scatter(
            sub["mh_pred"], sub["alpha_m_pred"],
            s=s, alpha=alpha, color=color, edgecolors="none",
            rasterized=True, label=f"{label}  n={len(sub):,}",
        )
    ax.axhline(0.15, color=PALETTE["ink"], lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(MH_LIM)
    ax.set_ylim(AM_LIM)
    ax.set_xlabel("[M/H] pred (dex)")
    ax.set_ylabel(r"[$\alpha$/M] pred (dex)")
    ax.set_title("all data, by tier", **_TITLE_KW)
    ax.legend(loc="upper right", fontsize=9, markerscale=4, frameon=False)


def main() -> int:
    apply_style()
    df = _load_holdout()
    if df.empty:
        print("[Y34] no holdout rows, aborting")
        return 1

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    # Top row, Kiel.
    _kiel_density(axes[0, 0], df, "all data")
    _kiel_density(axes[0, 1], df.loc[df["release_tier"] == 1], "Tier 1 only")
    _kiel_density(axes[0, 2], df.loc[df["release_tier"] == 2], "Tier 2 only")
    _kiel_density(axes[0, 3], df.loc[df["release_tier"] == 3], "Tier 3 only")
    _kiel_by_tier(axes[0, 4], df)

    # Bottom row, chemistry.
    _chem_density(axes[1, 0], df, "all data")
    _chem_density(axes[1, 1], df.loc[df["release_tier"] == 1], "Tier 1 only")
    _chem_density(axes[1, 2], df.loc[df["release_tier"] == 2], "Tier 2 only")
    _chem_density(axes[1, 3], df.loc[df["release_tier"] == 3], "Tier 3 only")
    _chem_by_tier(axes[1, 4], df)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.20)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.83, bottom=0.07,
                        hspace=0.40, wspace=0.32)
    headline(
        fig,
        "Stream 1, holdout: Kiel and chemistry sliced by release tier",
        f"val + test split, seed=0, n = {len(df):,}.  T1 = green, T2 = vermillion, T3 = red-purple.",
        top=0.83,
    )
    save(fig, "Y34_kiel_chem_by_tier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
