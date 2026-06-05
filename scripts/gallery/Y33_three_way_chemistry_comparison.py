"""Y33: Stream-1 Tier-1 holdout chemistry, GSP-Spec vs APOGEE DR19 vs model.

Three side-by-side panels of the [M/H] vs [α/M] (or [α/Fe] for GSP-Spec)
plane, in this order:
  - left:   Gaia DR3 GSP-Spec (RVS-derived chemistry)
  - middle: APOGEE DR19 (ground-truth high-resolution spectroscopy)
  - right:  ArqueoGal Pipeline 1 v1.1 prediction

All three panels are restricted to the Stream-1 Tier-1 hold-out cohort
(val + test split, seed=0). Stars without a GSP-Spec entry drop out of
the left panel only; the right two panels keep the full Tier-1 holdout.

Slide-friendly 16:6 layout.
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
GSPSPEC_S1 = REPO / "data/interim/stream1_gaia_dr3_raw.parquet"

MH_LIM = (-1.6, 0.55)
AM_LIM = (-0.10, 0.45)
HIST_BINS = 80


def _load_holdout() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "mh_apogee", "alpha_m_apogee",
             "fe_h_apogee", "teff_apogee", "b_deg"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)
    return df.loc[df["release_tier"] == 1].reset_index(drop=True)


def _load_gspspec(source_ids: np.ndarray) -> pd.DataFrame:
    gsp = pd.read_parquet(
        GSPSPEC_S1, columns=["source_id", "mh_gspspec", "alphafe_gspspec"]
    ).drop_duplicates("source_id")
    gsp = gsp.dropna(subset=["mh_gspspec", "alphafe_gspspec"])
    return gsp.loc[gsp["source_id"].isin(source_ids)].reset_index(drop=True)


_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _hist2d(ax, x: np.ndarray, y: np.ndarray, *, title: str,
            xlabel: str, ylabel: str) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    ax.hist2d(
        x[ok], y[ok],
        bins=HIST_BINS,
        range=[MH_LIM, AM_LIM],
        cmin=2,
        cmap="viridis",
    )
    ax.set_xlim(MH_LIM)
    ax.set_ylim(AM_LIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (n = {int(ok.sum()):,})", **_TITLE_KW)


def main() -> int:
    apply_style()
    df = _load_holdout()
    if df.empty:
        print("[Y33] no Tier-1 holdout rows, aborting")
        return 1
    gsp = _load_gspspec(df["source_id"].to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Gaia DR3 GSP-Spec.
    _hist2d(
        axes[0],
        gsp["mh_gspspec"].to_numpy(),
        gsp["alphafe_gspspec"].to_numpy(),
        title="Gaia DR3 GSP-Spec",
        xlabel=r"[M/H] (dex)",
        ylabel=r"[$\alpha$/Fe] (dex)",
    )
    # Panel 2: APOGEE DR19 truth.
    _hist2d(
        axes[1],
        df["mh_apogee"].to_numpy(),
        df["alpha_m_apogee"].to_numpy(),
        title="APOGEE DR19 (truth)",
        xlabel=r"[M/H] (dex)",
        ylabel=r"[$\alpha$/M] (dex)",
    )
    # Panel 3: ArqueoGal v1.1 prediction.
    _hist2d(
        axes[2],
        df["mh_pred"].to_numpy(),
        df["alpha_m_pred"].to_numpy(),
        title="ArqueoGal v1.1 prediction",
        xlabel=r"[M/H] (dex)",
        ylabel=r"[$\alpha$/M] (dex)",
    )

    for ax in axes:
        ax.grid(True, alpha=0.20)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.78, bottom=0.13, wspace=0.30)
    headline(
        fig,
        "Stream 1, Tier 1 holdout: chemistry plane across three sources",
        f"GSP-Spec n = {len(gsp):,};  APOGEE / ArqueoGal n = {len(df):,}.",
        top=0.78,
    )
    save(fig, "Y33_three_way_chemistry_comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
