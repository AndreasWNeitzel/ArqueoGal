"""Y11: Three-way chemistry-plane comparison, single talk slide.

Cohort: Stream-1 Tier-1 held-out ∩ GSP-Spec finite ∩ APOGEE truth finite.
Three side-by-side panels of [α/Fe] (or [α/M]) vs [M/H] under three
independent labellings of the same physical quantity.

Use this slide to make the case that the XP→ML predictions look like the
spectroscopic truth, while GSP-Spec on the same stars is noisier.
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


def _load() -> pd.DataFrame:
    pcols = [
        "source_id",
        "mh_pred",
        "alpha_m_pred",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    pred = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = [
        "source_id",
        "ra_deg",
        "dec_deg",
        "fe_h_apogee",
        "teff_apogee",
        "b_deg",
        "mh_apogee",
        "alpha_m_apogee",
    ]
    feat = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    ho = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)

    gsp = pd.read_parquet(GSPSPEC_S1, columns=["source_id", "mh_gspspec", "alphafe_gspspec"])
    gsp = gsp.dropna(subset=["mh_gspspec", "alphafe_gspspec"])
    df = df.merge(gsp, on="source_id", how="inner")
    df = df.dropna(subset=["mh_apogee", "alpha_m_apogee"]).reset_index(drop=True)
    return df


def _draw(ax, x, y, *, color_title, x_label, y_label, title, n):
    ok = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(
        x[ok],
        y[ok],
        gridsize=70,
        extent=(-1.6, 0.55, -0.10, 0.42),
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(-0.10, 0.42)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, color=color_title)
    ax.axhline(0.15, color=PALETTE["accent"], ls="--", lw=1.4, alpha=0.85)
    ax.text(
        0.02,
        0.97,
        f"n = {n:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["mist"]),
    )
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\log_{10}$ N", fontsize=10)


def main() -> int:
    apply_style()
    df = _load()
    n = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8.5))
    plt.subplots_adjust(wspace=0.32, left=0.05, right=0.97, bottom=0.10)
    _draw(
        axes[0],
        df["mh_pred"].to_numpy(),
        df["alpha_m_pred"].to_numpy(),
        color_title=PALETTE["ours"],
        x_label="[M/H]  (dex)",
        y_label=r"[$\alpha$/M]  (dex)",
        title="OURS, XP → MLP",
        n=n,
    )
    _draw(
        axes[1],
        df["mh_gspspec"].to_numpy(),
        df["alphafe_gspspec"].to_numpy(),
        color_title=PALETTE["gspspec"],
        x_label="[M/H]  (dex)",
        y_label=r"[$\alpha$/Fe]  (dex)",
        title="GAIA DR3 GSP-Spec, RVS",
        n=n,
    )
    _draw(
        axes[2],
        df["mh_apogee"].to_numpy(),
        df["alpha_m_apogee"].to_numpy(),
        color_title=PALETTE["apogee"],
        x_label="[M/H]  (dex)",
        y_label=r"[$\alpha$/M]  (dex)",
        title="APOGEE DR19, spectroscopic truth",
        n=n,
    )

    headline(
        fig,
        "Three labellings of the same stars",
        "Stream 1 Tier 1 held-out ∩ GSP-Spec finite ∩ APOGEE finite.  Same n in all three panels.",
        top=0.82,
    )
    save(fig, "Y11_three_way_chemistry_headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
