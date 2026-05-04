"""Y08: Chemistry plane headline, disc α-bimodality recovered from XP.

Single big panel of [α/M] vs [M/H] on Stream 1 Tier 1 held-out, alongside the
matched APOGEE truth. The point is to show that the model recovers the high-α /
low-α disc bimodality from XP coefficients alone.
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
    fcols = ["source_id", "fe_h_apogee", "teff_apogee", "b_deg", "mh_apogee", "alpha_m_apogee"]
    feat = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    ho = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)
    return df.loc[df["release_tier"] == 1].reset_index(drop=True)


def _draw(ax, x, y, *, title, n):
    ok = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(
        x[ok],
        y[ok],
        gridsize=80,
        extent=(-1.6, 0.55, -0.05, 0.42),
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    ax.set_xlabel("[M/H]  (dex)")
    ax.set_ylabel(r"[$\alpha$/M]  (dex)")
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(-0.05, 0.42)
    ax.set_title(title, color=PALETTE["navy"])
    # High-α / low-α visual divider, the canonical [α/M] = 0.15 dex line.
    ax.axhline(
        0.15,
        color=PALETTE["accent"],
        ls="--",
        lw=1.6,
        alpha=0.8,
        label=r"high-$\alpha$ / low-$\alpha$ divider",
    )
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.97,
        f"n = {n:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
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
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _draw(
        axes[0],
        df["mh_pred"].to_numpy(),
        df["alpha_m_pred"].to_numpy(),
        title="OURS, XP→MLP predictions",
        n=n,
    )
    _draw(
        axes[1],
        df["mh_apogee"].to_numpy(),
        df["alpha_m_apogee"].to_numpy(),
        title="APOGEE DR19, spectroscopic truth",
        n=n,
    )
    headline(
        fig,
        r"The disc $\alpha$-bimodality, recovered from Gaia XP alone",
        "Stream 1 Tier 1 held-out.  Two distinct sequences are visible in both panels.",
        top=0.84,
    )
    save(fig, "Y08_chemistry_plane_headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
