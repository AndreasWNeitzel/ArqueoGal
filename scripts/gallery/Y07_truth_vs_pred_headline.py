"""Y07: Truth-vs-prediction headline grid (Stream 1, Tier 1, held-out).

Five panels, one per label, with RMSE in a bold annotation box. Designed as
a single talk slide that summarises model performance.
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

LABEL_SPECS = [
    {
        "key": "teff",
        "name": r"$T_{\rm eff}$",
        "unit": "K",
        "extent": (3800, 5800),
        "rmse_unit": "K",
    },
    {"key": "logg", "name": r"$\log g$", "unit": "dex", "extent": (0.5, 3.7), "rmse_unit": "dex"},
    {"key": "mh", "name": "[M/H]", "unit": "dex", "extent": (-2.2, 0.6), "rmse_unit": "dex"},
    {
        "key": "alpha_m",
        "name": r"[$\alpha$/M]",
        "unit": "dex",
        "extent": (-0.10, 0.45),
        "rmse_unit": "dex",
    },
    {"key": "mg_h", "name": "[Mg/H]", "unit": "dex", "extent": (-2.0, 0.6), "rmse_unit": "dex"},
]


def _load() -> pd.DataFrame:
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
    pred = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = [
        "source_id",
        "fe_h_apogee",
        "teff_apogee",
        "b_deg",
        "logg_apogee",
        "mh_apogee",
        "alpha_m_apogee",
        "mg_h_apogee",
    ]
    feat = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    ho = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    return df


def main() -> int:
    apply_style()
    df = _load()
    n = len(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.42, hspace=0.40, top=0.91, bottom=0.06, left=0.05, right=0.97)
    axes = axes.ravel()
    summary_rows: list[str] = []
    for ax, spec in zip(axes[:5], LABEL_SPECS):
        key = spec["key"]
        truth = df[f"{key}_apogee"].to_numpy()
        pred = df[f"{key}_pred"].to_numpy()
        ok = np.isfinite(truth) & np.isfinite(pred)
        truth, pred = truth[ok], pred[ok]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        bias = float(np.mean(pred - truth))

        hb = ax.hexbin(
            truth,
            pred,
            gridsize=55,
            extent=(*spec["extent"], *spec["extent"]),
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        # Identity line.
        lo, hi = spec["extent"]
        ax.plot([lo, hi], [lo, hi], color=PALETTE["accent"], lw=2.0, ls="--", label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"truth {spec['name']} ({spec['unit']})")
        ax.set_ylabel(f"prediction {spec['name']} ({spec['unit']})")
        ax.set_title(spec["name"], color=PALETTE["navy"])

        # RMSE annotation.
        ax.text(
            0.04,
            0.96,
            f"RMSE = {rmse:.3g} {spec['rmse_unit']}\nbias = {bias:+.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["mist"]),
        )
        cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.06)
        cb.set_label(r"$\log_{10}$ N", fontsize=9)
        cb.ax.tick_params(labelsize=9)

        summary_rows.append(
            f"{spec['name']:<12s}  RMSE = {rmse:>7.3g} {spec['rmse_unit']:<3s}"
            f"   bias = {bias:+8.3g}"
        )

    # Sixth panel: summary table replacing the empty quadrant.
    ax = axes[5]
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "Per-label held-out summary\n"
        + "─" * 46
        + "\n"
        + "\n".join(summary_rows)
        + f"\n\nn = {n:,} stars\nStream 1 Tier 1, val + test, seed = 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=12,
        color=PALETTE["ink"],
    )

    headline(
        fig,
        "Truth vs prediction, Stream 1 Tier 1 held-out",
        f"All five labels.  n = {n:,} stars.  Diagonal is the ideal line.",
        top=0.90,
    )
    save(fig, "Y07_truth_vs_pred_headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
