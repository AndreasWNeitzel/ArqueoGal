"""C2: Train / val / test split predictions on Stream 1 (Kiel-bounded RGB pool).

Renders Kiel diagram + chemistry plane hex-density panels separately for
the 70/15/15 stratified train/val/test partitions of Stream 1, using the
exact same split logic the trainer applies (``stratified_split_ids`` with
the canonical ``split_seed=0`` and quantile bins on (Fe/H, Teff, |b|)).

Quick visual sanity check on the model BEFORE running inference on
Stream 2 / Stream 3: if the predicted chemistry plane on the test split
already shows no bimodality, the encoder representation is the limit; if
the test split looks the same as train, there is no leakage.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (truth labels +
  the columns the stratified split keys on).
- data/processed/pipeline1_predictions_stream1.parquet (regressor output
  for every Stream-1 source).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig

from arqueogal.xp_abundances.main.data import stratified_split_ids

KIEL_GRID = 70
CHEM_GRID = 80
KIEL_EXTENT = (3500, 6500, 0.5, 4.0)
CHEM_EXTENT = (-2.5, 0.6, -0.20, 0.55)
SPLIT_SEED = 0
FRACS = (0.70, 0.15, 0.15)


def main() -> int:
    apply_style()

    feat_path = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    pred_path = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
    if not feat_path.exists() or not pred_path.exists():
        print(
            f"Error: required inputs missing.\n  features: {feat_path}\n  predictions: {pred_path}"
        )
        return 1

    # Reproduce the trainer's stratified split exactly.
    feat_strat = pd.read_parquet(
        feat_path,
        columns=[
            "source_id",
            "fe_h_apogee",
            "teff_apogee",
            "b_deg",
            "logg_apogee",
            "mh_apogee",
            "alpha_m_apogee",
            "mg_h_apogee",
        ],
    )
    feat_strat = feat_strat.drop_duplicates(subset="source_id", keep="first")
    splits = stratified_split_ids(feat_strat, fracs=FRACS, seed=SPLIT_SEED)

    # Predictions: one row per source_id (assume training-time dedup is the
    # same as we use here — first occurrence per source_id).
    preds = pd.read_parquet(
        pred_path,
        columns=["source_id", "teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred"],
    )
    preds = preds.drop_duplicates(subset="source_id", keep="first")

    # Inner-join and split.
    merged = preds.merge(feat_strat, on="source_id", how="inner")
    print(f"[C2] merged Stream-1 cohort: n={len(merged):,}")
    print(
        f"[C2] split sizes: train={len(splits['train']):,}, "
        f"val={len(splits['val']):,}, test={len(splits['test']):,}"
    )

    split_dfs: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        ids = splits[name]
        sub = merged[merged["source_id"].isin(set(ids.tolist()))]
        split_dfs[name] = sub
        print(f"[C2] {name}: matched {len(sub):,} predictions / {len(ids):,} ids")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    splits_order = [
        ("train", "#1f77b4"),
        ("val", "#ff7f0e"),
        ("test", "#2ca02c"),
    ]

    # Row 0: predicted Kiel hex density per split.
    for col, (name, color) in enumerate(splits_order):
        ax = axes[0, col]
        df = split_dfs[name]
        if len(df) > 0:
            hb = ax.hexbin(
                df["teff_pred"],
                df["logg_pred"],
                gridsize=KIEL_GRID,
                cmap="viridis",
                mincnt=1,
                bins="log",
                extent=KIEL_EXTENT,
            )
            plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
        ax.set_xlabel(r"$T_\mathrm{eff,\, pred}$ (K)")
        ax.set_ylabel(r"$\log g_{\rm pred}$ (dex)")
        ax.set_title(f"{name} - predicted Kiel (n={len(df):,})", color=color)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

    # Row 1: predicted chemistry plane per split.
    for col, (name, color) in enumerate(splits_order):
        ax = axes[1, col]
        df = split_dfs[name]
        if len(df) > 0:
            hb = ax.hexbin(
                df["mh_pred"],
                df["alpha_m_pred"],
                gridsize=CHEM_GRID,
                cmap="viridis",
                mincnt=1,
                bins="log",
                extent=CHEM_EXTENT,
            )
            plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
        ax.axhline(
            0.15,
            color="white",
            lw=0.6,
            ls=":",
            alpha=0.7,
            label=r"$[\alpha/{\rm M}]=0.15$ (soft-ARI threshold)",
        )
        ax.set_xlabel(r"$[{\rm M/H}]_{\rm pred}$ (dex)")
        ax.set_ylabel(r"$[\alpha/{\rm M}]_{\rm pred}$ (dex)")
        ax.set_title(f"{name} - predicted chemistry (n={len(df):,})", color=color)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "C2 - Stream 1 predictions split by 70/15/15 stratified "
        f"train/val/test partition (seed {SPLIT_SEED})",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = REPO / "reports/gallery/C_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "C2_train_val_test_predictions", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
