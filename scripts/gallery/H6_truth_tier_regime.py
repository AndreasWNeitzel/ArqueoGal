"""H6: Stream 1 truth-vs-prediction comparison per release tier (held-out only).

Mirrors the H5 layout but adds the APOGEE truth panels next to the
predicted ones, restricted to held-out (val + test) stars so the model
has not seen these in training. Lets us verify per tier how cleanly the
encoder predicts truth on stars it never optimised against.

Layout (4 rows x 3 cols), rows correspond to (truth Kiel, predicted
Kiel, truth chemistry, predicted chemistry); columns are (T1, T2, T3).
A fifth row (split into two panels: Teff in K vs dex elements) carries
the per-tier predicted sigma.

Held-out partition is reproduced via the trainer's
``stratified_split_ids`` with ``seed=0`` and the canonical
``fracs=(0.70, 0.15, 0.15)`` defaults — so a star labelled "val" or
"test" here matches the trainer's "val" or "test" set exactly. Train
partition stars are excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig

from arqueogal.xp_abundances.main.data import stratified_split_ids

OUT = REPO / "reports/gallery/H_hybrid_release"
ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
ELEMENT_LABELS = ("Teff", r"$\log g$", "[M/H]", r"[$\alpha$/M]", "[Mg/H]")

SPLIT_SEED = 0
FRACS = (0.70, 0.15, 0.15)


def main() -> int:
    apply_style()
    print("[H6] Loading Stream 1 predictions + truth")

    pred_path = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
    truth_path = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not pred_path.exists() or not truth_path.exists():
        print(f"Error: required input missing\n  pred: {pred_path}\n  truth: {truth_path}")
        return 1

    import pyarrow.parquet as _pq

    avail = {f.name for f in _pq.ParquetFile(pred_path).schema_arrow}
    needed = [
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
    ]
    if "release_tier" in avail:
        needed.append("release_tier")
    for opt in ("ood_joint_flag", "label_extrapolation_flag"):  # tier drivers
        if opt in avail:
            needed.append(opt)
    pred = pd.read_parquet(pred_path, columns=needed)
    pred = pred.drop_duplicates("source_id", keep="first")
    pred["rmse_teff"] = pred["teff_sigma"].clip(10, 500)
    pred["rmse_logg"] = pred["logg_sigma"].clip(0.05, 2.0)
    pred["rmse_mh"] = pred["mh_sigma"].clip(0.01, 1.0)
    pred["rmse_alpha_m"] = pred["alpha_m_sigma"].clip(0.01, 1.0)
    pred["rmse_mg_h"] = pred["mg_h_sigma"].clip(0.01, 1.0)
    if "release_tier" in pred.columns:
        pred["tier"] = pred["release_tier"].astype(int)
    else:
        pred["tier"] = np.where(
            pred["rmse_teff"] < 100, 1, np.where(pred["rmse_teff"] < 200, 2, 3)
        ).astype(int)

    truth = pd.read_parquet(
        truth_path,
        columns=[
            "source_id",
            "fe_h_apogee",
            "teff_apogee",
            "logg_apogee",
            "b_deg",
            "mh_apogee",
            "alpha_m_apogee",
            "mg_h_apogee",
        ],
    )
    truth = truth.drop_duplicates("source_id", keep="first")
    df = pred.merge(truth, on="source_id", how="inner")
    print(f"[H6] joined cohort n={len(df):,}")

    # Reproduce the trainer's stratified split exactly so we can drop
    # train-partition rows.
    splits = stratified_split_ids(df, fracs=FRACS, seed=SPLIT_SEED)
    train_ids = set(splits["train"].tolist())
    df = df[~df["source_id"].isin(train_ids)].reset_index(drop=True)
    print(
        f"[H6] held-out (val+test) cohort n={len(df):,}; "
        f"tier counts {df['tier'].value_counts().to_dict()}"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    KIEL_GRID = 70
    CHEM_GRID = 70
    KIEL_EXTENT = (3500, 6500, 0.5, 4.0)
    CHEM_EXTENT = (-2.5, 0.6, -0.20, 0.55)
    TIERS = [(1, "T1 (science)"), (2, "T2 (caveat)"), (3, "T3 (do-not-trust)")]

    fig = plt.figure(figsize=(20, 19))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.30)

    # Row 0: TRUTH Kiel hex per tier.
    for col, (tier_val, tier_label) in enumerate(TIERS):
        ax = fig.add_subplot(gs[0, col])
        subset = df[df["tier"] == tier_val]
        n_in = int(len(subset))
        m = np.isfinite(subset["teff_apogee"].values) & np.isfinite(subset["logg_apogee"].values)
        if int(m.sum()) > 0:
            sub = subset.loc[m]
            hb = ax.hexbin(
                sub["teff_apogee"],
                sub["logg_apogee"],
                gridsize=KIEL_GRID,
                cmap="viridis",
                mincnt=1,
                bins="log",
                extent=KIEL_EXTENT,
            )
            plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
        ax.set_xlabel(r"$T_\mathrm{eff,\, APOGEE}$ (K)")
        ax.set_ylabel(r"$\log g_\mathrm{APOGEE}$ (dex)")
        ax.set_title(f"{tier_label} - TRUTH Kiel (n={n_in:,})")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

    # Row 1: PREDICTED Kiel hex per tier.
    for col, (tier_val, tier_label) in enumerate(TIERS):
        ax = fig.add_subplot(gs[1, col])
        subset = df[df["tier"] == tier_val]
        n_in = int(len(subset))
        if n_in > 0:
            hb = ax.hexbin(
                subset["teff_pred"],
                subset["logg_pred"],
                gridsize=KIEL_GRID,
                cmap="viridis",
                mincnt=1,
                bins="log",
                extent=KIEL_EXTENT,
            )
            plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
        ax.set_xlabel(r"$T_\mathrm{eff,\, pred}$ (K)")
        ax.set_ylabel(r"$\log g_\mathrm{pred}$ (dex)")
        ax.set_title(f"{tier_label} - PREDICTED Kiel (n={n_in:,})")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

    # Row 2: TRUTH chemistry plane per tier.
    for col, (tier_val, tier_label) in enumerate(TIERS):
        ax = fig.add_subplot(gs[2, col])
        subset = df[df["tier"] == tier_val]
        n_in = int(len(subset))
        m = np.isfinite(subset["mh_apogee"].values) & np.isfinite(subset["alpha_m_apogee"].values)
        if int(m.sum()) > 0:
            sub = subset.loc[m]
            hb = ax.hexbin(
                sub["mh_apogee"],
                sub["alpha_m_apogee"],
                gridsize=CHEM_GRID,
                cmap="viridis",
                mincnt=1,
                bins="log",
                extent=CHEM_EXTENT,
            )
            plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
        ax.axhline(0.15, color="white", lw=0.6, ls=":", alpha=0.7)
        ax.set_xlabel(r"$[\mathrm{M/H}]_\mathrm{APOGEE}$ (dex)")
        ax.set_ylabel(r"$[\alpha/\mathrm{M}]_\mathrm{APOGEE}$ (dex)")
        ax.set_title(f"{tier_label} - TRUTH Chemistry (n={n_in:,})")
        ax.grid(alpha=0.3)

    # Row 3: PREDICTED chemistry plane per tier.
    for col, (tier_val, tier_label) in enumerate(TIERS):
        ax = fig.add_subplot(gs[3, col])
        subset = df[df["tier"] == tier_val]
        n_in = int(len(subset))
        if n_in > 0:
            hb = ax.hexbin(
                subset["mh_pred"],
                subset["alpha_m_pred"],
                gridsize=CHEM_GRID,
                cmap="viridis",
                mincnt=1,
                bins="log",
                extent=CHEM_EXTENT,
            )
            plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
        ax.axhline(0.15, color="white", lw=0.6, ls=":", alpha=0.7)
        ax.set_xlabel(r"$[\mathrm{M/H}]_\mathrm{pred}$ (dex)")
        ax.set_ylabel(r"$[\alpha/\mathrm{M}]_\mathrm{pred}$ (dex)")
        ax.set_title(f"{tier_label} - PREDICTED Chemistry (n={n_in:,})")
        ax.grid(alpha=0.3)

    # Per-tier σ bar plots dropped 2026-05-03 — σ-thresholds are no longer
    # the T2 driver, so per-tier mean-σ is no longer the load-bearing summary.
    # Suptitle dropped per same edit.
    save_fig(fig, OUT / "H6_truth_tier_regime_stream1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
