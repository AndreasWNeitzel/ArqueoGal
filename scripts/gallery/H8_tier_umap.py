"""H8: UMAP of features and labels, color-coded by release tier and gate flags.

Operates on Stream 1 held-out only (val+test, 30 %, the same canonical
split seed=0 used by H6/H7) so that the tier-vs-truth comparison stays
honest. The composite ``release_tier`` is computed from the Stream-1
prediction parquet via ``release.assign_release_tier`` since Stream 1
does not have its own hybrid release run on disk.

Diagnoses the tiering geometry by projecting two spaces simultaneously:

- **Feature UMAP** (top row): UMAP on the 140-column feature vector that the
  encoder actually receives at inference, post FeatureScaler. Shows the
  geometry the OOD Mahalanobis gate operates in.
- **Label UMAP** (middle row): UMAP on the 5-D predicted-label vector
  (Teff, log g, [M/H], [α/M], [Mg/H]) z-scored. Shows where the regressor
  *outputs* its predictions, irrespective of whether they are trustworthy.
- **Chemistry plane** (bottom row): physical (M/H, α/M) for direct
  astrophysical interpretation.

Each row is shown four times, once per coloring scheme:

  col 1  release_tier ∈ {1, 2, 3}            (composite row-max tier)
  col 2  ood_joint_flag (XP-Mahalanobis)     (the hard reject gate)
  col 3  σ_α > 0.05 dex (α/M σ-inflation)    (the prior-collapse gate)
  col 4  kin_ood_flag (kinematic-aux OOD)    (the aux-channel demotion;
         skipped on Stream 1 since the column is absent — panel will be
         all-grey)

The hallucinated diagonal at M/H ∈ [-1.5, -0.5], α/M ∈ [0.0, 0.1] is the
prior-collapse signature where APOGEE coverage is sparse. The σ_α gate
catches it because posterior σ wider than σ_train means CMI(spectrum;
label | aux) ≈ 0 nats; release.py:294-310 documents the empirical-Bayes
calibration. The Mahalanobis gate catches the most extreme outliers in
XP space.

This figure replaces the implicit assumption that the tiering is a black
box. Each gate's geometric coverage is now visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig  # noqa: E402

from arqueogal.data.frozen_stats import (  # noqa: E402
    apply_frozen_zscore,
    load_frozen_zscore_stats,
)
from arqueogal.xp_abundances.main.data import (  # noqa: E402
    FeatureLayout,
    FeatureScaler,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

OUT = REPO / "reports/gallery/H_hybrid_release"
CKPT_PATH = REPO / (
    "models/main/xp_abundances/20260501_1d71682_26312a4_ensemble_5label/"
    "member_seed0/xp_abundances_main_ensemble_5label_seed0_best.pt"
)
PREDICTIONS_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEATURES_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
FROZEN_STATS = REPO / "data/processed/pipeline1_features_stream1.provenance.json"

N_TARGET = 40_000  # cap for UMAP runtime; held-out split is ~97k stars
SEED = 42
SIGMA_ALPHA_THRESHOLD = 0.05  # dex; release.py:_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD

LABEL_COLS = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")


def _load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, FeatureScaler]:
    if not (PREDICTIONS_S1.exists() and FEATURES_S1.exists() and CKPT_PATH.exists()):
        raise FileNotFoundError(
            f"missing inputs.\n"
            f"  preds:    {PREDICTIONS_S1}\n"
            f"  features: {FEATURES_S1}\n"
            f"  ckpt:     {CKPT_PATH}"
        )

    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    fs_blob = blob["feature_scaler"]
    fs = FeatureScaler(
        mean=np.asarray(fs_blob["mean"], dtype=np.float32),
        scale=np.asarray(fs_blob["scale"], dtype=np.float32),
        feature_names=tuple(fs_blob["feature_names"]),
        log10_mask=np.asarray(fs_blob["log10_mask"], dtype=bool),
        apply_mask=np.asarray(fs_blob["apply_mask"], dtype=bool),
    )

    layout = FeatureLayout()
    feat_cols_z = list(layout.all_required_columns)
    # Stream-1 features parquet stores c0_z directly (already z-scored at
    # build time), so no on-the-fly z-scoring is required for the c0 scalars.
    pred_cols = [
        "source_id",
        "ood_joint_flag",
        "ood_mahalanobis_score",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "teff_pred",
        "logg_pred",
        "mh_pred",
        "alpha_m_pred",
        "mg_h_pred",
        "label_extrapolation_flag",
        "label_mahalanobis_score",  # T2 colour scale
        "ood_mahalanobis_percentile",  # XP percentile (informational)
        "label_mahalanobis_percentile",  # label percentile (informational)
    ]
    df_p = pd.read_parquet(PREDICTIONS_S1, columns=pred_cols)
    feat_extra = ["fe_h_apogee", "teff_apogee", "b_deg"]
    df_f = pd.read_parquet(FEATURES_S1, columns=["source_id", *feat_cols_z, *feat_extra])
    df_f = df_f.drop_duplicates(subset="source_id", keep="first")
    df = df_f.merge(df_p, on="source_id", how="inner")
    print(f"[H8] joined {len(df):,} stars (stream-1 features x predictions)")

    # Restrict to the held-out split (val + test, 30 %, seed=0). This matches
    # H6/H7's "non-train only" framing.
    split_ids = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split_ids["val"], split_ids["test"]])
    holdout_mask = np.isin(df["source_id"].to_numpy(), holdout_ids)
    df = df.loc[holdout_mask].reset_index(drop=True)
    print(f"[H8] restricted to held-out (val+test) only: n={len(df):,}")

    # Compose composite release_tier from the per-element gates that release.py
    # implements. The Stream-1 prediction parquet has the σ columns and
    # ood_joint_flag but no kin_ood_flag (kinematic OOD is only emitted on
    # Stream 2/3); assign_release_tier handles the missing column gracefully.
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    if "kin_ood_flag" not in df.columns:
        df["kin_ood_flag"] = False
    tier_counts = df["release_tier"].value_counts().to_dict()
    print(f"[H8] held-out release_tier distribution: {tier_counts}")

    rng = np.random.default_rng(SEED)
    if len(df) > N_TARGET:
        idx = rng.choice(len(df), size=N_TARGET, replace=False)
        idx = np.sort(idx)
        df = df.iloc[idx].reset_index(drop=True)
        print(
            f"[H8] random subsample to {len(df):,} stars (NOT stratified — preserves true tier mix)"
        )

    # Stream-1 parquet stores all 140 cols in the schema FeatureLayout expects,
    # so no on-the-fly z-scoring is required (frozen stats were applied at
    # feature-build time).
    _ = (load_frozen_zscore_stats, apply_frozen_zscore)  # kept import for B17 parity
    X_raw = np.column_stack([df[c].to_numpy(dtype=np.float32) for c in feat_cols_z])
    X_feat = fs.transform(X_raw)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    Y = np.column_stack([df[c].to_numpy(dtype=np.float32) for c in LABEL_COLS])
    Y_mu = np.nanmean(Y, axis=0)
    Y_sd = np.nanstd(Y, axis=0)
    Y_sd = np.where(Y_sd > 1e-6, Y_sd, 1.0)
    Y = (Y - Y_mu) / Y_sd
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return df, X_feat, Y, fs


def _run_umap(X: np.ndarray, *, name: str) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.05,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        verbose=False,
        low_memory=True,
    )
    print(f"[H8] running UMAP on {name}: shape={X.shape}")
    Z = reducer.fit_transform(X)
    return np.asarray(Z, dtype=np.float32)


def _scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    c,
    *,
    vmin=None,
    vmax=None,
    cmap=None,
    s: float = 1.5,
    alpha: float = 0.45,
    labels=None,
    edgecolor="none",
):
    if cmap is None:
        sc = ax.scatter(x, y, c=c, s=s, alpha=alpha, edgecolors=edgecolor, linewidths=0.0)
    else:
        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=0.0,
        )
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    return sc


def _color_arrays(df: pd.DataFrame) -> dict[str, dict]:
    """2026-05-03 redesign: tier coloring + continuous Mahalanobis distance.

    Col 1 = composite tier (categorical, T1/T2/T3).
    Col 2 = continuous label-Mahalanobis distance (T2 score).
    Col 3 = continuous XP-Mahalanobis distance (T3 score).
    """
    tier = df["release_tier"].to_numpy(dtype=np.int8)
    # Okabe-Ito tier mapping: green=T1, vermilion=T2, red-purple=T3.
    tier_color = np.where(tier == 1, "#009E73", np.where(tier == 2, "#D55E00", "#CC79A7"))
    n1 = int((tier == 1).sum())
    n2 = int((tier == 2).sum())
    n3 = int((tier == 3).sum())

    label_pctl = (
        df["label_mahalanobis_percentile"].to_numpy(dtype=np.float64)
        if "label_mahalanobis_percentile" in df.columns
        else np.full(len(df), np.nan)
    )
    xp_pctl = (
        df["ood_mahalanobis_percentile"].to_numpy(dtype=np.float64)
        if "ood_mahalanobis_percentile" in df.columns
        else np.full(len(df), np.nan)
    )
    return {
        "tier": {
            "kind": "categorical",
            "c": tier_color,
            "mask_red": tier > 1,
            "title": (f"composite release_tier  (T1 n={n1:,}, T2 n={n2:,}, T3 n={n3:,})"),
            "legend": [
                ("Tier 1 (science-grade)", "#009E73"),
                ("Tier 2 (label-Mahal)", "#D55E00"),
                ("Tier 3 (XP-Mahal / NaN)", "#CC79A7"),
            ],
        },
        "label_mahal": {
            "kind": "continuous",
            "c": label_pctl,
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": 1.0,
            "label": r"label-Mahalanobis percentile",
            "title": "T2 score: label-space Mahalanobis percentile",
        },
        "xp_mahal": {
            "kind": "continuous",
            "c": xp_pctl,
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": 1.0,
            "label": r"XP-Mahalanobis percentile",
            "title": "T3 score: feature-space Mahalanobis percentile",
        },
    }


def _legend_proxies(spec: list[tuple[str, str]]):
    return [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=c,
            markeredgecolor="none",
            markersize=6,
            label=lbl,
        )
        for lbl, c in spec
    ]


def _plot_scatter_panel(ax, x, y, info, *, xlabel, ylabel):
    if info.get("kind") == "continuous":
        c = info["c"]
        ok = np.isfinite(c)
        # Order high-c on top so OOD-ish points dominate the visual.
        order = np.argsort(c[ok])
        sc = ax.scatter(
            x[ok][order],
            y[ok][order],
            c=c[ok][order],
            cmap=info["cmap"],
            vmin=info["vmin"],
            vmax=info["vmax"],
            s=1.6,
            alpha=0.55,
            edgecolors="none",
            rasterized=True,
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(info["label"], fontsize=8)
    else:
        mask_red = info["mask_red"]
        grey_idx = np.where(~mask_red)[0]
        red_idx = np.where(mask_red)[0]
        ax.scatter(
            x[grey_idx],
            y[grey_idx],
            c=info["c"][grey_idx],
            s=1.2,
            alpha=0.30,
            edgecolors="none",
            linewidths=0.0,
        )
        ax.scatter(
            x[red_idx],
            y[red_idx],
            c=info["c"][red_idx],
            s=2.0,
            alpha=0.65,
            edgecolors="none",
            linewidths=0.0,
        )
        ax.legend(
            handles=_legend_proxies(info["legend"]),
            loc="upper right",
            fontsize=7,
            frameon=False,
            markerscale=0.9,
        )
    ax.set_title(info["title"], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main() -> int:
    apply_style()
    df, X_feat, Y, _fs = _load_data()
    Z_feat = _run_umap(X_feat, name="features (140-D, post FeatureScaler)")
    Z_lab = _run_umap(Y, name="labels (5-D, z-scored predictions)")
    print("[H8] UMAP done")

    colors = _color_arrays(df)
    col_keys = ["tier", "label_mahal", "xp_mahal"]  # T1+composite, T2 score, T3 score

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)

    mh = df["mh_pred"].to_numpy(dtype=np.float32)
    am = df["alpha_m_pred"].to_numpy(dtype=np.float32)

    for j, key in enumerate(col_keys):
        info = colors[key]
        ax = fig.add_subplot(gs[0, j])
        _plot_scatter_panel(
            ax,
            Z_feat[:, 0],
            Z_feat[:, 1],
            info,
            xlabel="UMAP-1 (features)",
            ylabel="UMAP-2 (features)",
        )
        if j == 0:
            ax.set_ylabel("UMAP-2 (features)\n[140-D post-scaling]")

        ax = fig.add_subplot(gs[1, j])
        _plot_scatter_panel(
            ax, Z_lab[:, 0], Z_lab[:, 1], info, xlabel="UMAP-1 (labels)", ylabel="UMAP-2 (labels)"
        )
        if j == 0:
            ax.set_ylabel("UMAP-2 (labels)\n[5-D z-scored predictions]")

        ax = fig.add_subplot(gs[2, j])
        _plot_scatter_panel(
            ax, mh, am, info, xlabel="[M/H]_pred  (dex)", ylabel=r"$[\alpha/M]_{\rm pred}$  (dex)"
        )
        if j == 0:
            ax.set_ylabel(r"$[\alpha/M]_{\rm pred}$  (dex)" + "\n[chemistry plane]")
        ax.set_xlim(-2.0, 0.6)
        ax.set_ylim(-0.1, 0.45)

    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "H8_tier_umap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
