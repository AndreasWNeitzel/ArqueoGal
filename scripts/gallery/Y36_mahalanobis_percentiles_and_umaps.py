"""Y36: Stream-1 holdout, dual-Mahalanobis percentiles + UMAPs + Av lines.

Layout (1 row x 4 columns) on a slide-friendly 18:6 figure:

  1. label-Mahalanobis percentile vs XP-Mahalanobis percentile, both axes
     spanning the full [0, 1] range.  p = 0.99 cutoff lines marked.
  2. feature-space UMAP, fit on the encoder's full input vector (108 BP/RP
     normalised Hermite coefficients + the 23 auxiliary columns from
     DEFAULT_AUX_COLS: Gaia photometry, parallax + Bailer-Jones distance
     triple, dereddened IR photometry, and the multi-column extinction
     priors); each point coloured by XP-Mahalanobis percentile.  cmap = viridis.
  3. label-space UMAP, fit on the predicted (Teff, logg, [M/H], [alpha/M],
     [Mg/H]) tuple; each point coloured by label-Mahalanobis percentile.
     cmap = magma so the two UMAPs read as different views at a glance.
  4. dual-axis curve of the two Mahalanobis percentiles vs Av (line-of-
     sight extinction); blue axis = XP percentile, vermillion axis = label
     percentile.
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

from _presentation import OKABE_ITO, PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

UMAP_N = 15000
RNG_SEED = 0
N_COEF = 54

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _load_holdout() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "ood_mahalanobis_score", "label_mahalanobis_score",
        "ood_mahalanobis_percentile", "label_mahalanobis_percentile",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "fe_h_apogee", "teff_apogee", "b_deg", "av_los"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    return df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)


def _load_xp_features(source_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 108-D XP normalised-coefficient block, aligned to source_ids.

    BP norm 1..54 then RP norm 1..54 (same order the encoder sees in the
    XP-only sub-vector). No aux features are included; the feature-space
    UMAP is intentionally fit on the spectroscopic block alone.
    """
    bp = [f"bp_coef_norm_{i}" for i in range(1, N_COEF + 1)]
    rp = [f"rp_coef_norm_{i}" for i in range(1, N_COEF + 1)]
    cols = ["source_id"] + bp + rp
    df = pd.read_parquet(FEAT_S1, columns=cols).drop_duplicates("source_id")
    df = df.loc[df["source_id"].isin(source_ids)].reset_index(drop=True)
    X = df[bp + rp].to_numpy(dtype=np.float32)
    return X, df["source_id"].to_numpy()


def _impute_and_standardise(X: np.ndarray) -> np.ndarray:
    """Median-impute NaNs column-wise, then z-standardise."""
    X = X.astype(np.float32, copy=True)
    if not np.isfinite(X).all():
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for j in range(X.shape[1]):
            col = X[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = med[j]
                X[:, j] = col
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _umap_embed(X: np.ndarray) -> np.ndarray:
    X = _impute_and_standardise(X)
    try:
        import umap  # type: ignore
    except ImportError:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=RNG_SEED).fit_transform(X)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.10,
        random_state=RNG_SEED, metric="euclidean",
    )
    return reducer.fit_transform(X)


def main() -> int:
    apply_style()
    df = _load_holdout()
    if df.empty:
        print("[Y36] no holdout rows")
        return 1

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    # --- panel 1: dual-percentile scatter, restricted to the [0.5, 1.0]
    # tail (which is the only region the cutoff lines see) ---
    ax = axes[0]
    p_label = df["label_mahalanobis_percentile"].to_numpy()
    p_feat = df["ood_mahalanobis_percentile"].to_numpy()
    ok = np.isfinite(p_label) & np.isfinite(p_feat)
    ax.scatter(
        p_feat[ok], p_label[ok],
        s=2.0, alpha=0.10, color=OKABE_ITO[0],
        edgecolors="none", rasterized=True,
    )
    ax.axvline(0.99, color=PALETTE["tier3"], lw=1.4, ls="--",
               label="XP p = 99 (T3 cutoff)")
    ax.axhline(0.99, color=PALETTE["tier2"], lw=1.4, ls="--",
               label="label p = 99 (T2 cutoff)")
    ax.set_xlim(0.50, 1.0)
    ax.set_ylim(0.50, 1.0)
    ax.set_xlabel("XP-Mahalanobis percentile")
    ax.set_ylabel("label-Mahalanobis percentile")
    ax.set_title("Dual-Mahalanobis percentile cuts", **_TITLE_KW)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25, which="both")

    # Subsample for UMAP speed
    rng = np.random.default_rng(RNG_SEED)
    if len(df) > UMAP_N:
        idx = rng.choice(len(df), size=UMAP_N, replace=False)
        sub = df.iloc[idx].reset_index(drop=True)
    else:
        sub = df

    # --- panel 2: feature-space UMAP, XP normalised coefs only,
    # color = XP-Mahalanobis percentile ---
    ax = axes[1]
    X_feat, sid = _load_xp_features(sub["source_id"].to_numpy())
    sub_aligned = sub.set_index("source_id").loc[sid].reset_index()
    emb_feat = _umap_embed(X_feat)
    sc = ax.scatter(
        emb_feat[:, 0], emb_feat[:, 1],
        c=sub_aligned["ood_mahalanobis_percentile"].to_numpy(),
        s=3.0, alpha=0.65, cmap="plasma", vmin=0.0, vmax=1.0,
        edgecolors="none", rasterized=True,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("XP-Mahalanobis percentile", fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Feature-space UMAP  (108-D XP normalised coefs)",
                 **_TITLE_KW)
    ax.grid(True, alpha=0.20)

    # --- panel 3: label-space UMAP, color = label percentile ---
    ax = axes[2]
    X_lab = sub_aligned[
        ["teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred"]
    ].to_numpy(dtype=np.float32)
    emb_lab = _umap_embed(X_lab)
    sc = ax.scatter(
        emb_lab[:, 0], emb_lab[:, 1],
        c=sub_aligned["label_mahalanobis_percentile"].to_numpy(),
        s=3.0, alpha=0.65, cmap="cividis", vmin=0.0, vmax=1.0,
        edgecolors="none", rasterized=True,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("label-Mahalanobis percentile", fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Label-space UMAP  (5 predicted labels)", **_TITLE_KW)
    ax.grid(True, alpha=0.20)

    # --- panel 4: dual-axis Mahalanobis percentile vs Av ---
    ax = axes[3]
    av = df["av_los"].to_numpy()
    feat_pct = df["ood_mahalanobis_percentile"].to_numpy()
    lab_pct = df["label_mahalanobis_percentile"].to_numpy()
    ok = np.isfinite(av) & np.isfinite(feat_pct) & np.isfinite(lab_pct)
    av = av[ok]; feat_pct = feat_pct[ok]; lab_pct = lab_pct[ok]

    av_bins = np.linspace(np.nanpercentile(av, 1), np.nanpercentile(av, 99), 25)
    bin_id = np.digitize(av, av_bins) - 1
    centres, feat_med, feat_lo, feat_hi = [], [], [], []
    lab_med, lab_lo, lab_hi = [], [], []
    for k in range(len(av_bins) - 1):
        m = bin_id == k
        if m.sum() < 30:
            continue
        centres.append(0.5 * (av_bins[k] + av_bins[k + 1]))
        feat_med.append(np.nanmedian(feat_pct[m]))
        feat_lo.append(np.nanpercentile(feat_pct[m], 16))
        feat_hi.append(np.nanpercentile(feat_pct[m], 84))
        lab_med.append(np.nanmedian(lab_pct[m]))
        lab_lo.append(np.nanpercentile(lab_pct[m], 16))
        lab_hi.append(np.nanpercentile(lab_pct[m], 84))
    centres = np.asarray(centres)
    feat_med = np.asarray(feat_med); feat_lo = np.asarray(feat_lo); feat_hi = np.asarray(feat_hi)
    lab_med = np.asarray(lab_med); lab_lo = np.asarray(lab_lo); lab_hi = np.asarray(lab_hi)

    feat_color = OKABE_ITO[0]
    lab_color = OKABE_ITO[1]
    ax.fill_between(centres, feat_lo, feat_hi, color=feat_color, alpha=0.18)
    ln1, = ax.plot(centres, feat_med, color=feat_color, lw=2.0,
                   label="XP-Mahalanobis percentile")
    ax.set_xlabel(r"$A_V$ (mag, line of sight)")
    ax.set_ylabel("XP-Mahalanobis percentile", color=feat_color)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="y", colors=feat_color)
    ax.spines["left"].set_color(feat_color)

    ax2 = ax.twinx()
    ax2.fill_between(centres, lab_lo, lab_hi, color=lab_color, alpha=0.18)
    ln2, = ax2.plot(centres, lab_med, color=lab_color, lw=2.0,
                    label="label-Mahalanobis percentile")
    ax2.set_ylabel("label-Mahalanobis percentile", color=lab_color)
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis="y", colors=lab_color)
    ax2.spines["right"].set_color(lab_color)

    ax.set_title("Mahalanobis percentile vs extinction", **_TITLE_KW)
    ax.legend(handles=[ln1, ln2], loc="upper left", fontsize=9.5, frameon=False)
    ax.grid(True, alpha=0.20)

    fig.subplots_adjust(left=0.05, right=0.96, top=0.78, bottom=0.13, wspace=0.45)
    headline(
        fig,
        "Stream 1 holdout: dual-Mahalanobis cuts, UMAPs, and extinction trend",
        f"n = {len(df):,};  UMAP fit on n = {len(sub):,} subsample;  cutoffs at p = 99.",
        top=0.78,
    )
    save(fig, "Y36_mahalanobis_percentiles_and_umaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
