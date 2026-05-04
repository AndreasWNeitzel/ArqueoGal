"""Y25: UMAP of the XP feature space coloured by stellar labels.

Project the XP coefficient block (113 features) of a 30k subsample of
Stream-1 Tier-1 held-out stars into 2-D via UMAP. Four panels, same
embedding, different colour:

  Teff truth  /  log g truth  /  [M/H] truth  /  [α/M] truth

The point: smooth gradients in label space across the embedding mean the
encoder's input geometry already organises stars by parameter, before the
supervised head ever runs.

Cached embedding lives in ``data/processed/y25_umap_xp_holdout.parquet``
so re-runs are seconds, not minutes.
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
from _y_holdout import load_holdout  # noqa: E402

CACHE = REPO / "data/processed/y25_umap_xp_holdout.parquet"
N_TARGET = 30_000


def _build_or_load_embedding() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)

    import umap

    df = load_holdout()
    rng = np.random.default_rng(0)
    if len(df) > N_TARGET:
        idx = rng.choice(len(df), size=N_TARGET, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    feat_path = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    bp_cols = [f"bp_coef_{i}" for i in range(55)]
    rp_cols = [f"rp_coef_{i}" for i in range(55)]
    feat = pd.read_parquet(feat_path, columns=["source_id", *bp_cols, *rp_cols]).drop_duplicates(
        "source_id"
    )
    sub = df.merge(feat, on="source_id", how="inner")

    X = sub[bp_cols + rp_cols].to_numpy(dtype=np.float32)
    # Per-star standardisation since raw flux dominates the L2.
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)

    print(f"[Y25] running UMAP on {len(sub):,} stars × {X.shape[1]} features")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=0,
        verbose=False,
    )
    emb = reducer.fit_transform(X)

    out = sub[["source_id", "teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"]].copy()
    out["umap_x"] = emb[:, 0].astype(np.float32)
    out["umap_y"] = emb[:, 1].astype(np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


def _panel(ax, df, color_col, *, label, cmap, vlim=None):
    c = df[color_col].to_numpy()
    ok = np.isfinite(c)
    if vlim is None:
        vmin = float(np.nanpercentile(c, 1.0))
        vmax = float(np.nanpercentile(c, 99.0))
    else:
        vmin, vmax = vlim
    sc = ax.scatter(
        df["umap_x"].to_numpy()[ok],
        df["umap_y"].to_numpy()[ok],
        c=c[ok],
        cmap=cmap,
        s=2.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        edgecolor="none",
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(label, color=PALETTE["navy"])
    ax.set_aspect("equal")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=10)


def main() -> int:
    apply_style()
    df = _build_or_load_embedding()
    n = len(df)

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    plt.subplots_adjust(wspace=0.30, hspace=0.30, top=0.88, left=0.06, right=0.97, bottom=0.06)
    _panel(axes[0, 0], df, "teff_apogee", label=r"$T_{\rm eff}$  (K)", cmap="plasma_r")
    _panel(axes[0, 1], df, "logg_apogee", label=r"$\log g$  (dex)", cmap="viridis")
    _panel(axes[1, 0], df, "mh_apogee", label="[M/H]  (dex)", cmap="cividis")
    _panel(axes[1, 1], df, "alpha_m_apogee", label=r"[$\alpha$/M]  (dex)", cmap="magma")

    headline(
        fig,
        "UMAP of the XP feature space",
        f"Stream 1 Tier 1 held-out subsample, n = {n:,}.  Same 2-D embedding in all "
        "four panels; colour shows APOGEE truth labels.  Smooth gradients = the "
        "encoder's input geometry already organises stars by parameter.",
        top=0.88,
    )
    save(fig, "Y25_umap_label_colored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
