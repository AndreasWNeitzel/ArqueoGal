"""Stage 12: Pipeline-1 feature matrix across all three streams.

Outputs:
- ``feature_distributions.png`` — aux distributions overlaid across S1 / S2 / S3
  per feature.
- ``aux_correlation_heatmap.png`` — Stream-1 aux correlation (training-set
  reference; the model was trained against this correlation structure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig, PALETTE

OUT = REPO / "reports/gallery/12_feature_matrix"

STREAMS = [
    ("Stream 1", REPO / "data/processed/pipeline1_features_stream1.parquet",
     PALETTE["apogee"]),
    ("Stream 2", REPO / "data/processed/pipeline1_features_stream2.parquet",
     "#9467bd"),
    ("Stream 3", REPO / "data/processed/pipeline1_features_stream3.parquet",
     PALETTE["andrae_volume"]),
]

AUX_CANDIDATES = [
    "parallax_corr", "parallax_error",
    "g_mag", "bp_rp", "bp_g", "g_rp",
    "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
    "r_med_photogeo", "av_edenhofer", "av_lallement", "av_sfd",
    "av_nbhd_median", "bp_c0_z", "rp_c0_z",
]


def _load_aux(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    full = pd.read_parquet(path).iloc[:0].columns
    cols = [c for c in AUX_CANDIDATES if c in full]
    return pd.read_parquet(path, columns=cols)


def main() -> None:
    apply_style()

    loaded = []
    for name, path, color in STREAMS:
        df = _load_aux(path)
        if df is not None:
            loaded.append((name, color, df))

    # Union of features across streams (ordered)
    feat_set = set()
    for _, _, df in loaded:
        feat_set.update(df.columns)
    feats = [c for c in AUX_CANDIDATES if c in feat_set]

    # Per-feature overlay panels
    n = len(feats)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.6 * nrow),
                              constrained_layout=True)
    flat = axes.flatten()
    for i, c in enumerate(feats):
        ax = flat[i]
        # Robust range from union of finite values
        all_v = []
        for _, _, df in loaded:
            if c in df.columns:
                all_v.append(df[c].dropna().to_numpy())
        if not all_v:
            ax.set_visible(False)
            continue
        cat = np.concatenate(all_v)
        if len(cat) < 10:
            ax.set_visible(False); continue
        lo, hi = np.nanpercentile(cat, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            ax.set_visible(False); continue
        bins = np.linspace(lo, hi, 41)
        for name, color, df in loaded:
            if c not in df.columns:
                continue
            v = df[c].dropna().to_numpy()
            if len(v) < 10:
                continue
            ax.hist(v, bins=bins, density=True, histtype="step",
                     color=color, lw=1.2, label=name)
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", frameon=True,
                      framealpha=0.95, facecolor="white", edgecolor="0.4")
    for j in range(n, len(flat)):
        flat[j].set_visible(False)
    fig.suptitle("Stage 12 — aux feature distributions across S1 / S2 / S3 (overlay)",
                  fontsize=11)
    save_fig(fig, OUT / "feature_distributions.png")

    # Aux correlation (Stream 1 only — this is the training reference)
    s1_df = next((d for n, c, d in loaded if n == "Stream 1"), None)
    if s1_df is None:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    cols = [c for c in feats if c in s1_df.columns]
    sub = s1_df[cols].dropna()
    if len(sub):
        corr = sub.corr().to_numpy()
        names = list(sub.columns)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.04, label="Pearson")
        ax.set_title(f"Aux correlation (Stream 1 — training reference, n={len(sub):,})")
    save_fig(fig, OUT / "aux_correlation_heatmap.png")


if __name__ == "__main__":
    main()
