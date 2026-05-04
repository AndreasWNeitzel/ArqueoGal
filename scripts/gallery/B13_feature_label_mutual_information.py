"""B13: Feature - label discriminability scan.

Per encoder-input feature, compute a simple discriminability metric for
the bimodal disc components (truth [alpha/M] above vs below 0.15 dex)
on metal-poor stars ([M/H] < -0.5). The metric is the absolute Cohen's
d effect size:

    d = |mean(feature | high-α) - mean(feature | low-α)| /
        pooled_std(feature)

Features with d < 0.1 carry essentially no information about which disc
component a metal-poor star belongs to. This is the data-side
discriminability ceiling: SupCon cannot recover bimodality on a feature
set whose components are statistically indistinguishable.

Layout (3 x 1):
  panel 0 = bar chart of |Cohen's d| for all 140 encoder features,
            grouped by feature family (BP, RP, c0, residuals, aux).
  panel 1 = top-20 most discriminative features named.
  panel 2 = same on the **whole** Kiel pool (not metal-poor only) for
            comparison.
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
from arqueogal.xp_abundances.main.data import FeatureLayout

OUT = REPO / "reports/gallery/B_preprocessing"


def cohens_d(values: np.ndarray, labels: np.ndarray) -> float:
    a = values[labels & np.isfinite(values)]
    b = values[(~labels) & np.isfinite(values)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled <= 0:
        return float("nan")
    return float(abs(a.mean() - b.mean()) / pooled)


def main() -> int:
    apply_style()
    layout = FeatureLayout()
    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        print(f"Error: {parquet} not found")
        return 1
    cols = list(layout.all_required_columns) + ["mh_apogee", "alpha_m_apogee"]
    df = pd.read_parquet(parquet, columns=cols)
    print(f"[B13] cohort n={len(df):,}")

    feature_names = list(layout.all_required_columns)
    family = []
    bp_cols = set(layout.bp_coef_cols)
    rp_cols = set(layout.rp_coef_cols)
    scalar_cols = set(layout.xp_scalar_cols)
    res_cols = set(layout.residual_cols)
    for name in feature_names:
        if name in bp_cols:
            family.append("BP")
        elif name in rp_cols:
            family.append("RP")
        elif name in scalar_cols:
            family.append("c0")
        elif name in res_cols:
            family.append("residual")
        else:
            family.append("aux")
    family = np.array(family)
    family_color = {
        "BP": "#1f77b4", "RP": "#d62728", "c0": "#2ca02c",
        "residual": "#7f7f7f", "aux": "#9467bd",
    }

    alpha_thr = 0.15
    mh_thr = -0.5

    truth_alpha = df["alpha_m_apogee"].to_numpy(dtype=np.float64)
    truth_mh = df["mh_apogee"].to_numpy(dtype=np.float64)

    # Metal-poor mask
    mp_finite = (truth_mh < mh_thr) & np.isfinite(truth_alpha) & np.isfinite(truth_mh)
    high_alpha_mp = (truth_alpha >= alpha_thr) & mp_finite
    low_alpha_mp = (truth_alpha < alpha_thr) & mp_finite
    print(f"[B13] metal-poor cohort: high-α n={int(high_alpha_mp.sum()):,}, "
          f"low-α n={int(low_alpha_mp.sum()):,}")

    # Whole-pool mask
    all_finite = np.isfinite(truth_alpha) & np.isfinite(truth_mh)
    high_alpha_all = (truth_alpha >= alpha_thr) & all_finite
    low_alpha_all = (truth_alpha < alpha_thr) & all_finite
    print(f"[B13] full cohort:        high-α n={int(high_alpha_all.sum()):,}, "
          f"low-α n={int(low_alpha_all.sum()):,}")

    d_mp = np.full(len(feature_names), np.nan)
    d_all = np.full(len(feature_names), np.nan)
    for i, name in enumerate(feature_names):
        v = df[name].to_numpy(dtype=np.float64)
        # Combine into a binary label aligned with v
        # (high-α = 1, low-α = 0; only stars in either set count)
        sel_mp = high_alpha_mp | low_alpha_mp
        if sel_mp.sum() >= 10:
            d_mp[i] = cohens_d(v[sel_mp], high_alpha_mp[sel_mp])
        sel_all = high_alpha_all | low_alpha_all
        if sel_all.sum() >= 10:
            d_all[i] = cohens_d(v[sel_all], high_alpha_all[sel_all])

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 1, hspace=0.6)

    # Panel 0: full bar chart, MP cohort.
    ax = fig.add_subplot(gs[0, 0])
    colors = [family_color[f] for f in family]
    ax.bar(np.arange(len(feature_names)), d_mp, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.set_xlabel("Feature index (in encoder input order)")
    ax.set_ylabel("|Cohen's d|")
    ax.set_title(f"|Cohen's d| per encoder feature, metal-poor "
                 f"([M/H] < {mh_thr}) high-α vs low-α at threshold "
                 f"[α/M] = {alpha_thr}.  "
                 f"d > 0.2 = small; d > 0.5 = medium; d > 0.8 = large.")
    ax.axhline(0.1, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0.2, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0.5, color="r", lw=0.6, ls="--", alpha=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in family_color.values()]
    ax.legend(handles, list(family_color.keys()), fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Panel 1: top-20 most discriminative.
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(np.where(np.isnan(d_mp), -1, d_mp))[::-1][:20]
    bars = ax.barh(np.arange(20)[::-1], d_mp[order],
                   color=[family_color[family[i]] for i in order])
    ax.set_yticks(np.arange(20)[::-1])
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.set_xlabel("|Cohen's d|, metal-poor cohort")
    ax.set_title("Top-20 most disc-bimodality discriminative features (metal-poor)")
    ax.grid(axis="x", alpha=0.25)

    # Panel 2: same on full pool.
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(np.arange(len(feature_names)), d_all, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.set_xlabel("Feature index (encoder input order)")
    ax.set_ylabel("|Cohen's d|")
    ax.set_title(f"|Cohen's d| per encoder feature, FULL Kiel pool, "
                 f"high-α vs low-α at [α/M] = {alpha_thr}.")
    ax.axhline(0.1, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0.5, color="r", lw=0.6, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "B13 - Feature-level disc alpha-bimodality discriminability scan\n"
        f"Stream-1 training pool, n={len(df):,}.\n"
        "Bars = effect size of feature mean shift between high-alpha and low-alpha stars.\n"
        "Tall bars are useful; near-zero bars carry no information about disc component membership.",
        fontsize=11, fontweight="semibold", y=0.995,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B13_feature_label_mutual_information",
             formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
