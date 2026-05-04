"""B16: Mutual information of every encoder feature with truth [alpha/M].

Where Cohen's d in B13 measured how well a feature separates a binary
(high-/low-alpha) cut, mutual information measures the full continuous
dependency between a feature and truth [alpha/M] without imposing a
threshold. A feature with MI ~ 0 carries no information about
[alpha/M]; high-MI features should drive the encoder's [alpha/M]
representation.

Implementation: scikit-learn's ``mutual_info_regression`` (k-NN
estimator, Kraskov method). On stars with finite truth [alpha/M] and
finite feature value. Computed twice:
  panel 0 = full Stream-1 Kiel cohort
  panel 1 = metal-poor subset ([M/H] < -0.5) — the regime where the
            disc bimodality is most informative.

Output: bar chart of MI(feature; alpha/M) [nats], grouped by feature
family (BP / RP / c0 / residual / aux), with the top-15 most
informative features named.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig
from arqueogal.xp_abundances.main.data import FeatureLayout

OUT = REPO / "reports/gallery/B_preprocessing"

# Subsample for tractability: kNN MI estimator is O(N²) in N; cap at 30k.
SUBSAMPLE_N = 30_000
N_NEIGHBORS = 5


def per_feature_mi(
    df: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    rng_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    target = df[target_name].to_numpy(dtype=np.float64)
    finite_target = np.isfinite(target)
    if int(finite_target.sum()) < 100:
        return np.full(len(feature_names), np.nan)

    out = np.full(len(feature_names), np.nan)
    for i, name in enumerate(feature_names):
        v = df[name].to_numpy(dtype=np.float64)
        m = finite_target & np.isfinite(v)
        if int(m.sum()) < 200:
            continue
        idx = np.where(m)[0]
        if len(idx) > SUBSAMPLE_N:
            idx = rng.choice(idx, SUBSAMPLE_N, replace=False)
        x = v[idx].reshape(-1, 1)
        y = target[idx]
        # mutual_info_regression returns bits; convert to nats by /log2(e)
        mi_bits = mutual_info_regression(
            x, y, n_neighbors=N_NEIGHBORS, random_state=rng_seed,
        )[0]
        out[i] = float(mi_bits) * np.log(2)  # nats
    return out


def main() -> int:
    apply_style()
    layout = FeatureLayout()
    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        print(f"Error: {parquet} not found")
        return 1
    feature_names = list(layout.all_required_columns)
    cols = feature_names + ["mh_apogee", "alpha_m_apogee"]
    df = pd.read_parquet(parquet, columns=cols)
    df = df.drop_duplicates("source_id", keep="first") if "source_id" in df.columns else df
    print(f"[B16] cohort n={len(df):,}")

    # Family classification.
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

    print("[B16] computing MI on full Kiel cohort ...")
    mi_full = per_feature_mi(df, feature_names, "alpha_m_apogee", rng_seed=0)

    print("[B16] computing MI on metal-poor subset ([M/H] < -0.5) ...")
    mp = df[(df["mh_apogee"] < -0.5).fillna(False)]
    print(f"  metal-poor n={len(mp):,}")
    mi_mp = per_feature_mi(mp, feature_names, "alpha_m_apogee", rng_seed=0)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.7, 1.0, 0.7], hspace=0.65)

    # Panel 0: full-cohort MI bar chart.
    ax = fig.add_subplot(gs[0, 0])
    colors = [family_color[f] for f in family]
    ax.bar(np.arange(len(feature_names)), mi_full, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.set_xlabel("Feature index (encoder input order)")
    ax.set_ylabel("MI(feature; [alpha/M])  [nats]")
    ax.set_title(
        f"Full Kiel cohort - MI per encoder feature against truth [alpha/M]\n"
        f"(n={len(df):,}, kNN k={N_NEIGHBORS}, subsample cap {SUBSAMPLE_N:,})"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in family_color.values()]
    ax.legend(handles, list(family_color.keys()), fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Panel 1: top-15 named on full cohort.
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(np.where(np.isnan(mi_full), -1, mi_full))[::-1][:15]
    ax.barh(np.arange(15)[::-1], mi_full[order],
            color=[family_color[family[i]] for i in order])
    ax.set_yticks(np.arange(15)[::-1])
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.set_xlabel("MI [nats]")
    ax.set_title("Top-15 most informative features  (full cohort)")
    ax.grid(axis="x", alpha=0.25)

    # Panel 2: metal-poor MI bar chart.
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(np.arange(len(feature_names)), mi_mp, color=colors, width=0.95)
    ax.set_xlim(-1, len(feature_names))
    ax.set_xlabel("Feature index (encoder input order)")
    ax.set_ylabel("MI(feature; [alpha/M])  [nats]")
    ax.set_title(
        f"Metal-poor subset ([M/H] < -0.5) - MI per encoder feature against truth [alpha/M]\n"
        f"(n={len(mp):,}, kNN k={N_NEIGHBORS}, subsample cap {SUBSAMPLE_N:,})"
    )
    ax.legend(handles, list(family_color.keys()), fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: top-15 named on metal-poor subset.
    ax = fig.add_subplot(gs[3, 0])
    order_mp = np.argsort(np.where(np.isnan(mi_mp), -1, mi_mp))[::-1][:15]
    ax.barh(np.arange(15)[::-1], mi_mp[order_mp],
            color=[family_color[family[i]] for i in order_mp])
    ax.set_yticks(np.arange(15)[::-1])
    ax.set_yticklabels([feature_names[i] for i in order_mp], fontsize=9)
    ax.set_xlabel("MI [nats]")
    ax.set_title("Top-15 most informative features  (metal-poor cohort)")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "B16 - Per-feature mutual information vs truth [alpha/M]\n"
        "(Kraskov k-NN MI estimator, sklearn.feature_selection.mutual_info_regression)",
        fontsize=12, fontweight="semibold", y=0.995,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B16_mutual_information_alpha", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
