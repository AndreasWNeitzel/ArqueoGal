"""B14: NaN-imputation impact on encoder input distributions.

Training.py:155 calls ``np.nan_to_num(arrs['X'], nan=0.0)`` on the
encoder input. Aux features carry meaningful NaN populations (e.g.
``av_edenhofer`` is NaN where the Edenhofer dust map has no coverage,
~10-15% of the Kiel-bounded RGB pool). After ``nan_to_num``, every
NaN star gets feature value 0 — which is **inside the in-distribution
range** of most aux features, not outside. The encoder cannot
distinguish "I don't know this dust map at this position" from
"the dust map said zero here."

This plot shows the resulting artificial spike at zero for the aux
features whose NaN fraction is non-trivial.

Layout: one panel per aux feature with NaN fraction > 1%, plotting:
- the raw value distribution (pre-imputation, NaN dropped) in blue
- the post-imputation distribution (NaN -> 0.0) in red, log-y
- a vertical band marking the imputed value (x = 0) with the
  per-feature NaN fraction printed.
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


def main() -> int:
    apply_style()
    layout = FeatureLayout()
    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        print(f"Error: {parquet} not found")
        return 1
    aux_cols = list(layout.aux_cols)
    df = pd.read_parquet(parquet, columns=aux_cols)

    # Identify aux features with non-trivial NaN populations.
    nan_frac = {c: float(df[c].isna().mean()) for c in aux_cols}
    interesting = [(c, f) for c, f in nan_frac.items() if f >= 0.01]
    interesting.sort(key=lambda kv: kv[1], reverse=True)
    print(f"[B14] {len(interesting)} aux features with NaN fraction >= 1%:")
    for c, f in interesting:
        print(f"  {c:30s}  NaN frac = {f * 100:5.2f}%")

    if not interesting:
        print("[B14] No aux features with NaN fraction >= 1%; nothing to render.")
        return 0

    n = len(interesting)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows + 1))
    axes = axes.ravel() if n_rows > 1 else np.array([axes]).ravel()

    for i, (col, frac) in enumerate(interesting):
        ax = axes[i]
        v = df[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(v)
        v_finite = v[finite]
        if len(v_finite) == 0:
            continue
        # Bin range covers both the finite distribution AND zero.
        lo = float(min(0.0, np.nanpercentile(v_finite, 0.5)))
        hi = float(np.nanpercentile(v_finite, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 80)

        # Raw distribution (NaNs dropped).
        ax.hist(
            v_finite,
            bins=bins,
            color="#1f77b4",
            alpha=0.55,
            edgecolor="#1f77b4",
            lw=0.4,
            label=f"raw (NaN dropped, n={len(v_finite):,})",
        )

        # Post-imputation distribution: NaNs replaced with 0.0 by
        # training.py's np.nan_to_num call. Plot the imputed spike
        # honestly as a single tall bar at x=0 alongside the raw bulk.
        n_nan = int((~finite).sum())
        if n_nan > 0:
            # Place the imputation spike at exactly x=0 with bar width
            # matching one bin in the raw histogram.
            bar_w = bins[1] - bins[0]
            ax.bar(
                [0.0],
                [n_nan],
                width=bar_w * 0.9,
                color="#d62728",
                alpha=0.85,
                edgecolor="#d62728",
                lw=0.6,
                label=f"imputed spike at x=0  (n={n_nan:,})",
            )
        ax.axvline(0.0, color="r", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(
            f"{col}\nNaN fraction = {frac * 100:.2f}%  -> {n_nan:,} stars imputed to x=0",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    for j in range(len(interesting), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(
        "B14 - NaN imputation impact on encoder input.\n"
        "training.py:155 replaces NaN with 0.0 for aux features.\n"
        "Red dashed line marks x = 0; the histogram spike there is the\n"
        "artificial in-distribution mass introduced by imputation.",
        fontsize=11,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B14_nan_imputation_impact", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
