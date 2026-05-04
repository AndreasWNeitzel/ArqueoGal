"""Y06: What an XP coefficient vector actually looks like.

Three real RGB stars at nearly identical Teff and log g but different [M/H],
overlaid in a single panel. The point is that the metallicity signal lives in
~110 numbers per star (BP + RP coefficients) and is what the MLP learns to
read. Coefficients are read after the frozen-Hermite z-score so the y-axis is
on a comparable scale across stars.
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

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

# Three Stream-1 stars at Teff ≈ 4600-4700 K, log g ≈ 2.4-2.55, picked once
# from the parquet to give the figure a stable identity across re-runs.
TARGET_IDS = [
    2772886484650118144,  # [M/H] ≈ -0.65
    432157788257472128,  # [M/H] ≈ -0.05
    420889065383023744,  # [M/H] ≈ +0.32
]


def main() -> int:
    apply_style()

    df = pd.read_parquet(FEAT_S1)
    df = df.drop_duplicates("source_id", keep="first")
    df = df.set_index("source_id")
    # Some chosen IDs may not survive the dedup. Fall back to nearest-MH
    # giants in the same Teff/log g window in that case.
    miss = [sid for sid in TARGET_IDS if sid not in df.index]
    if miss:
        near = df[
            (df["teff_apogee"].between(4600, 4750)) & (df["logg_apogee"].between(2.4, 2.55))
        ].copy()
        # closest by [M/H] to (-0.65, -0.05, +0.32) for the missing slots.
        targets = {-0.65: 0, -0.05: 1, 0.32: 2}
        for mh_target, slot in list(targets.items())[: len(miss)]:
            best = (near["mh_apogee"] - mh_target).abs().idxmin()
            TARGET_IDS[slot] = best
    sub = df.loc[TARGET_IDS]

    bp_cols = [f"bp_coef_{i}" for i in range(55)]
    rp_cols = [f"rp_coef_{i}" for i in range(55)]
    bp_cols = [c for c in bp_cols if c in sub.columns]
    rp_cols = [c for c in rp_cols if c in sub.columns]

    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.18, width_ratios=[1, 1])
    ax_bp = fig.add_subplot(gs[0, 0])
    ax_rp = fig.add_subplot(gs[0, 1])

    colors = [PALETTE["navy"], PALETTE["accent"], PALETTE["tier3"]]
    labels = []
    for (_sid, row), color in zip(sub.iterrows(), colors):
        bp = row[bp_cols].to_numpy(dtype=float)
        rp = row[rp_cols].to_numpy(dtype=float)
        # Standardise per-star to remove the absolute-flux level — the
        # information we want to show is the SHAPE of the coefficient
        # vector across stars, not its overall scale.
        bp_z = (bp - np.nanmedian(bp)) / (np.nanstd(bp) or 1.0)
        rp_z = (rp - np.nanmedian(rp)) / (np.nanstd(rp) or 1.0)
        ax_bp.plot(
            np.arange(len(bp_z)),
            bp_z,
            "-",
            color=color,
            lw=2.0,
            marker="o",
            ms=3.5,
            alpha=0.85,
            label=f"[M/H] = {row['mh_apogee']:+.2f}",
        )
        ax_rp.plot(
            np.arange(len(rp_z)),
            rp_z,
            "-",
            color=color,
            lw=2.0,
            marker="o",
            ms=3.5,
            alpha=0.85,
            label=f"[M/H] = {row['mh_apogee']:+.2f}",
        )
        labels.append(f"[M/H] = {row['mh_apogee']:+.2f}")

    for ax, name, n in [(ax_bp, "BP", len(bp_cols)), (ax_rp, "RP", len(rp_cols))]:
        ax.set_xlabel(f"{name} coefficient index")
        ax.set_ylabel(f"per-star standardised {name} coefficient")
        ax.set_title(f"{name} block — {n} coefficients per star")
        ax.legend(title="APOGEE truth label", loc="upper right")
        ax.axhline(0.0, color=PALETTE["mist"], lw=0.8, zorder=0)

    headline(
        fig,
        "An XP coefficient vector — three real giants with different [M/H]",
        "Teff ≈ 4600-4750 K, log g ≈ 2.4-2.55.  Same star type, different metallicity.  "
        "All differences below feed straight into the MLP.",
        top=0.82,
    )
    save(fig, "Y06_xp_example_stars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
