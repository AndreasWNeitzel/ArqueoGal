"""Y12: Cohort sizes, single bar chart with tier breakdown.

For each of the three streams: total stars, plus a stacked bar showing the
T1 / T2 / T3 split. Tier numbers come from assign_release_tier on live
prediction parquets (no kin_ood_flag for Stream 1; pulled from the hybrid
release for Streams 2 and 3 when available).
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

from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED = {
    sid: REPO / f"data/processed/pipeline1_predictions_stream{sid}.parquet" for sid in (1, 2, 3)
}
HYBRID = {
    2: REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet",
    3: REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet",
}


def _tiers(sid: int) -> dict[int, int]:
    p = pd.read_parquet(
        PRED[sid],
        columns=[
            "source_id",
            "teff_sigma",
            "logg_sigma",
            "mh_sigma",
            "alpha_m_sigma",
            "mg_h_sigma",
            "ood_joint_flag",
            "label_extrapolation_flag",
        ],
    ).drop_duplicates("source_id")
    p["kin_ood_flag"] = False
    h = HYBRID.get(sid)
    if h is not None and h.exists():
        hd = pd.read_parquet(h, columns=["source_id", "kin_ood_flag"])
        p = p.merge(hd, on="source_id", how="left", suffixes=("", "_h"))
        if "kin_ood_flag_h" in p.columns:
            p["kin_ood_flag"] = p["kin_ood_flag_h"].fillna(False).astype(bool)
            p = p.drop(columns=["kin_ood_flag_h"])
    p["release_tier"] = assign_release_tier(p).astype(np.int8)
    return {
        1: int((p["release_tier"] == 1).sum()),
        2: int((p["release_tier"] == 2).sum()),
        3: int((p["release_tier"] == 3).sum()),
    }


def main() -> int:
    apply_style()
    counts = {sid: _tiers(sid) for sid in (1, 2, 3)}

    names = ["Stream 1\n(APOGEE × XP)", "Stream 2\n(TESS × XP)", "Stream 3\n(Andrae+23 × XP)"]
    sids = [1, 2, 3]
    n_total = [sum(counts[s].values()) for s in sids]
    t1 = np.array([counts[s][1] for s in sids])
    t2 = np.array([counts[s][2] for s in sids])
    t3 = np.array([counts[s][3] for s in sids])

    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(sids))
    width = 0.6

    ax.bar(
        x,
        t1,
        width,
        color=PALETTE["tier1"],
        edgecolor="white",
        linewidth=1.2,
        label="Tier 1 (science-grade)",
    )
    ax.bar(
        x,
        t2,
        width,
        bottom=t1,
        color=PALETTE["tier2"],
        edgecolor="white",
        linewidth=1.2,
        label="Tier 2 (caution)",
    )
    ax.bar(
        x,
        t3,
        width,
        bottom=t1 + t2,
        color=PALETTE["tier3"],
        edgecolor="white",
        linewidth=1.2,
        label="Tier 3 (do-not-trust)",
    )

    # Annotate each segment with absolute counts.
    for i, (n1, n2, n3) in enumerate(zip(t1, t2, t3)):
        ax.text(
            i,
            n1 / 2,
            f"{n1:,}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
        if n2 / max(n_total) > 0.015:
            ax.text(
                i,
                n1 + n2 / 2,
                f"{n2:,}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        if n3 / max(n_total) > 0.015:
            ax.text(
                i,
                n1 + n2 + n3 / 2,
                f"{n3:,}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        ax.text(
            i,
            n1 + n2 + n3,
            f"\nN = {n_total[i]:,}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("number of stars (Gaia DR3 source_ids)")
    ax.set_title("Per-stream cohort with release-tier breakdown", color=PALETTE["navy"])
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(n_total) * 1.18)

    headline(
        fig,
        "Where the data live",
        "Same model, three cohorts, single tier rule, applied at scale to ~980k Gaia DR3 stars.",
        top=0.84,
    )
    save(fig, "Y12_data_volume_bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
