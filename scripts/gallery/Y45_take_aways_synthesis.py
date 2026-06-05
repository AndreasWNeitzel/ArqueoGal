"""Y45: synthesis figure for the take-aways slide.

Three panels in a 1 x 3 layout, all tier-coloured per the v1.2 brief:

  - left:   per-label RMSE / APOGEE-floor ratio (compact restatement of
             Y43's headline bar; viridis-aware palette).
  - middle: sample-size advantage. log-log scatter of (training cohort
             size, inference cohort size) for ArqueoGal v1.1 alongside
             Andrae+2023, Zhang+2023, Khalatyan+2024, GSP-Spec, APOGEE
             DR19. Numbers are paper-quoted, not measured here.
  - right:  release-tier breakdown of the Stream-1 holdout as a single
             stacked horizontal bar. Same colour code as Y43.
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

# Same APOGEE floors as Y43.
_APOGEE_FLOORS = [
    ("teff",    r"$T_\mathrm{eff}$",  "K",   80.0),
    ("logg",    r"$\log g$",          "dex", 0.05),
    ("mh",      r"[M/H]",             "dex", 0.04),
    ("alpha_m", r"[$\alpha$/M]",      "dex", 0.03),
    ("mg_h",    r"[Mg/H]",            "dex", 0.03),
]

# Reference-paper sample sizes. All numbers are paper-quoted to within
# the table the original paper publishes; precise Gaia-DR3 cross-match
# yields differ by survey-year, here we take the headline figure.
# (label, n_train, n_inference, marker-color, marker)
_REF_SURVEYS = [
    ("ArqueoGal v1.1",    292_948,  87_882,    OKABE_ITO[0], "o"),
    ("Andrae+ 2023",       45_000,  175_000_000, OKABE_ITO[1], "s"),
    ("Zhang+ 2023",       300_000, 220_000_000, OKABE_ITO[2], "D"),
    ("Khalatyan+ 2024",   215_000, 175_000_000, OKABE_ITO[3], "^"),
    ("GSP-Spec (DR3)",          1, 5_500_000,  OKABE_ITO[4], "P"),
    ("APOGEE DR19",        720_000,    720_000, OKABE_ITO[5], "X"),
]

T1_COLOR = PALETTE["tier1"]
T2_COLOR = PALETTE["tier2"]
T3_COLOR = PALETTE["tier3"]

_TITLE_KW = dict(fontsize=12, fontweight="regular", color=PALETTE["ink"], pad=6)


def main() -> int:
    apply_style()

    pcols = ["source_id", "teff_pred", "logg_pred", "mh_pred",
             "alpha_m_pred", "mg_h_pred",
             "ood_joint_flag", "label_extrapolation_flag"]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "teff_apogee", "logg_apogee", "mh_apogee",
             "alpha_m_apogee", "mg_h_apogee", "fe_h_apogee", "b_deg"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(holdout)].reset_index(drop=True)
    t1 = df.loc[df["release_tier"] == 1]

    rmse: list[float] = []
    floors: list[float] = []
    for key, _tex, _unit, floor in _APOGEE_FLOORS:
        r = t1[f"{key}_pred"].to_numpy() - t1[f"{key}_apogee"].to_numpy()
        r = r[np.isfinite(r)]
        rmse.append(float(np.sqrt(np.mean(r * r))))
        floors.append(float(floor))
    ratio = np.asarray(rmse) / np.asarray(floors)

    n = len(df)
    n_by_tier = {k: int((df["release_tier"] == k).sum()) for k in (1, 2, 3)}
    pct = {k: v / n * 100.0 for k, v in n_by_tier.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4),
                              gridspec_kw=dict(width_ratios=[1.1, 1.2, 1.0]))

    # ---- (a) RMSE / APOGEE-floor ratio bar ----
    ax = axes[0]
    ypos = np.arange(len(_APOGEE_FLOORS))
    ax.barh(ypos, ratio, color=OKABE_ITO[0],
            edgecolor="white", linewidth=1.0)
    ax.axvline(1.0, color="#000000", lw=1.0, ls="--", alpha=0.85)
    for i, ratio_i in enumerate(ratio):
        ax.text(ratio_i + 0.03, ypos[i], f"{ratio_i:.2f}×",
                va="center", ha="left", fontsize=10.5, color=PALETTE["ink"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f[1] for f in _APOGEE_FLOORS], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(2.0, float(ratio.max()) * 1.4))
    ax.set_xlabel("ArqueoGal RMSE / APOGEE floor")
    ax.set_title("(a) precision vs APOGEE", **_TITLE_KW)
    ax.grid(True, axis="x", alpha=0.20)

    # ---- (b) sample-size scatter ----
    ax = axes[1]
    for label, n_train, n_inf, color, marker in _REF_SURVEYS:
        size = 220.0 if "ArqueoGal" in label else 90.0
        edgecolor = "#000000" if "ArqueoGal" in label else "white"
        lw = 1.2 if "ArqueoGal" in label else 0.8
        ax.scatter(n_train, n_inf, s=size, color=color, marker=marker,
                   edgecolor=edgecolor, linewidth=lw, zorder=3,
                   label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training cohort size")
    ax.set_ylabel("inference cohort size")
    ax.set_title("(b) sample size, ArqueoGal vs precedent", **_TITLE_KW)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=1)
    ax.grid(True, which="both", alpha=0.20)

    # ---- (c) release-tier stacked bar ----
    ax = axes[2]
    left = 0.0
    for tier, color, label in [
        (1, T1_COLOR, "Tier 1"),
        (2, T2_COLOR, "Tier 2"),
        (3, T3_COLOR, "Tier 3"),
    ]:
        w = pct[tier]
        ax.barh(0, w, left=left, color=color,
                edgecolor="white", linewidth=1.0)
        text_x = left + w / 2 if w >= 4.0 else left + w + 0.5
        text_color = "white" if w >= 4.0 else PALETTE["ink"]
        text_ha = "center" if w >= 4.0 else "left"
        ax.text(
            text_x, 0,
            f"{label}\n{pct[tier]:.2f}%",
            ha=text_ha, va="center", fontsize=10,
            color=text_color,
        )
        left += w
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("fraction of holdout (%)")
    ax.set_title(f"(c) release tiers, n = {n:,}", **_TITLE_KW)
    ax.grid(True, axis="x", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.78,
                        bottom=0.16, wspace=0.40)
    headline(
        fig,
        "Take-aways at a glance",
        "Precision vs APOGEE floor; sample-size advantage on the literature; "
        "tier composition of the holdout.",
        top=0.78,
    )
    save(fig, "Y45_take_aways_synthesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
