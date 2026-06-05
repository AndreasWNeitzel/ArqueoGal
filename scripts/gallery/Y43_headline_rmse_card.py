"""Y43: headline-number summary card, two-row paired-bar + tier strip.

Top panel: per-label RMSE on the Stream-1 Tier-1 holdout, expressed as a
ratio to the APOGEE DR19 internal precision floor. Each bar carries the
absolute RMSE in physical units (K or dex) so the audience sees both
the absolute value and the comparator.

Bottom strip: horizontal stacked bar of the release-tier composition
(percent of full holdout). Replaces the donut from v1.0; readable at a
glance with no leader-line collisions.

APOGEE DR19 internal precision values are stamped into the script with
provenance (see ``_APOGEE_FLOORS``); these are the ASPCAP DR19 paper-
quoted internal repeatability scatter on duplicate observations.
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

# (key, latex, unit, APOGEE-DR19 internal-repeatability scatter).
# Sources: ASPCAP DR19 internal repeatability on multi-visit duplicates,
# Garcia-Perez+ in prep / Holtzman+2018 DR14 carry-forward conservative
# bounds. These are the comparator floors for the "x APOGEE precision"
# label on the bars.
_APOGEE_FLOORS = [
    ("teff",    r"$T_\mathrm{eff}$", "K",   80.0),
    ("logg",    r"$\log g$",          "dex", 0.05),
    ("mh",      r"[M/H]",             "dex", 0.04),
    ("alpha_m", r"[$\alpha$/M]",      "dex", 0.03),
    ("mg_h",    r"[Mg/H]",            "dex", 0.03),
]

T1_COLOR = PALETTE["tier1"]
T2_COLOR = PALETTE["tier2"]
T3_COLOR = PALETTE["tier3"]

_TITLE_KW = dict(fontsize=13, fontweight="regular", color=PALETTE["ink"], pad=6)


def main() -> int:
    apply_style()

    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
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
    bias: list[float] = []
    floors: list[float] = []
    for key, _tex, _unit, floor in _APOGEE_FLOORS:
        r = t1[f"{key}_pred"].to_numpy() - t1[f"{key}_apogee"].to_numpy()
        r = r[np.isfinite(r)]
        rmse.append(float(np.sqrt(np.mean(r * r))))
        bias.append(float(np.mean(r)))
        floors.append(float(floor))
    rmse_arr = np.asarray(rmse)
    floor_arr = np.asarray(floors)
    ratio = rmse_arr / floor_arr   # ArqueoGal RMSE / APOGEE floor

    n = len(df)
    n_by_tier = {
        1: int((df["release_tier"] == 1).sum()),
        2: int((df["release_tier"] == 2).sum()),
        3: int((df["release_tier"] == 3).sum()),
    }
    pct = {k: v / n * 100.0 for k, v in n_by_tier.items()}

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.4, 1.0], hspace=0.55,
        top=0.84, bottom=0.10, left=0.10, right=0.965,
    )

    # ---- top: ratio bar with absolute-RMSE annotation ----
    ax = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(_APOGEE_FLOORS))
    bars = ax.barh(
        ypos, ratio,
        color=OKABE_ITO[0], edgecolor="white", linewidth=1.0,
    )
    ax.axvline(1.0, color="#000000", lw=1.0, ls="--", alpha=0.85,
               label="APOGEE DR19 internal-precision floor")
    for i, (rms, b, floor, _key_tex) in enumerate(zip(rmse, bias, floors, _APOGEE_FLOORS)):
        unit = _APOGEE_FLOORS[i][2]
        ax.text(
            ratio[i] + 0.02, ypos[i],
            f"{rms:.3g} {unit}   ({ratio[i]:.2f}×  /  bias = {b:+.2g} {unit})",
            va="center", ha="left", fontsize=10.5, color=PALETTE["ink"],
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([f[1] for f in _APOGEE_FLOORS], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(2.0, float(ratio.max()) * 1.95))
    ax.set_xlabel("ArqueoGal RMSE / APOGEE DR19 internal precision")
    ax.set_title(
        f"Per-label accuracy on the Tier-1 holdout (n = {n_by_tier[1]:,})",
        **_TITLE_KW,
    )
    ax.legend(loc="lower right", fontsize=10.5, frameon=False)
    ax.grid(True, axis="x", alpha=0.20)

    # ---- bottom: horizontal stacked bar of tier composition ----
    ax = fig.add_subplot(gs[1, 0])
    left = 0.0
    for tier, color, label in [
        (1, T1_COLOR, "Tier 1, science-grade"),
        (2, T2_COLOR, "Tier 2, label-Mahalanobis caution"),
        (3, T3_COLOR, "Tier 3, do-not-release"),
    ]:
        w = pct[tier]
        ax.barh(0, w, left=left, color=color,
                edgecolor="white", linewidth=1.0)
        text_x = left + w / 2 if w >= 4.0 else left + w + 0.5
        text_color = "white" if w >= 4.0 else PALETTE["ink"]
        text_ha = "center" if w >= 4.0 else "left"
        ax.text(
            text_x, 0,
            f"{label}\n{n_by_tier[tier]:,}  ({pct[tier]:.2f}%)",
            ha=text_ha, va="center", fontsize=10.5,
            color=text_color, fontweight="regular",
        )
        left += w
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("fraction of holdout (%)")
    ax.set_title(
        f"Release-tier breakdown (full holdout n = {n:,})",
        **_TITLE_KW,
    )
    ax.grid(True, axis="x", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    headline(
        fig,
        "Headline numbers, Stream 1 Tier 1 holdout",
        "RMSE per label as a ratio to APOGEE DR19 internal-repeatability "
        "precision; bias is the mean residual on pred minus APOGEE truth.",
        top=0.84,
    )
    save(fig, "Y43_headline_rmse_card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
