"""F06: per-label RMSE / APOGEE-floor (top) + train vs holdout (bottom)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import FEAT_S1, PRED_S1  # noqa: E402

from arqueogal.style import (  # noqa: E402
    ACCENT_SUCCESS, ACCENT_WARNING, CHROME, LABELS, OKABE_ITO,
    apply_style, save,
)
from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

FLOORS = [
    ("teff",    "Teff",    "K",   80.0),
    ("logg",    "logg",    "dex", 0.05),
    ("mh",      "Mh",      "dex", 0.04),
    ("alpha_m", "alpha_M", "dex", 0.03),
    ("mg_h",    "MgH",     "dex", 0.03),
]


def _split_rmse(df: pd.DataFrame, t1: pd.DataFrame, train_ids, holdout_ids):
    """RMSE per label on the in-sample (train) and holdout (val+test)
    Tier-1 partitions."""
    out_train, out_hold = [], []
    train_set = t1.loc[t1["source_id"].isin(set(train_ids))]
    hold_set = t1.loc[t1["source_id"].isin(set(holdout_ids))]
    for key, *_ in FLOORS:
        for sub, dst in ((train_set, out_train), (hold_set, out_hold)):
            r = sub[f"{key}_pred"].to_numpy() - sub[f"{key}_apogee"].to_numpy()
            r = r[np.isfinite(r)]
            dst.append(float(np.sqrt(np.mean(r * r))) if r.size else float("nan"))
    return np.asarray(out_train), np.asarray(out_hold)


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
    train_ids = split["train"]
    holdout_ids = np.concatenate([split["val"], split["test"]])

    t1 = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    rmse_train, rmse_hold = _split_rmse(df, t1, train_ids, holdout_ids)
    floors = np.asarray([f[3] for f in FLOORS])
    ratios_hold = rmse_hold / floors
    units = [f[2] for f in FLOORS]

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.5),
                              gridspec_kw=dict(height_ratios=[2.0, 1.6]),
                              layout="constrained")

    ax = axes[0]
    ypos = np.arange(len(FLOORS))
    colors = [ACCENT_SUCCESS if r < 1.0 else ACCENT_WARNING for r in ratios_hold]
    ax.barh(ypos, ratios_hold, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(1.0, color="#000000", lw=1.0, ls="--", alpha=0.7,
               label=r"APOGEE DR19 floor")
    for i, ratio_i in enumerate(ratios_hold):
        ax.text(ratio_i + 0.03, ypos[i],
                f"{rmse_hold[i]:.3g} {units[i]} ({ratio_i:.2f}x)",
                va="center", ha="left", fontsize=10,
                color=CHROME["body"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABELS[f[1]] for f in FLOORS], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.7)
    ax.set_xlabel(LABELS["rmse_apg"])
    ax.set_title(r"holdout RMSE / APOGEE DR19 floor")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="x", alpha=0.30)
    ax.grid(False, axis="y")

    ax = axes[1]
    width = 0.36
    x = np.arange(len(FLOORS))
    bar_train = ax.bar(x - width / 2, rmse_train / floors, width=width,
                        color=OKABE_ITO["blue"], edgecolor="white",
                        linewidth=0.8, label=r"train")
    bar_hold = ax.bar(x + width / 2, ratios_hold, width=width,
                       color=OKABE_ITO["vermillion"], edgecolor="white",
                       linewidth=0.8, label=r"holdout")
    for b, v_abs, unit in zip(bar_train, rmse_train, units):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                f"{v_abs:.3g}", ha="center", va="bottom", fontsize=8.5,
                color=CHROME["body"])
    for b, v_abs, unit in zip(bar_hold, rmse_hold, units):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                f"{v_abs:.3g}", ha="center", va="bottom", fontsize=8.5,
                color=CHROME["body"])
    ax.axhline(1.0, color="#000000", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f[1]] for f in FLOORS], fontsize=10)
    ax.set_ylabel(LABELS["rmse_apg"])
    ax.set_title(r"train vs holdout RMSE on Tier 1 (ratio to APOGEE floor)")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, max(2.7, float(ratios_hold.max() * 1.25)))
    ax.grid(True, axis="y", alpha=0.30)

    save(fig, "F06_headline_rmse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
