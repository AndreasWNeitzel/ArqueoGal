"""Y35: Stream-1 holdout residuals (pred - APOGEE truth), tiers overlaid.

Compact 2 x 2 layout, one subplot per released label (Teff, log g, [M/H],
[alpha/M]).  Each panel overlays:
  - all data (black step histogram)
  - Tier 1 only (filled, green)
  - Tier 2 only (filled, vermillion)
  - Tier 3 only (filled, red-purple)

Slide-friendly 14:8 layout.
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

LABELS = [
    ("teff",    "teff_pred",    "teff_apogee",    r"$T_{\rm eff}$ residual (K)",     (-400, 400)),
    ("logg",    "logg_pred",    "logg_apogee",    r"$\log g$ residual (dex)",        (-0.6, 0.6)),
    ("mh",      "mh_pred",      "mh_apogee",      r"[M/H] residual (dex)",           (-0.5, 0.5)),
    ("alpha_m", "alpha_m_pred", "alpha_m_apogee", r"[$\alpha$/M] residual (dex)",    (-0.20, 0.20)),
]

T1_COLOR = PALETTE["tier1"]
T2_COLOR = PALETTE["tier2"]
T3_COLOR = PALETTE["tier3"]
ALL_COLOR = OKABE_ITO[0]   # blue shaded background

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _load_holdout() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = [
        "source_id", "teff_apogee", "logg_apogee", "mh_apogee",
        "alpha_m_apogee", "mg_h_apogee", "fe_h_apogee", "b_deg",
    ]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    return df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)


def _stats(r: np.ndarray) -> tuple[float, float]:
    """Return (bias=mean, RMSE) over finite values."""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    bias = float(np.mean(r))
    rmse = float(np.sqrt(np.mean(r * r)))
    return bias, rmse


def _draw_panel(ax, df: pd.DataFrame, key: str, pcol: str, tcol: str,
                xlabel: str, xrange: tuple[float, float]) -> None:
    bins = np.linspace(xrange[0], xrange[1], 60)
    res_all = (df[pcol] - df[tcol]).to_numpy()

    # All data: shaded blue filled histogram in the background.
    b_all, s_all = _stats(res_all)
    ax.hist(
        res_all[np.isfinite(res_all)], bins=bins,
        color=ALL_COLOR, alpha=0.40, edgecolor="white", linewidth=0.4,
        zorder=1,
        label=f"all  bias={b_all:+.2g}, RMSE={s_all:.2g}",
    )

    # Per-tier step outlines on top.
    for tier, color, z in [(1, T1_COLOR, 3),
                           (2, T2_COLOR, 4),
                           (3, T3_COLOR, 5)]:
        sub = df.loc[df["release_tier"] == tier]
        if not len(sub):
            continue
        r = (sub[pcol] - sub[tcol]).to_numpy()
        b, s = _stats(r)
        ax.hist(
            r[np.isfinite(r)], bins=bins,
            histtype="step", color=color, lw=1.8, zorder=z,
            label=f"T{tier}  bias={b:+.2g}, RMSE={s:.2g}",
        )

    ax.axvline(0.0, color=PALETTE["ink"], lw=0.9, ls="-", alpha=0.6, zorder=2)
    ax.set_xlim(xrange)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log)")
    ax.set_title(xlabel.split(" residual")[0], **_TITLE_KW)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.20, which="both")


def main() -> int:
    apply_style()
    df = _load_holdout()
    if df.empty:
        print("[Y35] no holdout rows, aborting")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, (_key, pcol, tcol, xlabel, xrange) in zip(axes.ravel(), LABELS):
        _draw_panel(ax, df, _key, pcol, tcol, xlabel, xrange)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.10,
                        hspace=0.45, wspace=0.25)
    headline(
        fig,
        "Stream 1 holdout residuals: tiers overlaid (one panel per label)",
        f"shaded blue = all data (n = {len(df):,});  step = T1/T2/T3.  bias = mean, scatter = RMSE.",
        top=0.82,
    )
    save(fig, "Y35_residuals_by_tier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
