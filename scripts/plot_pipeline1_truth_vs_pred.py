"""Side-by-side truth vs predicted plots for the joint-loss pipeline.

Panels
------
Row 1: Kiel diagram (Teff, logg) — truth | pred
Row 2: Chemistry ([M/H], [α/M]) — truth | pred
Row 3: Chemistry ([M/H], [Mg/H]) — truth | pred
Row 4: Per-label histograms (truth filled, pred line) — 5 panels

Val-partition, 5-member joint ensemble (moment-matched).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_IN = _REPO / "reports/pipeline1/run_a/val_predictions.parquet"
_DEFAULT_OUT = _REPO / "reports/pipeline1/run_a/val_truth_vs_pred.png"

RANGE = {
    "teff": (4000, 5500),
    "logg": (0.9, 3.7),
    "mh": (-2.0, 0.5),
    "alpha_m": (-0.2, 0.5),
    "mg_h": (-2.0, 0.6),
}
PRETTY = {
    "teff": r"$T_{\rm eff}$ [K]",
    "logg": r"$\log g$ [dex]",
    "mh": r"$[{\rm M/H}]$ [dex]",
    "alpha_m": r"$[\alpha/{\rm M}]$ [dex]",
    "mg_h": r"$[{\rm Mg/H}]$ [dex]",
}
SHORT = {"teff": "Teff", "logg": "logg", "mh": "[M/H]", "alpha_m": "[α/M]", "mg_h": "[Mg/H]"}


def _pair_hex(
    axL,
    axR,
    x_t,
    y_t,
    x_p,
    y_p,
    xr,
    yr,
    xlab,
    ylab,
    title_t="truth",
    title_p="pred",
    invert_x=False,
    invert_y=False,
) -> None:
    x_lo, x_hi = sorted(xr)
    y_lo, y_hi = sorted(yr)
    common = dict(
        gridsize=60, mincnt=1, cmap="viridis", extent=(x_lo, x_hi, y_lo, y_hi), linewidths=0
    )
    hL = axL.hexbin(x_t, y_t, **common)
    hR = axR.hexbin(x_p, y_p, **common)
    vmax = max(hL.get_array().max(), hR.get_array().max())
    for h, ax in ((hL, axL), (hR, axR)):
        h.set_clim(1, vmax)
        ax.set_xlim(x_hi if invert_x else x_lo, x_lo if invert_x else x_hi)
        ax.set_ylim(y_hi if invert_y else y_lo, y_lo if invert_y else y_hi)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    axL.set_title(title_t)
    axR.set_title(title_p)


def _hist_overlay(ax, t, p, key) -> None:
    lo, hi = RANGE[key]
    m = np.isfinite(t) & np.isfinite(p)
    bins = np.linspace(lo, hi, 61)
    ax.hist(t[m], bins=bins, color="steelblue", alpha=0.45, density=True, label="truth")
    ax.hist(
        p[m],
        bins=bins,
        histtype="step",
        color="darkorange",
        linewidth=1.6,
        density=True,
        label="pred",
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel(PRETTY[key])
    ax.set_ylabel("density")
    ax.set_title(SHORT[key])
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-pred", type=Path, default=_DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.val_pred)

    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(4, 5, hspace=0.38, wspace=0.34)

    # Row 1 — Kiel (Teff/logg): truth | pred (use 2 columns each, span cols 0-1 and 2-3)
    axL = fig.add_subplot(gs[0, 0:2])
    axR = fig.add_subplot(gs[0, 2:4])
    _pair_hex(
        axL,
        axR,
        df["teff_truth"].to_numpy(),
        df["logg_truth"].to_numpy(),
        df["teff_pred"].to_numpy(),
        df["logg_pred"].to_numpy(),
        xr=RANGE["teff"],
        yr=RANGE["logg"],
        xlab=PRETTY["teff"],
        ylab=PRETTY["logg"],
        title_t="Kiel — truth",
        title_p="Kiel — pred",
        invert_x=True,
        invert_y=True,
    )
    # Residual Kiel (col 4)
    axD = fig.add_subplot(gs[0, 4])
    dT = df["teff_pred"].to_numpy() - df["teff_truth"].to_numpy()
    dG = df["logg_pred"].to_numpy() - df["logg_truth"].to_numpy()
    sc = axD.scatter(
        dT, dG, s=2, c=df["mh_truth"].to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=0.5, alpha=0.5
    )
    axD.axhline(0, color="k", linewidth=0.5)
    axD.axvline(0, color="k", linewidth=0.5)
    axD.set_xlabel(r"$\Delta T_{\rm eff}$ [K]")
    axD.set_ylabel(r"$\Delta \log g$ [dex]")
    axD.set_title(r"Kiel residuals (colour = truth [M/H])")
    axD.set_xlim(-400, 400)
    axD.set_ylim(-0.8, 0.8)
    plt.colorbar(sc, ax=axD, shrink=0.85, label="[M/H]")

    # Row 2 — chemistry [M/H] / [α/M]: truth | pred
    axL = fig.add_subplot(gs[1, 0:2])
    axR = fig.add_subplot(gs[1, 2:4])
    _pair_hex(
        axL,
        axR,
        df["mh_truth"].to_numpy(),
        df["alpha_m_truth"].to_numpy(),
        df["mh_pred"].to_numpy(),
        df["alpha_m_pred"].to_numpy(),
        xr=RANGE["mh"],
        yr=RANGE["alpha_m"],
        xlab=PRETTY["mh"],
        ylab=PRETTY["alpha_m"],
        title_t="[M/H] vs [α/M] — truth",
        title_p="[M/H] vs [α/M] — pred",
    )
    # Residual chemistry (col 4)
    axD = fig.add_subplot(gs[1, 4])
    dM = df["mh_pred"].to_numpy() - df["mh_truth"].to_numpy()
    dA = df["alpha_m_pred"].to_numpy() - df["alpha_m_truth"].to_numpy()
    sc = axD.scatter(
        dM, dA, s=2, c=df["mh_truth"].to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=0.5, alpha=0.5
    )
    axD.axhline(0, color="k", linewidth=0.5)
    axD.axvline(0, color="k", linewidth=0.5)
    axD.set_xlabel(r"$\Delta [{\rm M/H}]$ [dex]")
    axD.set_ylabel(r"$\Delta [\alpha/{\rm M}]$ [dex]")
    axD.set_title(r"chem residuals (colour = truth [M/H])")
    axD.set_xlim(-0.6, 0.6)
    axD.set_ylim(-0.3, 0.3)
    plt.colorbar(sc, ax=axD, shrink=0.85, label="[M/H]")

    # Row 3 — chemistry [M/H] / [Mg/H]: truth | pred
    axL = fig.add_subplot(gs[2, 0:2])
    axR = fig.add_subplot(gs[2, 2:4])
    _pair_hex(
        axL,
        axR,
        df["mh_truth"].to_numpy(),
        df["mg_h_truth"].to_numpy(),
        df["mh_pred"].to_numpy(),
        df["mg_h_pred"].to_numpy(),
        xr=RANGE["mh"],
        yr=RANGE["mg_h"],
        xlab=PRETTY["mh"],
        ylab=PRETTY["mg_h"],
        title_t="[M/H] vs [Mg/H] — truth",
        title_p="[M/H] vs [Mg/H] — pred",
    )
    # Residual [Mg/H] vs truth (col 4)
    axD = fig.add_subplot(gs[2, 4])
    dMg = df["mg_h_pred"].to_numpy() - df["mg_h_truth"].to_numpy()
    m = np.isfinite(dMg)
    axD.scatter(df["mh_truth"].to_numpy()[m], dMg[m], s=2, color="teal", alpha=0.4)
    axD.axhline(0, color="k", linewidth=0.5)
    axD.set_xlabel(PRETTY["mh"])
    axD.set_ylabel(r"$\Delta [{\rm Mg/H}]$ [dex]")
    axD.set_title(r"[Mg/H] residuals vs truth [M/H]")
    axD.set_xlim(RANGE["mh"])
    axD.set_ylim(-0.4, 0.4)

    # Row 4 — per-label histograms: truth vs pred
    for i, key in enumerate(("teff", "logg", "mh", "alpha_m", "mg_h")):
        ax = fig.add_subplot(gs[3, i])
        _hist_overlay(ax, df[f"{key}_truth"].to_numpy(), df[f"{key}_pred"].to_numpy(), key)

    n = len(df)
    fig.suptitle(
        f"Pipeline-1 joint — truth vs pred (val n={n:,}, 5-member ensemble)",
        fontsize=14,
        y=0.995,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
