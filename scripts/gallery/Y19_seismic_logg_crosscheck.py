"""Y19: Stream-2 asteroseismic log g cross-validation.

Stream 2 is the TESS-asteroseismic giants × Gaia DR3 XP cohort. The seismic
log g is computed from νmax via the standard scaling relation:

    log g_seismic = log g_sun + log10(numax / numax_sun)
                              + 0.5 * log10(Teff / Teff_sun)

with log g_sun = 4.438, numax_sun = 3090 μHz, Teff_sun = 5777 K. Teff is
taken from our XP-prediction (closed-loop on Teff but the seismic log g
depends only weakly on it: ±100 K → ±0.004 dex).

The figure compares our XP-predicted log g (which never sees νmax) to this
seismic log g for every Stream-2 star with a finite νmax. This is the
strongest single piece of evidence that the model recovers physical log g
rather than overfitting to the APOGEE log g pipeline.
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

PRED_S2 = REPO / "data/processed/pipeline1_predictions_stream2.parquet"
TESS_GAIA = REPO / "data/interim/stream2_tess_gaia.parquet"

LOGG_SUN = 4.438
NUMAX_SUN = 3090.0    # μHz
TEFF_SUN = 5777.0     # K


def main() -> int:
    apply_style()
    pred = pd.read_parquet(
        PRED_S2,
        columns=["source_id", "teff_pred", "logg_pred", "logg_sigma"],
    ).drop_duplicates("source_id")
    tess = pd.read_parquet(
        TESS_GAIA, columns=["source_id", "numax_muhz", "e_numax_muhz"],
    ).drop_duplicates("source_id")
    df = pred.merge(tess, on="source_id", how="inner")
    df = df.dropna(subset=["numax_muhz", "teff_pred", "logg_pred"])
    df = df.loc[df["numax_muhz"] > 0].reset_index(drop=True)

    # Seismic log g via νmax scaling.
    log_g_seis = (LOGG_SUN
                  + np.log10(df["numax_muhz"].to_numpy() / NUMAX_SUN)
                  + 0.5 * np.log10(df["teff_pred"].to_numpy() / TEFF_SUN))
    log_g_xp = df["logg_pred"].to_numpy()
    sig_xp = df["logg_sigma"].to_numpy()
    delta = log_g_xp - log_g_seis

    # Propagate seismic uncertainty (νmax-dominated, Teff-suppressed).
    rel_numax = df["e_numax_muhz"].to_numpy() / df["numax_muhz"].to_numpy()
    sig_seis = (rel_numax / np.log(10)).clip(min=0)  # dex

    bias = float(np.median(delta))
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    rsig = float(1.4826 * np.median(np.abs(delta - np.median(delta))))
    pull = delta / np.sqrt(sig_xp ** 2 + sig_seis ** 2)
    pull = pull[np.isfinite(pull)]
    pull_std = float(np.std(pull))

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.30)

    # --- Panel A: hexbin pred vs seismic.
    ax = fig.add_subplot(gs[0, 0])
    extent = (0.5, 3.7, 0.5, 3.7)
    hb = ax.hexbin(log_g_seis, log_g_xp, gridsize=70, extent=extent,
                   mincnt=1, bins="log", cmap="viridis")
    ax.plot([extent[0], extent[1]], [extent[0], extent[1]],
            color=PALETTE["accent"], ls="--", lw=2.0, label="1:1")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(r"seismic $\log g$ from $\nu_{\max}$ (dex)")
    ax.set_ylabel(r"XP-predicted $\log g$ (dex)")
    ax.set_title(r"$\log g$ from XP vs $\log g$ from seismology",
                 color=PALETTE["navy"])
    ax.legend(loc="lower right")
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\log_{10}$ N", fontsize=10)
    ax.text(
        0.02, 0.97,
        f"n = {len(df):,}\nbias = {bias:+.3f} dex\n"
        f"RMSE = {rmse:.3f} dex\n"
        rf"$\sigma_{{\rm MAD}}$ = {rsig:.3f} dex",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold", color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=PALETTE["mist"]),
    )

    # --- Panel B: residual histogram.
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(delta, bins=80, range=(-0.5, 0.5),
            color=PALETTE["navy_light"], edgecolor="white", linewidth=0.4,
            density=True, label=r"$\Delta = \log g_{\rm XP} - \log g_{\rm seis}$")
    xs = np.linspace(-0.5, 0.5, 400)
    gauss = (1.0 / (rsig * np.sqrt(2 * np.pi))) \
            * np.exp(-0.5 * ((xs - bias) / rsig) ** 2)
    ax.plot(xs, gauss, color=PALETTE["accent"], lw=2.4,
            label=rf"$N(\rm bias, \sigma_{{\rm MAD}})$")
    ax.axvline(0.0, color=PALETTE["ink"], lw=1.0, alpha=0.5)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"$\Delta \log g$ (dex)")
    ax.set_ylabel("density")
    ax.set_title("Residual against asteroseismic truth",
                 color=PALETTE["navy"])
    verdict = (PALETTE["tier1"] if 0.85 <= pull_std <= 1.20
               else PALETTE["tier2"])
    ax.text(
        0.97, 0.97,
        rf"pull $\sigma$ = {pull_std:.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=verdict,
                  edgecolor="none"),
    )
    ax.legend(loc="upper left", fontsize=10)

    headline(
        fig,
        r"Independent-physics test — $\log g$ from XP vs from $\nu_{\max}$",
        "Stream 2 (TESS asteroseismic giants).  Seismic log g uses only ν_max + Teff "
        "via the standard scaling relation; the model never saw it.",
        top=0.84,
    )
    save(fig, "Y19_seismic_logg_crosscheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
