"""F13: Stream-2 calibration vs asteroseismic log g (slide 14)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import (  # noqa: E402
    load_s2_predictions, load_s2_seismic, seismic_logg,
)
from arqueogal.style import (  # noqa: E402
    ACCENT_PRIMARY, LABELS, annotate_corner, apply_style, colorbar,
    hexbin_density, save,
)


def main() -> int:
    apply_style()
    pred = load_s2_predictions()
    seism = load_s2_seismic()
    df = pred.merge(seism, on="source_id", how="inner")
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    df = df.dropna(subset=["numax_muhz", "teff_pred", "logg_pred",
                            "logg_sigma"])
    df["logg_seis"] = seismic_logg(df["numax_muhz"].to_numpy(),
                                     df["teff_pred"].to_numpy())
    df = df.loc[(df["numax_muhz"] > 0)].reset_index(drop=True)
    if len(df) > 8000:
        df = df.sample(n=8000, random_state=0).reset_index(drop=True)

    res = df["logg_pred"].to_numpy() - df["logg_seis"].to_numpy()
    sigma = df["logg_sigma"].to_numpy()
    ok = np.isfinite(res) & np.isfinite(sigma) & (sigma > 0)
    df = df.loc[ok].reset_index(drop=True)
    res = res[ok]; sigma = sigma[ok]
    pull = res / sigma

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5),
                              layout="constrained")

    ax = axes[0]
    extent = (1.0, 4.0, 1.0, 4.0)
    hb = hexbin_density(
        ax, df["logg_seis"].to_numpy(), df["logg_pred"].to_numpy(),
        gridsize=40, mincnt=4, extent=extent,
    )
    colorbar(ax, hb, LABELS["counts_log"])
    ax.plot([1.0, 4.0], [1.0, 4.0], color="#000000", lw=0.8, ls="--")
    rmse = float(np.sqrt(np.mean(res * res)))
    annotate_corner(ax,
                     f"RMSE = {rmse:.3f} dex\nn = {len(df):,}",
                     loc="upper left", fontsize=10)
    ax.set_xlim(1.0, 4.0); ax.set_ylim(1.0, 4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"asteroseismic $\log g$ [dex]")
    ax.set_ylabel(r"JANUS $\log g$ [dex]")
    ax.set_title(r"JANUS vs asteroseismic")
    ax.grid(False)

    ax = axes[1]
    bins = np.linspace(-4, 4, 51)
    ax.hist(pull, bins=bins, density=True, color=ACCENT_PRIMARY,
            alpha=0.6, edgecolor="#1A2B4C", lw=0.8)
    x = np.linspace(-4, 4, 200)
    ax.plot(x, norm.pdf(x, 0, 1), color="#000000", lw=1.0, ls="--")
    annotate_corner(
        ax, rf"std = {pull.std():.2f}" "\n" rf"mean = {pull.mean():+.2f}",
        loc="upper right", fontsize=10,
    )
    ax.set_xlim(-4, 4)
    ax.set_xlabel(LABELS["pull"])
    ax.set_ylabel(r"density")
    ax.set_title(r"pull distribution")
    ax.grid(True, alpha=0.30)

    ax = axes[2]
    bins_pct = np.percentile(sigma, np.linspace(0, 100, 11))
    bins_pct = np.unique(bins_pct)
    idx = np.digitize(sigma, bins_pct) - 1
    sigma_med, rmse_obs = [], []
    for k in range(len(bins_pct) - 1):
        m = idx == k
        if m.sum() < 30:
            continue
        sigma_med.append(np.median(sigma[m]))
        r = res[m]
        rmse_obs.append(np.sqrt(np.mean(r * r)))
    sigma_med = np.asarray(sigma_med); rmse_obs = np.asarray(rmse_obs)
    ax.plot(sigma_med, rmse_obs, "o-", color=ACCENT_PRIMARY, lw=1.4,
             markersize=5)
    if rmse_obs.size:
        lo = float(min(sigma_med.min(), rmse_obs.min()))
        hi = float(max(sigma_med.max(), rmse_obs.max()))
        ax.plot([lo, hi], [lo, hi], color="#000000", lw=0.8, ls="--")
        annotate_corner(
            ax, f"RMSE / sigma = {float(np.median(rmse_obs / sigma_med)):.2f}",
            loc="upper left", fontsize=10,
        )
    ax.set_xlabel(r"predicted $\sigma_{\log g}$ [dex]")
    ax.set_ylabel(r"observed RMSE [dex]")
    ax.set_title(r"reliability")
    ax.grid(True, alpha=0.30)

    save(fig, "F13_stream2_calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
