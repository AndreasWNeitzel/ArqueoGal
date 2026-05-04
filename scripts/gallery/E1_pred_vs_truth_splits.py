"""Stream 1 (APOGEE DR19) — pred vs truth across the holdout test set.

Visualizes pred-vs-truth as 3x5 grid of hexbin plots for the holdout test set
across 5 elements: [Teff, log g, [M/H], [alpha/M], [Mg/H]].

Usage:
  python E1_pred_vs_truth_splits.py --n-stars 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import load_stream1_holdout

from arqueogal.utils.plotting import save_figure, set_aa_style


def stats(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float, float, int]:
    """Compute RMSE, bias, std, and count of finite pairs."""
    finite = np.isfinite(yt) & np.isfinite(yp)
    if not finite.any():
        return np.nan, np.nan, np.nan, 0
    yt_f, yp_f = yt[finite], yp[finite]
    res = yp_f - yt_f
    rmse = float(np.sqrt(np.mean(res**2)))
    bias = float(np.median(res))
    std = float(np.std(res - bias))
    n = int(finite.sum())
    return rmse, bias, std, n


def main(n_stars: int | None = None) -> None:
    set_aa_style(font_size=9.0)

    data = load_stream1_holdout()

    if n_stars is not None and n_stars < len(data):
        data = data.sample(n=n_stars, random_state=42)

    Yt_test = data[
        ["teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee"]
    ].values

    Yp_test = data[["teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred"]].values

    labels_info = [
        ("Teff", r"$T_{\rm eff}$ [K]", (3500, 6500), "K", ".0f"),
        (r"$\log g$", r"$\log g$ (dex)", (0.5, 4.0), "dex", ".2f"),
        ("[M/H]", "[M/H] (dex)", (-2.5, 0.6), "dex", ".2f"),
        (r"[$\alpha$/M]", r"[$\alpha$/M] (dex)", (-0.2, 0.5), "dex", ".2f"),
        ("[Mg/H]", "[Mg/H] (dex)", (-1.5, 0.5), "dex", ".2f"),
    ]

    # 2x5: top row pred-vs-truth (existing), bottom row residual-vs-truth
    # with bias line and ±1σ shaded band.
    fig, axes = plt.subplots(2, 5, figsize=(18.0, 8.4), gridspec_kw={"height_ratios": [1.4, 1.0]})
    fig.suptitle(
        "Stream 1 (APOGEE DR19) - pred vs truth (holdout test set)\n"
        r"top: pred vs truth, hex = $\log_{10}$ N.  "
        r"bottom: residual vs truth, white line = bias, shaded = $\pm 1\sigma$.",
        fontsize=10,
        y=0.995,
    )

    Yt = Yt_test
    Yp = Yp_test
    for c in range(5):
        _, label_str, lim, unit, fmt = labels_info[c]
        ax = axes[0, c]

        rmse, bias, std, n = stats(Yt[:, c], Yp[:, c])
        finite = np.isfinite(Yt[:, c]) & np.isfinite(Yp[:, c])

        # --- top: pred vs truth.
        ax.hexbin(
            Yt[finite, c],
            Yp[finite, c],
            gridsize=40,
            mincnt=3,
            cmap="viridis",
            bins="log",
            extent=[lim[0], lim[1], lim[0], lim[1]],
            edgecolors="face",
        )
        ax.plot(lim, lim, color="red", lw=0.6, ls="--", zorder=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"truth {label_str}", fontsize=9)
        ax.set_ylabel(f"pred {label_str}", fontsize=9)
        if not np.isnan(rmse):
            txt = f"n={n:,}\nRMSE={rmse:{fmt}} {unit}\nbias={bias:+{fmt}}\nstd={std:{fmt}}"
            ax.text(
                0.05,
                0.95,
                txt,
                transform=ax.transAxes,
                fontsize=7,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=2),
            )
        ax.tick_params(labelsize=7)

        # --- bottom: residual vs truth.
        ax_r = axes[1, c]
        resid = Yp[finite, c] - Yt[finite, c]
        # Symmetric residual y-range scaled to ±3σ around bias for legibility.
        if not np.isnan(std) and std > 0:
            r_half = max(3.0 * std, abs(bias) + 2.5 * std)
        else:
            r_half = 1.0
        rlim = (-r_half, r_half)
        ax_r.hexbin(
            Yt[finite, c],
            resid,
            gridsize=40,
            mincnt=3,
            cmap="plasma",
            bins="log",
            extent=[lim[0], lim[1], rlim[0], rlim[1]],
            edgecolors="face",
        )
        ax_r.axhline(0, color="white", lw=0.5, ls="--", alpha=0.4, zorder=4)
        # Bias line + ±1σ shaded band.
        ax_r.axhspan(
            bias - std,
            bias + std,
            color="white",
            alpha=0.20,
            zorder=3,
            label=rf"bias $\pm 1\sigma$  ({bias:+{fmt}}, $\sigma={std:{fmt}}$)",
        )
        ax_r.axhline(bias, color="white", lw=1.4, ls="-", alpha=0.95, zorder=5)
        ax_r.set_xlim(lim)
        ax_r.set_ylim(rlim)
        ax_r.set_xlabel(f"truth {label_str}", fontsize=9)
        ax_r.set_ylabel(f"residual {unit}", fontsize=8)
        ax_r.legend(fontsize=6, loc="upper right", framealpha=0.85)
        ax_r.tick_params(labelsize=7)
        ax_r.grid(True, alpha=0.25)

    out_dir = REPO / "reports/gallery/E_validation"
    paths = save_figure(fig, out_dir / "E1_pred_vs_truth_splits", formats=("pdf", "png"))
    for p in paths:
        print(f"[E1] wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-stars",
        type=int,
        default=None,
        help="Optional: downsample to N stars (default: use all)",
    )
    args = parser.parse_args()

    main(n_stars=args.n_stars)
