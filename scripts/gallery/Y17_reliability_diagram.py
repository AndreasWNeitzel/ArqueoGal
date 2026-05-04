"""Y17: Reliability diagram — predicted σ vs observed RMSE per σ-bin.

For each label, bin stars by predicted σ (10 quantile bins). For each bin,
compute the empirical RMSE of the residuals and plot it against the bin's
mean predicted σ. A perfectly calibrated model lies on the diagonal.

This is the canonical calibration diagnostic for heteroscedastic regressors
(Kuleshov+2018; Levi+2020). Combined with Y16 (pulls), it gives the full
picture: Y16 says "is the average σ right", Y17 says "does σ track risk".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402
from _y_holdout import LABELS, load_holdout  # noqa: E402

N_BINS = 10


def _sigma_axis_max(key: str) -> float:
    return {"teff": 250.0, "logg": 0.45, "mh": 0.30, "alpha_m": 0.10, "mg_h": 0.30}[key]


def main() -> int:
    apply_style()
    df = load_holdout()
    n = len(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.40, hspace=0.40)
    axes = axes.ravel()
    axes[5].axis("off")

    for ax, spec in zip(axes[:5], LABELS):
        k = spec["key"]
        delta = df[f"{k}_pred"].to_numpy() - df[f"{k}_apogee"].to_numpy()
        sig = df[f"{k}_sigma"].to_numpy()
        ok = np.isfinite(delta) & np.isfinite(sig) & (sig > 0)
        delta, sig = delta[ok], sig[ok]

        # Quantile-bin by predicted σ.
        edges = np.quantile(sig, np.linspace(0.0, 1.0, N_BINS + 1))
        edges[0] -= 1e-6
        bin_idx = np.clip(np.searchsorted(edges, sig, side="right") - 1, 0, N_BINS - 1)
        sig_centres = np.array([float(np.mean(sig[bin_idx == b])) for b in range(N_BINS)])
        rmse_emp = np.array(
            [float(np.sqrt(np.mean(delta[bin_idx == b] ** 2))) for b in range(N_BINS)]
        )
        # Bootstrap 1σ error band on RMSE per bin.
        rng = np.random.default_rng(0)
        n_boot = 200
        rmse_boot = np.empty((n_boot, N_BINS), dtype=np.float64)
        for b in range(N_BINS):
            d_b = delta[bin_idx == b]
            for j in range(n_boot):
                draw = rng.choice(d_b, size=len(d_b), replace=True)
                rmse_boot[j, b] = float(np.sqrt(np.mean(draw**2)))
        lo = np.quantile(rmse_boot, 0.16, axis=0)
        hi = np.quantile(rmse_boot, 0.84, axis=0)

        smax = _sigma_axis_max(k)
        ax.plot(
            [0, smax],
            [0, smax],
            color=PALETTE["accent"],
            ls="--",
            lw=2.0,
            label="ideal (calibrated)",
            zorder=1,
        )
        ax.fill_between(
            sig_centres,
            lo,
            hi,
            color=PALETTE["navy_light"],
            alpha=0.30,
            zorder=2,
            label="1σ bootstrap",
        )
        ax.plot(
            sig_centres,
            rmse_emp,
            "o-",
            color=PALETTE["navy"],
            lw=2.2,
            ms=8,
            mec="white",
            mew=1.3,
            label="empirical",
            zorder=3,
        )
        ax.set_xlim(0, smax)
        ax.set_ylim(0, smax)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"predicted $\sigma$ {spec['name']} ({spec['unit']})")
        ax.set_ylabel(rf"observed RMSE {spec['name']} ({spec['unit']})")
        ax.set_title(spec["name"], color=PALETTE["navy"])
        ax.legend(loc="upper left", fontsize=9.5)

        # Mean ratio annotation.
        ratio = float(np.mean(rmse_emp / np.where(sig_centres > 0, sig_centres, np.nan)))
        verdict_color = PALETTE["tier1"] if 0.8 <= ratio <= 1.2 else PALETTE["tier2"]
        ax.text(
            0.97,
            0.04,
            f"⟨RMSE/σ⟩ = {ratio:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=verdict_color, edgecolor="none"),
        )

    headline(
        fig,
        "Reliability — does σ track risk?",
        f"Stream 1 Tier 1 held-out, n = {n:,}.  "
        "Per σ-quantile bin: mean predicted σ vs observed RMSE.  "
        "Diagonal is perfect calibration.",
        top=0.90,
    )
    save(fig, "Y17_reliability_diagram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
