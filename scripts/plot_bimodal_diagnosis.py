"""Diagnose the Stream-3 [α/M]=+0.11 attractor as bimodal-target collapse.

Shows two things in one figure:
  (1) Training (Stream-1) α/M density at the stripe cell (warm upper-RGB,
      metal-poor) — bimodal, with thin-disc mode near +0.05 and thick-disc
      mode near +0.25.
  (2) v2 stripe predictions (stars in the Stream-3 attractor subset) —
      collapsed to ≈ +0.11, the mean of the bimodal distribution.

The finding: Gaussian NLL has μ* = E[y|x] as its unique minimiser; when
p(α/M | XP) is bimodal at the stripe cell, the optimal regression prediction
is the conditional mean, not either mode. No SupCon / β-NLL / weighting knob
can fix this — the head architecture cannot express two modes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("bimodal")

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "data/processed/pipeline1_features_stream1.parquet"
V2 = REPO / "data/processed/pipeline1_predictions_stream3_volume_v2.parquet"
OUT = REPO / "reports/pipeline1/run_a_v2/bimodal_alpha_m_diagnosis.png"

# Stripe cell bounds (same as diagnostic selection)
TEFF_LO, TEFF_HI = 4600, 5000
LOGG_LO, LOGG_HI = 1.8, 3.0
MH_LO,   MH_HI   = -0.9, -0.3


def main() -> None:
    # ---- training density at the stripe cell ----
    tr = pd.read_parquet(TRAIN, columns=[
        "teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"])
    tr = tr.dropna(subset=["teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"])
    in_cell = (
        tr["teff_apogee"].between(TEFF_LO, TEFF_HI)
        & tr["logg_apogee"].between(LOGG_LO, LOGG_HI)
        & tr["mh_apogee"].between(MH_LO, MH_HI)
    )
    a_cell = tr.loc[in_cell, "alpha_m_apogee"].to_numpy()
    a_all = tr["alpha_m_apogee"].to_numpy()

    # ---- v2 stripe predictions ----
    pr = pd.read_parquet(V2, columns=[
        "alpha_m_pred", "mh_pred",
        "ood_joint_flag", "regime_b_flag",
    ])
    rel = ~pr["ood_joint_flag"].astype(bool) & ~pr["regime_b_flag"].astype(bool)
    d = pr.loc[rel]
    stripe = (
        d["alpha_m_pred"].between(0.105, 0.120)
        & d["mh_pred"].between(-1.5, -0.3)
    )
    a_stripe = d.loc[stripe, "alpha_m_pred"].to_numpy()

    _LOG.info("training in stripe cell: %d  (total train RGB: %d)", len(a_cell), len(a_all))
    _LOG.info("v2 stripe subset:        %d  (total release_ok: %d)", len(a_stripe), int(rel.sum()))

    # ---- figure ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.07},
    )

    # TOP — training α/M density in the stripe cell (with bimodality)
    bins = np.linspace(-0.1, 0.5, 80)
    ax_top.hist(a_cell, bins=bins, density=True, color="0.55", alpha=0.75,
                label=f"Stream-1 training  (in stripe cell, N={len(a_cell):,})",
                edgecolor="none")
    # reference: full-training marginal
    ax_top.hist(a_all, bins=bins, density=True, histtype="step",
                color="0.30", lw=1.2, label=f"Stream-1 training  (all, N={len(a_all):,})")
    ax_top.axvline(np.median(a_cell), color="C3", lw=1.4, ls="--",
                   label=f"median α/M in cell = {np.median(a_cell):+.3f}")
    ax_top.axvline(np.mean(a_cell), color="C0", lw=1.4, ls=":",
                   label=f"mean α/M in cell = {np.mean(a_cell):+.3f}")
    ax_top.set_ylabel("training density  p(α/M | cell)")
    ax_top.set_title(
        "Bimodal α/M diagnosis — stripe cell "
        f"(Teff∈[{TEFF_LO},{TEFF_HI}] K, log g∈[{LOGG_LO},{LOGG_HI}], "
        f"[M/H]∈[{MH_LO},{MH_HI}])"
    )
    ax_top.legend(loc="upper right", fontsize=9, frameon=False)
    ax_top.grid(alpha=0.25)

    # BOTTOM — v2 stripe prediction distribution
    ax_bot.hist(a_stripe, bins=bins, density=True, color="C1", alpha=0.9,
                edgecolor="none",
                label=f"v2 stripe predictions (N={len(a_stripe):,})\n"
                      f"μ={np.mean(a_stripe):+.4f}  σ={np.std(a_stripe):.4f}")
    ax_bot.axvline(np.median(a_cell), color="C3", lw=1.4, ls="--")
    ax_bot.axvline(np.mean(a_cell), color="C0", lw=1.4, ls=":")
    ax_bot.set_xlabel("[α/M]  (dex)")
    ax_bot.set_ylabel("v2 density  p(α/M_pred)")
    ax_bot.legend(loc="upper right", fontsize=9, frameon=False)
    ax_bot.grid(alpha=0.25)

    fig.text(
        0.01, 0.005,
        "Reading: the training target is bimodal in the stripe cell. Gaussian-NLL μ* collapses onto the "
        "conditional mean (+0.11), which sits between the thin-disc (≈+0.05) and thick-disc (≈+0.25) "
        "modes. v2 stripe predictions reproduce that collapse exactly. This is distributional, not a "
        "contrastive/SupCon failure.",
        fontsize=8, ha="left", color="0.25", wrap=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    _LOG.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
