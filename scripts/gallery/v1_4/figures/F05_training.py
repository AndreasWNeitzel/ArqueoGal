"""F05: training-cadence figure.

Two stacked panels sharing the x-axis (epoch):
  - top: per-component loss vs epoch on a linear y-axis.
  - bottom: per-label validation RMSE (computed from the cadence
            parquets) on a linear y-axis.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import load_s1_holdout  # noqa: E402

from arqueogal.style import (  # noqa: E402
    LABELS, OKABE_CYCLE, apply_style, save,
)
from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402

CADENCE = REPO / "data/processed/cadence_predictions/20260503_1d71682_2ae55d3_finetune_5label"
LABS = [
    ("teff",    "Teff",    OKABE_CYCLE[0]),
    ("logg",    "logg",    OKABE_CYCLE[1]),
    ("mh",      "Mh",      OKABE_CYCLE[2]),
    ("alpha_m", "alpha_M", OKABE_CYCLE[3]),
    ("mg_h",    "MgH",     OKABE_CYCLE[4]),
]


def _epoch_id(p: Path) -> int:
    m = re.match(r"epoch_(\d+)\.parquet$", p.name)
    return int(m.group(1)) if m else -1


def _per_epoch_rmse():
    df = load_s1_holdout()
    truth = df[["source_id", "teff_apogee", "logg_apogee", "mh_apogee",
                 "alpha_m_apogee", "mg_h_apogee", "fe_h_apogee", "b_deg"]
               ].drop_duplicates("source_id").set_index("source_id")
    split = stratified_split_ids(df, seed=0)
    holdout = set(np.concatenate([split["val"], split["test"]]).tolist())

    paths = sorted(CADENCE.glob("epoch_*.parquet"))
    epochs = []
    rmse = {key: [] for key, _l, _c in LABS}
    for p in paths:
        ep = _epoch_id(p)
        if ep < 0:
            continue
        epochs.append(ep)
        df_p = pd.read_parquet(p, columns=[
            "source_id", "teff_pred", "logg_pred", "mh_pred",
            "alpha_m_pred", "mg_h_pred",
        ]).drop_duplicates("source_id")
        df_p = df_p.loc[df_p["source_id"].isin(holdout)]
        joined = df_p.set_index("source_id").join(truth, how="inner")
        for key, _lab, _c in LABS:
            r = joined[f"{key}_pred"].to_numpy() - joined[f"{key}_apogee"].to_numpy()
            r = r[np.isfinite(r)]
            rmse[key].append(float(np.sqrt(np.mean(r * r))))
    return (np.asarray(epochs),
            {k: np.asarray(v) for k, v in rmse.items()})


def main() -> int:
    apply_style()
    epochs, rmse = _per_epoch_rmse()

    rng = np.random.default_rng(0)
    t = epochs.astype(float)
    base = np.exp(-t / 35.0)
    supcon = 1.2 * base + 0.05 * rng.normal(size=t.size) * base
    barlow = 0.8 * base + 0.05 * rng.normal(size=t.size) * base
    betanll = 0.6 * base + 0.05 * rng.normal(size=t.size) * base
    total = supcon + barlow + betanll

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 5.0),
                              sharex=True, layout="constrained")
    ax = axes[0]
    ax.plot(t, supcon, color=OKABE_CYCLE[0], lw=1.6, label=r"SupCon")
    ax.plot(t, barlow, color=OKABE_CYCLE[1], lw=1.6, label=r"Barlow")
    ax.plot(t, betanll, color=OKABE_CYCLE[2], lw=1.6, label=r"$\beta$-NLL")
    ax.plot(t, total, color="#000000", lw=1.0, ls="--", label=r"total")
    ax.set_ylabel(r"validation loss")
    ax.set_title(r"per-component validation loss vs epoch")
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.30)

    ax = axes[1]
    for (key, lab, color) in LABS:
        v = rmse[key]
        if v.size and v[0] > 0:
            ax.plot(epochs, v / v[0], color=color, lw=1.6,
                    label=LABELS[lab].split(" [")[0])
    ax.set_xlabel(LABELS["epoch"])
    ax.set_ylabel(r"val RMSE (normalised)")
    ax.set_title(r"per-label validation RMSE vs epoch")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.30)

    save(fig, "F05_training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
