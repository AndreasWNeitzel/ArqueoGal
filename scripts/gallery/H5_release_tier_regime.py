"""H5: Release tier regime visualisation, with Tier 2 split by demoting gate.

Per release.py:569-672, a star is in Tier 2 because at least one of the
following demotion gates fired:

  T2-σ_Teff      teff_sigma   > 150 K       (per-element σ-inflation)
  T2-σ_logg      logg_sigma   > 0.30 dex
  T2-σ_M/H       mh_sigma     > 0.20 dex
  T2-σ_α/M       alpha_m_sigma > 0.05 dex
  T2-σ_Mg/H      mg_h_sigma   > 0.20 dex
  T2-kin_ood     kin_ood_flag = True        (aux-channel demotion;
                                              demotes [α/M] and [Mg/H] only)

A single star may fire multiple gates simultaneously and therefore appear in
multiple sub-population panels. T1 (no gate fired) and T3 (XP-Mahalanobis OOD
or per-element NaN) are mutually exclusive with the T2 sub-populations.

Layout per stream (one figure each):
  Row 1 (Kiel)      :  one panel per sub-population — T1, T2-σ_*, T2-kin_ood, T3
  Row 2 (chemistry) :  same column ordering, [M/H] vs [α/M]
  Row 3 (σ bars)    :  per-sub-population mean predicted σ, Teff (K) and dex
                       elements split into two adjacent sub-axes so the dex
                       bars are not crushed by the K scale.
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

from _common import apply_style, save_fig  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

OUT = REPO / "reports/gallery/H_hybrid_release"
ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
ELEMENT_LABELS = ("Teff", r"$\log g$", "[M/H]", r"[$\alpha$/M]", "[Mg/H]")

# Per-element σ-inflation thresholds — release.py:_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD
SIGMA_THRESHOLDS = {
    "teff": 150.0,
    "logg": 0.30,
    "mh": 0.20,
    "alpha_m": 0.05,
    "mg_h": 0.20,
}

PRED_PATHS = {
    1: REPO / "data/processed/pipeline1_predictions_stream1.parquet",
    2: REPO / "data/processed/pipeline1_predictions_stream2.parquet",
    3: REPO / "data/processed/pipeline1_predictions_stream3.parquet",
}
HYBRID_PATHS = {
    2: REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet",
    3: REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet",
}


def _load_stream(stream_id: int) -> pd.DataFrame:
    pred_path = PRED_PATHS[stream_id]
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    needed = [
        "source_id", "teff_pred", "logg_pred",
        "mh_pred", "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma", "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    try:
        df = pd.read_parquet(pred_path, columns=needed)
    except (KeyError, ValueError):
        df = pd.read_parquet(pred_path, columns=needed[:-1])
    df = df.drop_duplicates(subset="source_id", keep="first")
    # kin_ood retired 2026-05-03; tier no longer reads it.
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    return df


def _subgroup_masks(df: pd.DataFrame, has_kin: bool = False) -> dict[str, np.ndarray]:
    """Tier partition (T1 / T2 / T3) on the new label-Mahalanobis tier scheme.

    The 2026-05-03 redesign collapsed the previous σ-threshold + kin_ood
    sub-populations into a single T2 driver: label-extrapolation flag
    (5-D Mahalanobis on the predicted label vector vs APOGEE-truth
    envelope). So T2 is no longer a multi-panel breakout.
    """
    tier = df["release_tier"].to_numpy()
    return {
        "T1": tier == 1,
        "T2": tier == 2,
        "T3": tier == 3,
    }


def _column_label(name: str) -> str:
    pretty = {
        "T1": "T1 (science)",
        "T2": "T2 (label-Mahal extrapolation)",
        "T3": "T3 (input OOD)",
    }
    return pretty.get(name, name)


def _column_color(name: str) -> str:
    if name == "T1":
        return "#2ca02c"
    if name == "T3":
        return "#d62728"
    return "#ff7f0e"


def main(argv: list[str] | None = None) -> int:
    apply_style()
    print("[H5] Loading per-stream predictions for tier-composition diagnostic")

    streams: dict[int, pd.DataFrame] = {}
    for sid in (1, 2, 3):
        try:
            streams[sid] = _load_stream(sid)
            tc = streams[sid]["release_tier"].value_counts().to_dict()
            print(f"  Stream {sid}: {len(streams[sid]):,} stars; tier counts {tc}")
        except FileNotFoundError as e:
            print(f"  Warning: Stream {sid} unavailable: {e}")

    if not streams:
        print("Error: no streams found")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    KIEL_GRID = 70
    CHEM_GRID = 70
    KIEL_EXTENT = (3500, 6500, 0.5, 4.0)
    CHEM_EXTENT = (-2.5, 0.6, -0.20, 0.55)

    for sid, df in streams.items():
        masks = _subgroup_masks(df)
        # Drop any subgroup that is empty so the panel grid stays compact.
        cols = [name for name, m in masks.items() if int(m.sum()) > 0]
        ncol = len(cols)
        # 2 rows × ncol: Kiel + chemistry per tier subgroup. Bar plots dropped
        # 2026-05-03 — with σ-thresholds gone, mean-σ per tier is no longer
        # the load-bearing diagnostic. Suptitle dropped per same edit.
        fig, axes = plt.subplots(2, ncol, figsize=(5.5 * ncol, 9.5),
                                 squeeze=False)

        # Row 0: Kiel hex per subgroup.
        for j, name in enumerate(cols):
            ax = axes[0, j]
            sub = df.loc[masks[name]]
            n = int(len(sub))
            if n > 0:
                hb = ax.hexbin(
                    sub["teff_pred"], sub["logg_pred"],
                    gridsize=KIEL_GRID, cmap="viridis", mincnt=1,
                    bins="log", extent=KIEL_EXTENT,
                )
                plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
            ax.set_xlabel(r"$T_{\rm eff}$ (K)")
            if j == 0:
                ax.set_ylabel(r"$\log g$ (dex)")
            ax.set_title(f"Stream {sid} — {_column_label(name)}  n={n:,}",
                         fontsize=11)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.grid(alpha=0.3)

        # Row 1: chemistry hex per subgroup.
        for j, name in enumerate(cols):
            ax = axes[1, j]
            sub = df.loc[masks[name]]
            n = int(len(sub))
            if n > 0:
                hb = ax.hexbin(
                    sub["mh_pred"], sub["alpha_m_pred"],
                    gridsize=CHEM_GRID, cmap="viridis", mincnt=1,
                    bins="log", extent=CHEM_EXTENT,
                )
                plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N")
            ax.axhline(0.15, color="white", lw=0.6, ls=":", alpha=0.7)
            ax.set_xlabel("[M/H] (dex)")
            if j == 0:
                ax.set_ylabel(r"[$\alpha$/M] (dex)")
            ax.grid(alpha=0.3)

        fig.tight_layout()
        save_fig(fig, OUT / f"H5_release_tier_regime_stream{sid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
