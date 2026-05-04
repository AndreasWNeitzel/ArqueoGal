"""Shared loader: Stream-3 kinematics ⨝ predictions ⨝ Tier 1 mask.

Used by Y26-Y29. Returns one DataFrame keyed on source_id with all the
kinematic + chemistry columns needed for Toomre / E-Lz / action-space /
R-z plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S3 = REPO / "data/processed/pipeline1_predictions_stream3.parquet"
KIN_S3 = REPO / "data/processed/pipeline2_kinematics_stream3_volume.parquet"
HYBRID_S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"

# Solar circular velocity in McMillan+2017 potential (used by kinematics.py).
V_LSR_KMS = 233.1


def load_kin_chem() -> pd.DataFrame:
    pred_cols = [
        "source_id",
        "teff_pred",
        "logg_pred",
        "mh_pred",
        "alpha_m_pred",
        "mg_h_pred",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    pred = pd.read_parquet(PRED_S3, columns=pred_cols).drop_duplicates("source_id")
    pred["kin_ood_flag"] = False
    if HYBRID_S3.exists():
        h = pd.read_parquet(HYBRID_S3, columns=["source_id", "kin_ood_flag"])
        pred = pred.merge(h, on="source_id", how="left", suffixes=("", "_h"))
        if "kin_ood_flag_h" in pred.columns:
            pred["kin_ood_flag"] = pred["kin_ood_flag_h"].fillna(False).astype(bool)
            pred = pred.drop(columns=["kin_ood_flag_h"])
    pred["release_tier"] = assign_release_tier(pred).astype(np.int8)

    kin = pd.read_parquet(KIN_S3)
    df = pred.merge(kin, on="source_id", how="inner")
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    return df
