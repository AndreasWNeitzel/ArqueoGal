"""Shared helper: load Stream-1 Tier-1 held-out predictions + APOGEE truth.

Used by Y15 / Y16 / Y17 / Y18, same cohort, different metric. Centralised
here so a single edit updates the calibration story coherently across the
four diagnostic figures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

# Env override lets the eval figures read from a non-canonical predictions
# parquet without editing every Y script. Set ARQUEOGAL_PRED_S1=path.parquet
# to evaluate a different model run; unset to use the canonical 5-ensemble.
PRED_S1 = Path(os.environ.get(
    "ARQUEOGAL_PRED_S1",
    str(REPO / "data/processed/pipeline1_predictions_stream1.parquet")))
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

LABELS = (
    {"key": "teff", "name": r"$T_{\rm eff}$", "unit": "K",  "rmse_unit": "K"},
    {"key": "logg", "name": r"$\log g$",       "unit": "dex","rmse_unit": "dex"},
    {"key": "mh",   "name": "[M/H]",            "unit": "dex","rmse_unit": "dex"},
    {"key": "alpha_m", "name": r"[$\alpha$/M]","unit": "dex","rmse_unit": "dex"},
    {"key": "mg_h", "name": "[Mg/H]",           "unit": "dex","rmse_unit": "dex"})


def load_holdout() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma", "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    pred = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = [
        "source_id", "fe_h_apogee", "teff_apogee", "b_deg",
        "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee",
    ]
    feat = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    ho = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)
    return df.loc[df["release_tier"] == 1].reset_index(drop=True)
