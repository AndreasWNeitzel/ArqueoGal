"""Shared data loaders for v1.4 figure scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
PRED_S2 = REPO / "data/processed/pipeline1_predictions_stream2.parquet"
FEAT_S2 = REPO / "data/processed/pipeline1_features_stream2.parquet"
GAIA_RAW_S1 = REPO / "data/interim/stream1_gaia_dr3_raw.parquet"
GAIA_RAW_S2 = REPO / "data/interim/stream2_gaia_dr3_raw.parquet"
APOGEE_DR19 = REPO / "data/interim/apogee_dr19_corrected.parquet"
TESS_GAIA = REPO / "data/interim/stream2_tess_gaia.parquet"

OUT_FIGS = REPO / "reports" / "gallery" / "v1_4" / "figs"
LIT_CSV = REPO / "data" / "processed" / "literature_rmse.csv"


def load_s1_holdout() -> pd.DataFrame:
    """Stream-1 holdout (val + test, seed=0) with truth, prediction, tier."""
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma",
        "ood_joint_flag", "label_extrapolation_flag",
        "ood_mahalanobis_score", "label_mahalanobis_score",
        "ood_mahalanobis_percentile", "label_mahalanobis_percentile",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = [
        "source_id", "teff_apogee", "logg_apogee", "mh_apogee",
        "alpha_m_apogee", "mg_h_apogee", "fe_h_apogee", "b_deg",
        "ra_deg", "dec_deg", "r_med_photogeo", "av_los",
    ]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout = np.concatenate([split["val"], split["test"]])
    return df.loc[df["source_id"].isin(holdout)].reset_index(drop=True)


def load_s2_predictions() -> pd.DataFrame:
    """Stream-2 predictions + tier."""
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S2, columns=pcols).drop_duplicates("source_id")
    p["release_tier"] = assign_release_tier(p).astype(np.int8)
    return p


def load_s1_gspspec(source_ids) -> pd.DataFrame:
    cols = ["source_id", "mh_gspspec", "alphafe_gspspec",
            "teff_gspspec", "logg_gspspec"]
    df = pd.read_parquet(GAIA_RAW_S1, columns=cols).drop_duplicates("source_id")
    return df.loc[df["source_id"].isin(source_ids)].reset_index(drop=True)


def load_s1_apogee(source_ids) -> pd.DataFrame:
    """APOGEE DR19 cross-match for Stream-1 Tier-1 source_ids."""
    cols = ["source_id", "teff", "logg", "m_h_atm", "alpha_m_atm"]
    df = pd.read_parquet(APOGEE_DR19, columns=cols).drop_duplicates("source_id")
    df = df.loc[df["source_id"].isin(source_ids)].reset_index(drop=True)
    return df.dropna(subset=cols[1:]).reset_index(drop=True)


def load_s2_gspspec() -> pd.DataFrame:
    cols = ["source_id", "mh_gspspec", "alphafe_gspspec",
            "teff_gspspec", "logg_gspspec"]
    df = pd.read_parquet(GAIA_RAW_S2, columns=cols).drop_duplicates("source_id")
    return df.dropna(subset=cols[1:]).reset_index(drop=True)


def load_s2_apogee(source_ids) -> pd.DataFrame:
    cols = ["source_id", "teff", "logg", "m_h_atm", "alpha_m_atm"]
    df = pd.read_parquet(APOGEE_DR19, columns=cols).drop_duplicates("source_id")
    df = df.loc[df["source_id"].isin(source_ids)].reset_index(drop=True)
    return df.dropna(subset=cols[1:]).reset_index(drop=True)


def load_s2_seismic() -> pd.DataFrame:
    cols = ["source_id", "numax_muhz", "e_numax_muhz"]
    return pd.read_parquet(
        TESS_GAIA, columns=cols).drop_duplicates("source_id").reset_index(drop=True)


def seismic_logg(numax_muhz: np.ndarray, teff_k: np.ndarray) -> np.ndarray:
    """Asteroseismic log g via Kjeldsen & Bedding (1995) scaling."""
    NUMAX_SUN = 3090.0
    TEFF_SUN = 5777.0
    LOGG_SUN = 4.438
    return (LOGG_SUN
            + np.log10(numax_muhz / NUMAX_SUN)
            + 0.5 * np.log10(teff_k / TEFF_SUN))
