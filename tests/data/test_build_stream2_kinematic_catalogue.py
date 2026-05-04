"""Tests for the Stream-2 kinematic-catalogue orchestrator.

Three layers:

1. ``assign_distance_trust_flags`` correctness on hand-built fixtures.
2. ``build_stream2_kinematic_catalogue`` end-to-end on a synthetic
   Stream-2-shaped frame: BJ21 + av-layer joined → trust flags →
   extinction → galpy. Verifies the parquet + sidecar are emitted.
3. Graceful-degradation when BJ21 or av_layer is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.build_stream2_kinematic_catalogue import (
    DEFAULT_DIST_RELATIVE_SPREAD_FLAG,
    assign_distance_trust_flags,
    build_stream2_kinematic_catalogue,
)

# --- assign_distance_trust_flags ---------------------------------------------


def test_trust_flags_positive_case():
    df = pd.DataFrame(
        {
            "r_med_photogeo": [1.0, 2.0],
            "r_lo_photogeo": [0.95, 1.95],
            "r_hi_photogeo": [1.05, 2.05],
            "parallax": [1.0, 0.5],
        }
    )
    flags = assign_distance_trust_flags(df)
    assert flags["dist_has_bj21"].tolist() == [True, True]
    assert flags["dist_relative_spread_high"].tolist() == [False, False]
    assert flags["dist_negative_parallax"].tolist() == [False, False]
    assert flags["dist_trustworthy"].tolist() == [True, True]


def test_trust_flags_high_spread_demotes():
    df = pd.DataFrame(
        {
            "r_med_photogeo": [1.0],
            "r_lo_photogeo": [0.5],
            "r_hi_photogeo": [1.5],  # spread = 1.0 / 1.0 = 1.0 > 0.30
            "parallax": [1.0],
        }
    )
    flags = assign_distance_trust_flags(df)
    assert flags["dist_relative_spread_high"][0]
    assert not flags["dist_trustworthy"][0]


def test_trust_flags_negative_parallax_demotes():
    df = pd.DataFrame(
        {
            "r_med_photogeo": [1.0],
            "r_lo_photogeo": [0.95],
            "r_hi_photogeo": [1.05],
            "parallax": [-0.5],
        }
    )
    flags = assign_distance_trust_flags(df)
    assert flags["dist_negative_parallax"][0]
    assert not flags["dist_trustworthy"][0]


def test_trust_flags_missing_bj21_columns():
    df = pd.DataFrame({"source_id": [1, 2]})
    flags = assign_distance_trust_flags(df)
    assert flags["dist_has_bj21"].tolist() == [False, False]
    assert flags["dist_relative_spread_high"].tolist() == [True, True]
    assert flags["dist_trustworthy"].tolist() == [False, False]


def test_trust_flags_threshold_override():
    df = pd.DataFrame(
        {
            "r_med_photogeo": [1.0],
            "r_lo_photogeo": [0.85],
            "r_hi_photogeo": [1.15],  # spread 0.3 = exactly the default threshold
            "parallax": [1.0],
        }
    )
    # default 0.30: spread > 0.30 ⇒ False (the comparison is strict >)
    flags = assign_distance_trust_flags(df)
    assert not flags["dist_relative_spread_high"][0]  # spread == 0.30, not > 0.30
    flags_lax = assign_distance_trust_flags(df, relative_spread_threshold=0.5)
    assert not flags_lax["dist_relative_spread_high"][0]


# --- build_stream2_kinematic_catalogue ---------------------------------------


@pytest.fixture
def synthetic_stream2():
    """A frame shaped like ingest_stream2 output, plus optional BJ21/av frames."""
    rng = np.random.default_rng(20260429)
    n = 32
    source_id = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    stream2 = pd.DataFrame(
        {
            "source_id": source_id,
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-90, 90, n),
            "parallax": rng.uniform(0.5, 5.0, n),
            "parallax_over_error": rng.uniform(8.0, 50.0, n),
            "pmra": rng.standard_normal(n) * 5.0,
            "pmdec": rng.standard_normal(n) * 5.0,
            "radial_velocity": rng.standard_normal(n) * 30.0,
            "g_mag": rng.uniform(11.0, 14.0, n),
            "j_mag": rng.uniform(10.0, 13.0, n),
            "h_mag": rng.uniform(9.5, 12.5, n),
            "k_mag": rng.uniform(9.3, 12.3, n),
            "w1_mag": rng.uniform(9.2, 12.2, n),
            "w2_mag": rng.uniform(9.15, 12.15, n),
        }
    )
    bj21 = pd.DataFrame(
        {
            "source_id": source_id,
            "r_med_photogeo": rng.uniform(0.4, 2.5, n),
            "r_lo_photogeo": np.nan,
            "r_hi_photogeo": np.nan,
        }
    )
    bj21["r_lo_photogeo"] = bj21["r_med_photogeo"] * 0.95
    bj21["r_hi_photogeo"] = bj21["r_med_photogeo"] * 1.05
    av_layer = pd.DataFrame(
        {
            "source_id": source_id,
            "av_edenhofer": np.where(
                bj21["r_med_photogeo"] <= 1.25, rng.uniform(0.0, 0.5, n), np.nan
            ),
            "av_lallement": np.where(
                (bj21["r_med_photogeo"] > 1.25) & (bj21["r_med_photogeo"] <= 3.0),
                rng.uniform(0.2, 1.0, n),
                np.nan,
            ),
            "av_sfd": rng.uniform(0.4, 1.5, n),
            "av_nbhd_median": rng.uniform(0.3, 1.0, n),
            "av_nbhd_std": rng.uniform(0.05, 0.4, n),
        }
    )
    return stream2, bj21, av_layer


def test_build_writes_parquet_and_sidecar(synthetic_stream2, tmp_path: Path):
    stream2, bj21, av = synthetic_stream2
    out_path = tmp_path / "stream2_kinematic.parquet"
    result = build_stream2_kinematic_catalogue(
        stream2,
        output_path=out_path,
        bj21_df=bj21,
        av_layer_df=av,
    )
    assert result == out_path
    assert out_path.exists()

    sidecar = out_path.with_suffix(".provenance.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["row_count_before"] == len(stream2)
    assert payload["row_count_after"] == len(stream2)
    extinction = payload["extra"]["extinction_correction"]
    assert extinction["applied"] is True
    assert extinction["law"]["name"] == "CCM89_RV3.1_Yuan2013"
    assert "dist_trustworthy" in payload["extra"]["distance_trust_counts"]


def test_build_emits_dereddened_columns(synthetic_stream2, tmp_path: Path):
    stream2, bj21, av = synthetic_stream2
    out_path = tmp_path / "out.parquet"
    build_stream2_kinematic_catalogue(stream2, output_path=out_path, bj21_df=bj21, av_layer_df=av)
    written = pd.read_parquet(out_path)
    for band in ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"):
        assert f"{band}_dered" in written.columns
    for flag in (
        "dist_has_bj21",
        "dist_relative_spread_high",
        "dist_negative_parallax",
        "dist_trustworthy",
        "av_is_neighborhood_fallback",
        "av_distance_prior_dominated",
        "av_neighbourhood_high_dispersion",
    ):
        assert flag in written.columns


def test_build_skips_extinction_when_no_av_layer(synthetic_stream2, tmp_path: Path):
    stream2, bj21, _ = synthetic_stream2
    out_path = tmp_path / "no_av.parquet"
    build_stream2_kinematic_catalogue(stream2, output_path=out_path, bj21_df=bj21, av_layer_df=None)
    written = pd.read_parquet(out_path)
    # av_los should not be added when no av layer is supplied.
    assert "av_los" not in written.columns
    sidecar = json.loads(out_path.with_suffix(".provenance.json").read_text())
    assert sidecar["extra"]["extinction_correction"]["applied"] is False


def test_build_skips_actions_when_no_bj21(synthetic_stream2, tmp_path: Path):
    stream2, _, av = synthetic_stream2
    out_path = tmp_path / "no_bj.parquet"
    build_stream2_kinematic_catalogue(stream2, output_path=out_path, bj21_df=None, av_layer_df=av)
    written = pd.read_parquet(out_path)
    assert "r_med_photogeo" in written.columns  # column injected as NaN
    assert written["r_med_photogeo"].isna().all()
    assert not written["dist_trustworthy"].any()


def test_build_uses_threshold_override(synthetic_stream2, tmp_path: Path):
    """Setting a tighter threshold should demote more stars."""
    stream2, bj21, _ = synthetic_stream2
    # Inflate the spread on half of the stars so the threshold matters.
    bj21 = bj21.copy()
    bj21.loc[bj21.index[::2], "r_hi_photogeo"] = bj21.loc[bj21.index[::2], "r_med_photogeo"] * 1.2
    bj21.loc[bj21.index[::2], "r_lo_photogeo"] = bj21.loc[bj21.index[::2], "r_med_photogeo"] * 0.7

    out_loose = tmp_path / "loose.parquet"
    build_stream2_kinematic_catalogue(
        stream2,
        output_path=out_loose,
        bj21_df=bj21,
        relative_spread_threshold=0.6,
    )
    out_tight = tmp_path / "tight.parquet"
    build_stream2_kinematic_catalogue(
        stream2,
        output_path=out_tight,
        bj21_df=bj21,
        relative_spread_threshold=0.10,
    )
    n_loose = int(pd.read_parquet(out_loose)["dist_trustworthy"].sum())
    n_tight = int(pd.read_parquet(out_tight)["dist_trustworthy"].sum())
    assert n_loose >= n_tight


def test_default_threshold_is_documented():
    assert DEFAULT_DIST_RELATIVE_SPREAD_FLAG == 0.30
