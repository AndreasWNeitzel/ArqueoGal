"""Tests for arqueogal.data.extinction (interstellar dereddening).

Five layers of coverage:

1. Frozen-constants regression on Yuan+2013 ratios (any drift fails loud).
2. ``ExtinctionLaw`` immutability + sidecar-fingerprint round-trip.
3. ``select_av`` priority logic across the dust-map fusion regimes.
4. ``assign_av_quality`` flag semantics (neighbourhood fallback, prior-
   dominated distance, high-dispersion sightline).
5. ``deredden_broadband`` + ``apply_extinction_corrections`` end-to-end:
   linear correctness, NaN propagation, train/inference parity, MappingProxy
   immutability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.extinction import (
    AV_SOURCE_CODES,
    DEFAULT_EXTINCTION_LAW,
    YUAN2013_AV_RATIOS,
    ExtinctionLaw,
    apply_extinction_corrections,
    assign_av_quality,
    deredden_broadband,
    dereddened_column_names,
    select_av,
)

# --- Frozen constants ---------------------------------------------------------


def test_yuan2013_ratios_match_published_values():
    """Regression guard: Yuan+2013 ratios are part of the v1 contract."""
    expected = {
        "j_mag": 0.276,
        "h_mag": 0.176,
        "k_mag": 0.112,
        "w1_mag": 0.063,
        "w2_mag": 0.050,
    }
    assert dict(YUAN2013_AV_RATIOS) == expected


def test_yuan2013_ratios_are_immutable():
    """MappingProxyType blocks accidental drift post-import."""
    with pytest.raises(TypeError):
        YUAN2013_AV_RATIOS["j_mag"] = 0.30  # type: ignore[index]


def test_default_law_uses_ccm89_rv31():
    assert DEFAULT_EXTINCTION_LAW.name == "CCM89_RV3.1_Yuan2013"
    assert DEFAULT_EXTINCTION_LAW.r_v == 3.1
    assert DEFAULT_EXTINCTION_LAW.av_ratios is YUAN2013_AV_RATIOS


def test_extinction_law_fingerprint_is_jsonable():
    fp = DEFAULT_EXTINCTION_LAW.fingerprint()
    import json

    payload = json.dumps(fp)
    restored = json.loads(payload)
    assert restored["name"] == "CCM89_RV3.1_Yuan2013"
    assert restored["r_v"] == 3.1
    assert restored["av_ratios"]["j_mag"] == pytest.approx(0.276)


def test_extinction_law_is_frozen_dataclass():
    """Frozen + slots prevents post-construction mutation."""
    with pytest.raises(AttributeError):
        DEFAULT_EXTINCTION_LAW.r_v = 5.0  # type: ignore[misc]


# --- select_av ---------------------------------------------------------------


def _fixture(distance_kpc, av_eden=np.nan, av_lall=np.nan, av_sfd=np.nan, av_nbhd=np.nan):
    return pd.DataFrame(
        {
            "r_med_photogeo": np.atleast_1d(distance_kpc).astype(np.float64),
            "av_edenhofer": np.atleast_1d(av_eden).astype(np.float64),
            "av_lallement": np.atleast_1d(av_lall).astype(np.float64),
            "av_sfd": np.atleast_1d(av_sfd).astype(np.float64),
            "av_nbhd_median": np.atleast_1d(av_nbhd).astype(np.float64),
        }
    )


def test_select_av_prefers_edenhofer_inside_125_kpc():
    df = _fixture(distance_kpc=0.5, av_eden=0.3, av_lall=0.5, av_sfd=1.0)
    av, src = select_av(df)
    assert av[0] == pytest.approx(0.3)
    assert src[0] == AV_SOURCE_CODES["edenhofer"]


def test_select_av_falls_to_lallement_in_mid_regime():
    df = _fixture(distance_kpc=2.0, av_eden=np.nan, av_lall=0.6, av_sfd=1.5)
    av, src = select_av(df)
    assert av[0] == pytest.approx(0.6)
    assert src[0] == AV_SOURCE_CODES["lallement"]


def test_select_av_uses_sfd_beyond_3_kpc():
    df = _fixture(distance_kpc=5.0, av_lall=np.nan, av_sfd=1.2)
    av, src = select_av(df)
    assert av[0] == pytest.approx(1.2)
    assert src[0] == AV_SOURCE_CODES["sfd"]


def test_select_av_uses_neighborhood_when_all_3d_maps_missing():
    df = _fixture(distance_kpc=2.5, av_nbhd=0.7)
    av, src = select_av(df)
    assert av[0] == pytest.approx(0.7)
    assert src[0] == AV_SOURCE_CODES["neighborhood_median"]


def test_select_av_marks_missing_when_every_layer_is_nan():
    df = _fixture(distance_kpc=1.0)
    av, src = select_av(df)
    assert np.isnan(av[0])
    assert src[0] == AV_SOURCE_CODES["missing"]


def test_select_av_handles_distanceless_stars():
    """No distance ⇒ priority walk (Edenhofer → Lallement → SFD)."""
    df = _fixture(distance_kpc=np.nan, av_eden=np.nan, av_lall=0.4, av_sfd=0.9)
    av, src = select_av(df)
    assert av[0] == pytest.approx(0.4)
    assert src[0] == AV_SOURCE_CODES["lallement"]


def test_select_av_cross_fills_when_preferred_layer_is_nan():
    """1.0 kpc star with Edenhofer NaN must fall to Lallement, not give up."""
    df = _fixture(distance_kpc=1.0, av_eden=np.nan, av_lall=0.5)
    av, src = select_av(df)
    assert av[0] == pytest.approx(0.5)
    assert src[0] == AV_SOURCE_CODES["lallement"]


# --- assign_av_quality -------------------------------------------------------


def test_av_quality_flags_neighborhood_fallback():
    av = np.array([0.3, 0.5])
    src = np.array(
        [AV_SOURCE_CODES["edenhofer"], AV_SOURCE_CODES["neighborhood_median"]],
        dtype=np.int8,
    )
    flags = assign_av_quality(av, src)
    assert flags["av_is_neighborhood_fallback"].tolist() == [False, True]


def test_av_quality_flags_distance_prior_dominated():
    av = np.array([0.2, 0.4, 0.6])
    src = np.full(3, AV_SOURCE_CODES["edenhofer"], dtype=np.int8)
    flags = assign_av_quality(
        av,
        src,
        parallax_over_error=np.array([10.0, 3.0, np.nan]),
        parallax_snr_floor=5.0,
    )
    assert flags["av_distance_prior_dominated"].tolist() == [False, True, False]


def test_av_quality_flags_high_dispersion_sightline():
    av = np.array([0.2, 0.4])
    src = np.full(2, AV_SOURCE_CODES["edenhofer"], dtype=np.int8)
    flags = assign_av_quality(
        av,
        src,
        av_neighbourhood_std=np.array([0.05, 0.7]),
        nbhd_std_high_mag=0.5,
    )
    assert flags["av_neighbourhood_high_dispersion"].tolist() == [False, True]


# --- deredden_broadband ------------------------------------------------------


def test_deredden_broadband_linear_correctness():
    """mag_dered = mag − A_V × (A_λ / A_V) for every band."""
    n = 4
    av = np.array([0.0, 0.5, 1.0, 2.0])
    raw = pd.DataFrame(
        {
            "j_mag": np.full(n, 14.00),
            "h_mag": np.full(n, 13.50),
            "k_mag": np.full(n, 13.20),
            "w1_mag": np.full(n, 13.10),
            "w2_mag": np.full(n, 13.05),
        }
    )
    out = deredden_broadband(raw, av)
    np.testing.assert_allclose(out["j_mag_dered"], 14.00 - av * 0.276, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["h_mag_dered"], 13.50 - av * 0.176, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["k_mag_dered"], 13.20 - av * 0.112, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["w1_mag_dered"], 13.10 - av * 0.063, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["w2_mag_dered"], 13.05 - av * 0.050, rtol=0, atol=1e-12)


def test_deredden_broadband_propagates_nan_av():
    """Stars with NaN A_V get NaN dereddened columns, raw kept intact."""
    raw = pd.DataFrame({"j_mag": [12.0, 12.0]})
    av = np.array([0.5, np.nan])
    out = deredden_broadband(raw, av)
    assert out["j_mag_dered"].iloc[0] == pytest.approx(12.0 - 0.5 * 0.276)
    assert np.isnan(out["j_mag_dered"].iloc[1])
    # Raw column retained.
    assert out["j_mag"].tolist() == [12.0, 12.0]


def test_deredden_broadband_propagates_nan_raw_mag():
    """NaN raw magnitude propagates to NaN dereddened, with finite Av."""
    raw = pd.DataFrame({"j_mag": [12.0, np.nan]})
    av = np.array([0.5, 0.5])
    out = deredden_broadband(raw, av)
    assert out["j_mag_dered"].iloc[0] == pytest.approx(12.0 - 0.5 * 0.276)
    assert np.isnan(out["j_mag_dered"].iloc[1])


def test_deredden_broadband_skips_absent_band():
    """Missing band ⇒ all-NaN dereddened column, no exception."""
    raw = pd.DataFrame({"j_mag": [12.0, 12.5]})
    av = np.array([0.0, 0.5])
    out = deredden_broadband(raw, av, bands=("j_mag", "h_mag"))
    assert "h_mag_dered" in out.columns
    assert out["h_mag_dered"].isna().all()


def test_deredden_broadband_train_inference_parity():
    """Same A_V + same raw mag ⇒ byte-identical dereddened mag at train and inference.

    This is the load-bearing invariant: the transform is deterministic, no
    fitted parameters, no per-call state, no random seed. Dataset drift is
    impossible without a deliberate change to YUAN2013_AV_RATIOS.
    """
    raw = pd.DataFrame({"j_mag": [12.0, 13.0, 14.0], "h_mag": [11.5, 12.5, 13.5]})
    av = np.array([0.1, 0.5, 1.5])
    train = deredden_broadband(raw.copy(), av)
    infer = deredden_broadband(raw.copy(), av)
    pd.testing.assert_frame_equal(train, infer)


def test_deredden_broadband_rejects_av_shape_mismatch():
    raw = pd.DataFrame({"j_mag": [12.0, 13.0]})
    with pytest.raises(ValueError, match="av must be 1-D with length"):
        deredden_broadband(raw, np.array([0.5]))


def test_deredden_broadband_rejects_unknown_band_in_law():
    raw = pd.DataFrame({"u_mag": [12.0]})
    av = np.array([0.5])
    with pytest.raises(KeyError, match="A_lambda/A_V ratio for u_mag"):
        deredden_broadband(raw, av, bands=("u_mag",))


def test_dereddened_column_names_default():
    assert dereddened_column_names() == (
        "j_mag_dered",
        "h_mag_dered",
        "k_mag_dered",
        "w1_mag_dered",
        "w2_mag_dered",
    )


# --- apply_extinction_corrections (end-to-end) -------------------------------


def test_apply_extinction_corrections_end_to_end():
    """Full ingestion stanza: select_av → flags → deredden → all columns set."""
    n = 3
    df = pd.DataFrame(
        {
            "r_med_photogeo": [0.5, 2.0, 5.0],
            "parallax_over_error": [10.0, 4.0, 8.0],
            "av_edenhofer": [0.3, np.nan, np.nan],
            "av_lallement": [np.nan, 0.6, np.nan],
            "av_sfd": [np.nan, np.nan, 1.4],
            "av_nbhd_median": [np.nan, np.nan, np.nan],
            "av_nbhd_std": [0.05, 0.6, 0.10],
            "j_mag": np.full(n, 13.0),
            "h_mag": np.full(n, 12.5),
            "k_mag": np.full(n, 12.2),
            "w1_mag": np.full(n, 12.1),
            "w2_mag": np.full(n, 12.05),
        }
    )
    out = apply_extinction_corrections(df)

    # av_los populated from the right map per regime.
    np.testing.assert_allclose(out["av_los"].to_numpy(), [0.3, 0.6, 1.4])
    np.testing.assert_array_equal(
        out["av_los_source"].to_numpy(),
        np.array(
            [
                AV_SOURCE_CODES["edenhofer"],
                AV_SOURCE_CODES["lallement"],
                AV_SOURCE_CODES["sfd"],
            ],
            dtype=np.int8,
        ),
    )

    # Trust flags emitted.
    assert out["av_is_neighborhood_fallback"].tolist() == [False, False, False]
    assert out["av_distance_prior_dominated"].tolist() == [False, True, False]
    assert out["av_neighbourhood_high_dispersion"].tolist() == [False, True, False]

    # Dereddened columns linear in av_los.
    np.testing.assert_allclose(
        out["j_mag_dered"].to_numpy(),
        13.0 - np.array([0.3, 0.6, 1.4]) * 0.276,
    )
    np.testing.assert_allclose(
        out["w2_mag_dered"].to_numpy(),
        12.05 - np.array([0.3, 0.6, 1.4]) * 0.050,
    )
    # Raw columns retained.
    assert out["j_mag"].tolist() == [13.0, 13.0, 13.0]


def test_apply_extinction_corrections_does_not_mutate_input_by_default():
    df = pd.DataFrame(
        {
            "r_med_photogeo": [0.5],
            "av_edenhofer": [0.3],
            "av_lallement": [np.nan],
            "av_sfd": [np.nan],
            "av_nbhd_median": [np.nan],
            "j_mag": [12.0],
            "h_mag": [11.5],
            "k_mag": [11.0],
            "w1_mag": [10.9],
            "w2_mag": [10.85],
        }
    )
    snap = df.copy()
    _ = apply_extinction_corrections(df)
    pd.testing.assert_frame_equal(df, snap)


def test_apply_extinction_corrections_alternative_law_changes_output():
    """Sanity: a different ratio set produces a different dereddened value.

    Used by future contributors who want to compare CCM89 vs Wang+Chen 2019:
    the framework is parameterised on ``ExtinctionLaw`` so the comparison is
    a one-liner (instantiate a new law, re-run apply_extinction_corrections).
    """
    df = pd.DataFrame(
        {
            "r_med_photogeo": [0.5],
            "av_edenhofer": [1.0],
            "av_lallement": [np.nan],
            "av_sfd": [np.nan],
            "av_nbhd_median": [np.nan],
            "j_mag": [12.0],
            "h_mag": [11.5],
            "k_mag": [11.0],
            "w1_mag": [10.9],
            "w2_mag": [10.85],
        }
    )
    wang_chen_2019 = ExtinctionLaw(
        name="Wang_Chen_2019_RV3.1",
        r_v=3.1,
        av_ratios={
            "j_mag": 0.282,
            "h_mag": 0.190,
            "k_mag": 0.118,
            "w1_mag": 0.061,
            "w2_mag": 0.046,
        },
    )
    yuan = apply_extinction_corrections(df.copy())
    wang = apply_extinction_corrections(df.copy(), law=wang_chen_2019)
    assert yuan["j_mag_dered"].iloc[0] != wang["j_mag_dered"].iloc[0]
    assert yuan["j_mag_dered"].iloc[0] == pytest.approx(12.0 - 1.0 * 0.276)
    assert wang["j_mag_dered"].iloc[0] == pytest.approx(12.0 - 1.0 * 0.282)
