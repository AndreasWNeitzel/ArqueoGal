"""Tests for ``xp_abundances.main.release``.

Coverage:
- Tier 1 / 2 / 3 decision on hand-crafted rows
- Hard-kill precedence (OOD or NaN prediction trumps caveat)
- Missing-column tolerance (missing flag → treated as False, not crash)
- ``annotate_parquet`` round-trip + sidecar emission
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.xp_abundances.main.release import (
    _coerce_flag_series,
    annotate_parquet,
    assign_g_mag_bin,
    assign_kin_ood_flag,
    assign_per_element_release_tier,
    assign_prediction_sigma_inflated,
    assign_release_tier,
    assign_xp_abundance_type,
    tier_counts,
)


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        # Mixed bool/int/None object Series (from heterogeneous parquet joins),
        # this is the case that broke the prior astype("boolean") implementation.
        (pd.Series([True, 0, False, 1, None], dtype=object), [True, False, False, True, False]),
        # Pure bool Series.
        (pd.Series([True, False, True]), [True, False, True]),
        # Nullable BooleanDtype Series with pd.NA.
        (pd.Series([True, pd.NA, False], dtype="boolean"), [True, False, False]),
        # Plain int 0/1.
        (pd.Series([1, 0, 1]), [True, False, True]),
        # Float series with NaN, NaN should map to False.
        (pd.Series([1.0, np.nan, 0.0]), [True, False, False]),
    ])
def test_coerce_flag_series_handles_every_upstream_dtype(series, expected):
    """The release pipeline receives flag columns under several dtypes; they
    must all collapse to a clean ``bool`` Series. Regression guard against the
    pandas-2.2 FutureWarning fix that originally rejected mixed object series.
    """
    out = _coerce_flag_series(series)
    assert out.dtype == bool
    assert out.tolist() == expected


def _row(
    *,
    ood_joint=False,
    latent_support=False,
    regime_b=False,
    mode_ambiguous=False,
    ood_disagreement=False,
    aux_missing=False,
    pred_nan=False,
    label_extrap=False):
    return {
        "source_id": 0,
        "label_extrapolation_flag": label_extrap,
        "teff_pred": np.nan if pred_nan else 4500.0,
        "logg_pred": 2.5,
        "mh_pred": -0.2,
        "fe_h_pred": -0.15,
        "alpha_m_pred": 0.05,
        "mg_h_pred": -0.15,
        "c_h_pred": 0.0,
        "n_h_pred": 0.1,
        "o_h_pred": 0.05,
        "na_h_pred": -0.2,
        "al_h_pred": 0.1,
        "si_h_pred": 0.0,
        "s_h_pred": 0.05,
        "k_h_pred": -0.1,
        "ca_h_pred": 0.08,
        "ti_h_pred": 0.0,
        "v_h_pred": 0.02,
        "cr_h_pred": -0.05,
        "mn_h_pred": -0.15,
        "ni_h_pred": 0.0,
        "ce_h_pred": 0.1,
        "ood_joint_flag": ood_joint,
        "latent_support_flag": latent_support,
        "regime_b_flag": regime_b,
        "mode_ambiguous_flag": mode_ambiguous,
        "ood_disagreement_flag": ood_disagreement,
        "aux_missing_any": aux_missing,
    }


def test_clean_row_is_tier_1():
    df = pd.DataFrame([_row()])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 1


def test_mode_ambiguous_is_diagnostic_only_in_v6():
    """v6 (2026-05-03, ADR-0016): mode_ambiguous_flag is diagnostic-only.
    The α/M per-element caveat carve-out from v5 was retired because the flag
    fires on ~46 % of the cohort (the disc is genuinely bimodal at fixed
    Teff/log g/[M/H]) and demoting half the catalog was not justified.
    Every element should remain Tier 1 when only this flag fires."""
    df = pd.DataFrame([_row(mode_ambiguous=True)])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, (
            f"{elem} unexpectedly demoted on diagnostic-only mode_ambiguous_flag (v6)"
        )
    composite = assign_release_tier(df)
    assert composite.iloc[0] == 1


@pytest.mark.parametrize(
    "diagnostic_kw",
    [
        {"regime_b": True},
        {"ood_disagreement": True},
        {"aux_missing": True},
        {"latent_support": True},
    ])
def test_diagnostic_only_flags_do_not_change_tier(diagnostic_kw):
    """v5 (2026-04-26): regime_b_flag, ood_disagreement_flag, aux_missing_any,
    latent_support_flag, ood_aux_mahalanobis_flag, dist_prior_dominated are
    retired from tier gating. They are kept as diagnostic columns but no
    longer affect release_tier. See release/test_ablations_2026-04-26/REPORT.md
    for the empirical justification."""
    df = pd.DataFrame([_row(**diagnostic_kw)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 1, (
        f"diagnostic-only flag {list(diagnostic_kw)[0]} unexpectedly demoted "
        f"the row to Tier {tier.iloc[0]}"
    )


@pytest.mark.parametrize(
    "kill_kw",
    [
        {"ood_joint": True},
        {"pred_nan": True},
    ])
def test_single_hard_kill_demotes_to_tier_3(kill_kw):
    df = pd.DataFrame([_row(**kill_kw)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 3


def test_hard_kill_trumps_caveat():
    df = pd.DataFrame([_row(mode_ambiguous=True, ood_joint=True)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 3


def test_missing_diagnostic_flag_columns_are_treated_as_false():
    """The diagnostic-only flag columns may be absent (older parquets); they
    must not affect tier assignment. label_extrapolation_flag remains
    required (see test_assign_release_tier_missing_label_extrap_raises)."""
    row = _row()
    for key in (
        "regime_b_flag",
        "mode_ambiguous_flag",
        "ood_disagreement_flag",
        "aux_missing_any",
        "latent_support_flag"):
        row.pop(key)
    df = pd.DataFrame([row])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 1


def test_assign_release_tier_missing_label_extrap_raises():
    """v6 (2026-05-03, ADR-0016): label_extrapolation_flag is the sole T2
    driver. Absence of the column would silently promote T2 candidates to
    T1; the validation guard in assign_per_element_release_tier MUST raise."""
    row = _row()
    row.pop("label_extrapolation_flag")
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="label_extrapolation_flag"):
        assign_release_tier(df)


def test_label_extrapolation_flag_demotes_to_tier_2_globally():
    """v6 (2026-05-03): a True label_extrapolation_flag demotes EVERY released
    element to Tier 2 (it is a global output-OOD gate, not per-element).
    Composite tier is therefore 2."""
    df = pd.DataFrame([_row(label_extrap=True)])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 2, (
            f"{elem} should be Tier 2 when label_extrapolation_flag fires"
        )
    composite = assign_release_tier(df)
    assert composite.iloc[0] == 2


def test_tier_counts():
    rows = [
        _row(),  # tier 1 (clean)
        _row(regime_b=True),  # tier 1 (regime_b is diagnostic-only)
        _row(mode_ambiguous=True),  # tier 1 (mode_ambiguous diagnostic-only in v6)
        _row(label_extrap=True),  # tier 2 (label-Mahalanobis output-OOD)
        _row(ood_joint=True),  # tier 3
        _row(pred_nan=True),  # tier 3 (NaN on Teff demotes Teff to T3)
    ]
    df = pd.DataFrame(rows)
    df["release_tier"] = assign_release_tier(df)
    counts = tier_counts(df)
    assert counts == {1: 3, 2: 1, 3: 2}


def test_annotate_parquet_round_trip(tmp_path: Path):
    rows = [
        _row(),  # tier 1
        _row(label_extrap=True),  # tier 2 (v6: label-Mahalanobis output-OOD)
        _row(ood_joint=True),  # tier 3
    ]
    df = pd.DataFrame(rows)
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)

    summary = annotate_parquet(pq)

    out = pd.read_parquet(pq)
    assert "release_tier" in out.columns
    assert out["release_tier"].dtype == np.int8
    assert summary["n_rows"] == 3
    assert summary["counts"] == {1: 1, 2: 1, 3: 1}

    sidecar = pq.with_name("pred.release_tier.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["n_rows"] == 3
    assert payload["counts"] == {"1": 1, "2": 1, "3": 1}
    assert "ood_joint_flag" in payload["ood_flags_considered"]


def test_annotate_parquet_is_idempotent(tmp_path: Path):
    df = pd.DataFrame([_row(), _row(label_extrap=True)])
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)
    s1 = annotate_parquet(pq)
    s2 = annotate_parquet(pq)
    assert s1 == s2
    out = pd.read_parquet(pq)
    # Exactly one release_tier column, not duplicated
    assert list(out.columns).count("release_tier") == 1


def test_assign_xp_abundance_type():
    df = pd.DataFrame([_row(), _row()])
    types = assign_xp_abundance_type(df)

    # v6 schema: all 21 elements covered
    expected_elements = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "alpha_m",
        "mg_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    assert set(types.keys()) == expected_elements

    # Spectrum-dominant: teff, logg, mh, and all new elements (pending audit)
    spectrum_dominant = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    for elem in spectrum_dominant:
        assert (types[elem] == "spectrum_dominant").all(), f"{elem} should be spectrum_dominant"

    # Aux-assisted: alpha_m, mg_h (only ones audit-confirmed)
    aux_assisted = {"alpha_m", "mg_h"}
    for elem in aux_assisted:
        assert (types[elem] == "aux_assisted").all(), f"{elem} should be aux_assisted"


def test_assign_kin_ood_flag():
    df = pd.DataFrame([_row(), _row(), _row()])
    flag = assign_kin_ood_flag(df)

    assert flag.dtype == np.bool_
    assert len(flag) == 3
    # v1 placeholder: all False
    assert not flag.any()


def test_assign_g_mag_bin():
    rows = [
        {**_row(), "phot_g_mean_mag_corr": 14.5},
        {**_row(), "phot_g_mean_mag_corr": 15.0},
        {**_row(), "phot_g_mean_mag_corr": 15.5},
        {**_row(), "phot_g_mean_mag_corr": 16.0},
        {**_row(), "phot_g_mean_mag_corr": 16.5},
        {**_row(), "phot_g_mean_mag_corr": 17.0},
    ]
    df = pd.DataFrame(rows)
    bins = assign_g_mag_bin(df)

    assert bins[0] == "bright"
    assert bins[1] == "bright"
    assert bins[2] == "mid"
    assert bins[3] == "mid"
    assert bins[4] == "faint"
    assert bins[5] == "faint"


def test_assign_g_mag_bin_missing_column():
    """When no G column is present, every row gets ``"unknown"`` (graceful degrade,
    not raise) so the consumer can filter on it without aborting the pipeline."""
    df = pd.DataFrame([_row(), _row()])
    bins = assign_g_mag_bin(df)
    assert (bins == "unknown").all()
    assert bins.dtype == "string"


def test_annotate_parquet_adds_all_release_columns(tmp_path: Path):
    rows = [_row(), _row(regime_b=True)]
    df = pd.DataFrame(rows)
    df["phot_g_mean_mag_corr"] = [15.2, 16.3]
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)

    annotate_parquet(pq)

    out = pd.read_parquet(pq)
    # Check all new columns exist
    assert "release_tier" in out.columns
    assert "xp_abundance_type__teff" in out.columns
    assert "xp_abundance_type__logg" in out.columns
    assert "xp_abundance_type__mh" in out.columns
    assert "xp_abundance_type__alpha_m" in out.columns
    assert "xp_abundance_type__mg_h" in out.columns
    # kin_ood_flag is NOT auto-emitted by annotate_parquet as of v6 (2026-05-03).
    # If upstream joined it in, it survives; otherwise it is absent.
    assert "kin_ood_flag" not in out.columns
    assert "g_mag_bin" in out.columns

    # Check dtypes
    assert out["release_tier"].dtype == np.int8
    assert out["g_mag_bin"].dtype == "string"
    # XP abundance types should be string dtype
    for elem in ["teff", "logg", "mh", "alpha_m", "mg_h"]:
        assert out[f"xp_abundance_type__{elem}"].dtype == "string"

    # Check sidecar includes schema version
    sidecar = pq.with_name("pred.release_tier.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert "catalog_schema_version" in payload
    # Schema version bumped to 6 (21-element expansion, 2026-04-28): extended to all
    # 21 labels, using placeholder σ-thresholds (0.15 dex) pending 21-label audit.
    assert payload["catalog_schema_version"] == 6
    # v5 sidecar advertises retired-but-emitted diagnostic flags separately.
    assert "diagnostic_only_columns" in payload
    assert "per_element_caveat_flags" in payload
    assert "release_columns_added" in payload
    assert "tier_gating_logic" in payload
    assert "aux_assisted_elements" in payload
    # Aux-assisted set must be exactly {alpha_m, mg_h} per the [α/M] reframing.
    assert sorted(payload["aux_assisted_elements"]) == ["alpha_m", "mg_h"]
    # v3+v4 columns must be advertised.
    for new_col in (
        "release_tier__teff",
        "release_tier__alpha_m",
        "dist_prior_dominated",
        "prediction_sigma_inflated__teff",
        "prediction_sigma_inflated__alpha_m",
        "prediction_sigma_inflated_any"):
        assert new_col in payload["release_columns_added"], (
            f"{new_col} missing from sidecar manifest"
        )
    # v4 thresholds must be advertised.
    assert "prediction_sigma_inflated_thresholds" in payload
    thr = payload["prediction_sigma_inflated_thresholds"]
    assert thr["teff"] == 150.0
    assert thr["mh"] == 0.20
    assert thr["alpha_m"] == 0.05


def test_aux_assisted_labels_nonzero_uncertainty(tmp_path: Path):
    """Aux-assisted labels should carry valid uncertainty estimates."""
    rows = [
        {
            **_row(),
            "alpha_m_pred": 0.05,
            "mg_h_pred": -0.15,
        }
    ]
    df = pd.DataFrame(rows)
    df["phot_g_mean_mag_corr"] = [15.5]
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)

    annotate_parquet(pq)

    out = pd.read_parquet(pq)
    # Aux-assisted columns should have sensible values
    assert out["xp_abundance_type__alpha_m"].iloc[0] == "aux_assisted"
    assert out["xp_abundance_type__mg_h"].iloc[0] == "aux_assisted"
    # Predictions themselves should be non-NaN
    assert not pd.isna(out["alpha_m_pred"].iloc[0])
    assert not pd.isna(out["mg_h_pred"].iloc[0])


# -----------------------------------------------------------------------------
# Per-element tier-gating tests (Phase A2-followup, META_META §14.3)
# -----------------------------------------------------------------------------


def test_per_element_tier_clean_row_all_tier_1():
    """A clean row with no flags has every per-element tier == 1 (all 21 elements)."""
    df = pd.DataFrame([_row()])
    per_element = assign_per_element_release_tier(df)
    expected_elements = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "alpha_m",
        "mg_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    assert set(per_element) == expected_elements
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, f"{elem} should be Tier 1 on clean row"


def test_per_element_tier_kin_ood_is_diagnostic_only_in_v6():
    """v6 (2026-05-03, ADR-0016): kin_ood_flag is diagnostic-only. The aux-assisted
    demotion was retired because halo / accreted-debris stars are exactly the
    science target for users who want them; demoting them by default was the
    wrong move. The label-Mahalanobis output-OOD gate replaces this path."""
    row = _row()
    row["kin_ood_flag"] = True
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, (
            f"{elem} unexpectedly demoted on diagnostic-only kin_ood_flag (v6)"
        )


def test_per_element_tier_diagnostic_flags_do_not_demote():
    """v5 (2026-04-26): the v3 'global caveats' (regime_b, aux_missing,
    ood_disagreement) are diagnostic-only, they must not change any per-element
    tier. Replaces the v3 ``test_per_element_tier_global_caveat_demotes_all``
    which asserted the opposite."""
    for flag_kw in (
        {"regime_b": True},
        {"aux_missing": True},
        {"ood_disagreement": True}):
        row = _row(**flag_kw)
        df = pd.DataFrame([row])
        per_element = assign_per_element_release_tier(df)
        for elem, series in per_element.items():
            assert series.iloc[0] == 1, (
                f"{elem} unexpectedly demoted on diagnostic-only flag {list(flag_kw)[0]}"
            )


def test_per_element_tier_ood_demotes_all_to_tier_3():
    """A joint OOD flag demotes every element to Tier 3."""
    row = _row(ood_joint=True)
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 3, f"{elem} should be Tier 3 on ood_joint"


def test_per_element_tier_per_element_nan_only_demotes_that_element():
    """A NaN in alpha_m_pred only demotes alpha_m to Tier 3, leaves others Tier 1."""
    row = _row()
    row["alpha_m_pred"] = np.nan
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    assert per_element["teff"].iloc[0] == 1
    assert per_element["logg"].iloc[0] == 1
    assert per_element["mh"].iloc[0] == 1
    assert per_element["alpha_m"].iloc[0] == 3  # NaN prediction
    assert per_element["mg_h"].iloc[0] == 1


def test_assign_release_tier_is_row_max_of_per_element():
    """The composite assign_release_tier returns the per-row max across elements.
    v6 (2026-05-03): only ood_joint_flag (T3) and label_extrapolation_flag (T2)
    drive tier; mode_ambiguous and kin_ood are diagnostic-only."""
    rows = [
        _row(),  # all Tier 1 → composite 1
        _row(label_extrap=True),  # all Tier 2 (global output-OOD) → composite 2
        _row(mode_ambiguous=True),  # diagnostic-only in v6 → all Tier 1 → composite 1
        _row(ood_joint=True),  # all Tier 3 → composite 3
    ]
    df = pd.DataFrame(rows)
    composite = assign_release_tier(df)
    assert composite.tolist() == [1, 2, 1, 3]


def test_aux_mahalanobis_ood_flag_is_diagnostic_only_in_v5():
    """v5 (2026-04-26): ``ood_aux_mahalanobis_flag`` was retired from the OOD set.
    The column may still be present on the parquet (it's emitted by upstream
    annotation as a diagnostic) but it must NOT trigger Tier 3."""
    row = _row()
    row["ood_aux_mahalanobis_flag"] = True
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, (
            f"{elem} unexpectedly demoted on retired ood_aux_mahalanobis_flag"
        )


def test_dist_prior_dominated_is_diagnostic_only_in_v5():
    """v5 (2026-04-26): ``dist_prior_dominated`` was retired from the caveat set.
    The column is still emitted by ``annotate_parquet`` for diagnostic purposes
    but it must NOT trigger Tier 2."""
    row = _row()
    row["dist_prior_dominated"] = True
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, f"{elem} unexpectedly demoted on retired dist_prior_dominated"


def test_mode_ambiguous_is_diagnostic_only_v6():
    """v6 (2026-05-03, ADR-0016): mode_ambiguous_flag was retired as a per-element
    α/M caveat because the underlying ambiguity reflects real disc bimodality at
    fixed (Teff, log g, [M/H]) and fires on ~46 % of the cohort. No element
    demotes when this flag is set."""
    row = _row(mode_ambiguous=True)
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, (
            f"{elem} unexpectedly demoted on diagnostic-only mode_ambiguous_flag"
        )


def test_annotate_parquet_emits_per_element_tier_columns(tmp_path):
    """annotate_parquet adds release_tier__<element> columns for all 21 elements."""
    rows = [_row()]
    df = pd.DataFrame(rows)
    df["phot_g_mean_mag_corr"] = [14.0]
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)

    annotate_parquet(pq)

    out = pd.read_parquet(pq)
    all_elements = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "alpha_m",
        "mg_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    for elem in all_elements:
        col = f"release_tier__{elem}"
        assert col in out.columns, f"{col} missing"
        assert out[col].iloc[0] in (1, 2, 3)
    # Composite still present and equals row-max.
    assert "release_tier" in out.columns
    expected = max(out[f"release_tier__{e}"].iloc[0] for e in all_elements)
    assert out["release_tier"].iloc[0] == expected


# v6 schema tests: 21-element expansion (2026-04-28)
# Demonstrates per-element tier independence and row-max promotion for all 21 labels


def test_21_element_per_element_tiers_all_present():
    """v6 schema: assign_per_element_release_tier returns all 21 elements."""
    df = pd.DataFrame([_row()])
    per_element = assign_per_element_release_tier(df)

    expected_elements = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "alpha_m",
        "mg_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    assert set(per_element.keys()) == expected_elements
    # All clean row, all should be Tier 1
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, f"{elem} should be Tier 1 on clean row"


def test_21_element_per_element_nan_demotes_only_that_element():
    """A NaN in fe_h_pred only demotes fe_h to Tier 3, leaves others Tier 1."""
    row = _row()
    row["fe_h_pred"] = np.nan
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)

    # Only fe_h should be Tier 3; all others Tier 1
    assert per_element["fe_h"].iloc[0] == 3
    for elem in per_element:
        if elem != "fe_h":
            assert per_element[elem].iloc[0] == 1, f"{elem} should remain Tier 1 when fe_h is NaN"


def test_21_element_row_max_composite_tier():
    """Composite release_tier is the per-row max across all 21 per-element tiers."""
    rows = [
        # Row 0: all Tier 1
        _row_with_sigma(),
        # Row 1: one new element (ce_h) has NaN (Tier 3), composite should be 3
        {**_row_with_sigma(), "ce_h_pred": np.nan},
        # Row 2: one mid-tier element (ca_h) has sigma-inflated, composite should be 2
        {**_row_with_sigma(), "ca_h_sigma": 0.30},  # well above 0.15 dex threshold
    ]
    df = pd.DataFrame(rows)

    # Composite should be max across all 21 elements.
    # v6 (2026-05-03): σ-inflation is diagnostic-only; the row-2 ca_h_sigma=0.30
    # is no longer a T2 driver, so the composite stays Tier 1.
    composite = assign_release_tier(df)
    assert composite.iloc[0] == 1  # all Tier 1
    assert composite.iloc[1] == 3  # ce_h is Tier 3 → row is Tier 3
    assert composite.iloc[2] == 1  # ca_h σ-inflation is diagnostic-only in v6


def test_21_element_abundance_types_v6():
    """v6 schema: all 21 elements have xp_abundance_type assigned."""
    df = pd.DataFrame([_row()])
    types = assign_xp_abundance_type(df)

    expected_spectrum_dominant = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    expected_aux_assisted = {"alpha_m", "mg_h"}

    for elem in expected_spectrum_dominant:
        assert (types[elem] == "spectrum_dominant").all(), f"{elem} should be spectrum_dominant"

    for elem in expected_aux_assisted:
        assert (types[elem] == "aux_assisted").all(), f"{elem} should be aux_assisted"


# -----------------------------------------------------------------------------
# σ-threshold caveat tests (Phase A2-followup-2, 2026-04-25)
# Per HIGH_SIGMA_RESCUE_REPORT: demote stars with inflated per-element σ to
# Tier 2 because their regression head has collapsed to the training-distribution
# prior mean instead of reading information from the spectrum.
# v6 extension: all 21 elements now have σ-thresholds (v1.1 placeholders for new ones)
# -----------------------------------------------------------------------------


def _row_with_sigma(**sigma_overrides):
    """Like ``_row`` but defaults all per-element σ columns to small (in-range) values
    and lets callers override individual elements to inflated values.

    v6 extension: includes all 21 sigma columns with conservative defaults (1/3 of
    the v1.1 placeholder thresholds for new elements; 1/2 for tuned ones).
    """
    row = _row()
    base = {
        "teff_sigma": 80.0,
        "logg_sigma": 0.10,
        "mh_sigma": 0.08,
        "fe_h_sigma": 0.05,
        "alpha_m_sigma": 0.03,
        "mg_h_sigma": 0.10,
        "c_h_sigma": 0.05,
        "n_h_sigma": 0.05,
        "o_h_sigma": 0.05,
        "na_h_sigma": 0.05,
        "al_h_sigma": 0.05,
        "si_h_sigma": 0.05,
        "s_h_sigma": 0.05,
        "k_h_sigma": 0.05,
        "ca_h_sigma": 0.05,
        "ti_h_sigma": 0.05,
        "v_h_sigma": 0.05,
        "cr_h_sigma": 0.05,
        "mn_h_sigma": 0.05,
        "ni_h_sigma": 0.05,
        "ce_h_sigma": 0.05,
    }
    base.update(sigma_overrides)
    row.update(base)
    return row


def test_assign_prediction_sigma_inflated_below_threshold_is_false():
    """All 21 elements below their σ threshold → every flag is False."""
    df = pd.DataFrame([_row_with_sigma()])
    flags = assign_prediction_sigma_inflated(df)
    expected_elements = {
        "teff",
        "logg",
        "mh",
        "fe_h",
        "alpha_m",
        "mg_h",
        "c_h",
        "n_h",
        "o_h",
        "na_h",
        "al_h",
        "si_h",
        "s_h",
        "k_h",
        "ca_h",
        "ti_h",
        "v_h",
        "cr_h",
        "mn_h",
        "ni_h",
        "ce_h",
    }
    assert set(flags) == expected_elements
    for elem, series in flags.items():
        assert series.iloc[0] is np.False_ or bool(series.iloc[0]) is False, (
            f"{elem} should be False on in-range σ"
        )
        assert series.dtype == bool


@pytest.mark.parametrize(
    "elem,sigma_col,inflated_value",
    [
        ("teff", "teff_sigma", 200.0),  # > 150 K (Stream-1 tuned)
        ("logg", "logg_sigma", 0.40),  # > 0.30 dex (Stream-1 tuned)
        ("mh", "mh_sigma", 0.25),  # > 0.20 dex (Stream-1 tuned)
        ("alpha_m", "alpha_m_sigma", 0.08),  # > 0.05 dex (Stream-1 tuned, tight)
        ("mg_h", "mg_h_sigma", 0.25),  # > 0.20 dex (Stream-1 tuned)
        ("fe_h", "fe_h_sigma", 0.20),  # > 0.15 dex (v1.1 placeholder)
        ("ca_h", "ca_h_sigma", 0.20),  # > 0.15 dex (v1.1 placeholder)
        ("ce_h", "ce_h_sigma", 0.20),  # > 0.15 dex (v1.1 placeholder)
    ])
def test_assign_prediction_sigma_inflated_above_threshold_is_true(elem, sigma_col, inflated_value):
    """A single element above its threshold lights only that element's flag (21 elements)."""
    df = pd.DataFrame([_row_with_sigma(**{sigma_col: inflated_value})])
    flags = assign_prediction_sigma_inflated(df)
    assert bool(flags[elem].iloc[0]) is True, f"{elem} should be True when σ={inflated_value}"
    for other in set(flags) - {elem}:
        assert bool(flags[other].iloc[0]) is False, (
            f"{other} should not fire on isolated {elem} σ inflation"
        )


def test_assign_prediction_sigma_inflated_missing_sigma_column_is_false():
    """Missing σ columns → flag is False (conservative; do not erroneously demote)."""
    df = pd.DataFrame([_row()])  # no *_sigma columns at all
    flags = assign_prediction_sigma_inflated(df)
    for elem, series in flags.items():
        assert bool(series.iloc[0]) is False, f"{elem} must default to False when σ missing"


def test_assign_prediction_sigma_inflated_nan_is_false():
    """NaN in σ → flag is False (NaN is not 'above threshold')."""
    row = _row_with_sigma(teff_sigma=np.nan, mh_sigma=np.nan)
    df = pd.DataFrame([row])
    flags = assign_prediction_sigma_inflated(df)
    assert bool(flags["teff"].iloc[0]) is False
    assert bool(flags["mh"].iloc[0]) is False


def test_assign_prediction_sigma_inflated_exactly_at_threshold_is_false():
    """The implementation uses strict ``>``, so σ exactly at threshold must NOT
    fire. This pins the boundary condition: a star at exactly 150 K Teff σ is
    not yet 'inflated'; the demotion only kicks in when σ exceeds the
    threshold."""
    row = _row_with_sigma(
        teff_sigma=150.0,  # exactly at threshold
        logg_sigma=0.30,  # exactly at threshold
        mh_sigma=0.20,  # exactly at threshold
        alpha_m_sigma=0.05,  # exactly at threshold
        mg_h_sigma=0.20,  # exactly at threshold
    )
    df = pd.DataFrame([row])
    flags = assign_prediction_sigma_inflated(df)
    for elem, series in flags.items():
        assert bool(series.iloc[0]) is False, (
            f"{elem} fired at σ exactly = threshold; expected strict >, got >="
        )


def test_sigma_inflated_is_diagnostic_only_in_v6():
    """v6 (2026-05-03, ADR-0016): per-element σ-inflation flags are
    diagnostic-only. They are still emitted (tested by
    test_sigma_inflated_emits_columns_in_annotate_parquet) but they no
    longer demote to Tier 2; the label_extrapolation_flag (output-OOD) is the
    sole T2 driver. A row with inflated σ but no other flags stays Tier 1."""
    row = _row_with_sigma(teff_sigma=200.0, mh_sigma=0.25)  # both inflated
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, (
            f"{elem} unexpectedly demoted on diagnostic-only σ-inflation (v6)"
        )


def test_sigma_inflated_does_not_promote_over_tier_3():
    """Even though σ-inflation is diagnostic-only in v6, a NaN prediction must
    still demote to Tier 3, this protects the row-level NaN-as-T3 invariant."""
    row = _row_with_sigma(teff_sigma=200.0)
    row["teff_pred"] = np.nan  # hard-kill on teff
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    assert per_element["teff"].iloc[0] == 3  # NaN trumps everything


def test_sigma_inflated_emits_columns_in_annotate_parquet(tmp_path):
    """``annotate_parquet`` writes ``prediction_sigma_inflated__<elem>`` and the
    ``prediction_sigma_inflated_any`` aggregate, with correct values per row."""
    rows = [
        _row_with_sigma(),  # all in range
        _row_with_sigma(teff_sigma=200.0),  # teff inflated only
        _row_with_sigma(alpha_m_sigma=0.15, mg_h_sigma=0.30),  # two aux-assisted inflated
    ]
    df = pd.DataFrame(rows)
    df["phot_g_mean_mag_corr"] = [14.5, 15.2, 16.1]
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)

    annotate_parquet(pq)
    out = pd.read_parquet(pq)

    for elem in ("teff", "logg", "mh", "alpha_m", "mg_h"):
        col = f"prediction_sigma_inflated__{elem}"
        assert col in out.columns, f"{col} missing from annotated parquet"
        assert out[col].dtype == np.bool_

    assert out["prediction_sigma_inflated__teff"].tolist() == [False, True, False]
    assert out["prediction_sigma_inflated__alpha_m"].tolist() == [False, False, True]
    assert out["prediction_sigma_inflated__mg_h"].tolist() == [False, False, True]

    # Aggregate is row-OR across elements.
    assert "prediction_sigma_inflated_any" in out.columns
    assert out["prediction_sigma_inflated_any"].tolist() == [False, True, True]

    # v6 (2026-05-03): σ-inflation is diagnostic-only, it no longer demotes
    # to Tier 2. The columns are still emitted (asserted above) but tier
    # assignment ignores them. All three rows therefore stay Tier 1.
    assert out["release_tier__teff"].tolist() == [1, 1, 1]
    assert out["release_tier__alpha_m"].tolist() == [1, 1, 1]
    assert out["release_tier__mg_h"].tolist() == [1, 1, 1]
    assert out["release_tier"].tolist() == [1, 1, 1]
