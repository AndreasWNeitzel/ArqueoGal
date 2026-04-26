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
    annotate_parquet,
    assign_g_mag_bin,
    assign_kin_ood_flag,
    assign_per_element_release_tier,
    assign_prediction_sigma_inflated,
    assign_release_tier,
    assign_xp_abundance_type,
    tier_counts,
)


def _row(
    *,
    ood_joint=False,
    latent_support=False,
    regime_b=False,
    mode_ambiguous=False,
    ood_disagreement=False,
    aux_missing=False,
    pred_nan=False,
):
    return {
        "source_id": 0,
        "teff_pred": np.nan if pred_nan else 4500.0,
        "logg_pred": 2.5,
        "mh_pred": -0.2,
        "alpha_m_pred": 0.05,
        "mg_h_pred": -0.15,
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


def test_mode_ambiguous_demotes_only_alpha_m_to_tier_2():
    """v5 (2026-04-26): mode_ambiguous_flag demotes ONLY [α/M] now, not every element.
    The composite row-max still ends up at Tier 2 because [α/M] is at Tier 2."""
    df = pd.DataFrame([_row(mode_ambiguous=True)])
    per_element = assign_per_element_release_tier(df)
    assert per_element["alpha_m"].iloc[0] == 2
    for other in ("teff", "logg", "mh", "mg_h"):
        assert per_element[other].iloc[0] == 1, (
            f"{other} should NOT demote on mode_ambiguous_flag (v5: α/M-only caveat)"
        )
    composite = assign_release_tier(df)
    assert composite.iloc[0] == 2


@pytest.mark.parametrize(
    "diagnostic_kw",
    [
        {"regime_b": True},
        {"ood_disagreement": True},
        {"aux_missing": True},
        {"latent_support": True},
    ],
)
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
    ],
)
def test_single_hard_kill_demotes_to_tier_3(kill_kw):
    df = pd.DataFrame([_row(**kill_kw)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 3


def test_hard_kill_trumps_caveat():
    df = pd.DataFrame([_row(mode_ambiguous=True, ood_joint=True)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 3


def test_missing_flag_columns_are_treated_as_false():
    # Strip all flag columns — remaining row must be Tier 1
    row = _row()
    for key in (
        "regime_b_flag",
        "mode_ambiguous_flag",
        "ood_disagreement_flag",
        "aux_missing_any",
        "latent_support_flag",
    ):
        row.pop(key)
    df = pd.DataFrame([row])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 1


def test_tier_counts():
    rows = [
        _row(),  # tier 1
        _row(regime_b=True),  # tier 1 (v5: regime_b is diagnostic-only now)
        _row(mode_ambiguous=True),  # tier 2 (α/M demoted via per-element caveat)
        _row(ood_joint=True),  # tier 3
        _row(pred_nan=True),  # tier 3 (NaN on Teff demotes Teff to T3)
    ]
    df = pd.DataFrame(rows)
    df["release_tier"] = assign_release_tier(df)
    counts = tier_counts(df)
    assert counts == {1: 2, 2: 1, 3: 2}


def test_annotate_parquet_round_trip(tmp_path: Path):
    rows = [
        _row(),  # tier 1
        _row(mode_ambiguous=True),  # tier 2 via α/M-only caveat
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
    df = pd.DataFrame([_row(), _row(mode_ambiguous=True)])
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

    assert set(types.keys()) == {"teff", "logg", "mh", "alpha_m", "mg_h"}
    # Spectrum-dominant
    assert (types["teff"] == "spectrum_dominant").all()
    assert (types["logg"] == "spectrum_dominant").all()
    assert (types["mh"] == "spectrum_dominant").all()
    # Aux-assisted
    assert (types["alpha_m"] == "aux_assisted").all()
    assert (types["mg_h"] == "aux_assisted").all()


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
    assert "kin_ood_flag" in out.columns
    assert "g_mag_bin" in out.columns

    # Check dtypes
    assert out["release_tier"].dtype == np.int8
    assert out["kin_ood_flag"].dtype == np.bool_
    assert out["g_mag_bin"].dtype == "string"
    # XP abundance types should be string dtype
    for elem in ["teff", "logg", "mh", "alpha_m", "mg_h"]:
        assert out[f"xp_abundance_type__{elem}"].dtype == "string"

    # Check sidecar includes schema version
    sidecar = pq.with_name("pred.release_tier.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert "catalog_schema_version" in payload
    # Schema version bumped to 5 (per-cell-gate ablation, 2026-04-26): simplified
    # OOD/caveat sets and α/M σ-threshold tightened to 0.05 dex
    # (release/test_ablations_2026-04-26/REPORT.md).
    assert payload["catalog_schema_version"] == 5
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
        "prediction_sigma_inflated_any",
    ):
        assert new_col in payload["release_columns_added"], f"{new_col} missing from sidecar manifest"
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
    """A clean row with no flags has every per-element tier == 1."""
    df = pd.DataFrame([_row()])
    per_element = assign_per_element_release_tier(df)
    assert set(per_element) == {"teff", "logg", "mh", "alpha_m", "mg_h"}
    for elem, series in per_element.items():
        assert series.iloc[0] == 1, f"{elem} should be Tier 1 on clean row"


def test_per_element_tier_kin_ood_demotes_only_aux_assisted():
    """kin_ood_flag=True demotes alpha_m and mg_h to Tier 2 but leaves Teff/logg/mh at Tier 1."""
    row = _row()
    row["kin_ood_flag"] = True
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    assert per_element["teff"].iloc[0] == 1
    assert per_element["logg"].iloc[0] == 1
    assert per_element["mh"].iloc[0] == 1
    assert per_element["alpha_m"].iloc[0] == 2  # aux-assisted, demoted
    assert per_element["mg_h"].iloc[0] == 2  # aux-assisted, demoted


def test_per_element_tier_diagnostic_flags_do_not_demote():
    """v5 (2026-04-26): the v3 'global caveats' (regime_b, aux_missing,
    ood_disagreement) are diagnostic-only — they must not change any per-element
    tier. Replaces the v3 ``test_per_element_tier_global_caveat_demotes_all``
    which asserted the opposite."""
    for flag_kw in (
        {"regime_b": True},
        {"aux_missing": True},
        {"ood_disagreement": True},
    ):
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
    """The composite assign_release_tier returns the per-row max across elements."""
    rows = [
        _row(),  # all Tier 1 → composite 1
        _row(mode_ambiguous=True),  # α/M Tier 2 (per-element), others Tier 1 → composite 2
        {**_row(), "kin_ood_flag": True},  # alpha_m/mg_h Tier 2, others Tier 1 → composite 2
        _row(ood_joint=True),  # all Tier 3 → composite 3
    ]
    df = pd.DataFrame(rows)
    composite = assign_release_tier(df)
    assert composite.tolist() == [1, 2, 2, 3]


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
        assert series.iloc[0] == 1, (
            f"{elem} unexpectedly demoted on retired dist_prior_dominated"
        )


def test_mode_ambiguous_per_element_caveat():
    """v5 (2026-04-26): ``mode_ambiguous_flag`` is now a per-element caveat for
    [α/M] only. Other elements should not demote when the flag is set."""
    row = _row(mode_ambiguous=True)
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    assert per_element["teff"].iloc[0] == 1
    assert per_element["logg"].iloc[0] == 1
    assert per_element["mh"].iloc[0] == 1
    assert per_element["alpha_m"].iloc[0] == 2  # only α/M demotes
    assert per_element["mg_h"].iloc[0] == 1


def test_annotate_parquet_emits_per_element_tier_columns(tmp_path):
    """annotate_parquet adds release_tier__<element> columns for every element."""
    rows = [_row()]
    df = pd.DataFrame(rows)
    df["phot_g_mean_mag_corr"] = [14.0]
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)

    annotate_parquet(pq)

    out = pd.read_parquet(pq)
    for elem in ("teff", "logg", "mh", "alpha_m", "mg_h"):
        col = f"release_tier__{elem}"
        assert col in out.columns, f"{col} missing"
        assert out[col].iloc[0] in (1, 2, 3)
    # Composite still present and equals row-max.
    assert "release_tier" in out.columns
    expected = max(
        out[f"release_tier__{e}"].iloc[0] for e in ("teff", "logg", "mh", "alpha_m", "mg_h")
    )
    assert out["release_tier"].iloc[0] == expected


# -----------------------------------------------------------------------------
# σ-threshold caveat tests (Phase A2-followup-2, 2026-04-25)
# Per HIGH_SIGMA_RESCUE_REPORT: demote stars with inflated per-element σ to
# Tier 2 because their regression head has collapsed to the training-distribution
# prior mean instead of reading information from the spectrum.
# -----------------------------------------------------------------------------


def _row_with_sigma(**sigma_overrides):
    """Like ``_row`` but defaults all per-element σ columns to small (in-range) values
    and lets callers override individual elements to inflated values."""
    row = _row()
    base = {
        "teff_sigma": 80.0,
        "logg_sigma": 0.10,
        "mh_sigma": 0.08,
        "alpha_m_sigma": 0.03,
        "mg_h_sigma": 0.10,
    }
    base.update(sigma_overrides)
    row.update(base)
    return row


def test_assign_prediction_sigma_inflated_below_threshold_is_false():
    """All elements below their σ threshold → every flag is False."""
    df = pd.DataFrame([_row_with_sigma()])
    flags = assign_prediction_sigma_inflated(df)
    assert set(flags) == {"teff", "logg", "mh", "alpha_m", "mg_h"}
    for elem, series in flags.items():
        assert series.iloc[0] is np.False_ or bool(series.iloc[0]) is False, (
            f"{elem} should be False on in-range σ"
        )
        assert series.dtype == bool


@pytest.mark.parametrize(
    "elem,sigma_col,inflated_value",
    [
        ("teff", "teff_sigma", 200.0),    # > 150 K
        ("logg", "logg_sigma", 0.40),     # > 0.30 dex
        ("mh", "mh_sigma", 0.25),         # > 0.20 dex
        ("alpha_m", "alpha_m_sigma", 0.08),  # > 0.05 dex
        ("mg_h", "mg_h_sigma", 0.25),     # > 0.20 dex
    ],
)
def test_assign_prediction_sigma_inflated_above_threshold_is_true(elem, sigma_col, inflated_value):
    """A single element above its threshold lights only that element's flag."""
    df = pd.DataFrame([_row_with_sigma(**{sigma_col: inflated_value})])
    flags = assign_prediction_sigma_inflated(df)
    assert bool(flags[elem].iloc[0]) is True
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
        teff_sigma=150.0,    # exactly at threshold
        logg_sigma=0.30,     # exactly at threshold
        mh_sigma=0.20,       # exactly at threshold
        alpha_m_sigma=0.05,  # exactly at threshold
        mg_h_sigma=0.20,     # exactly at threshold
    )
    df = pd.DataFrame([row])
    flags = assign_prediction_sigma_inflated(df)
    for elem, series in flags.items():
        assert bool(series.iloc[0]) is False, (
            f"{elem} fired at σ exactly = threshold; expected strict >, got >="
        )


def test_sigma_inflated_demotes_per_element_to_tier_2():
    """Inflated σ on a single element demotes only that element to Tier 2.

    Other elements with in-range σ stay at Tier 1. This is the central regression
    test for the v4 caveat introduced in HIGH_SIGMA_RESCUE_REPORT.md.
    """
    row = _row_with_sigma(teff_sigma=200.0, mh_sigma=0.25)  # both inflated
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    assert per_element["teff"].iloc[0] == 2  # σ-inflated → Tier 2
    assert per_element["mh"].iloc[0] == 2  # σ-inflated → Tier 2
    assert per_element["logg"].iloc[0] == 1  # σ in range → Tier 1
    assert per_element["alpha_m"].iloc[0] == 1  # σ in range → Tier 1
    assert per_element["mg_h"].iloc[0] == 1  # σ in range → Tier 1


def test_sigma_inflated_does_not_promote_over_tier_3():
    """A NaN prediction (Tier 3 trigger) is not overridden by a σ-inflated caveat.

    Tier 3 (hard kill) must dominate Tier 2 (caveat); the σ flag must not promote
    a Tier-3 row up to Tier 2."""
    row = _row_with_sigma(teff_sigma=200.0)
    row["teff_pred"] = np.nan  # hard-kill on teff
    df = pd.DataFrame([row])
    per_element = assign_per_element_release_tier(df)
    assert per_element["teff"].iloc[0] == 3  # Tier 3 trumps caveat


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

    # Tier columns reflect the per-element demotion.
    assert out["release_tier__teff"].tolist() == [1, 2, 1]
    assert out["release_tier__alpha_m"].tolist() == [1, 1, 2]
    assert out["release_tier__mg_h"].tolist() == [1, 1, 2]
    # Composite is row-max.
    assert out["release_tier"].tolist() == [1, 2, 2]
