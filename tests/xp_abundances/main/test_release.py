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
    assign_release_tier,
    tier_counts,
)


def _row(
    *,
    ood_joint=False, latent_support=False,
    regime_b=False, mode_ambiguous=False,
    ood_disagreement=False, aux_missing=False,
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


@pytest.mark.parametrize("caveat_kw", [
    {"regime_b": True},
    {"mode_ambiguous": True},
    {"ood_disagreement": True},
    {"aux_missing": True},
])
def test_single_caveat_demotes_to_tier_2(caveat_kw):
    df = pd.DataFrame([_row(**caveat_kw)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 2


@pytest.mark.parametrize("kill_kw", [
    {"ood_joint": True},
    {"latent_support": True},
    {"pred_nan": True},
])
def test_single_hard_kill_demotes_to_tier_3(kill_kw):
    df = pd.DataFrame([_row(**kill_kw)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 3


def test_hard_kill_trumps_caveat():
    df = pd.DataFrame([_row(regime_b=True, ood_joint=True)])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 3


def test_missing_flag_columns_are_treated_as_false():
    # Strip all caveat flags — remaining row must be Tier 1
    row = _row()
    for key in ("regime_b_flag", "mode_ambiguous_flag", "ood_disagreement_flag",
                "aux_missing_any", "latent_support_flag"):
        row.pop(key)
    df = pd.DataFrame([row])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 1


def test_tier_counts():
    rows = [
        _row(),                         # tier 1
        _row(regime_b=True),            # tier 2
        _row(mode_ambiguous=True),      # tier 2
        _row(ood_joint=True),           # tier 3
        _row(pred_nan=True),            # tier 3
    ]
    df = pd.DataFrame(rows)
    df["release_tier"] = assign_release_tier(df)
    counts = tier_counts(df)
    assert counts == {1: 1, 2: 2, 3: 2}


def test_annotate_parquet_round_trip(tmp_path: Path):
    rows = [
        _row(),
        _row(regime_b=True),
        _row(ood_joint=True),
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
    df = pd.DataFrame([_row(), _row(regime_b=True)])
    pq = tmp_path / "pred.parquet"
    df.to_parquet(pq)
    s1 = annotate_parquet(pq)
    s2 = annotate_parquet(pq)
    assert s1 == s2
    out = pd.read_parquet(pq)
    # Exactly one release_tier column, not duplicated
    assert list(out.columns).count("release_tier") == 1
