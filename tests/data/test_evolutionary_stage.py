"""Tests for the RGB+HeCB inference filter.

Covers:

1. Default filter: Andrae label authoritative when present, atmospheric
   box used as fallback when not. Documented as the production policy.
2. ``require_andrae=True``: strict mode for catalogues where the Andrae
   label is mandatory.
3. Custom Andrae-label set, custom Teff/log g bounds.
4. NaN propagation and missing columns.
5. ``filter_to_rgb_or_hecb`` count provenance.
6. Frozen-dataclass immutability and JSON fingerprint.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.evolutionary_stage import (
    ACCEPTED_ANDRAE_STAGES,
    DEFAULT_EVOLUTIONARY_STAGE_FILTER,
    EvolutionaryStageFilter,
    filter_to_rgb_or_hecb,
    is_rgb_or_hecb,
)


def test_accepted_stages_are_rgb_rc_and_candidate():
    assert {"RGB", "RC", "RGB_candidate"} == ACCEPTED_ANDRAE_STAGES


def test_filter_is_frozen():
    with pytest.raises(AttributeError):
        DEFAULT_EVOLUTIONARY_STAGE_FILTER.teff_max = 9000  # type: ignore[misc]


def test_fingerprint_is_jsonable():
    fp = DEFAULT_EVOLUTIONARY_STAGE_FILTER.fingerprint()
    payload = json.dumps(fp)
    restored = json.loads(payload)
    assert restored["teff_min"] == 4000.0
    assert restored["teff_max"] == 5500.0
    assert restored["logg_min"] == 1.0
    assert restored["logg_max"] == 3.5
    assert sorted(restored["accepted_andrae"]) == ["RC", "RGB", "RGB_candidate"]
    assert restored["require_andrae"] is False


# --- is_rgb_or_hecb ----------------------------------------------------------


def test_andrae_label_accepts_rgb_rc_candidate():
    df = pd.DataFrame(
        {
            "evolutionary_stage_andrae": ["RGB", "RC", "RGB_candidate", "AGB", None, "MS"],
            "teff": [4500, 4800, 4600, 4700, 4500, 5400],
            "logg": [2.5, 2.5, 2.5, 2.5, 2.5, 4.5],
        }
    )
    flags = is_rgb_or_hecb(df)
    assert flags["andrae_accepted"].tolist() == [True, True, True, False, False, False]


def test_atmospheric_box_accepts_training_pool():
    df = pd.DataFrame(
        {
            "teff": [4500, 5400, 3900, 5600, np.nan, 4500],
            "logg": [2.0, 3.4, 2.5, 2.0, 2.5, 0.5],
        }
    )
    flags = is_rgb_or_hecb(df)
    # In-box: row 0, 1; out-of-box: row 2 (cool), row 3 (hot), row 4 (NaN), row 5 (low logg).
    assert flags["atmospheric_accepted"].tolist() == [True, True, False, False, False, False]


def test_default_policy_falls_back_to_atmospheric_when_andrae_missing():
    """Star with no Andrae label but in the atmospheric box passes the gate."""
    df = pd.DataFrame(
        {
            "evolutionary_stage_andrae": [None, "AGB", None],
            "teff": [4500, 4500, 5800],
            "logg": [2.5, 2.5, 4.5],
        }
    )
    flags = is_rgb_or_hecb(df)
    # Row 0: no Andrae, in box → pass.
    # Row 1: AGB Andrae label → reject (does NOT fall back to atmospheric).
    # Row 2: no Andrae, out of box → reject.
    assert flags["rgb_or_hecb"].tolist() == [True, False, False]


def test_require_andrae_strict_mode():
    df = pd.DataFrame(
        {
            "evolutionary_stage_andrae": [None, "RGB", "RC"],
            "teff": [4500, 4500, 4500],
            "logg": [2.5, 2.5, 2.5],
        }
    )
    strict = EvolutionaryStageFilter(require_andrae=True)
    flags = is_rgb_or_hecb(df, filt=strict)
    # Star without Andrae label is rejected even though atmospheric box passes.
    assert flags["rgb_or_hecb"].tolist() == [False, True, True]


def test_handles_missing_andrae_column_via_atmospheric_fallback():
    df = pd.DataFrame({"teff": [4500, 5800], "logg": [2.5, 4.5]})
    flags = is_rgb_or_hecb(df)
    assert flags["andrae_accepted"].tolist() == [False, False]
    assert flags["rgb_or_hecb"].tolist() == [True, False]


def test_handles_missing_teff_logg_columns():
    df = pd.DataFrame({"evolutionary_stage_andrae": ["RGB", "AGB"]})
    flags = is_rgb_or_hecb(df)
    assert flags["atmospheric_accepted"].tolist() == [False, False]
    # With Andrae present, RGB still passes via the Andrae path.
    assert flags["rgb_or_hecb"].tolist() == [True, False]


def test_custom_atmospheric_bounds():
    df = pd.DataFrame({"teff": [4200, 4800, 6200], "logg": [2.0, 2.0, 2.0]})
    custom = EvolutionaryStageFilter(teff_min=3500.0, teff_max=6000.0)
    flags = is_rgb_or_hecb(df, filt=custom)
    # Wider Teff window keeps the 6200 K star out (above 6000); 4200 K stays in.
    assert flags["atmospheric_accepted"].tolist() == [True, True, False]


def test_filter_to_rgb_or_hecb_returns_filtered_frame_and_counts():
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "evolutionary_stage_andrae": ["RGB", "AGB", None, "RC", None],
            "teff": [4500, 4500, 4500, 4800, 7000],
            "logg": [2.5, 2.5, 2.5, 2.5, 4.5],
        }
    )
    out, counts = filter_to_rgb_or_hecb(df)
    # RGB (Andrae) + RC (Andrae) + Andrae-less in box = 3 rows.
    assert counts["n_in"] == 5
    assert counts["n_out"] == 3
    assert counts["n_andrae_accepted"] == 2
    assert out["source_id"].tolist() == [1, 3, 4]


def test_strict_filter_rejects_andraeless_in_filter_to_rgb_or_hecb():
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "evolutionary_stage_andrae": ["RGB", None, "AGB"],
            "teff": [4500, 4500, 4500],
            "logg": [2.5, 2.5, 2.5],
        }
    )
    strict = EvolutionaryStageFilter(require_andrae=True)
    out, counts = filter_to_rgb_or_hecb(df, filt=strict)
    assert out["source_id"].tolist() == [1]
    assert counts["n_out"] == 1
