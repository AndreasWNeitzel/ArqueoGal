"""Tests for ``arqueogal.data.release_pipeline``.

Covers the iter-1 hardening that added provenance-sidecar emission, cardinality
assertion, and frozen-stats fingerprint cross-referencing to
``join_predictions_with_features``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.release_pipeline import (
    _FEATURE_JOIN_COLS,
    _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD,
    attach_hybrid_columns,
    join_predictions_with_features,
)


def _make_predictions(n: int, *, source_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Build a tiny predictions parquet matching the Pipeline 1 Stream 3 schema."""
    if source_ids is None:
        source_ids = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "source_id": source_ids,
            "teff_pred": rng.normal(4500, 200, n).astype(np.float32),
            "logg_pred": rng.normal(2.5, 0.4, n).astype(np.float32),
            "mh_pred": rng.normal(-0.4, 0.3, n).astype(np.float32),
            "alpha_m_pred": rng.normal(0.1, 0.1, n).astype(np.float32),
            "mg_h_pred": rng.normal(-0.3, 0.2, n).astype(np.float32),
            "ood_joint_flag": np.zeros(n, dtype=bool),
        },
    )


def _make_features(n: int, *, source_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Build a tiny features parquet covering the columns release_pipeline needs."""
    if source_ids is None:
        source_ids = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(1)
    cols = {c: np.full(n, np.nan, dtype=np.float32) for c in _FEATURE_JOIN_COLS}
    cols["source_id"] = source_ids
    cols["g_mag"] = rng.uniform(11, 16, n).astype(np.float32)
    cols["bp_mag"] = cols["g_mag"] + 0.5
    cols["rp_mag"] = cols["g_mag"] - 0.5
    cols["bp_rp"] = (cols["bp_mag"] - cols["rp_mag"]).astype(np.float32)
    cols["parallax_corr"] = rng.uniform(0.1, 5.0, n).astype(np.float32)
    cols["parallax_error"] = rng.uniform(0.01, 0.1, n).astype(np.float32)
    cols["r_med_photogeo"] = (1000.0 / cols["parallax_corr"]).astype(np.float32)
    return pd.DataFrame(cols)


def test_join_writes_sidecar_with_expected_keys(tmp_path: Path) -> None:
    """Happy path: matched 100-row predictions × features yields a clean sidecar.

    Verifies that write_sidecar is invoked, the file lands at the conventional
    `<stem>.provenance.json` location, and the JSON carries the keys downstream
    auditors expect (input SHA-256s, n_predictions_in/joined, frozen-stats hook).
    """
    n = 100
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "predictions_with_features.parquet"

    _make_predictions(n).to_parquet(pred_path)
    _make_features(n).to_parquet(feat_path)

    summary = join_predictions_with_features(pred_path, feat_path, out_path)

    assert summary["n_joined"] == n
    assert summary["n_predictions_in"] == n
    assert summary["n_features_in"] == n

    sidecar = out_path.with_name(out_path.stem + ".provenance.json")
    assert sidecar.exists(), f"sidecar not emitted at {sidecar}"

    payload = json.loads(sidecar.read_text())
    assert payload["output_file"] == str(out_path)
    assert payload["script"].endswith("release_pipeline.py")
    assert payload["row_count_before"] == n
    assert payload["row_count_after"] == n
    sources = {s["name"]: s for s in payload["sources"]}
    assert "Pipeline 1 Stream 3 predictions" in sources
    assert "Pipeline 1 Stream 3 features (subset)" in sources
    assert len(sources["Pipeline 1 Stream 3 predictions"]["sha256"]) == 64
    assert "feature_columns_carried" in payload["extra"]
    assert payload["extra"]["frozen_stats_basis_fingerprint_sha256"] is None


def test_join_raises_on_silent_inner_shrink(tmp_path: Path) -> None:
    """Inner join with a prediction missing from features must raise.

    Otherwise the release would silently shrink (`612k → 611k`) without any
    log entry. The assertion at release_pipeline.py:184 is the hard contract.
    """
    pred_ids = np.arange(50, dtype=np.int64)
    feat_ids = np.arange(48, dtype=np.int64)  # two predictions have no feature row
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "out.parquet"

    _make_predictions(len(pred_ids), source_ids=pred_ids).to_parquet(pred_path)
    _make_features(len(feat_ids), source_ids=feat_ids).to_parquet(feat_path)

    with pytest.raises(ValueError, match="silent data-loss bug"):
        join_predictions_with_features(pred_path, feat_path, out_path)


def test_join_preserves_frozen_stats_fingerprint_from_predictions_sidecar(
    tmp_path: Path,
) -> None:
    """If the predictions sidecar carries a frozen-stats fingerprint, it surfaces.

    The release-stage sidecar must record the basis fingerprint that was active
    at inference time, so a downstream consumer can re-verify against the
    Stream-1 frozen stats independently. This is the load-bearing reproducibility
    invariant.
    """
    n = 30
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "out.parquet"

    _make_predictions(n).to_parquet(pred_path)
    _make_features(n).to_parquet(feat_path)

    fp = "0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7"
    pred_sidecar = pred_path.with_name(pred_path.name + ".provenance.json")
    pred_sidecar.write_text(
        json.dumps(
            {
                "output_file": str(pred_path),
                "frozen_stats": {"basis_fingerprint_sha256": fp},
            },
        ),
    )

    join_predictions_with_features(pred_path, feat_path, out_path)

    sidecar = out_path.with_name(out_path.stem + ".provenance.json")
    payload = json.loads(sidecar.read_text())
    assert payload["extra"]["frozen_stats_basis_fingerprint_sha256"] == fp


def test_join_left_preserves_unmatched_predictions(tmp_path: Path) -> None:
    """``how="left"`` keeps every prediction even when its feature row is missing.

    The cardinality assertion in `join_predictions_with_features` is gated on
    `how == "inner"`, so the left-join path is the recommended escape hatch
    when the caller explicitly tolerates feature-side gaps. Verifies the row
    count survives and feature columns are NaN for the unmatched rows.
    """
    pred_ids = np.arange(20, dtype=np.int64)
    feat_ids = np.arange(15, dtype=np.int64)
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "out.parquet"

    _make_predictions(len(pred_ids), source_ids=pred_ids).to_parquet(pred_path)
    _make_features(len(feat_ids), source_ids=feat_ids).to_parquet(feat_path)

    summary = join_predictions_with_features(
        pred_path, feat_path, out_path, how="left",
    )

    assert summary["n_joined"] == len(pred_ids)
    assert summary["join_how"] == "left"
    out = pd.read_parquet(out_path)
    # Predictions 15-19 had no feature row; their parallax_corr should be NaN.
    unmatched = out[out["source_id"].isin(np.arange(15, 20))]
    assert unmatched["parallax_corr"].isna().all()


def test_join_raises_on_duplicate_source_ids(tmp_path: Path) -> None:
    """``validate="one_to_one"`` raises if either side has duplicate source_ids.

    Stream 3 features and predictions are 1-to-1 by construction. If a future
    upstream regression introduces duplicates on either side, the merge must
    raise. This is the load-bearing safety net behind the join contract.
    """
    pred_ids = np.array([1, 2, 3, 3, 4], dtype=np.int64)  # duplicate 3
    feat_ids = np.arange(5, dtype=np.int64)
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "out.parquet"

    _make_predictions(len(pred_ids), source_ids=pred_ids).to_parquet(pred_path)
    _make_features(len(feat_ids), source_ids=feat_ids).to_parquet(feat_path)

    with pytest.raises(pd.errors.MergeError):
        join_predictions_with_features(pred_path, feat_path, out_path)


def test_join_handles_malformed_predictions_sidecar(tmp_path: Path) -> None:
    """Invalid JSON in the predictions sidecar must degrade gracefully.

    The fingerprint cross-reference is best-effort; if the predictions sidecar
    is corrupted, the join should still succeed and emit its own sidecar with
    `frozen_stats_basis_fingerprint_sha256: null` rather than raising.
    """
    n = 20
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "out.parquet"
    _make_predictions(n).to_parquet(pred_path)
    _make_features(n).to_parquet(feat_path)

    # Write a malformed JSON sidecar.
    pred_path.with_name(pred_path.name + ".provenance.json").write_text(
        "{this is not: valid json,",
    )

    join_predictions_with_features(pred_path, feat_path, out_path)
    sidecar = out_path.with_name(out_path.stem + ".provenance.json")
    payload = json.loads(sidecar.read_text())
    assert payload["extra"]["frozen_stats_basis_fingerprint_sha256"] is None


def test_join_skips_provenance_when_disabled(tmp_path: Path) -> None:
    """``write_provenance=False`` is an escape hatch for unit tests / dry runs."""
    n = 10
    pred_path = tmp_path / "predictions.parquet"
    feat_path = tmp_path / "features.parquet"
    out_path = tmp_path / "out.parquet"
    _make_predictions(n).to_parquet(pred_path)
    _make_features(n).to_parquet(feat_path)

    join_predictions_with_features(
        pred_path, feat_path, out_path, write_provenance=False,
    )
    assert not out_path.with_name(out_path.stem + ".provenance.json").exists()


# -----------------------------------------------------------------------------
# Hybrid composer tests (Phase A2-followup-2, 2026-04-25)
# Per HIGH_SIGMA_RESCUE_REPORT: when σ_<elem> > threshold, substitute the
# latent-kNN median (bounded by training-set support) for that element.
# -----------------------------------------------------------------------------


def test_hybrid_thresholds_match_release():
    """The local copy of σ thresholds must match release.py to avoid drift.

    Both the orchestrator (release_pipeline) and the release-tier annotator
    (release.py) consume the same per-element σ thresholds. release_pipeline
    keeps a local copy to avoid pulling in torch via the package import; this
    test catches the case where one diverges from the other."""
    from arqueogal.xp_abundances.main.release import _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD as canon
    assert _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD == canon


def _make_annotated_with_sigma(n: int = 6) -> pd.DataFrame:
    """Build a tiny annotated parquet with σ in known positions."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "teff_pred": rng.normal(4500, 200, n).astype(np.float32),
            "logg_pred": rng.normal(2.5, 0.4, n).astype(np.float32),
            "mh_pred": rng.normal(-0.4, 0.3, n).astype(np.float32),
            "alpha_m_pred": rng.normal(0.1, 0.1, n).astype(np.float32),
            "mg_h_pred": rng.normal(-0.3, 0.2, n).astype(np.float32),
            "teff_sigma": np.array([80.0, 200.0, 80.0, 80.0, 80.0, 80.0], dtype=np.float32),
            "logg_sigma": np.full(n, 0.10, dtype=np.float32),
            "mh_sigma": np.full(n, 0.08, dtype=np.float32),
            "alpha_m_sigma": np.array([0.03, 0.03, 0.15, 0.03, 0.03, 0.03], dtype=np.float32),
            "mg_h_sigma": np.full(n, 0.10, dtype=np.float32),
        }
    )
    return df


def _make_knn_rescue(source_ids: np.ndarray) -> pd.DataFrame:
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": source_ids,
            "knn_teff_med": np.full(n, 4800.0, dtype=np.float32),
            "knn_teff_iqr": np.full(n, 200.0, dtype=np.float32),
            "knn_logg_med": np.full(n, 2.7, dtype=np.float32),
            "knn_logg_iqr": np.full(n, 0.20, dtype=np.float32),
            "knn_mh_med": np.full(n, -0.5, dtype=np.float32),
            "knn_mh_iqr": np.full(n, 0.15, dtype=np.float32),
            "knn_alpha_m_med": np.full(n, 0.20, dtype=np.float32),
            "knn_alpha_m_iqr": np.full(n, 0.08, dtype=np.float32),
            "knn_mg_h_med": np.full(n, -0.20, dtype=np.float32),
            "knn_mg_h_iqr": np.full(n, 0.15, dtype=np.float32),
        }
    )


def test_hybrid_emits_columns_and_uses_regressor_when_sigma_in_range(tmp_path: Path):
    """In-range σ → hybrid_pred == regressor pred, hybrid_source == 'regressor'."""
    df = _make_annotated_with_sigma(n=6)
    annotated_path = tmp_path / "annotated.parquet"
    df.to_parquet(annotated_path)

    knn = _make_knn_rescue(df["source_id"].to_numpy())
    knn_path = tmp_path / "knn.parquet"
    knn.to_parquet(knn_path)

    summary = attach_hybrid_columns(annotated_path, knn_path)
    out = pd.read_parquet(annotated_path)

    # Row 0 has all σ in range → all five elements use the regressor.
    for elem in ("teff", "logg", "mh", "alpha_m", "mg_h"):
        assert out[f"{elem}_hybrid_source"].iloc[0] == "regressor"
        assert out[f"{elem}_hybrid_tier"].iloc[0] == 1
        np.testing.assert_allclose(
            out[f"{elem}_hybrid_pred"].iloc[0], out[f"{elem}_pred"].iloc[0], rtol=1e-5
        )

    assert summary["n_rows"] == 6
    assert summary["knn_attached"] is True


def test_hybrid_substitutes_knn_when_sigma_inflated(tmp_path: Path):
    """High σ AND kNN finite → hybrid_pred == knn_med, source == 'knn', tier 2."""
    df = _make_annotated_with_sigma(n=6)
    annotated_path = tmp_path / "annotated.parquet"
    df.to_parquet(annotated_path)

    knn = _make_knn_rescue(df["source_id"].to_numpy())
    knn_path = tmp_path / "knn.parquet"
    knn.to_parquet(knn_path)

    attach_hybrid_columns(annotated_path, knn_path)
    out = pd.read_parquet(annotated_path)

    # Row 1 has teff_sigma=200 > 150 → kNN substitution.
    assert out["teff_hybrid_source"].iloc[1] == "knn"
    assert out["teff_hybrid_tier"].iloc[1] == 2
    np.testing.assert_allclose(out["teff_hybrid_pred"].iloc[1], 4800.0, rtol=1e-5)
    # σ becomes IQR / 1.349.
    np.testing.assert_allclose(out["teff_hybrid_sigma"].iloc[1], 200.0 / 1.349, rtol=1e-5)
    # Other elements at row 1 still use regressor.
    for other in ("logg", "mh", "alpha_m", "mg_h"):
        assert out[f"{other}_hybrid_source"].iloc[1] == "regressor"

    # Row 2 has alpha_m_sigma=0.15 > 0.05 → kNN substitution for alpha_m.
    assert out["alpha_m_hybrid_source"].iloc[2] == "knn"
    np.testing.assert_allclose(out["alpha_m_hybrid_pred"].iloc[2], 0.20, rtol=1e-5)
    # Teff at row 2 unaffected (sigma in range).
    assert out["teff_hybrid_source"].iloc[2] == "regressor"


def test_hybrid_falls_back_to_regressor_caveat_when_knn_missing(tmp_path: Path):
    """High σ AND no kNN → hybrid_pred == regressor pred, source == 'regressor_caveat'."""
    df = _make_annotated_with_sigma(n=6)
    annotated_path = tmp_path / "annotated.parquet"
    df.to_parquet(annotated_path)

    summary = attach_hybrid_columns(annotated_path, knn_rescue_path=None)
    out = pd.read_parquet(annotated_path)

    # Row 1 (teff_sigma=200) → regressor_caveat (no kNN available).
    assert out["teff_hybrid_source"].iloc[1] == "regressor_caveat"
    assert out["teff_hybrid_tier"].iloc[1] == 2
    np.testing.assert_allclose(
        out["teff_hybrid_pred"].iloc[1], df["teff_pred"].iloc[1], rtol=1e-5
    )
    assert summary["knn_attached"] is False


def test_hybrid_falls_back_to_caveat_when_knn_value_is_nan(tmp_path: Path):
    """High σ AND kNN-NaN → hybrid_pred == regressor pred, source == 'regressor_caveat'."""
    df = _make_annotated_with_sigma(n=6)
    annotated_path = tmp_path / "annotated.parquet"
    df.to_parquet(annotated_path)

    knn = _make_knn_rescue(df["source_id"].to_numpy())
    # Wipe out kNN values for row 1.
    knn.loc[1, ["knn_teff_med", "knn_teff_iqr"]] = np.nan
    knn_path = tmp_path / "knn.parquet"
    knn.to_parquet(knn_path)

    attach_hybrid_columns(annotated_path, knn_path)
    out = pd.read_parquet(annotated_path)

    assert out["teff_hybrid_source"].iloc[1] == "regressor_caveat"


def test_hybrid_per_element_summary_counts(tmp_path: Path):
    """Summary block accurately tracks regressor / knn / regressor_caveat counts."""
    df = _make_annotated_with_sigma(n=6)
    annotated_path = tmp_path / "annotated.parquet"
    df.to_parquet(annotated_path)
    knn = _make_knn_rescue(df["source_id"].to_numpy())
    knn_path = tmp_path / "knn.parquet"
    knn.to_parquet(knn_path)

    summary = attach_hybrid_columns(annotated_path, knn_path)
    teff_summary = summary["per_element"]["teff"]
    assert teff_summary["regressor"] == 5  # only row 1 has inflated teff_sigma
    assert teff_summary["knn"] == 1
    assert teff_summary["regressor_caveat"] == 0

    alpha_m_summary = summary["per_element"]["alpha_m"]
    assert alpha_m_summary["knn"] == 1  # row 2 has inflated alpha_m_sigma
    assert alpha_m_summary["regressor"] == 5
