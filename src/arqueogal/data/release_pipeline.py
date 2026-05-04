"""End-to-end release pipeline: predictions × features → annotated catalog → derivatives.

The Stream 3 inference output (`pipeline1_predictions_stream3.parquet`) is a thin
table, predictions, sigmas, covariances, OOD flags. It does NOT carry the
auxiliary photometry / parallax / distance columns that the release-side flag
computation (`assign_g_mag_bin`, `assign_dist_prior_dominated`) needs. Those live
in the features parquet (`pipeline1_features_stream3.parquet`).

This module joins the two on `source_id`, runs `release.annotate_parquet`, and
optionally builds the five derivative artefacts plus the FITS / VOTable exports.

Why this exists
---------------

The iter-1 real-data run found that without the join, `g_mag_bin` collapses to
100% "unknown" and `dist_prior_dominated` to 100% False. The join is a
prerequisite for honest release-side annotation; this module makes it the
canonical orchestrator.

Design choices
--------------

- **Inner join on source_id.** Stream 3 features and predictions should be
  one-to-one by construction, but missing rows are flagged in the manifest.
- **Subset of feature columns.** We only need photometry (g_mag, bp_mag, rp_mag,
  J/H/K, W1/W2), parallax (parallax_corr, parallax_error, ruwe), and distance
  (r_med_photogeo). XP coefficients themselves are not needed at the release
  stage. This keeps the joined parquet small (~1 GB instead of 5 GB).
- **Idempotent.** The driver re-runs cleanly: existing outputs are overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.provenance import (
    LocalSource,
    Provenance,
    write_sidecar,
)

# The auxiliary feature columns we need at the release stage. Subset of the
# 275-column Stream 3 features parquet.
_FEATURE_JOIN_COLS = (
    "source_id",
    "ra_deg",
    "dec_deg",
    "b_deg",
    "g_mag",
    "bp_mag",
    "rp_mag",
    "bp_rp",
    "parallax_corr",
    "parallax_error",
    "parallax_raw",
    "ruwe",
    "r_med_photogeo",
    "r_lo_photogeo",
    "r_hi_photogeo",
    "distance_pc",
    "j_mag",
    "h_mag",
    "k_mag",
    "w1_mag",
    "w2_mag",
    "ag_gspphot",
    "teff_gspphot",
)


def _load_release_module(src_root: Path) -> types.ModuleType:
    """Load release.py directly to avoid the package __init__'s torch import."""
    spec = importlib.util.spec_from_file_location(
        "ag_release_pipeline_release",
        src_root / "arqueogal" / "xp_abundances" / "main" / "release.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load release.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ag_release_pipeline_release"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_artefacts_module(src_root: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ag_release_pipeline_artefacts",
        src_root / "arqueogal" / "data" / "release_artefacts.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load release_artefacts.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ag_release_pipeline_artefacts"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def join_predictions_with_features(
    predictions_path: Path,
    features_path: Path,
    output_path: Path,
    *,
    feature_columns: tuple[str, ...] = _FEATURE_JOIN_COLS,
    how: str = "inner",
    write_provenance: bool = True,
) -> dict[str, int | float | str]:
    """Inner-join predictions with the feature columns needed for release annotation.

    Parameters
    ----------
    predictions_path
        Stream 3 predictions parquet.
    features_path
        Stream 3 features parquet (typically 5x larger than predictions).
    output_path
        Output joined parquet path.
    feature_columns
        Subset of feature columns to carry into the output. Default is the
        photometry + parallax + distance set required by the release-side flag
        computations.
    how
        Join how. Default "inner". Use "left" if you want to preserve every
        prediction even when a feature row is missing.
    write_provenance
        When True (default), emits a ``*.provenance.json`` sidecar next to the
        joined parquet via :func:`arqueogal.data.provenance.write_sidecar`. The
        sidecar pins input SHA-256s, join parameters, row counts, and (when
        available) the frozen-stats basis fingerprint cross-referenced from the
        predictions sidecar.

    Returns
    -------
    dict
        Summary: row counts and timing.

    Raises
    ------
    ValueError
        If predictions and features have unequal cardinality after the
        ``how="inner"`` validate=one_to_one merge silently drops rows. Catching
        this prevents a release-time silent shrink (e.g. 613939 → 613850 due
        to a single missing feature row going unnoticed).
    """
    t0 = time.time()
    preds = pd.read_parquet(predictions_path)
    n_pred = len(preds)

    # Load only the requested feature columns, dropping any that the
    # features parquet does not actually carry. Stream 1 / Stream 2
    # parquets lack a few Stream-3-only columns (e.g. ``parallax_raw``);
    # the release pipeline can proceed without them and downstream
    # consumers fall back to ``parallax_corr`` when ``parallax_raw`` is
    # absent.
    import pyarrow.parquet as _pq
    feat_schema_cols = set(_pq.read_schema(features_path).names)
    requested = [c for c in feature_columns if c]
    available_cols = [c for c in requested if c in feat_schema_cols]
    missing_cols = [c for c in requested if c not in feat_schema_cols]
    if missing_cols:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "join_predictions_with_features: features parquet %s missing "
            "columns %s; release pipeline will fall back to default values",
            features_path,
            sorted(missing_cols),
        )
    feats = pd.read_parquet(features_path, columns=list(available_cols))
    n_feat = len(feats)

    # Verify source_id is unique on both sides.
    n_unique_pred = preds["source_id"].nunique()
    n_unique_feat = feats["source_id"].nunique()

    joined = preds.merge(feats, on="source_id", how=how, validate="one_to_one")
    n_joined = len(joined)

    # Catch silent row drops that validate="one_to_one" misses (it catches
    # duplicates on either side but not asymmetric coverage). For an inner join
    # we require every prediction to find a feature row; otherwise the release
    # would silently shrink. Tolerated only when how != "inner" (caller asked
    # for left-join or similar).
    if how == "inner" and n_pred != n_joined:
        raise ValueError(
            f"join_predictions_with_features: inner-join shrunk row count from "
            f"{n_pred} (predictions) to {n_joined} (joined); "
            f"{n_pred - n_joined} predictions had no matching feature row. "
            f"This is a silent data-loss bug. Verify the features parquet "
            f"covers every source_id in the predictions parquet, or pass "
            f'how="left" to retain unmatched predictions explicitly.'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_path)

    summary: dict[str, int | float | str] = {
        "n_predictions_in": n_pred,
        "n_features_in": n_feat,
        "n_unique_pred": n_unique_pred,
        "n_unique_feat": n_unique_feat,
        "n_joined": n_joined,
        "join_how": how,
        "feature_columns": list(available_cols),
        "elapsed_seconds": float(round(time.time() - t0, 2)),
        "output_path": str(output_path),
    }

    if write_provenance:
        # Cross-reference the frozen-stats fingerprint from the predictions
        # sidecar if it exists. This pins the basis used during inference into
        # the release artefact so a downstream consumer can re-verify.
        frozen_fp = None
        pred_sidecar = predictions_path.with_name(
            predictions_path.name + ".provenance.json",
        )
        if pred_sidecar.exists():
            try:
                pred_payload = json.loads(pred_sidecar.read_text())
                frozen_fp = pred_payload.get("frozen_stats", {}).get("basis_fingerprint_sha256")
            except (json.JSONDecodeError, KeyError, TypeError):
                frozen_fp = None

        prov = Provenance(
            output_file=str(output_path),
            script="src/arqueogal/data/release_pipeline.py",
            sources=[
                LocalSource(
                    name="Pipeline 1 Stream 3 predictions",
                    path=str(predictions_path),
                    sha256=_sha256(predictions_path),
                ),
                LocalSource(
                    name="Pipeline 1 Stream 3 features (subset)",
                    path=str(features_path),
                    sha256=_sha256(features_path),
                ),
            ],
            corrections=[
                "join: inner on source_id, validate=one_to_one (predictions × features)",
            ],
            row_count_before=n_pred,
            row_count_after=n_joined,
            notes=(
                "Stream 3 predictions joined with the photometry / parallax / "
                "distance columns the release annotator needs. The predictions "
                "parquet is a thin output; this join reattaches the auxiliary "
                "Gaia and DR19 catalog columns required by assign_g_mag_bin "
                "and assign_dist_prior_dominated."
            ),
            extra={
                "feature_columns_carried": list(available_cols),
                "n_unique_source_id_predictions": int(n_unique_pred),
                "n_unique_source_id_features": int(n_unique_feat),
                "frozen_stats_basis_fingerprint_sha256": frozen_fp,
                "elapsed_seconds": float(round(time.time() - t0, 2)),
            },
        )
        write_sidecar(prov)

    return summary


def attach_kin_ood_flag(joined_path: Path, kin_ood_path: Path) -> dict[str, object]:
    """Inject ``kin_ood_flag`` into the joined release parquet from a lookup table.

    The Phase B kinematic-OOD detector (``scripts/build_kin_ood_flag.py``) writes
    a parquet with ``[source_id, kin_ood_flag, kin_ood_score]`` covering the
    Stream-3 kinematic-ready subset (~249 k of 614 k stars). Stars without
    kinematics are not in that table; the left-join here leaves them at False
    (the conservative default that keeps them in Tier 1 for aux-assisted
    elements). The injected column overrides the v1 placeholder filled by
    ``release.assign_kin_ood_flag``.
    """
    if not kin_ood_path.exists():
        return {"injected": False, "reason": f"kin_ood lookup parquet absent: {kin_ood_path}"}
    df = pd.read_parquet(joined_path)
    lookup = pd.read_parquet(kin_ood_path)
    n_pre = (
        int(df.get("kin_ood_flag", pd.Series(dtype=bool)).sum())
        if "kin_ood_flag" in df.columns
        else 0
    )
    if "kin_ood_flag" in df.columns:
        df = df.drop(columns=["kin_ood_flag"])
    df = df.merge(lookup[["source_id", "kin_ood_flag"]], on="source_id", how="left")
    # Route through float64 to keep mixed bool/int/None object Series working:
    # ``fillna(False).astype(bool)`` raises a FutureWarning in pandas ≥ 2.1
    # and will hard-break in 2.2, while ``astype("boolean")`` chokes on a
    # mixed bool/int object Series. float64 accepts every shape upstream
    # emits (see release._coerce_flag_series for the same reasoning).
    df["kin_ood_flag"] = df["kin_ood_flag"].astype("float64").fillna(0.0).astype(bool)
    if "kin_ood_score" in lookup.columns:
        df = df.drop(columns=[c for c in ("kin_ood_score",) if c in df.columns])
        df = df.merge(lookup[["source_id", "kin_ood_score"]], on="source_id", how="left")
    df.to_parquet(joined_path)
    n_post = int(df["kin_ood_flag"].sum())
    return {
        "injected": True,
        "lookup_path": str(kin_ood_path),
        "n_lookup_rows": int(len(lookup)),
        "n_release_rows": int(len(df)),
        "n_kin_ood_pre": n_pre,
        "n_kin_ood_post": n_post,
        "kin_ood_pct_post": float(round(100.0 * n_post / len(df), 3)),
    }


def run_release_pipeline(
    predictions_path: Path,
    features_path: Path,
    output_dir: Path,
    *,
    src_root: Path | None = None,
    build_derivatives: bool = True,
    build_partition: bool = True,
    kin_ood_path: Path | None = None,
) -> dict[str, object]:
    """End-to-end release pipeline: join → kin_ood inject → annotate → derivatives → partition.

    If ``kin_ood_path`` is provided and exists, the per-star ``kin_ood_flag`` from
    that lookup parquet is injected into the joined release table BEFORE
    ``release.annotate_parquet`` runs, so the per-element tier-gating logic
    consumes real kinematic-OOD information instead of the v1 all-False placeholder.

    Returns a manifest with all stage summaries.
    """
    if src_root is None:
        src_root = Path(__file__).resolve().parents[2]

    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "predictions_with_features.parquet"

    join_summary = join_predictions_with_features(
        predictions_path,
        features_path,
        joined_path,
    )

    if kin_ood_path is None:
        # Default: try the canonical artefact path written by build_kin_ood_flag.py.
        candidate = src_root.parent / "data/processed/pipeline1_kin_ood_flag.parquet"
        kin_ood_path = candidate if candidate.exists() else None
    kin_ood_summary = (
        attach_kin_ood_flag(joined_path, kin_ood_path)
        if kin_ood_path
        else {
            "injected": False,
            "reason": "no kin_ood_path provided",
        }
    )

    release_mod = _load_release_module(src_root)
    t0 = time.time()
    annotate_summary = release_mod.annotate_parquet(joined_path)
    annotate_summary["elapsed_seconds"] = float(round(time.time() - t0, 2))

    manifest: dict[str, object] = {
        "join": join_summary,
        "kin_ood_inject": kin_ood_summary,
        "annotate": annotate_summary,
        "joined_path": str(joined_path),
    }

    annotated = pd.read_parquet(joined_path)
    flag_distribution: dict[str, object] = {
        "g_mag_bin": annotated["g_mag_bin"].value_counts().to_dict(),
        "dist_prior_dominated_pct": float(
            100 * annotated["dist_prior_dominated"].mean(),
        ),
        "kin_ood_flag_pct": float(100 * annotated["kin_ood_flag"].mean()),
        "release_tier_distribution": annotated["release_tier"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict(),
    }
    manifest["flag_distribution"] = flag_distribution

    if build_derivatives:
        artefacts_mod = _load_artefacts_module(src_root)
        derivatives_dir = output_dir / "derivatives"
        derivatives_dir.mkdir(exist_ok=True)
        derivatives_summary: dict[str, object] = {}
        derivatives_summary["hrd"] = artefacts_mod.build_hrd_ready_subset(
            joined_path,
            derivatives_dir / "hrd_ready.parquet",
        )
        derivatives_summary["kinematic"] = artefacts_mod.build_kinematic_ready_subset(
            joined_path,
            derivatives_dir / "kinematic_ready.parquet",
        )
        derivatives_summary["tier1"] = artefacts_mod.build_tier1_only_subset(
            joined_path,
            derivatives_dir / "tier1_only_full.parquet",
        )
        derivatives_summary["per_cell"] = artefacts_mod.build_per_cell_summary(
            joined_path,
            derivatives_dir / "per_cell_summary.parquet",
        )
        derivatives_summary["per_magnitude"] = artefacts_mod.build_per_magnitude_reliability(
            joined_path,
            derivatives_dir / "per_magnitude_reliability.parquet",
        )
        manifest["derivatives"] = derivatives_summary

        if build_partition:
            artefacts_mod = _load_artefacts_module(src_root)
            partition_dir = output_dir / "partitioned_by_g_mag_bin"
            try:
                partition_summary = artefacts_mod.partition_by_g_mag_bin(
                    joined_path,
                    partition_dir,
                )
                manifest["partition"] = partition_summary
            except Exception as e:
                manifest["partition_error"] = f"{type(e).__name__}: {e}"

    manifest_path = output_dir / "release_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


_HYBRID_ELEMENTS: tuple[str, ...] = (
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
)
_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD: dict[str, float] = {
    "teff": 150.0,
    "logg": 0.30,
    "mh": 0.20,
    "fe_h": 0.15,  # dex; v1.1 placeholder (empirical-Bayes ceiling ≈ 1·σ_train)
    "alpha_m": 0.05,  # tightened 2026-04-26 from 0.10 → 0.05; see release.py
    "mg_h": 0.20,
    "c_h": 0.15,
    "n_h": 0.15,
    "o_h": 0.15,
    "na_h": 0.15,
    "al_h": 0.15,
    "si_h": 0.15,
    "s_h": 0.15,
    "k_h": 0.15,
    "ca_h": 0.15,
    "ti_h": 0.15,
    "v_h": 0.15,
    "cr_h": 0.15,
    "mn_h": 0.15,
    "ni_h": 0.15,
    "ce_h": 0.15,
}
"""Mirror of release._PER_ELEMENT_SIGMA_INFLATED_THRESHOLD. Duplicated rather
than imported to avoid coupling the release-pipeline orchestrator to the heavy
release.py module (which transitively pulls torch via the model imports).
Kept in sync by the test ``test_hybrid_thresholds_match_release``.

v1.1 (2026-04-29): expanded from 5 to 21 elements to match the 21-label
production head. The 5 Stream-1-tuned entries keep their empirical-Bayes-
calibrated thresholds; the 16 new entries use a 0.15 dex placeholder
pending the §3.3 promotion-protocol re-run on the 21-label model."""


def _hybrid_compose_per_element(
    df: pd.DataFrame,
    *,
    elem: str,
    sigma_threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compose a hybrid (regressor + kNN) prediction for a single element.

    For each row, decide which surface is the released value:

    - ``regressor`` if ``<elem>_sigma <= sigma_threshold``: regression head is
      not in prior-collapse, trust it.
    - ``knn`` if ``<elem>_sigma > sigma_threshold`` AND the kNN columns are
      present and finite: substitute the latent-kNN median (bounded by
      training-set support).
    - ``regressor_caveat`` if ``<elem>_sigma > sigma_threshold`` but kNN is
      not available: fall back to the regressor with a Tier-2 demotion.

    Returns
    -------
    pred : pd.Series
        Hybrid point-estimate (per-row choice between regressor and kNN).
    sigma : pd.Series
        Hybrid σ. Regressor σ when regressor was used; kNN IQR/1.349 (Gaussian-
        equivalent σ from the IQR) when kNN was used.
    source : pd.Series
        Per-row source label, one of ``{"regressor", "knn", "regressor_caveat"}``.
    tier : pd.Series
        Per-element tier ``int8``. Tier 1 if regressor was used, Tier 2
        otherwise. NaN-prediction handling is downstream
        (``release.assign_per_element_release_tier``).
    """
    idx = df.index
    pred_col = f"{elem}_pred"
    sigma_col = f"{elem}_sigma"
    knn_med_col = f"knn_{elem}_med"
    knn_iqr_col = f"knn_{elem}_iqr"

    pred = (
        df[pred_col].astype("float32")
        if pred_col in df.columns
        else pd.Series(np.nan, index=idx, dtype="float32")
    )
    sigma = (
        df[sigma_col].astype("float32")
        if sigma_col in df.columns
        else pd.Series(np.nan, index=idx, dtype="float32")
    )
    inflated = sigma.fillna(0.0) > sigma_threshold

    if knn_med_col in df.columns and knn_iqr_col in df.columns:
        knn_med = df[knn_med_col].astype("float32")
        knn_iqr = df[knn_iqr_col].astype("float32")
        knn_finite = knn_med.notna() & knn_iqr.notna()
        use_knn = inflated & knn_finite

        hybrid_pred = pred.copy()
        hybrid_pred[use_knn] = knn_med[use_knn]
        # Convert IQR to Gaussian-equivalent σ: σ ≈ IQR / 1.349.
        knn_sigma = knn_iqr / 1.349
        hybrid_sigma = sigma.copy()
        hybrid_sigma[use_knn] = knn_sigma[use_knn]

        source = pd.Series("regressor", index=idx, dtype="string")
        source[inflated & ~knn_finite] = "regressor_caveat"
        source[use_knn] = "knn"

        tier = pd.Series(1, index=idx, dtype="int8")
        tier[inflated] = 2
    else:
        # No kNN available — degrade gracefully: regressor with caveat tier
        # for inflated rows.
        hybrid_pred = pred
        hybrid_sigma = sigma
        source = pd.Series("regressor", index=idx, dtype="string")
        source[inflated] = "regressor_caveat"
        tier = pd.Series(1, index=idx, dtype="int8")
        tier[inflated] = 2

    return hybrid_pred, hybrid_sigma, source, tier


def attach_hybrid_columns(
    annotated_path: Path,
    knn_rescue_path: Path | None,
) -> dict[str, object]:
    """Attach hybrid (regressor + kNN) columns to an annotated catalog parquet.

    Reads the annotated catalog (already carrying ``release_tier__<elem>`` and
    ``prediction_sigma_inflated__<elem>``), joins the kNN-rescue parquet on
    ``source_id``, and writes back the parquet with new per-element columns:

    - ``<elem>_hybrid_pred``: the chosen hybrid point estimate.
    - ``<elem>_hybrid_sigma``: σ matching the hybrid choice.
    - ``<elem>_hybrid_source``: ``{"regressor", "knn", "regressor_caveat"}``.
    - ``<elem>_hybrid_tier``: per-element hybrid tier (int8).

    Parameters
    ----------
    annotated_path
        Path to the annotated catalog parquet (output of
        ``release.annotate_parquet`` after ``join_predictions_with_features``).
    knn_rescue_path
        Path to the kNN-rescue parquet (output of ``run_knn_rescue.py``).
        If None, the function still emits the hybrid columns but every
        per-element source defaults to ``regressor`` / ``regressor_caveat``.

    Returns
    -------
    dict
        Summary of the hybrid composition: per-element source counts, total
        rows, and the kNN-rescue file used (or null if absent).
    """
    annotated = pd.read_parquet(annotated_path)
    if knn_rescue_path is not None and Path(knn_rescue_path).exists():
        knn = pd.read_parquet(knn_rescue_path)
        annotated = annotated.merge(knn, on="source_id", how="left", suffixes=("", "_knn"))
        knn_attached = True
    else:
        knn_attached = False

    summary: dict[str, object] = {
        "n_rows": int(len(annotated)),
        "knn_rescue_path": str(knn_rescue_path) if knn_rescue_path else None,
        "knn_attached": knn_attached,
        "per_element": {},
    }

    for elem in _HYBRID_ELEMENTS:
        pred, sigma, source, tier = _hybrid_compose_per_element(
            annotated, elem=elem, sigma_threshold=_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD[elem]
        )
        annotated[f"{elem}_hybrid_pred"] = pred
        annotated[f"{elem}_hybrid_sigma"] = sigma
        annotated[f"{elem}_hybrid_source"] = source
        annotated[f"{elem}_hybrid_tier"] = tier
        summary["per_element"][elem] = {
            "regressor": int((source == "regressor").sum()),
            "knn": int((source == "knn").sum()),
            "regressor_caveat": int((source == "regressor_caveat").sum()),
            "tier_distribution": tier.astype(int).value_counts().sort_index().to_dict(),
        }

    annotated.to_parquet(annotated_path)
    return summary


def run_hybrid_release_pipeline(
    predictions_path: Path,
    features_path: Path,
    knn_rescue_path: Path | None,
    output_dir: Path,
    *,
    src_root: Path | None = None,
    build_derivatives: bool = True,
    build_partition: bool = True,
) -> dict[str, object]:
    """End-to-end hybrid release pipeline: regressor → kNN-rescue → composer.

    Wraps :func:`run_release_pipeline` and adds the hybrid composition stage
    (regressor + kNN → ``<elem>_hybrid_*`` columns). The hybrid surface is
    the recommended user-facing prediction for D-Cat-b consumers; the
    regressor-only surface is preserved for methodology comparison.

    Parameters
    ----------
    predictions_path, features_path, output_dir, src_root, build_derivatives,
    build_partition
        Forwarded to :func:`run_release_pipeline`.
    knn_rescue_path
        Path to the kNN-rescue parquet from ``scripts/run_knn_rescue.py``.
        ``None`` skips the kNN attachment but still emits ``regressor_caveat``
        labels on σ-inflated rows so consumers can filter accordingly.

    Returns
    -------
    dict
        The standard release-pipeline manifest, augmented with a top-level
        ``hybrid`` block reporting per-element source counts and tier
        distribution under the hybrid recipe.
    """
    manifest = run_release_pipeline(
        predictions_path,
        features_path,
        output_dir,
        src_root=src_root,
        build_derivatives=build_derivatives,
        build_partition=build_partition,
    )
    joined_path = Path(manifest["joined_path"])
    hybrid_summary = attach_hybrid_columns(joined_path, knn_rescue_path)
    manifest["hybrid"] = hybrid_summary
    manifest_path = output_dir / "release_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/pipeline1_predictions_stream3.parquet"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/pipeline1_features_stream3.parquet"),
    )
    parser.add_argument(
        "--knn-rescue",
        type=Path,
        default=None,
        help="Optional kNN-rescue parquet from scripts/run_knn_rescue.py. If "
        "omitted, the legacy regressor-only release is built.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release/D-Cat-b/pipeline_run"),
    )
    parser.add_argument("--no-derivatives", action="store_true")
    parser.add_argument("--no-partition", action="store_true")
    args = parser.parse_args()

    if args.knn_rescue is not None:
        manifest = run_hybrid_release_pipeline(
            args.predictions,
            args.features,
            args.knn_rescue,
            args.output_dir,
            build_derivatives=not args.no_derivatives,
            build_partition=not args.no_partition,
        )
    else:
        manifest = run_release_pipeline(
            args.predictions,
            args.features,
            args.output_dir,
            build_derivatives=not args.no_derivatives,
            build_partition=not args.no_partition,
        )
    print(json.dumps(manifest, indent=2, default=str))


__all__ = [
    "attach_hybrid_columns",
    "join_predictions_with_features",
    "run_hybrid_release_pipeline",
    "run_release_pipeline",
]


if __name__ == "__main__":
    main()
