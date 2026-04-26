# Integration Test Audit: ArqueoGal

Date: 2026-04-26  
Auditor: Claude Haiku 4.5

## Executive Summary

The integration test tree (`tests/integration/`) is severely underdeveloped. It contains **only one test file** with seven GPU stress-battery tests covering kNN-hybrid validation, but **zero tests exercise the end-to-end predictions → annotate_parquet → attach_hybrid_columns chain**, **v5 schema validation is absent**, **kNN rescue integration is not tested**, and **cross-stream invariants have no coverage**. The existing tests use relativistic path construction (`REPO = Path(__file__).resolve().parents[2]`) that is CI-safe, but the referenced checkpoint and feature files may not exist in CI environments without explicit setup. The test tree mirrors neither the main module structure nor the experimental module, and critical integration gaps remain unfilled.

---

## Detailed Findings

### A. Coverage Gaps

#### (1) Missing End-to-End Pipeline Tests

**Problem:** No test exercises the full `predictions → join_predictions_with_features → annotate_parquet → attach_hybrid_columns` chain on production-shaped inputs.

- `tests/integration/test_hybrid_stress_battery.py` tests kNN consistency, CV stability, and permutation importance in isolation.
- The release pipeline (`src/arqueogal/data/release_pipeline.py`) has unit tests in `tests/data/test_release_pipeline.py::test_hybrid_*` (lines 248–407) covering small synthetic dataframes (n=6), but these do not validate:
  - **Shape preservation** through the join on real 600k+ row predictions × features.
  - **Provenance sidecar correctness** when frozen-stats fingerprints are carried through the chain.
  - **Tier assignment correctness** on a production-sized held-out split.
  - **Hybrid column alignment** (pred, sigma, source, tier) per element on the combined regressor + kNN surface.

**Impact:** A regression in the join cardinality assertion, tier computation, or hybrid composition logic would not be detected until the full Stream 3 release run.

**Recommendation:** Add `tests/integration/test_release_pipeline_e2e.py` that:
1. Loads a small (10k row) slice of real Stream 1 predictions and features.
2. Runs `join_predictions_with_features`, `annotate_parquet`, `attach_hybrid_columns` in sequence.
3. Asserts on output schema (column names, dtypes), row counts, and tier distributions.

---

#### (2) No v5 Schema Validation Tests

**Problem:** v5 schema (2026-04-26) simplified the OOD flag set and retired global caveats (regime_b, ood_disagreement, aux_missing_any, dist_prior_dominated). No integration test validates that:

- The new `ood_joint_flag` (XP-Mahalanobis only) is the sole per-row OOD gate feeding Tier 3 assignment.
- Mode-ambiguous flag is confined to `alpha_m` only; other elements ignore it.
- Per-element `sigma_inflated_<elem>` thresholds match the hardcoded values in `release.py` (line 136-145).
- The hybrid columns (`teff_hybrid_pred`, `teff_hybrid_sigma`, `teff_hybrid_source`, `teff_hybrid_tier`) exist and have the correct dtypes (float32, float32, string, int8).

**Impact:** A v4→v5 schema migration (if introduced) or a threshold drift between `release_pipeline.py` and `release.py` would silently corrupt the release.

**Recommendation:**
- Add `tests/integration/test_v5_schema.py` that loads a 1k-row production predictions parquet and validates column presence, dtype, and cardinality.
- The test `tests/data/test_release_pipeline.py::test_hybrid_thresholds_match_release` (line 248) already cross-checks `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD`; extend it to a full v5 contract check.

---

#### (3) kNN Rescue Path Not Tested End-to-End

**Problem:** The `attach_hybrid_columns` path branches based on whether `knn_rescue_path` is None:

- Lines 534-536 in `release_pipeline.py`: if kNN rescue parquet exists, merge; else use `regressor` / `regressor_caveat` only.
- Unit tests in `test_release_pipeline.py` cover the three cases (lines 299–388): regressor-in-range, kNN-substitution, and regressor_caveat fallback.
- **But:** no integration test validates that a real kNN rescue parquet (output from `run_knn_rescue.py`) produces consistent tier distributions and predictions when attached to a real annotated catalog.

**Impact:** If `run_knn_rescue.py` changes the kNN output schema (e.g., renames `knn_teff_med` → `teff_knn_med`), the merge at line 536 will silently produce all NaN hybrid columns with no error raised (pandas merge with `suffixes` allows unmatched columns).

**Recommendation:** Add an integration test that:
1. Generates a small real kNN rescue parquet using `knn_rescue.write_artifact()` or a fixture.
2. Calls `attach_hybrid_columns` with both annotated and kNN paths.
3. Asserts that per-element source counts sum to n_rows and that no hybrid columns are all-NaN.

---

#### (4) Cross-Stream Invariants Not Validated

**Problem:** The CLAUDE.md (section 16, hard invariant) states "Frozen Hermite z-score stats across runs. Stream 3 must load v1's per-coefficient z-score stats." The integration tests do not validate that Stream 1 and Stream 3 feature parquets:

- Use the same Hermite basis fingerprint in their provenance sidecars.
- Have identical feature column names and dtypes.
- Can be loaded with the same `FeatureLayout` and `LabelTiers` (used in line 88 of `test_hybrid_stress_battery.py`).

**Impact:** A silent schema mismatch between S1 and S3 (e.g., reordering of XP coefficient columns or a dtype change) would break the kNN rescue on Stream 3 inference without an explicit failure.

**Recommendation:** Add `tests/integration/test_cross_stream_schema.py` that:
1. Loads both `pipeline1_features_stream1.parquet` and `pipeline1_features_stream3.parquet` sidecars.
2. Asserts their `frozen_stats_basis_fingerprint_sha256` values match (or that S3 references S1's fingerprint).
3. Loads both with the same `FeatureLayout` and compares column names / dtypes / Hermite coef ranges.

---

### B. Checkpoint and Artifact Path Dependencies

#### (1) Absolute Path References in test_hybrid_stress_battery.py

**Lines 44–53:**
```python
REPO = Path(__file__).resolve().parents[2]
ENCODER_DIR = (
    REPO
    / "models/main/xp_abundances/strong_contrastive_2026-04-25"
    / "20260425_6b96c06_dbcbc09_ensemble_5label"
    / "member_seed0"
)
TRAIN_PARQUET = REPO / "data/processed/pipeline1_features_stream1.parquet"
S3_PARQUET = REPO / "data/processed/pipeline1_features_stream3.parquet"
FROZEN_STATS = REPO / "data/processed/pipeline1_features_stream1.provenance.json"
```

**Status:** All paths exist on the current machine (verified 2026-04-26):
- ✓ `models/main/xp_abundances/strong_contrastive_2026-04-25/20260425_6b96c06_dbcbc09_ensemble_5label/member_seed0/` exists (4 files).
- ✓ `data/processed/pipeline1_features_stream1.parquet` exists (512 MB).
- ✓ `data/processed/pipeline1_features_stream3.parquet` exists (731 MB).

**Risk:** In CI environments without these artefacts staged, the `@pytest.mark.stress` tests skip gracefully (lines 69–70, 76, 190–191), so there is no silent failure. However:

- The checkpoint path is pinned to a specific commit hash (`6b96c06_dbcbc09`). If the checkpoint is deleted during model retrain, the stress battery cannot run even locally.
- The feature parquets are large (1.2 GB combined). CI runners may not have disk quota or may timeout downloading them.

**Recommendation:**
- Document in `CLAUDE.md` §13 (hard invariants) that the stress battery requires the three artefacts to be present for `--run-stress` to succeed.
- Add a CI stage that downloads / stages the frozen checkpoint and feature parquets (or skips the stress battery).
- Consider parametrizing the test with a fixture that uses smaller subsets (e.g., `pipeline1_features_stream1_subset20k.parquet`, which exists at 33 MB).

---

#### (2) Fixture Dependencies on Production-Size Inputs

**Lines 66–101:**
```python
@pytest.fixture(scope="module")
def training_arrays():
    if not TRAIN_PARQUET.exists():
        pytest.skip(f"training parquet missing: {TRAIN_PARQUET}")
    from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers, load_arrays

    layout = FeatureLayout()
    tiers = LabelTiers.five_label()
    arr = load_arrays(TRAIN_PARQUET, layout, tiers, include_label_errors=False)
    X = np.asarray(arr["X"])
    Y = np.asarray(arr["Y"])
    sid = np.asarray(arr["source_id"])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(Y).all(axis=1)
    X, Y, sid = X[keep], Y[keep], sid[keep]
    _, fi = np.unique(sid, return_index=True)
    fi = np.sort(fi)
    return X[fi], Y[fi], sid[fi]
```

**Status:** The fixture loads ~500k rows of Stream 1 training data. This is intentional (stress battery needs large n for stable CV folds and cell statistics). However:

- **No docstring explaining the fixture's scope or why it must be production-sized.** A developer might assume a 100-row synthetic fixture is sufficient.
- **The load_arrays call is expensive** (10–30 s on disk I/O). If run in CI, this could timeout a single test that depends on the fixture.

**Recommendation:**
- Add a docstring to the `training_arrays` fixture explaining that it intentionally uses the full Stream 1 dataset for statistical validity of the CV and cell-calibration tests.
- Consider adding a `@pytest.fixture(scope="session")` wrapper to cache it across the module.

---

### C. Test Markers and Organizational Issues

#### (1) Missing @gpu Marker on test_7_permutation_importance

**Line 350:**
```python
def test_7_permutation_importance(encoder, training_arrays, device):
```

The test calls `_encode(encoder, ...)` (line 374–375) which requires the GPU. The `pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.stress]` at module level (line 58) does apply, but the individual test lacks an explicit `@pytest.mark.gpu` decorator. This is not a failure (module-level marks apply), but it can be confusing.

**Recommendation:** Add explicit `@pytest.mark.gpu` to each test for clarity.

---

#### (2) No Tests for Stress Markers

The `conftest.py` (lines 17–32) registers a `--run-stress` option, but there is no test in `tests/` that validates the marker filtering logic itself. If a developer adds a test with `@pytest.mark.stress` but forgets to run with `--run-stress`, the test silently skips.

**Recommendation:** Add a test in `tests/test_conftest.py` (if it exists) or `tests/integration/test_integration_markers.py` that verifies:
- Tests marked `@pytest.mark.stress` are skipped by default.
- Tests marked `@pytest.mark.stress` run when `--run-stress` is passed.

---

### D. Testing Conventions Violations

#### (1) Helper Functions Not Mirrored from External Source

**Line 105:**
```python
# Helpers (mirrored from .expert_review_2026-04-24/.../hybrid_stress_battery.py)
```

The functions `_encode`, `_gpu_knn`, `_knn_summary`, `_metrics` are duplicated from an expert-review artefact, not from the main source tree. This violates the CLAUDE.md code-style principle: "No committed duplication of load-bearing logic."

**Impact:** If `_encode` or `_gpu_knn` logic changes in the main codebase, the test's helper functions go stale.

**Recommendation:**
- Move `_encode`, `_gpu_knn`, `_knn_summary` to `src/arqueogal/xp_abundances/main/knn_rescue.py` or a new `src/arqueogal/utils/knn_helpers.py`.
- Import them from the source tree in the test.
- Keep `_metrics` local to the test (it is test-specific).

---

#### (2) No Type Hints on Test Functions

**Lines 165–389:**
Test functions lack type hints. CLAUDE.md (code-style section) states "Type hints where they clarify."

**Recommendation:** Add return type hints (all return `None`) and parameter type hints to test functions.

---

### E. Documentation and Transparency Issues

#### (1) No README for tests/integration/

The `tests/integration/` directory has only `conftest.py`, `__init__.py`, and one test file. There is no README explaining:
- What the stress battery is meant to validate (answer: the seven-test protocol from `research_brief.md` §3.3).
- Why the tests are expensive (GPU, large feature matrices).
- How to run them locally vs. in CI.
- What artefacts must exist (`ENCODER_DIR`, `TRAIN_PARQUET`, etc.).

**Recommendation:** Add `tests/integration/README.md` with:
1. Purpose of the integration test tree.
2. List of seven stress-battery tests.
3. Prerequisites (GPU, checkpoint, data parquets).
4. Run instructions (`pytest tests/integration --run-stress`).
5. Expected runtime (e.g., "~5 minutes on RTX 3060").

---

#### (2) Stress Battery Tests Lack Data-Driven Assertions

**Lines 180–183, 213–215, etc.:**
Hard-coded tolerances (e.g., `cv_ratio < 0.10`, `rms < tol[lbl]`) lack citations or references to where these numbers come from. A few have inline comments (e.g., line 229), but most do not.

**Recommendation:**
- Add a docstring to each test function explaining the tolerance and its source (e.g., "based on Stream 1 held-out stress-battery results" or "empirical Bayes shrinkage ceiling").
- Consider moving tolerances to a module-level constant dict:
  ```python
  _STRESS_BATTERY_TOLERANCES = {
      "cv_ratio": 0.10,  # CV RMSE spread; see research_brief.md §3.3
      "leakage_rms": {...},
      ...
  }
  ```

---

### F. Unused Test Fixtures and Stale Imports

#### (1) FROZEN_STATS Variable Never Used

**Line 53:**
```python
FROZEN_STATS = REPO / "data/processed/pipeline1_features_stream1.provenance.json"
```

This constant is defined but never used in any test. The frozen stats are loaded indirectly via `FeatureLayout` (which reads from `src/arqueogal/data/frozen_stats.py`), not from the parquet sidecar.

**Recommendation:** Remove the unused constant or add a test that validates the frozen stats fingerprint in the parquet sidecar.

---

### G. Summary of Missing Test Files

| Test File | Purpose | Lines | Status |
|-----------|---------|-------|--------|
| `test_release_pipeline_e2e.py` | End-to-end chain validation | ~100 | MISSING |
| `test_v5_schema.py` | v5 schema and tier assignment | ~50 | MISSING |
| `test_knn_rescue_integration.py` | kNN rescue attachment on real data | ~80 | MISSING |
| `test_cross_stream_schema.py` | S1 / S3 frozen-stats and column alignment | ~60 | MISSING |
| `test_integration_markers.py` | Marker filtering for stress battery | ~30 | MISSING |
| `README.md` | Documentation of integration test tree | ~40 | MISSING |

---

## Audit Verdict

**Pass with Required Additions:**

The existing integration test (`test_hybrid_stress_battery.py`) is well-structured and covers kNN hybrid validation rigorously. The path construction is CI-safe and the fixture handling is correct for large data. However, the integration test tree is incomplete and lacks:

1. End-to-end pipeline validation (predictions → tier assignment → hybrid columns).
2. v5 schema validation.
3. kNN rescue integration testing.
4. Cross-stream schema invariants.
5. Clear documentation.

**Recommendation:** Before the D5.1 release (Dec 2026), add the four missing integration test files and the README. Until then, the release pipeline must be validated manually or via the expert-review artefact.

---

## Files Cited

- `tests/integration/conftest.py` — lines 17–32 (marker registration)
- `tests/integration/test_hybrid_stress_battery.py` — lines 44–53 (paths), 66–101 (fixture), 105 (duplication comment), 165–389 (tests), 350 (permutation importance)
- `tests/data/test_release_pipeline.py` — lines 248–407 (hybrid tests), 248 (threshold cross-check)
- `src/arqueogal/xp_abundances/main/release.py` — lines 136–145 (sigma thresholds), 56–72 (v5 OOD flags)
- `src/arqueogal/data/release_pipeline.py` — lines 502–564 (attach_hybrid_columns), 534–536 (kNN merge)
- `CLAUDE.md` — section 16 (frozen stats invariant), code-style (conventions)
