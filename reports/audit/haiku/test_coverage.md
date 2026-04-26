# Test Coverage Audit (2026-04-26)

## Executive Summary

The ArqueoGal test suite has decent breadth but significant policy violations and gaps: 4 xp_abundances/main modules lack tests entirely; all data-layer tests use 100-300 row toy fixtures instead of the required 5k-row stratified smoke tests; and experimental tree is empty despite the policy requiring mirror structure.

## Critical Gaps (Policy Violations)

### Missing Test Files (Hard Stop Violations)

1. **`src/arqueogal/xp_abundances/main/config.py`** — No test file exists.
   - 34 lines of configuration dataclass definitions.
   - No contract test for field names, types, or defaults.

2. **`src/arqueogal/xp_abundances/main/sanity.py`** — No test file exists.
   - Pre-training gate that asserts schema, distributions, coverage.
   - Critical invariant: failure blocks training. Missing tests mean regressions go undetected.
   - Status: Mentioned in CLAUDE.md as a known gap.

3. **`src/arqueogal/xp_abundances/main/kinematic_ood.py`** — No test file exists.
   - Out-of-distribution detection via kinematic features.
   - 130 lines; not trivial.

4. **`src/arqueogal/xp_abundances/main/ensemble_diagnostics.py`** — No test file exists.
   - 200+ lines. Likely computes aggregate metrics across ensemble members.
   - No verification that output contracts match expected columns/dtypes.

5. **`src/arqueogal/data/release_artefacts.py`** — No test file exists.
   - Data emission module (parquet + sidecar).
   - Provenance sidecar contract is non-negotiable per CLAUDE.md invariant #14.

### Empty Test Trees

**`tests/xp_abundances/experimental/__init__.py`** — Empty. `src/arqueogal/xp_abundances/experimental/` is mirrored in policy but has no test file. Current state: only `__init__.py` exists in tests/experimental; src/experimental has real modules if any exist.

## Production-Size Smoke Test Gaps (Policy Violation #1)

**CLAUDE.md invariant:** "Production-size stratified smoke test is required for any new data-layer module. 100-row tests on gaiaxpy or external libraries are integration placebos; they miss the edge cases that trigger in 5k-row batches."

### Current State: All data-layer tests use toy fixtures

1. **`tests/data/test_gaia_xp.py`** — Only 514 lines.
   - Uses `_fake_xp_df()` with rows **not stratified by regime, magnitude bins, or |b| bins**.
   - Hermite reprojection tested on 1-5 synthetic rows (test_reproject_reproduces_pure_hermite_input_exactly: 1 row; test_reproject_residual_rms_is_nonnegative_and_float32: 5 rows).
   - Ye+2024 NN correction never called on >10 rows.
   - **Missing:** 5k-row stratified batch with regime cells, magnitude bins, |b| bins to catch XP edge cases that appear only in production scale.

2. **`tests/data/test_ingest_stream1.py`** (291 lines) — 5 APOGEE rows total.
   - `_apogee_df()` creates 5-row fixture (hardcoded in patched_pipeline fixture line 80).
   - Tests orchestration but never exercises real batch sizes.

3. **`tests/data/test_ingest_stream3.py`** (245 lines) — 300 Andrae rows.
   - Better than stream1, but still toy scale (300 < 5k).
   - Stratification checked? No: `_andrae_df()` uses `rng.uniform()` without regime/magnitude/|b| bins.

4. **`tests/data/test_ingest_xp.py`** (241 lines) — Toy scale.
   - No production batch test.

5. **`tests/data/test_crossmatch.py`** (297 lines) — Tested on synthetic 100-row DataFrames.
   - Example (test_dr2_to_dr3_many_to_one_tie_break_by_mag): 10-15 row fixtures.

6. **`tests/data/test_dedup.py`** (176 lines) — Toy rows; no stratification.

### Consequence

The policy states: "100-row smoke tests on gaiaxpy or external libraries are integration placebos — they miss the edge cases that trigger in 5 k-row batches." All these tests are placebo-grade. NaN propagation at scale, regime-cell edge cases, and many-to-one join tie-breaking under load will not be caught.

## Parametrize Coverage

**Only one test file uses `@pytest.mark.parametrize`:** `test_release.py`.

All other test files use single-case assertions or fixture-based loops. No parametrized failure-mode matrices for:
- NaN masks (V, Mg, Fe, α/M per-element cases in losses).
- DR2→DR3 join tie-break modes (|Δmag| vs proper-motion distance).
- Preprocessing ordering (Ye+2024 → normalise → log/z-score → per-coef z-score).
- Regime-B exclusion edge cases (|b|<5°, warm upper-RGB).

## Fixture and Path Brittleness

1. **Hard-coded relative paths in REPO calculations:**
   - `tests/integration/test_hybrid_stress_battery.py` lines 44-53:
     ```python
     REPO = Path(__file__).resolve().parents[2]
     ENCODER_DIR = (
         REPO / "models/main/xp_abundances/strong_contrastive_2026-04-25"
         / "20260425_6b96c06_dbcbc09_ensemble_5label" / "member_seed0"
     )
     TRAIN_PARQUET = REPO / "data/processed/pipeline1_features_stream1.parquet"
     S3_PARQUET = REPO / "data/processed/pipeline1_features_stream3.parquet"
     ```
   - If directory structure changes (e.g., model path refactoring), tests silently skip with `pytest.skip()` rather than fail loudly.

2. **No centralized fixture directory:** CLAUDE.md invariant #4 states "No mocked database for integration tests if the test exists to verify the schema or the column-shape contract. Use a real fixture file (small but realistic) loaded from `tests/fixtures/`." The `tests/fixtures/` directory does not exist.

3. **Monkeypatch-heavy architecture:** All stream1/stream3/ingest tests monkeypatch network I/O and heavy computations. This is correct for unit tests, but the consequence is that no test actually runs the full orchestration pipeline with real data flow. The provenance sidecar tests (`tests/data/test_provenance.py`) are correct offline—but they never test that provenance is *correctly filled* during an actual ingest run (row counts, source URLs, corrections applied).

## DESIGN.md Schema Contract Tests

Auditing drift between emitted DataFrame columns and DESIGN.md contracts:

1. **`tests/xp_abundances/main/test_knn_rescue.py`** has:
   ```python
   test_artifact_columns_align_with_master_schema()
   ```
   This is good. But it exists in only this one module.

2. **Missing for:**
   - `tests/data/test_gaia_xp.py` — Never verifies that `zscore_c0()` output matches §6.4 column contract.
   - `tests/data/test_ingest_stream1.py` — Never asserts final parquet columns match DESIGN.md.
   - `tests/data/test_ingest_stream3.py` — Same.
   - `tests/xp_abundances/main/test_release.py` — Has 629 lines but no explicit DESIGN.md drift test (assert emitted columns vs DESIGN.md schema_version).

## NaN Propagation and Edge Case Blind Spots

1. **NaN handling at the train/inference boundary:** CLAUDE.md invariant mentions "`nan_to_num` train/inference boundary" footgun. `tests/xp_abundances/main/test_training.py` loads real data and runs training, but does not parametrize over:
   - Aux feature with single NaN → should be sanitised to 0.0 before trunk.
   - Per-element label NaN (V, Mg, Fe, α/M) → `beta_nll_block_cholesky` mask.

2. **`beta_nll_block_cholesky` with element-specific NaN masks:** Tests exist (`tests/xp_abundances/main/test_losses.py`), but do not parametrize all combinations:
   - Teff present, logg NaN, mh present, alpha_m NaN, mg_h present (only test: all present or all NaN-masked).
   - Loss must be finite for all combinations; current test coverage is weak.

3. **AIP TAP inline-IN 504 → UPLOAD:** `tests/data/test_tap.py` exists (verified via line count check above, 200+ lines) but does not mock the ~100 KB ceiling condition to force upload path. No test that asserts `tap.batched_upload_fetch_df()` is called when the IN list exceeds threshold.

4. **DR2→DR3 many-to-one join tie-break:** `tests/data/test_crossmatch.py` has `test_dr2_to_dr3_many_to_one_tie_break_by_mag` but tests on 10-15 rows. Never exercises the tie-break when 100+ sources share the same DR3 partner at the boundary of the 300 mas / 0.1 mag tie-break distance.

## Provenance Sidecar Coverage

Per CLAUDE.md invariant #14: "Provenance sidecars are non-negotiable. Every Parquet emits a `*.provenance.json` with source URL, query, row counts, cuts, corrections, git SHA, timestamp, input SHA-256. Without it, the artefact is not reproducible."

**Test presence:**
- `tests/data/test_provenance.py` (130 lines) — Tests the Provenance dataclass serialisation and sidecar emission logic in isolation. Good.
- **But:** No test that verifies a full ingest run (e.g., `ingest_stream1()`) actually emits the sidecar with correct `row_count_before/after`, `corrections`, and input SHA-256.

**Gaps:**
- `test_ingest_stream1.py::test_provenance_has_local_and_tap_sources()` exists and checks that sources list is non-empty. But does not verify:
  - Input file SHA-256 matches the actual APOGEE FITS.
  - Corrections list reflects what was actually applied (Mészáros+2025, Lindegren zpt, Riello G-mag).
  - Row counts match the parquet on disk.

## Recommendations (Priority Order)

1. **Add provenance full-stack tests:** Each ingest module (stream1, stream3, xp, etc.) should have a test that verifies the sidecar contains correct row counts, corrections, and input SHA-256 after a patched-but-complete orchestration run.

2. **Add production-size stratified smoke tests:** For each data-layer module (gaia_xp, ingest_stream1, ingest_stream3, crossmatch, dedup, distances, kinematics, selection_function), create a 5k-row test with stratification on regime cells, magnitude bins, |b| bins. Use real or realistic data distributions (not uniform random).

3. **Create `tests/fixtures/` with realistic small parquets:** E.g., 100-200 row representative samples of each major table (Stream 1 + XP, Stream 3 + enrichment, Andrae+2023). Load these in tests that need schema verification without mocking the entire pipeline.

4. **Add missing test files:**
   - `tests/xp_abundances/main/test_config.py` — Dataclass fields, defaults.
   - `tests/xp_abundances/main/test_sanity.py` — Sanity battery behavior; each gate fires correctly.
   - `tests/xp_abundances/main/test_kinematic_ood.py` — OOD detection thresholds, output shape.
   - `tests/xp_abundances/main/test_ensemble_diagnostics.py` — Metric outputs match contract.
   - `tests/data/test_release_artefacts.py` — Sidecar and parquet schema.

5. **Add parametrized failure-mode tests:** Use `@pytest.mark.parametrize` for:
   - Per-element NaN combinations in loss tests.
   - Preprocessing order violations (flip Ye+2024 and normalise; should break).
   - DR2→DR3 tie-break edge cases at scale.
   - Regime-B warm upper-RGB exclusion (|b|<5°, logg<1.5, Teff>4500).

6. **Add DESIGN.md drift detection:** Each module with a DESIGN.md schema should have a test that asserts emitted columns and dtypes match the contract. Template:
   ```python
   def test_schema_matches_design_md():
       df = emit_output()
       assert set(df.columns) == set(DESIGN_MD_COLUMNS)
       for col, expected_dtype in DESIGN_MD_SCHEMA.items():
           assert df[col].dtype == expected_dtype
   ```

7. **Document fixtures in a README:** If `tests/fixtures/` is created, document the shape, provenance, and stratification of each fixture parquet.

## Files Affected

- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/config.py` — No test.
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/sanity.py` — No test.
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/kinematic_ood.py` — No test.
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/ensemble_diagnostics.py` — No test.
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/release_artefacts.py` — No test.
- `/home/aneitzel/projects/ArqueoGal/tests/data/test_gaia_xp.py` — Toy fixtures (1-5 rows), no stratification, no production smoke test.
- `/home/aneitzel/projects/ArqueoGal/tests/data/test_ingest_stream1.py` — 5-row fixture, no smoke test.
- `/home/aneitzel/projects/ArqueoGal/tests/data/test_ingest_stream3.py` — 300-row fixture (better), no stratification, no schema contract test.
- `/home/aneitzel/projects/ArqueoGal/tests/xp_abundances/experimental/__init__.py` — Empty; src/experimental status unknown.
- `/home/aneitzel/projects/ArqueoGal/tests/integration/test_hybrid_stress_battery.py` — Hard-coded model path; silent skip if structure changes.
