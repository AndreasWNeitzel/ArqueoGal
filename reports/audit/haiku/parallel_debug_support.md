# Parallel Debugging Infrastructure Audit — ArqueoGal
**Date:** 2026-04-26  
**Auditor:** Claude Haiku 4.5  
**Scope:** Hypothesis-driven debugging support across production scripts and test harnesses

---

## Executive Summary

The codebase has **strong support for isolated ablation testing** (test_ablations/run_per_cell_ablations.py is a model architecture), but **critical production drivers tightly couple multiple failure domains**, requiring the full ensemble + TAP stack to debug single-component failures. Most monolithic scripts (run_pipeline1_inference.py, run_calibration.py, run_knn_rescue.py) lack unit-testable breakpoints; hypothesis-driven bisection would need refactoring to extract thin wrappers around high-risk subsystems (Mahalanobis OOD, z-score stats loading, feature assembly).

---

## Part 1: Model Pattern — test_ablations/

### run_per_cell_ablations.py (Lines 1–415)

**Design:** Post-hoc ablation harness that reads pre-computed predictions + truth labels, then applies gate-toggle logic without recomputation.

**Strengths (hypothesis-debugging friendly):**
- **Zero re-inference:** All hypotheses tested post-hoc on the same inference output (`predictions_stream1.parquet`), so a single hypothesis test does not require GPU/model reload.
- **Isolated gate logic:** `assign_tier()` (line 122–163) encapsulates all tier-assignment logic in one function; hypotheses manipulate only the config object (`AblationConfig`).
- **Parametric configs:** 19 distinct `AblationConfig` instances (lines 266–388) each represent a single hypothesis; no conditional branching inside metrics computation.
- **Reproducible split:** Test split recomputed deterministically from same seed + stratification used at training time (lines 237–246).

**Gaps:**
- **Cannot debug model output failures:** If a hypothesis is "the predictions themselves are wrong," you can't toggle it post-hoc. A separate inference harness with the same ablation structure is needed.
- **No per-hypothesis timing:** Script runs all 19 configs sequentially (line 393); no breakdown of compute cost per hypothesis.
- **Assumes input integrity:** No smoke-test on the input parquets (count mismatches, column schema drift, NaN inflation). A hypothesis like "the truth labels are corrupted" would silently produce misleading metrics.

---

## Part 2: Production Drivers — Tight Coupling Analysis

### run_pipeline1_inference.py (1,255 lines)

**Full dependency chain at `main()` (line 1192–1251):**

```
main()
├─ _resolve_device() [line 305–308] ✓ trivial, unit-testable
├─ run_inference() [calls line 1239–1250]
│  ├─ load_ensemble() [line 154, imports] → 5× checkpoint load + model reconstruction
│  ├─ load_frozen_zscore_stats() [line 143, imports]
│  ├─ verify_basis_fingerprint() [line 144, imports] ✗ Hard failure on drift
│  ├─ _detect_input_schema() [line 342–374] ✓ Unit-testable
│  ├─ apply_frozen_zscore() [line 143] ✓ Unit-testable (but embedded in 600-line fn)
│  ├─ _assemble_feature_matrix() [line 380–400] ✓ Unit-testable
│  ├─ XpAbundanceDataset [line 149] → DataLoader setup
│  ├─ predict_ensemble() [line 155] ✗ GPU full-ensemble inference
│  ├─ _un_scale_predictions() ✓ Unit-testable
│  ├─ fit_mahalanobis_ood() [line 165] ✗ Requires training-set parquet on first call
│  ├─ flag_mahalanobis_ood() [line 165]
│  ├─ combined_ood_status() [line 163]
│  ├─ RegimeBEnvelope [line 168] ✓ Unit-testable
│  ├─ BimodalityGrid [line 148] ✓ Unit-testable
│  ├─ score_selection_prob() [line 147] ✓ Unit-testable
│  └─ _atomic_write_parquet() [line 325–336] ✓ Unit-testable (but hard to fail cleanly)
```

**Critical tight-coupling observations:**

1. **TAP wrapper failure → full re-inference cost:** If a hypothesis is "the Mahalanobis OOD fitting is broken," a debugger must:
   - Load the 5-member ensemble (5–10 min wall-clock)
   - Run full inference on 1.3M Stream-3 rows (20–30 min on RTX 3060)
   - Then bisect OOD fitting (1 min)
   
   No way to test OOD in isolation without re-running inference.

2. **Frozen stats loading is all-or-nothing:** `verify_basis_fingerprint()` (line 144) hard-fails on any drift. A hypothesis like "the basis fingerprint is off by one character" cannot be tested with a fallback — the script halts before predictable error reporting.

3. **Feature assembly is embedded:** `_assemble_feature_matrix()` spans lines 380–450+ and mixes:
   - Schema detection (testable)
   - z-score stat application (testable)
   - NaN-sanitization logic (testable)
   - DataFrame→NumPy casting (testable)
   
   But the entire chain must run before you know if a hypothesis like "NaN sanitization is wrong" is the culprit.

4. **No intermediate checkpoint dumps:** If a hypothesis is "the predictions are NaN-propagating from aux features," you'd need to:
   - Modify the source to save `X` after assembly (line ~450)
   - Re-run the full script (30+ min)
   - Inspect the saved array
   
   Contrast with test_ablations.py, which reads pre-computed predictions from disk.

**What would help:**
- Extract `run_inference()` into a pure function with signature:
  ```python
  def run_inference(
      ensemble_checkpoints: list[Path],
      feature_matrix: np.ndarray,  # Pre-assembled
      frozen_stats: FrozenZScoreStats,
      label_scaler: LabelScaler,
      training_data_for_ood: pd.DataFrame | None = None,  # Optional, cached
  ) -> EnsemblePrediction
  ```
  Allow callers to assemble features once, then test multiple OOD/Regime-B hypotheses without re-inference.

- Move `fit_mahalanobis_ood()` to a separate pre-compute step (like `build_ood_bundle.py`) and cache the bundle as a pickle. Make inference read a pre-fitted bundle.

---

### run_calibration.py (1,062 lines)

**Full dependency chain at `main()` (line 650+, not shown):**

```
main()
├─ load_checkpoint() [line 44] ✗ 5 checkpoints loaded + Mahalanobis OOD bundle fit
├─ build_dataloaders() [line 44] ✗ Requires full training parquet + feature engineering
├─ bin_by_cells() [line 50] ✓ Unit-testable (pure NumPy/Pandas logic)
├─ temperature_scaling_per_cell() [line 54] ✓ Unit-testable in isolation
├─ shrunken_per_cell_per_label_scale() [line 52] ✓ Unit-testable
└─ coverage_at_levels() [line 51] ✓ Unit-testable
```

**Critical tight-coupling:**

- **Binary failure mode:** Either the full validation-split calibration succeeds or fails. No way to test per-element (Teff vs logg) hypothesis failures without running the full loader.
- **14 ADR-gated decisions locked in:** The script depends on 14 constants and choices (DESIGN.md); testing a hypothesis like "should we use GP-smoothed σ instead of empirical-Bayes?" requires re-running the entire validation loop with a flag toggle — no isolated testbed.
- **Flags parsed, not parameterized:** The `--apply-gp-smoothing` CLI flag (visible in docstring, line 2) toggles logic deep in `_reconstruct_model()` (lines 91–); a debug harness would need to re-instantiate the entire calibration pipeline.

---

### run_knn_rescue.py (800+ lines)

**Full dependency chain:**

```
main()
├─ load_ensemble() [line 34] ✗ 5 checkpoints
├─ load_frozen_zscore_stats() [line 32] ✓ Unit-testable
├─ apply_frozen_zscore() [line 32]
├─ compute_latents() [line 36] ✗ GPU inference
├─ gpu_knn_search() [line 36] ✗ GPU KNN on 1.3M × latent_dim space
└─ summarize_neighbors() [line 38] ✓ Unit-testable
```

**Coupling issue:**

- **Encoder latent computation cannot be tested in isolation:** If a hypothesis is "the encoder is returning NaN latents," you must load the ensemble and run inference (15–20 min), then inspect the latent matrix. No way to mock the encoder or test `gpu_knn_search()` with synthetic latents without modifying the script.

---

## Part 3: Test Infrastructure

### Unit Tests (tests/xp_abundances/main/, tests/data/)

**Status:** 62 test files, comprehensive coverage for library modules.

**Strengths:**
- `tests/scripts/test_run_pipeline1_inference.py` (47 KB, 400+ lines) is a model:
  - Injects a **tiny 14-D FeatureLayout** (line 113–119) to avoid loading the full ensemble.
  - Pre-builds a mock 2-member ensemble (lines 122–150+) with synthetic checkpoints.
  - Tests `run_inference()` function without CLI (entrypoint decoupled).
  - Raises `FrozenStatsMismatchError` on drift (line 49, tests line ~200+).

**Gaps:**
- **No xp_abundances/experimental/ tests:** The `experimental/` module has no test mirror (CLAUDE.md §5 notes this). OOD detector and kin-OOD hypotheses cannot be unit-tested.
- **No calibration harness tests:** `run_calibration.py` has no test file; the full validation-split pipeline is untested.
- **No kNN rescue tests:** `run_knn_rescue.py` has no tests; `gpu_knn_search()` is black-box.
- **Sanity.py is untested:** CLAUDE.md notes this; production-size smoke tests on external libraries are missing.

### Integration Tests (tests/integration/)

**File:** `test_hybrid_stress_battery.py`

**Status:** Single integration test that exercises the full Pipeline 1 stack (10k-row subset).

**What it does right:**
- Runs a complete inference → calibration → audit cycle on a fixed 10k subset.
- Verifies column schema and output integrity end-to-end.

**What it doesn't do:**
- Does not test **hypothesis bisection:** A hypothesis like "Regime-B envelope flag logic is wrong" cannot be isolated without running the full stack.
- No ablation-style parametrization: A single test path, no hypothesis-per-test-case branching.

---

## Part 4: Specific Debugging Scenarios

### Scenario 1: "OOD Mahalanobis scores are suspiciously high"

**Current workflow (no parallel-debug support):**
1. Modify `run_pipeline1_inference.py` to save the 108-D XP feature block before Mahalanobis scoring.
2. Re-run full inference (30 min).
3. Inspect the saved features + covariance matrix.
4. Hypothesis test: compute Mahalanobis with a different SVD threshold (line ~800).
5. Re-run to check output (another 30 min).

**What would help:**
- Separate `build_mahalanobis_bundle.py` that pre-computes + caches the bundle once.
- Modify `run_pipeline1_inference.py` to accept a `--mahalanobis-bundle` argument.
- Create `test_ablations/test_mahalanobis_hypotheses.py` that:
  - Reads pre-computed predictions + the cached bundle.
  - Recomputes Mahalanobis scores with alternative thresholds / algorithms.
  - Measures tier-promotion changes (post-hoc, no re-inference).

---

### Scenario 2: "Z-score stats are incorrect for Stream 3"

**Current workflow:**
1. Hypothesis: "the basis fingerprint verification is too strict; I should allow a 1-char drift."
2. Edit line 144 of `run_pipeline1_inference.py` (verify_basis_fingerprint call).
3. Re-run (30 min) to see if results change.

**What would help:**
- Extract `load_frozen_stats()` into a separate unit-testable function with a `strict=True` fallback mode.
- Create `test_ablations/test_zscoring_hypotheses.py` that:
  - Loads frozen stats with both strict and lenient modes.
  - Applies both to a reference dataset.
  - Compares output differences.

---

### Scenario 3: "Kin-OOD demotion for aux-assisted elements is too aggressive"

**Current workflow (requires re-inference):**
1. Modify `run_pipeline1_inference.py` or the underlying tier-promotion logic.
2. Re-run full inference (30 min).
3. Check tier distributions.

**With parallel-debug support (post-hoc, no re-inference):**
- Use `test_ablations/run_per_cell_ablations.py` pattern:
  - Read pre-computed predictions.
  - Toggle `use_kin_ood=False` in a config.
  - Compare tier promotion rates.
  - Time: <1 min instead of 30+ min.

**Current blocker:** `kin_ood_flag` must be computed at inference time (embedded in `run_pipeline1_inference.py`). To test a kin-OOD hypothesis post-hoc, the flag must already be in the parquet. See DESIGN.md for whether this is output.

---

## Part 5: Recommendations for Hypothesis-Driven Debug Harnesses

### Priority 1: Extract TAP-Wrapper Tests

**For OOD / Regime-B / Bimodality hypotheses (currently untestable without re-inference):**

1. **Create `test_ablations/test_ood_hypotheses.py`:**
   - Input: `predictions_stream1.parquet` (with OOD columns already populated).
   - Logic: Re-compute OOD flags with alternative Mahalanobis thresholds, ensemble-disagreement thresholds, and joint-flag logic.
   - Output: JSON summary of how tier promotion changes per hypothesis.

2. **Create `test_ablations/test_regime_b_hypotheses.py`:**
   - Input: Predictions + predicted Teff/log g + observed b_deg.
   - Logic: Toggle Regime-B envelope boundaries (5° → 3°, 4750 K → 4500 K, etc.).
   - Output: Tier promotion changes per hypothesis.

3. **Create `test_ablations/test_bimodality_hypotheses.py`:**
   - Input: Predictions + mode_ambiguous_grid.npz.
   - Logic: Recompute mode_ambiguous_flag with different grid definitions or thresholds.
   - Output: Tier changes for affected elements (alpha_m, mg_h).

### Priority 2: Modularize Inference for Unit-Testable Subsystems

**For feature assembly / z-scoring failures (currently embedded in 1,255-line driver):**

1. **Extract `src/arqueogal/xp_abundances/main/feature_assembly.py`:**
   - Function: `assemble_and_validate_features(df, layout, frozen_stats, schema) -> np.ndarray`
   - Unit tests in `tests/xp_abundances/main/test_feature_assembly.py`.
   - Testable hypotheses: NaN propagation, z-score application, column-order errors.

2. **Extract `src/arqueogal/xp_abundances/main/ood_bundler.py`:**
   - Function: `build_and_cache_mahalanobis_bundle(training_parquet, layout, frozen_stats) -> MahalanobisOODBundle`
   - Pre-compute once; cache as pickle.
   - Unit tests for bundle fitting with synthetic data.

3. **Extract `src/arqueogal/xp_abundances/main/tier_assignment.py`:**
   - Move logic from `test_ablations/run_per_cell_ablations.py:assign_tier()` into production code.
   - Reuse from `run_pipeline1_inference.py` during output assembly.
   - Enable post-hoc ablation testing (as test_ablations does now).

### Priority 3: Test Coverage for Experimental OOD Detectors

**For kin-OOD / calibration analysis / ensemble disagreement hypotheses:**

1. **Create `tests/xp_abundances/main/test_kin_ood.py`:**
   - Unit tests for `kin_ood_detector.py` (if it exists).
   - Testable hypotheses: kinematic thresholds, velocity-dispersion assumptions.

2. **Create `tests/scripts/test_run_calibration.py`:**
   - Minimal calibration harness on 100-row synthetic validation set.
   - Tests for temperature scaling, empirical-Bayes shrinkage, coverage.

3. **Create `tests/scripts/test_run_knn_rescue.py`:**
   - Mock encoder latents; test `gpu_knn_search()` with synthetic latents.
   - Unit tests for `summarize_neighbors()` logic.

---

## Part 6: Metrics for "Hypothesis-Debug-Ready" Code

| Criterion | Current State | Target |
|-----------|---------------|--------|
| **Post-hoc ablation test harnesses** | 1 (test_ablations/) | 3–4 (OOD, Regime-B, Bimodality, Calibration) |
| **Unit-testable feature assembly** | Embedded in 1,255-line driver | Extracted to <200-line function + tests |
| **OOD bundle caching** | Fit on every inference run | Pre-fitted, pickled, read at inference time |
| **Tier-assignment logic in production** | Ablation-only (post-hoc) | Importable + testable + reused in inference output |
| **Test mirror for experimental/ module** | None (0 tests) | Full coverage (estimated 10+ test files) |
| **Integration test parametrization** | Single test path | 5–6 hypotheses per scenario |
| **Failure-case unit tests** | Schema drift, NaN propagation | OOD edge cases, Regime-B boundary, bimodality grid misses |

---

## Part 7: File-by-File Citation Summary

### Production Drivers (Monolithic, Hard to Bisect)
- `/home/aneitzel/projects/ArqueoGal/scripts/run_pipeline1_inference.py:1255` — 5-domain coupling: ensemble load, feature assembly, OOD, Regime-B, output writing.
- `/home/aneitzel/projects/ArqueoGal/scripts/run_calibration.py:1062` — Dataloader + calibration tightly coupled; no hypothesis parametrization.
- `/home/aneitzel/projects/ArqueoGal/scripts/run_knn_rescue.py:~800` — Encoder latent + KNN tightly coupled; no synthetic-latent testing.

### Model Pattern (Post-Hoc Ablation)
- `/home/aneitzel/projects/ArqueoGal/scripts/test_ablations/run_per_cell_ablations.py:415` — Ablation configs (line 266–388), assign_tier() logic (line 122–163), metrics (line 166–216).
- `/home/aneitzel/projects/ArqueoGal/tests/scripts/test_run_pipeline1_inference.py:47KB` — Tiny FeatureLayout injection (line 101–119), mock ensemble (lines 122+).

### Missing Test Mirrors
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/experimental/` — No `/home/aneitzel/projects/ArqueoGal/tests/xp_abundances/experimental/`.
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/sanity.py` — CLAUDE.md acknowledges lack of tests.
- `/home/aneitzel/projects/ArqueoGal/scripts/run_calibration.py` — No `/home/aneitzel/projects/ArqueoGal/tests/scripts/test_run_calibration.py`.
- `/home/aneitzel/projects/ArqueoGal/scripts/run_knn_rescue.py` — No corresponding test file.

---

## Conclusion

The codebase has **demonstrated the ability to design hypothesis-driven test harnesses** (test_ablations/run_per_cell_ablations.py is exemplary), but **production drivers violate the principle by coupling multiple failure domains**. A debug subagent cannot bisect a "Mahalanobis OOD is broken" hypothesis without spinning up the entire 5-member ensemble and running inference on 1.3M rows (30+ min). Hypothesis-driven debugging would require:

1. Extracting OOD / Regime-B / Bimodality logic into post-hoc ablation harnesses (following test_ablations.py pattern).
2. Caching expensive intermediate results (ensemble latents, Mahalanobis bundles, preprocessed features).
3. Building unit-testable wrappers for feature assembly, tier assignment, and calibration logic.
4. Filling test gaps in experimental/ and script drivers.

**Estimated refactoring effort:** 2–3 weeks for full hypothesis-driven debug support (10–15 new test files, 3–5 extracted modules, caching infrastructure). **Payoff:** 3–5x faster root-cause analysis on future model/data failures.
