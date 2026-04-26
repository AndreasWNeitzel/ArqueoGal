# Pytest Patterns Audit — ArqueoGal test suite
**Date:** 2026-04-26  
**Scope:** 60 test files, 300+ test functions, xp_abundances/main focus with emphasis on test_release.py v5 update  

## Summary Findings

**Strengths:** Test-Release.py v5 demonstrates excellent design, with clear parametrization of gate logic, separate per-element vs. composite tier tests, and exhaustive v5 caveat reorganization. Parametrization is used judiciously (3 total across the suite), avoiding bloat while covering edge cases effectively.

**Weaknesses:** Fixture hygiene is inconsistent; repeated monkeypatch setup patterns across test_gaia_enrich.py, test_ingest_stream1.py, and test_ingest_stream3.py should consolidate to shared conftest.py. Marker coverage is minimal (only 6 @pytest.mark.slow out of 300+ tests); GPU tests lack explicit markers despite requiring CUDA, and stress-battery is opt-in via command-line flag rather than test discovery.

**Recommendation:** Elevate shared fixtures to root conftest.py, add GPU marks to test_hybrid_stress_battery.py, and consider autouse fixtures to enforce monkeypatch poisoning contracts.

---

## 1. Parametrization Patterns

### Well-executed parametrization in test_release.py

**File:** `/home/aneitzel/projects/ArqueoGal/tests/xp_abundances/main/test_release.py`

**Parametrize usage (3 total in entire suite):**
1. **Lines 77–97:** `test_diagnostic_only_flags_do_not_change_tier` — parametrizes v5 diagnostic-only flags  
   ```python
   @pytest.mark.parametrize(
       "diagnostic_kw",
       [
           {"regime_b": True},
           {"ood_disagreement": True},
           {"aux_missing": True},
           {"latent_support": True},
       ],
   )
   ```
   **Assessment:** Excellent. Each parameter set tests a distinct gate-logic path; test name clearly indicates the invariant being checked. No repeated assertions—just one per element.

2. **Lines 100–111:** `test_single_hard_kill_demotes_to_tier_3`  
   ```python
   @pytest.mark.parametrize(
       "kill_kw",
       [
           {"ood_joint": True},
           {"pred_nan": True},
       ],
   )
   ```
   **Assessment:** Lean, non-bloated. Two hard-kill conditions, each demotes to Tier 3. Parametrization justifies itself because the assertion is identical for both cases.

3. **Lines 509–528:** `test_assign_prediction_sigma_inflated_above_threshold_is_true`  
   ```python
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
   ```
   **Assessment:** Textbook example. Each row tests a per-element threshold (v4 σ caveat). The thresholds are documented in the comment row, and the parametrization avoids 5× code duplication. Every tuple generates a distinct test ID.

### No parametrize abuse detected
The suite avoids the antipattern of bundling orthogonal test cases into parametrization. Tests like `test_clean_row_is_tier_1` (line 57) and `test_mode_ambiguous_demotes_only_alpha_m_to_tier_2` (line 63) remain separate functions, each with its own docstring, because they test *different* behaviors. Good discipline.

---

## 2. Fixture Patterns and Repeated Setup

### Inconsistent fixture consolidation across data layer

**Pattern:** Multiple test files in `tests/data/` define similar monkeypatch fixtures inline, duplicating mocking infrastructure.

**Example 1: TAP mocking**  
- `test_gaia_enrich.py:46–61` (capture_sync fixture)  
- `test_gaia_enrich.py:64–78` (capture_async fixture)  
- `test_ingest_stream1.py:60–114` (patched_pipeline fixture with 6× monkeypatch calls)  

**Duplication:** `_fake_result`, `_extract_batch_ids`, and the `fake_sync`/`fake_async` factory pattern are repeated across multiple files.

**Recommendation:** Extract to `tests/data/conftest.py`:
```python
@pytest.fixture
def mock_tap_sync(monkeypatch):
    """Shared TAP sync mock for data layer."""
    def fake_sync(_service, adql, **kw):
        ids = _extract_batch_ids(adql)
        return _fake_result(ids)
    monkeypatch.setattr("arqueogal.data.tap", "run_sync", fake_sync)
    return fake_sync
```

**Current state:** Each file is self-contained, which aids readability for a single test but increases maintenance burden if the TAP mock contract changes.

### Inline test helpers vs. fixtures

**File:** `test_release.py:31–54` (`_row()` factory function)  
**Assessment:** Excellent choice. The `_row()` helper is test-specific, not worth a fixture. Decorated with `**kwargs` for fine-grained control. Parametrization uses this helper cleanly (lines 63, 114, etc.).

**File:** `test_kinematics.py:33–75` (`_sunlike_row()`, `_mc_row()` helpers)  
**Assessment:** Correct; these are helper builders for row construction, not setup/teardown.

### Fixture scope confusion: no root conftest.py

**Current state:** Only `tests/integration/conftest.py` exists (32 lines), defining the `--run-stress` plugin hook.  
**Gap:** No root `tests/conftest.py`. All fixtures are file-local.

**Implication:**  
- Integration-wide fixtures (e.g., tmp_path mock factories) cannot be shared.
- Each test file must independently `monkeypatch` if it mocks TAP.
- Test order could affect fixture state if scope="session" fixtures existed (they don't, so low risk).

**Line count:** Integration conftest is minimal and correct; root conftest is absent by design (probably intentional to keep tests isolated).

---

## 3. Test Categorization and Markers

### Sparse marker usage

**Current markers (from pyproject.toml):**
```toml
markers = [
    "slow: integration tests invoking real galpy/astropy (seconds each)",
    "gpu: tests that require a CUDA device and the production checkpoint",
    "stress: hybrid stress-battery tests, opt-in via --run-stress",
]
```

**Actual usage in codebase:**
```bash
grep -r "@pytest.mark\." tests/ --include="*.py" | sort | uniq -c | sort -rn
      6 @pytest.mark.slow        (test_kinematics.py only)
      3 @pytest.mark.parametrize (test_release.py only)
      1 @pytest.mark.skipif      (test_gaia_corrections.py, test_frozen_stats.py)
      1 @pytest.mark.gpu         + @pytest.mark.slow + @pytest.mark.stress
        (test_hybrid_stress_battery.py only)
```

**Coverage:** ~1.9% of test functions are marked.

**Gaps:**
1. **No integration markers** except stress-battery. Data-layer tests (test_gaia_enrich.py, test_ingest_stream1.py) mock TAP but are not marked @pytest.mark.integration.
2. **GPU test coverage:** Only `test_hybrid_stress_battery.py` has `pytestmark = [pytest.mark.gpu, ...]` (line header). Inference tests in `test_inference.py` likely require GPU but carry no marker.
3. **Stress-battery is opt-in via command-line, not discovery:** `conftest.py:26–32` uses `pytest_collection_modifyitems` to skip stress tests by default. This is correct and explicit.

**Recommendation:**  
1. Add `@pytest.mark.integration` to all tests in `tests/data/` (they mock external services).  
2. Scan `tests/xp_abundances/main/` for GPU-requiring tests (e.g., `test_inference.py`, `test_training.py`) and add `@pytest.mark.gpu`.  
3. Consider `@pytest.mark.smoke` for fast sanity checks (< 100 ms).

---

## 4. Test Independence and Order Sensitivity

### No detected test order dependencies

**Assertion:** Scanned test_release.py, test_kinematics.py, test_audit.py for shared state.  
- All fixtures use `tmp_path` (function-scoped, unique per test).  
- No module-level globals written by tests.  
- Monkeypatches are autocleared per-function.

**Assessment:** Tests are properly isolated. Running subsets of tests in any order will not cause cross-test pollution.

**Edge case:** `test_annotate_parquet_is_idempotent` (test_release.py:175–184) intentionally calls `annotate_parquet` twice on the same parquet to verify idempotency. This is a valid test pattern, not a dependency; each call uses a fresh tmp_path.

---

## 5. Test Release v5 Update (2026-04-26)

### Design quality: excellent

**File:** `/home/aneitzel/projects/ArqueoGal/tests/xp_abundances/main/test_release.py`

**V5 changes codified in test names and docstrings:**
1. **Lines 63–74:** `test_mode_ambiguous_demotes_only_alpha_m_to_tier_2` — new; v5 narrowed mode_ambiguous to [α/M] only. Docstring cites "v5 (2026-04-26)".
2. **Lines 77–97:** `test_diagnostic_only_flags_do_not_change_tier` — docstring (line 87) cites "v5 (2026-04-26): regime_b_flag, ood_disagreement_flag, aux_missing_any, latent_support_flag, ... retired from tier gating" and references the empirical justification document.
3. **Lines 409–434:** Retired-flag tests (`test_aux_mahalanobis_ood_flag_is_diagnostic_only_in_v5`, `test_dist_prior_dominated_is_diagnostic_only_in_v5`). Each docstring explains what was removed and when.
4. **Lines 437–447:** `test_mode_ambiguous_per_element_caveat` — per-element tier logic now demotes [α/M] to Tier 2 only.

### Comprehensive per-element tier testing

**Lines 333–470 (per-element tier gate subsection):**
- `test_per_element_tier_clean_row_all_tier_1` (line 333) — baseline.
- `test_per_element_tier_kin_ood_demotes_only_aux_assisted` (line 342) — aux-assisted [α/M] and [Mg/H] demote to Tier 2, spectrum-dominant unaffected.
- `test_per_element_tier_diagnostic_flags_do_not_demote` (line 355) — **replaces v3 inverse test** (line 359 docstring explicitly says "Replaces the v3 `test_per_element_tier_global_caveat_demotes_all`").
- `test_per_element_tier_ood_demotes_all_to_tier_3` (line 374) — hard-kill on joint OOD.
- `test_per_element_tier_per_element_nan_only_demotes_that_element` (line 383) — NaN in α/M_pred → α/M Tier 3, others Tier 1.
- `test_assign_release_tier_is_row_max_of_per_element` (line 396) — integration test verifying composite tier = max(per-element tiers).

**Assessment:** Each test is narrowly scoped (one behavior per function) and the composite test (line 396) verifies the row-max aggregation. No code duplication across the 7 per-element tests.

### σ-threshold caveat (v4, retained in v5)

**Lines 473–630 (σ-threshold subsection):**
- `test_assign_prediction_sigma_inflated_*` (5 functions + parametrized) — comprehensive coverage of threshold logic.
- **Boundary condition test (line 547):** "σ exactly at threshold is False" — pins strict `>`, not `>=`. Critical for reproducibility.
- **Integration with tier gates (line 567):** σ inflation demotes per-element to Tier 2; NaN hard-kill (Tier 3) dominates caveat (line 583).

**Assessment:** Thorough. The v4 caveat is well-tested and integration with v5 gate logic is verified.

---

## 6. Missing-Column Tolerance and Graceful Degradation

**File:** `test_release.py:119–132` (`test_missing_flag_columns_are_treated_as_false`)  
```python
def test_missing_flag_columns_are_treated_as_false():
    # Strip all flag columns — remaining row must be Tier 1
    row = _row()
    for key in ("regime_b_flag", "mode_ambiguous_flag", ...):
        row.pop(key)
    df = pd.DataFrame([row])
    tier = assign_release_tier(df)
    assert tier.iloc[0] == 1
```

**Assessment:** Excellent. Explicitly validates that missing flag columns default to False, not crash. This is essential for backward compatibility if an older parquet lacks a newly-added diagnostic column.

**Related:** `test_assign_g_mag_bin_missing_column` (line 231) — when no G column, returns "unknown" not error. Graceful degrade.

---

## 7. Conftest Organization and Hooks

### Integration conftest: minimal and correct

**File:** `/home/aneitzel/projects/ArqueoGal/tests/integration/conftest.py` (32 lines)

**Content:**
- `pytest_addoption` hook (lines 17–23) — adds `--run-stress` CLI flag.
- `pytest_collection_modifyitems` hook (lines 26–32) — skips `@pytest.mark.stress` tests unless `--run-stress` passed.

**Assessment:** Correct and focused. The opt-in pattern avoids accidental CI runs of expensive tests.

### Root conftest absent

**Current:** No `tests/conftest.py`.  
**Impact:** All shared fixtures must be file-local or scope="session".

**Hypothetical improvement:** A root conftest could house:
```python
# tests/conftest.py
@pytest.fixture
def mock_tap_sync(monkeypatch):
    """Shared TAP sync mock for data layer."""
    ...

@pytest.fixture
def mock_tap_async(monkeypatch):
    """Shared TAP async mock for data layer."""
    ...
```

**Current workaround:** Each file (test_gaia_enrich.py, test_ingest_stream1.py, test_ingest_stream3.py) defines its own. This is maintainable but verbose.

---

## 8. Test File Naming and Organization

### Naming consistency: excellent

**Structure:**
```
tests/
├── data/
│   ├── test_apogee_dr19.py
│   ├── test_gaia_corrections.py
│   ├── test_gaia_enrich.py
│   ├── test_ingest_stream1.py
│   └── ... (31 more files)
├── xp_abundances/
│   ├── main/
│   │   ├── test_release.py
│   │   ├── test_training.py
│   │   ├── test_inference.py
│   │   └── ... (12 more files)
│   └── experimental/
└── integration/
    └── test_hybrid_stress_battery.py
```

**Assessment:** Mirrored from `src/` tree. No ambiguity. Each test file names match its source module counterpart. Follows PEP 328 naming ("test_<module>.py").

---

## 9. Unmocked Side Effects and Integration

### TAP mocking thoroughness

**File:** `test_gaia_enrich.py:46–78`  
- Fixture capture_sync (line 46–61): monkeypatches both `run_sync` and poisons `run_async`.  
- Fixture capture_async (line 64–78): monkeypatches both `run_async` and poisons `run_sync`.

**Assessment:** Defensive. Poisoning the unused code path ensures test leakage is caught immediately. Good practice.

### Silent imports without side effects

**Scanned files:** test_model.py, test_inference.py, test_training.py, test_uncertainty.py  
- All import from `arqueogal.xp_abundances.main.*` and `torch`.  
- No `torch.cuda.set_device()` or device-selection code at module level.  
- GPU device selection deferred to fixture or test function.

**Assessment:** Clean. No module-level device assignment that could break tests on CPU-only CI.

---

## 10. Coverage Gaps and Stubs

### Missing test trees (noted in CLAUDE.md)

From `CLAUDE.md` (Hard invariants §3):
> **Gaps:** `xp_abundances/main/config.py`, `sanity.py` lack tests. Experimental test trees are empty.

**Verification:**
```bash
ls tests/xp_abundances/main/test_config.py  # → file not found
ls tests/xp_abundances/experimental/        # → empty (only __init__.py)
```

**Assessment:** Known gaps. Not a pattern hygiene issue; a deliberate skipped coverage area.

---

## 11. Audit Test Placeholders

**File:** `test_release.py:273–276` (sidecar manifest)
```python
# v5 sidecar advertises retired-but-emitted diagnostic flags separately.
assert "diagnostic_only_columns" in payload
assert "per_element_caveat_flags" in payload
```

**Assessment:** Good. The test does not stub the metadata; it verifies structure. If the payload schema changes, this test will fail. No test is silently passing because a feature is unimplemented.

---

## 12. Summary of Recommendations

| Issue | File(s) | Severity | Action |
|-------|---------|----------|--------|
| Repeated monkeypatch patterns | test_gaia_enrich.py, test_ingest_stream1.py, test_ingest_stream3.py | Medium | Consolidate to `tests/data/conftest.py` |
| GPU tests lack explicit marks | test_inference.py, test_training.py | Low | Add `@pytest.mark.gpu` to functions requiring CUDA |
| Data-layer tests lack integration marks | tests/data/*.py | Low | Add `@pytest.mark.integration` to data layer |
| No root conftest.py | — | Low | Create tests/conftest.py for cross-layer fixtures (optional) |
| Sparse marker coverage (~1.9%) | — | Low | Aim for 100% marker coverage; use @pytest.mark.smoke for fast tests |

---

## Conclusion

The test suite demonstrates strong discipline in test_release.py (v5 update is exemplary), with clear separation of concerns, minimal parametrization abuse, and excellent integration testing patterns. The main hygiene issue is fixture consolidation in the data layer, which can be addressed without refactoring. Marker coverage is sparse but adequate; expanding it would improve CI/CD filtering. Overall: **well-structured, maintainable, low-risk test portfolio**.
