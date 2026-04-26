# Data Quality Audit — Sanity Battery Integration

**Audit date:** 2026-04-26  
**Auditor:** Claude Haiku 4.5  
**Scope:** `src/arqueogal/xp_abundances/main/sanity.py`, `scripts/run_pretraining_sanity.py`, integration with training pipeline

---

## A. Gate Enforcement: Is Sanity Battery a Real Training Blocker?

### Finding: Sanity battery is NOT integrated as a training halt mechanism.

The `sanity.py` module defines a complete battery (`run_battery`, `BatteryVerdict`) with hard-fail and soft-fail semantics, and the runner script (`run_pretraining_sanity.py:411–412`) correctly raises `SystemExit` when `verdict.overall == "HARD-FAIL"`. However:

1. **No training-pipeline integration:** The `src/arqueogal/xp_abundances/main/training.py` contains an unrelated `_first_epoch_sanity_check` (lines 320-347) that validates prediction-space statistics on the first training epoch (mean/std of predictions vs. truth). It does NOT call `sanity.run_battery()`.

2. **Orphaned pre-training gate:** The `run_pretraining_sanity.py` script is designed to run standalone and emit a report (`reports/sanity_battery/pretraining_audit.md`), but there is no CI/CD hook, makefile target, or documented required-before-training entry point. The script exists as an offline diagnostic tool, not a training-time gate.

3. **Training proceeds despite failures:** A user can manually run `run_pretraining_sanity.py`, see "HARD-FAIL", ignore the report, and proceed directly to `python -m arqueogal.xp_abundances.main.training`. No subprocess call or checkpoint validation stops them.

**Impact:** The battery's hard-fail semantics are aspirational rather than enforced. Tier 1 completeness checks and parameter bounds are well-designed (lines 157–259), but they do not halt training unless a user manually inspects the report and respects it.

**Recommendation:** Either (i) integrate `run_battery()` into the training entrypoint with early exit, or (ii) demote the battery from a "halt gate" to a "pre-training diagnostic" and update the docstring (line 4–5) to clarify the distinction.

---

## B. Column Contract Coverage: v5 Schema vs. Sanity Checks

### Finding: Sanity battery covers ~30% of required training-schema columns.

**PIPELINE1_TRAINING_SCHEMA v3 required columns (master_schema.py:197–217):**

- Identifiers: 3 columns (`source_id`, `apogee_id`, `sdss_id`)
- Astrometry: 8 columns (`ra`, `dec`, `parallax_corr`, `parallax_error`, `pmra`, `pmra_error`, `pmdec`, `pmdec_error`)
- Astrometric covariance: 10 columns (GAIA_ASTROMETRY_COV_COLS)
- Photometry: 4 columns (`phot_g_mean_mag_corr`, `bp_rp`, `bp_g`, `g_rp`)
- XP arrays: 4 columns (bp_coeffs_norm, rp_coeffs_norm, bp_coeff_errs_norm, rp_coeff_errs_norm)
- XP scalars: 2 columns (bp_c0_z, rp_c0_z)
- Distance: 3 columns (r_med_photogeo, r_lo_photogeo, r_hi_photogeo)
- Extinction: 4 columns (av_los, av_los_source, av_nbhd_median, av_nbhd_std)
- APOGEE atmospheric: 8 columns (teff_apogee, e_teff_apogee, logg_apogee, e_logg_apogee, mh_apogee, e_mh_apogee, alpha_m_apogee, e_alpha_m_apogee)
- APOGEE per-element: 34 columns (17 elements × 2: [X/H]_apogee, e_[X/H]_apogee)
- Flags: 2 columns (flag_bad, ruwe)

**Total required: 82 columns.**

**Sanity battery explicit coverage (sanity.py):**

| Check | Columns Tested | Count |
|-------|---|---|
| `check_xp_feature_nan_invariant` (lines 76–135) | bp_coef_norm_1..54, rp_coef_norm_1..54, bp_c0_z, rp_c0_z, ye2024_flag, xp_fit_flag_residual_high, bp_coef_0, rp_coef_0 | 112 + flags |
| `check_tier1_label_completeness` (lines 157–193) | teff_apogee, logg_apogee, mh_apogee, flag_bad | 4 |
| `check_parameter_bounds` (lines 228–259) | teff_apogee, logg_apogee, fe_h_apogee, mh_apogee, alpha_m_apogee, flag_bad | 6 |
| `check_per_element_nan_rates` (lines 265–310) | fe_h_apogee + tier2 (8 elements) + tier3 (8 elements), flag_bad | 17 |
| `check_zscore_validity` (lines 319–370) | bp_c0_z, rp_c0_z, ye2024_flag, xp_fit_flag_residual_high, bp_coef_0, rp_coef_0 | 6 |
| `check_dedup_idempotency` (lines 376–408) | source_id | 1 |
| `check_checkpoint_label_scaler` (lines 414–518) | label_scaler_mean, label_scaler_scale, label_names | 3 metadata |

**Coverage: Direct testing of ~130 column references, but most are:**
- XP coefficient scalars (implicit via normalized coef checks, not explicit range validation)
- APOGEE labels (atmospheric + element-wise NaN checks, NOT range validation for per-element [X/H])
- Flags (ye2024_flag, xp_fit_flag_residual_high, flag_bad, but NOT ruwe)

**Gaps:**

1. **NO validation of Gaia astrometry** (ra, dec, parallax_corr, pmra, pmdec, parallax_error, pmra_error, pmdec_error): No finite checks, no bounds checks, no plausibility checks (e.g. parallax > 0).

2. **NO validation of astrometric covariance** (10 correlation columns): No check that correlations ∈ [−1, +1].

3. **NO validation of Gaia photometry** (phot_g_mean_mag_corr, bp_rp, bp_g, g_rp): No finite checks, no color-magnitude consistency.

4. **NO validation of distance columns** (r_med_photogeo, r_lo_photogeo, r_hi_photogeo): No ordering checks (lo ≤ med ≤ hi), no finite checks.

5. **NO validation of extinction columns** (av_los, av_los_source, av_nbhd_median, av_nbhd_std): No physical bounds (Av ≥ 0), no source consistency.

6. **NO validation of per-element [X/H] ranges**: The gate (check 3, line 228) validates only Tier-1 atmospheric bounds. Tier-2 and Tier-3 elements are NaN-counted (check 4) but never range-checked. An [X/H] value of 99.9 dex would pass the battery.

7. **NO row-level cross-checks:** No validation that (Teff, log g) pairs are physically consistent (e.g. cool main-sequence stars are rare); no check that distance is plausible given G and extinction; no check that color residuals are small after dereddening.

8. **NO measurement-error validation:** Error columns (e_teff_apogee, e_logg_apogee, etc., 8 Tier-1 errors + 34 per-element errors) are not checked for finiteness, positivity, or plausibility.

---

## C. Tier-1 Atmospheric Completeness Check

### Finding: Check is correct and explicit; constant TIER1_ATMOSPHERIC is well-documented.

**Verification:**

- Constant definition (lines 140–154): Explicitly gates on `(teff_apogee, logg_apogee, mh_apogee)` only, excluding `fe_h_apogee`.
- Rationale (lines 145–154): Clear justification referencing ASPCAP DR19 per-element fitting behavior and research_brief §3.2.
- Implementation (lines 157–193): Hard-fail (line 185), checks all rows with `flag_bad == 0` (line 167), reports per-label NaN counts (lines 170–174).
- Test coverage: Covered in `tests/xp_abundances/main/test_data.py` (LabelTiers tests) but NO dedicated unit test for `check_tier1_label_completeness()` itself.

**Strength:** The check's logic and documentation are sound. The exclusion of `fe_h_apogee` is well-reasoned and citable.

**Weakness:** Without a unit test, future refactoring could silently break the hard-fail semantics or the row-filtering logic.

---

## D. Test Coverage: Sanity Module Tests

### Finding: Sanity module has ZERO unit tests.

**Evidence:**

- Files in `tests/xp_abundances/main/`: test_adapter.py, test_audit.py, test_bimodality.py, test_data.py, test_halfway_umap.py, test_inference.py, test_knn_rescue.py, test_losses.py, test_model.py, test_ood.py, test_release.py, test_release_pipeline.py, test_tier_promotion.py, test_training.py, test_uncertainty.py.

- NO test_sanity.py.

- Grep for `run_battery` or `BatteryVerdict` in the test tree (lines shown above): zero results.

**Impact:** 

1. Hard-fail checks (lines 549–555 in run_battery) are never unit-tested, so refactoring risk is high.
2. Edge cases (NaN-in-coefs-when-flagged, boundary parameter values, per-element rate calculations) are untested.
3. Integration with the runner script is not validated — e.g., the expected_dedup_rows constant (line 57 of run_pretraining_sanity.py) can drift without triggering a test failure.

---

## E. Silent Data-Quality Failures vs. Hard Errors

### Finding: XP coefficient NaN contract relies on side-channel flag columns that may not be synchronized.

**Issue (lines 89–98 of sanity.py):**

The check `check_xp_feature_nan_invariant` defines the expected mask:

```python
ye_bad = (df["ye2024_flag"] != 0)                           # A
strat_bad = ~(strat_flag == 0) | NA                         # B
c0_bad = (~isfinite(bp/rp_coef_0)) | (bp/rp_coef_0 <= 0)   # C
flagged_out = ye_bad | strat_bad | c0_bad
```

and asserts that all NaN in XP columns appear only where `flagged_out==True`.

**Risk:** If the emit pipeline (`data/gaia_xp.py`, `data/ingest_xp.py`) changes the condition under which it sets `ye2024_flag` or `xp_fit_flag_residual_high`, or if it skips setting one of these flags but still emits NaN, the check will catch it. However, the check is **silent in the NaN-to-flag direction**: it does not verify that *every* flagged-out row actually has NaN in the expected columns.

Consider: if a row is flagged as `ye2024_flag != 0` but all 108 XP feature columns are finite (a data-plumbing bug in the emit pipeline), the check passes, and the inconsistency propagates to training. The row gets marked "bad" but carries valid features, creating a silent inconsistency.

**Current behavior (line 114):** The check only reports surprise NaN (present where flagged_out==False). It does not report missing NaN (absent where flagged_out==True).

**Recommendation:** Add a counter for (flagged_out==True but all 108 XP columns are finite) and report it as a detail-level warning, or extend the pass criterion to require at least one NaN per flagged row.

---

## F. Soft-Fail Checks: Logging vs. Raising

### Finding: Soft-fail checks (5–6) are logged by the runner but do not halt training.

**Location:** Lines 321–356 (runner output) and lines 319–408 (checks 5–6).

**Behavior:**

- `check_zscore_validity` (line 319): Soft-fail, computes statistics, returns False if mean/std deviate from 0/1 by more than tolerance, but training continues.
- `check_dedup_idempotency` (line 376): Soft-fail, compares dedup output to expected row count.

Both are rendered in markdown (runner lines 207–209) with summary and JSON details, but neither raises an exception or logs a warning at the WARNING or ERROR level. The runner script logs at INFO level (line 51) and just appends results to the markdown report (line 219).

**Impact:** A production run with soft-fail output (e.g. z-score mean = 0.15 instead of 0.001) produces a markdown report that only the operator reading the file will discover. If the report is not read, the training proceeds with silently miscalibrated features.

**Recommendation:** Add a `logger.warning()` call in the runner for any soft-fail check (lines 358–361), and raise an exception if soft-fail checks are present (optional exit-code 1 for non-blocking CI runs).

---

## G. Provenance and Column Fingerprinting

### Finding: Sanity battery does NOT verify that the emitted parquet matches the declared schema.

The parquet file carries a provenance sidecar (`pipeline1_features_stream1.parquet.provenance.json`) per CLAUDE.md invariant 14, but the battery does NOT:

1. Load and validate the sidecar against the current data.
2. Check that the column set matches the schema (PIPELINE1_TRAINING_SCHEMA).
3. Verify that the frozen Hermite z-score stats (basis fingerprint) used to compute `bp_c0_z` / `rp_c0_z` match the constants expected for Stream 3 inference (CLAUDE.md §16).

**Opportunity:** Add a check that loads `provenance.json` and validates (i) column count and names, (ii) input row counts, (iii) corrections applied (Ye2024 NN flux, Riello+2021 G-mag, Mészáros+2025 [X/M]). This would catch upstream emit-pipeline drifts at gate time.

---

## H. Great Expectations / Pandera Hardening Opportunity

### Finding: Plain assertions and conditional loops are prone to silent failures.

The battery is written in procedural numpy/pandas (no schema validation library). Specific vulnerabilities:

1. **NaN detection (line 108):** `df[col].isna().to_numpy()` silently returns an empty array if `col` is missing from the dataframe. A missing column would cause an AttributeError, but a mislabeled column (e.g. `bp_coef_norm_1` vs. `bp_coeff_norm_1`) would not.

2. **Flag composition (lines 89–98):** Three separate boolean masks are OR'd with no explicit validation that they are all the same length.

3. **Parameter bounds loop (lines 232–245):** The loop iterates over PARAMETER_BOUNDS items, but if a key is missing from the dataframe, it raises KeyError (which is correct) but the error message could be clearer.

**Hardening options (in priority order):**

- **Option A (Lightweight):** Wrap the battery in a schema pre-check (`PIPELINE1_TRAINING_SCHEMA.validate(df)`) before running checks.

- **Option B (Pandera):** Define a Pandera schema for the training parquet and run it before the battery. Pandera provides column/dtype/range validation in a declarative form.

- **Option C (Great Expectations):** Migrate the battery to Great Expectations Expectations Suite (lines define expected column names, NaN rates, value ranges, correlations). GX is heavier but provides better introspection, auto-generated docs, and integration with orchestration tools (Dagster, Prefect).

**Recommendation:** At minimum, add Option A (schema pre-check); Option B is worthwhile if the battery grows to include extinction/distance/photometry checks.

---

## I. Data-Quality Issues Silently Logged

### Finding: The audit module (audit.py) reports at INFO level for non-blocking diagnostics.

While not part of the pre-training sanity battery, the post-training audit (audit.py §9.2 validation tests) logs diagnostics via Python logger at INFO level. For instance, stubbed test 3 (SHAP) and test 6 (cross-catalogue consistency) are noted in the code but not enforced; the report just says "5/6 coverage" (lines 92–94 of audit.py).

**Risk:** A future audit run that fails test 5 (CMI) or passes test 6 with low signal will be visible in the audit report JSON but not in the training logs unless explicitly grepped. The tier-promotion gate (tier_promotion.py lines 9–20) references the stubbed tests, but calling code must check the JSON output to know if a label was actually promoted.

**Recommendation:** Audit-level pass/fail decisions (tier promotion) should raise or log at WARNING level, not INFO. Alternatively, the release script (release.py) should validate that all tiers were promoted with complete test coverage (6/6) and fail if coverage is 5/6.

---

## J. Summary of Data-Quality Hardening Recommendations

| Priority | Category | Finding | Action |
|---|---|---|---|
| **Critical** | Integration | Sanity battery is not a training halt gate. | Integrate `run_battery()` into training entrypoint or clarify as pre-deployment diagnostic. |
| **Critical** | Coverage | No validation of 50+ astrometry, photometry, distance, extinction columns. | Add schema pre-check and/or basic bounds checks for physical validity. |
| **Critical** | Testing | Zero unit tests for sanity module. | Add test_sanity.py with fixtures for edge cases (NaN boundary, parameter limits). |
| **High** | Errors | Silent NaN-to-flag mismatches not detected. | Extend check_xp_feature_nan_invariant to report flagged rows with ALL-finite XP columns. |
| **High** | Logging | Soft-fail checks do not warn at WARNING level. | Add logger.warning() for soft-fail verdicts in runner script. |
| **Medium** | Schema | No validation against declared PIPELINE1_TRAINING_SCHEMA. | Call MasterSchema.validate(df) as the first battery step. |
| **Medium** | Provenance | Frozen Hermite z-score stats not verified. | Add check that loads provenance.json and validates input metadata. |
| **Low** | Maturity | Procedural assertions vs. declarative schema validation. | Evaluate Pandera or Great Expectations for future generalization. |

---

## References

- `src/arqueogal/xp_abundances/main/sanity.py` — sanity battery module (576 lines)
- `scripts/run_pretraining_sanity.py` — runner script (417 lines)
- `src/arqueogal/data/master_schema.py` — column contract definitions (424 lines)
- `docs/research_brief.md` — §3.3 tier promotion, §9.2 audit protocol
- `CLAUDE.md` — §3 Hard invariants, §14 Tier promotion test 6 stub status
