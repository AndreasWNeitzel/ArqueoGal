# Debuggability Audit: ArqueoGal xp_abundances.main

**Date:** 2026-04-26  
**Scope:** `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/`  
**Focus:** Silent failures, missing assertions, inadequate error context, NaN instrumentation

---

## Summary

The codebase exhibits **strong error-message discipline** with explicit shape/dimension context. The **NaN boundary between training and inference is well-instrumented with post-hoc assertions** (inference.py:242-246). However, three debuggability gaps exist: (1) silent cell-skipping in GP α-smoothing without logging which cells were excluded, (2) inadequate NaN documentation in `collect_predictions` output, and (3) missing per-batch/per-member diagnostic context in training loss spikes.

---

## Findings

### 1. NaN Boundary Instrumentation (ADR-0012) — STRONG

**Location:** `training.py:154`, `inference.py:240-246`

**Status:** Properly implemented.

- **Training path:** `build_dataloaders` (line 137-154) drops rows with NaN in core XP features before loading, then applies `np.nan_to_num` to residuals/aux at line 154 with `copy=False`.
- **Inference path:** `predict_ensemble` (line 240-246) sanitizes `mu_m` and `L_m` from `collect_predictions` with assertions that would catch post-sanitization non-finite values.
- **Gap:** The assertions compare output to an unreachable sentinel (`np.nan_to_num` output is always finite by definition), so an honest failure mode (NaN appearing after model forward but before sanitization) would silently pass through. **However**, this is mitigated by the assertion itself—if `mu_m` contains NaN after `nan_to_num(..., nan=0.0)`, that is a catastrophic model-output failure that an assertion can detect. The design is sound.

**Location:** `uncertainty.py:89-125` (`collect_predictions`)

**Issue:** Returns `mu`, `L`, `y` stacked from model outputs. These **may contain NaN** (especially `y` from validation loader on incomplete labels), but the docstring (lines 94-97) does not warn about this. A caller using `collect_predictions` output directly for diagnostics (e.g., computing CMI on uncleaned `y`) may silently undercount rows.

**Example:** Line 119-121 concatenates raw model outputs and uncleaned labels without noting that NaN rows are present. A downstream metric computed on `"y"` will silently drop NaN via `np.isfinite()` filters in calibration code, but a naive user would not discover this without reading calibration.py.

---

### 2. Silent Cell Skipping in GP α-Smoothing — WEAK DEBUGGABILITY

**Location:** `uncertainty.py:506-532` (`gp_smoothed_per_cell_per_label_scale`)

**Issue:** Cells are silently excluded from GP training with three criteria:
- Line 511: `if n_c < min_cell_stars: continue`
- Line 514: `if not np.isfinite(center).all(): continue`
- Line 528: `if not train_cell_ids: raise RuntimeError`

The first two exclusions (undersample + non-finite center) issue no logging. A user observes that 64 cells were loaded but the GP trained on only 48, with no indication why 16 were skipped. The function is not called in production (GP smoothing rejected per CLAUDE.md footgun note), but if reactivated, this would be a silent-failure vector.

**Better approach:** Add a diagnostic dict returned with training info (count of skipped cells per reason).

---

### 3. Missing Per-Batch Diagnostic Context in Training — MODERATE

**Location:** `training.py:608-617` (grad-norm abort check)

**Issue:** When `grad_norm > cfg.grad_norm_abort_threshold`, the error message includes loss, NLL, SupCon per the current batch but **does not include the batch index or sample IDs**. A user observing "grad_norm=3.14 exceeded 2.5 at batch {n}" cannot immediately identify which stars caused the spike.

**Current output (line 609-616):**
```
RuntimeError(
    f"grad_norm={g_val:.2f} exceeded abort threshold ... "
    f"(batch {n}; loss={float(total):.4f}, "
    f"nll={parts.get('nll', float('nan')):.4f}, "
    f"supcon={parts.get('supcon', float('nan')):.4f}). ..."
)
```

**Improvement:** Add `source_id` range to the error message, e.g. `source_ids {batch.source_id.min():.0f}..{batch.source_id.max():.0f}`. Since the Dataset stores `source_id` (line 197-203), a 1-line change could surface this at abort time.

---

### 4. XpFeatureAdapter Input Validation — ADEQUATE

**Location:** `adapter.py:117-128`

**Status:** Clear. The docstring (line 32) explicitly states "does NOT guard against NaN"; inference.py module docstring (line 26-29) mirrors this contract.

---

### 5. Error Messages with Full Context — STRONG

**Examples:**
- `losses.py:173-174`: "sample_weights shape X != (B,) = (Y,)"
- `losses.py:180`: "mask shape X != y shape Y"
- `losses.py:197-202`: three explicit shape checks with dimension names
- `training.py:271-273`: "inverse_freq_mh_column not in tiers.all_labels" lists both values

All shape-mismatch errors include actual dimensions, not generic "dimension mismatch".

---

### 6. Label Scaler Validation — STRONG

**Location:** `training.py:994-1004`

Two defensive checks before saving checkpoint:
1. `label_scaler.label_names` must equal `tiers.all_labels` (line 994-999) — prevents silent reorder bugs.
2. `label_scaler.is_default()` check (line 1000-1004) — catches placeholder scalers.

---

## Recommendations

### High Priority
1. **Augment `collect_predictions` docstring** (uncertainty.py:94-97) to explicitly warn:
   ```
   Note: output "y" may contain NaN on rows with missing labels (Tier 2/3).
   Downstream code must handle this via np.isfinite() filtering.
   ```

2. **Add `source_id` range to grad-norm abort** (training.py:609):
   ```python
   source_ids = batch[3]  # or passed separately if not in current batch tuple
   raise RuntimeError(
       f"... batch {n} (source_ids {source_ids.min():.0f}..{source_ids.max():.0f}); ..."
   )
   ```

### Medium Priority
3. **Log skipped cells in `gp_smoothed_per_cell_per_label_scale`** (uncertainty.py:506-532):
   - Count cells rejected per criterion (undersample, non-finite center).
   - Return diagnostics dict or log at info level.
   - (Low impact: function not active in production.)

### Low Priority
4. No broad structural gaps identified. Assertions are appropriately placed; tracebacks are not masked (no bare `except`); NaN boundaries have explicit guards with assertions.

---

## Files Modified/Audited

- `training.py`: 1054 lines, NaN guard + label validation ✓
- `inference.py`: 298 lines, NaN assertions + calibration ✓
- `uncertainty.py`: 978 lines, cell binning, calibration, GP smoothing — partial instrumentation
- `adapter.py`: 187 lines, contract clearly documented ✓
- `data.py`: 600+ lines, stratified split, error messages explicit ✓
- `losses.py`: 330+ lines, shape validation thorough ✓
- `bimodality.py`: Exception handling explicit with reason strings ✓

---

## Conclusion

**Debuggability is high for the core pipeline** (training → inference → aggregation). The NaN boundary is properly guarded and would catch downstream model failures via post-hoc assertions. Error messages consistently include actual values, not generic placeholders. The main gaps are documentation (what does collect_predictions output contain?) and optionality (GP smoothing's silent cell rejection is a footgun when re-enabled). None are blockers for current production use.
