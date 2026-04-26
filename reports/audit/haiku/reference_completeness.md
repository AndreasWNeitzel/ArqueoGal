# Reference Documentation Completeness Audit
**Date:** 2026-04-26  
**Scope:** ArqueoGal user-facing reference surface (CATALOG_SCHEMA.md, CLI scripts, public __all__ exports, DESIGN.md)

---

## Findings

### 1. Undocumented Public Functions in release.py

**Status:** ✓ PASS

The `src/arqueogal/xp_abundances/main/release.py` module exports 9 functions via `__all__`:
- `annotate_parquet`, `assign_dist_prior_dominated`, `assign_g_mag_bin`, `assign_kin_ood_flag`
- `assign_per_element_release_tier`, `assign_prediction_sigma_inflated`, `assign_release_tier`, `tier_counts`

All are documented in DESIGN.md (§ "Module layout") with their roles clearly stated. The internal constants `_PER_ELEMENT_CAVEAT_FLAGS`, `_OOD_FLAGS`, `_CAVEAT_FLAGS`, and `_PRED_COLS` are documented as module-level module docstring constants with their semantics and ablation-study provenance (ADR-0015, release/test_ablations_2026-04-26/REPORT.md references).

**Note:** The `_PER_ELEMENT_CAVEAT_FLAGS` is correctly marked as `Final[dict[str, tuple[str, ...]]]` and is reference-documented in the module docstring at lines 90–101. It is not exported via `__all__` (which is correct — it is a private implementation detail).

### 2. CLI Scripts Help Text

**Status:** PARTIAL — 1 script notably deficient

Script `/home/aneitzel/projects/ArqueoGal/scripts/run_pipeline1_inference.py` carries an **extremely verbose, multi-paragraph docstring that is actually the entire --help text**, yet the argparse definition is sparse. Specifically:

- The docstring in the script header is 900+ lines and covers the logic in narrative form (well-written, comprehensive).
- Individual argument `--help` strings in argparse are minimal (1–2 lines where they exist).
- **Gap:** A user running `--help` sees the short argparse help, not the comprehensive docstring. The full docstring is only visible by reading the file directly.

**Recommendation:** Migrate key docstring sections into argparse `.description` or `.epilog` to make the full narrative available via `--help`.

### 3. CATALOG_SCHEMA.md Completeness vs. Column Emissions

**Status:** ISSUE FOUND — One column not documented

The column `parallax_over_error` is assigned in release.py but **not documented** in CATALOG_SCHEMA.md. Grep search:
```
release.py:   df['parallax_over_error'] = (df['parallax_corr'] / df['parallax_error']).astype('float32')
CATALOG_SCHEMA.md: no mention of parallax_over_error
```

This appears to be a post-parquet-join diagnostic column (not in the main tier-annotation pipeline) used for Mahalanobis distance computation. **Recommendation:** Add a subsection in CATALOG_SCHEMA.md under "Quality Flags" or "Diagnostics" to document this column, or confirm it is transient and remove it from the release pipeline.

### 4. v5 Retired-but-Emitted Diagnostic Flags — CATALOG_SCHEMA.md Completeness

**Status:** ✓ PASS (but could be more explicit)

The CATALOG_SCHEMA.md correctly marks v5 changes at the schema header (line 5, version 5 tag) and within each column's row-comment. For example:

- `latent_support_flag` (line 131): **v5 diagnostic-only (retired from gating).** ✓ Marked.
- `ood_aux_mahalanobis_flag` (line 132): **v5 diagnostic-only (retired from gating).** ✓ Marked.
- `regime_b_flag` (line 144): **v5 diagnostic-only (retired from gating).** ✓ Marked.
- `aux_missing_any` (line 147): **v5 diagnostic-only (retired from gating).** ✓ Marked.

The DESIGN.md also provides the ablation-study reference (release/test_ablations_2026-04-26/REPORT.md) for why they were retired. However, the CATALOG_SCHEMA.md does not explicitly state the **sidecar manifest strategy** (distinguishing `expected_upstream_columns` vs `diagnostic_only_columns` in JSON metadata). That distinction is mentioned only in DESIGN.md, not CATALOG_SCHEMA.md.

**Recommendation:** Add a brief note in CATALOG_SCHEMA.md §2 (Overview or Sidecar section) that the accompanying JSON sidecar (`*.release_tier.json`) carries the authoritative list of which columns are active gates vs. diagnostic-only.

### 5. Hybrid Columns Cross-Reference

**Status:** ✓ PASS (but inverse reference missing)

CATALOG_SCHEMA.md documents hybrid columns (§3.6, lines 259–273):
- `<elem>_hybrid_pred`, `<elem>_hybrid_sigma`, `<elem>_hybrid_source`, `<elem>_hybrid_tier`

The documentation correctly notes they are emitted by `release_pipeline.attach_hybrid_columns` and references the "Gaussian-quantile inversion" caveat on the σ calculation.

**Gap (Minor):** The CATALOG_SCHEMA.md does not back-reference the corresponding section in DESIGN.md (§ "v5 schema additions — hybrid composer columns"). An inverse citation (DESIGN.md → CATALOG_SCHEMA.md §3.6) is also missing.

The logic lives in `src/arqueogal/data/release_pipeline.py`, not in the xp_abundances.main module. This separation is architecturally correct (release_pipeline is the composer, xp_abundances is the producer), but the cross-module citation chain is loose. **Recommendation:** Add a note in both docs linking the two perspectives: "For the producer-side (per-element kNN artifact structure), see DESIGN.md § 'Latent-kNN rescue contract'; for the consumer-side (hybrid column composition), see data/release_pipeline.py::attach_hybrid_columns."

### 6. Master Schema (master_schema.py) Documentation Alignment

**Status:** ✓ PASS

`master_schema.py` defines three frozen schema contracts:
- `PIPELINE1_TRAINING_SCHEMA`
- `PIPELINE1_INFERENCE_SCHEMA`
- `PIPELINE2_FEATURES_SCHEMA` (retained for historical compatibility, not written by production code)

All three are aligned with CATALOG_SCHEMA.md via the column registries (e.g., `GAIA_ASTROMETRY_COV_COLS`, `XP_ARRAY_COLS`, `XP_SCALAR_COLS`, `APOGEE_ELEMENT_LABELS`). The docstrings reference data_acquisition.md correctly.

### 7. Data Acquisition Reference (data_acquisition.md)

**Status:** ✓ PASS

data_acquisition.md §10 correctly cross-references the master_schema.py contracts and provides the data-layer picture that CATALOG_SCHEMA.md (release-layer) and DESIGN.md (module-layer) do not attempt to cover.

---

## Summary Table

| Aspect | Status | Notes |
|---|---|---|
| Public functions in xp_abundances.main/__init__.py | ✓ | All in __all__, documented in DESIGN.md |
| CLI --help completeness (run_pipeline1_inference.py) | ⚠️ | Verbose docstring exists but not exposed via argparse help |
| CATALOG_SCHEMA.md vs. emitted columns | ⚠️ | `parallax_over_error` undocumented; transient vs. permanent unclear |
| v5 retired flags marked in schema | ✓ | All marked as "diagnostic-only"; sidecar strategy not explicit |
| Hybrid columns cross-reference | ⚠️ | DESIGN.md ↔ CATALOG_SCHEMA.md bidirectional links missing |
| master_schema.py alignment | ✓ | Fully aligned; correctly documented |
| Tier definitions & gates | ✓ | Clear per-element decision tree; ADR-0015 linked |

---

## Recommendations (Priority Order)

1. **Clarify `parallax_over_error`:** Document or remove from release pipeline.
2. **Migrate argparse help:** Move run_pipeline1_inference.py docstring content into `.description` and `.epilog` for visibility.
3. **Add sidecar note to CATALOG_SCHEMA.md:** Explain how the JSON sidecar exposes the distinction between active and diagnostic-only columns.
4. **Add cross-module citations:** Link DESIGN.md ↔ CATALOG_SCHEMA.md ↔ release_pipeline.py for hybrid column provenance.

---

## Test Coverage

The following tests enforce documentation-code alignment:

- `tests/xp_abundances/main/test_knn_rescue.py::test_artifact_columns_align_with_master_schema` — KNN artifact columns vs. master schema
- `tests/release/test_release_pipeline.py::test_hybrid_thresholds_match_release` — σ-inflation threshold consistency
- Implicit: CI harness validates parquet provenance sidecars against schema version

**Recommendation:** Add a smoke test to `tests/test_master_schema.py` that loads a release parquet and verifies all advertised columns in CATALOG_SCHEMA.md are present.
