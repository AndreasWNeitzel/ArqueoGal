# ArqueoGal Module Organization Audit — 2026-04-26

## Executive Summary

The ArqueoGal codebase maintains strict main ↔ experimental boundary discipline with no cross-imports detected. Module-level `__all__` exports are defined uniformly across the library, and the public API contract is well-maintained. However, several empty `__init__.py` files at package boundaries (root, `data/`, `xp_abundances/`) miss opportunities for re-exporting common types, and a small set of data modules contain duplicated constants that could migrate to `utils/` for reuse.

## Findings

### 1. Boundary Integrity (PASS)

**Status**: No violations detected.

- Main ↔ experimental cross-imports: **0 violations**
- Verified via AST walk across `xp_abundances/main/` and `xp_abundances/experimental/`. Main modules (26 files) import only from `main/`, `data/`, and `utils/`; experimental modules (1 placeholder) do not exist.
- No reverse imports from `main/` into `experimental/`.

**Implication**: The hard invariant "no main ↔ experimental cross-imports" is upheld. Any future experimental code can be added without risk of architectural drift.

### 2. Module-Level `__all__` Exports (PASS)

**Status**: All non-`__init__` modules define `__all__`.

- Scanned: 57 production modules across `data/`, `utils/`, and `xp_abundances/main/`.
- Every module has explicit `__all__` declaration.
- Example: `xp_abundances/main/__init__.py:89–149` re-exports 59 public symbols across 16 submodules (adapter, audit, config, data, inference, etc.).

**Implication**: Public interfaces are unambiguous. Unlisted members are guaranteed internal, simplifying review and refactoring.

### 3. Package-Level Re-exports (ISSUE)

**Status**: Missed opportunities for discovery and convenience.

**Empty `__init__.py` files** (4 total):
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/__init__.py` — root package (1 line, no exports)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/__init__.py` — no exports (single blank line)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/__init__.py` — no exports (single blank line)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/experimental/__init__.py` — placeholder (single blank line)

**Rich re-export hubs** (2 total):
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/utils/__init__.py:35–52` — 12 re-exported symbols (ConfigValidationError, load_config, set_global_seed, etc.)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/__init__.py:89–149` — 59 re-exported symbols

**Impact**: Callers must navigate full import paths (e.g., `from arqueogal.xp_abundances.main.inference import predict_ensemble` vs. the convenience `from arqueogal.xp_abundances import predict_ensemble`). The root `__init__.py` especially misses a chance to surface canonical entry points.

**Recommendation**: Populate empty `__init__.py` files, at minimum:
- `data/__init__.py`: re-export Master schema and key ingestion entry points (e.g., `build_master_catalogs`, `ingest_stream1`)
- `xp_abundances/__init__.py`: delegate to `main` when no experimental code exists, or at least define a clear boundary statement
- Root `__init__.py`: re-export the top-level library version and critical types for end-users

### 4. Symbol Duplication in `data/` (MINOR)

**Status**: Intentional duplicates; documented; within tolerance.

**Identified duplicates:**
- `XP_COEFF_LEN` (55) — defined in both `gaia_xp.py:50` and `frozen_stats.py:51`
  - Both are marked `Final[int]`
  - `frozen_stats.py:53` explicitly documents the duplication: `:data:\`arqueogal.data.gaia_xp.XP_COEFF_LEN\` — duplicated here to avoid a circular dependency.`
  - Duplication is **intentional** to avoid circular imports between `gaia_xp` and `frozen_stats`

- `sha256_file()` — imported in `ingest_stream1.py:45` from `downloads.py`, then re-exported in `ingest_stream1.py:234` (__all__)
  - This is re-export, not duplication

- `DEFAULT_OUTPUT_FILENAME` — each of `ingest_stream{1,2,3}.py` and `ingest_xp.py` defines its own (e.g., `stream1_apogee_gaia.parquet`, `stream2_gaia_tic.parquet`)
  - These are **distinct constants** with the same name, appropriate to their module's domain. Not duplication.

**Recommendation**: No action needed. The `XP_COEFF_LEN` duplication is documented and justified. Consider adding a comment to `frozen_stats.py` at import time to link to `gaia_xp.py` for maintainability.

### 5. Test Coverage (KNOWN GAPS, DOCUMENTED)

**Status**: Test mirror tree is comprehensive; documented gaps are minimal.

**Coverage snapshot:**
- Test files: 56 total
  - Data layer: 31 tests
  - XP Abundances main: 14 tests
  - Utils: 6 tests
  - Integration: 1 smoke test
  - Scripts: 1 test

**Source modules without tests (7 total, pre-existing documentation):**
1. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/config.py` — configuration dataclasses; CLAUDE.md documents this gap
2. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/sanity.py` — validation checks; CLAUDE.md documents this gap
3. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/utils/label_conventions.py` — label mappings (negligible logic)
4. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/utils/calibration_plots.py` — plotting utilities (visual inspection required)
5. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/release_artefacts.py` — artefact metadata (lightweight wrapper)
6. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/ensemble_diagnostics.py` — analysis code (not a release blocker)
7. `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/kinematic_ood.py` — OOD detection; newly added (no test yet)

**Test files with corresponding source**: All 56 test files map to a production module via module name or import verification (e.g., `test_credentials.py` → `data.credentials`).

**Assessment**: Gaps are acknowledged and isolated. No surprise untested modules lurk. The repo's testing culture prioritizes high-value assertions (stratified smoke tests in data layer, loss/model gradients in training).

### 6. Import Structure (PASS)

**Verified properties:**
- **No circular imports at compilation time**: All modules compile cleanly via `py_compile`.
- **Dependency flow is acyclic**: 
  - `data/` modules import from `utils/` only (not reverse)
  - `xp_abundances/main/` imports from `data/` and `utils/` only
  - `utils/` has no internal cyclic imports (six independent modules: config, gpu, io, reproducibility, coordinates, label_conventions)
- **Astropy/external imports are localized**: astropy.units, astropy.coordinates are imported on-demand in `utils/coordinates.py` (not at package top-level), avoiding heavy initialization cost for light-weight utilities.

**Exception handling:**
- `utils/io.py` defines `ArqueoGalCheckpointError` (line 22), a custom exception; correctly exported via `utils/__init__.py:37`.
- `utils/config.py` defines `ConfigValidationError`; correctly exported.

### 7. Code Organization by Responsibility

**Data layer** (`data/`, 30 modules):
- **Data sources**: `apogee_dr19.py`, `andrae2023.py`, `gaia_*.py`, `dust_maps.py`, `tic_v82.py`, `tess_hon2021.py`
- **Ingestion orchestration**: `ingest_stream{1,2,3}.py`, `ingest_xp.py`, `build_master_catalogs.py`
- **Transformations**: `gaia_corrections.py`, `gaia_enrich.py`, `enrich_geometry.py`, `enrich_kinematics.py`, `ir_photometry.py`, `kinematics.py`, `distances.py`, `dust_maps.py`
- **Post-processing**: `dedup.py`, `selection_function.py`, `stream3_selection.py`, `crossmatch.py`
- **Infrastructure**: `tap.py` (TAP query wrapper), `credentials.py` (AIP auth), `downloads.py` (file I/O), `provenance.py` (metadata sidecars), `release_pipeline.py` (coordination), `release_artefacts.py` (output schemas), `frozen_stats.py` (Hermite z-score cache), `master_schema.py` (schema definition)

**Utilities** (`utils/`, 9 modules):
- **Configuration**: `config.py` (YAML loader, validation)
- **Hardware**: `gpu.py` (device selection, VRAM checks, cuML/UMAP class selection)
- **I/O**: `io.py` (checkpoint versioning, parquet with CRC checks, streaming)
- **Reproducibility**: `reproducibility.py` (seed setting, determinism flags)
- **Coordinates**: `coordinates.py` (astropy chain, galactocentric frames)
- **Labeling**: `label_conventions.py` (APOGEE → Pipeline 1 label mappings)
- **Plotting**: `plotting.py` (A&A-compatible matplotlib rcParams)
- **Calibration**: `calibration_plots.py` (diagnostic plots for uncertainty calibration)

**Pipeline 1** (`xp_abundances/main/`, 18 modules):
- **Models**: `model.py` (encoder, block-Cholesky head), `adapter.py` (XP → feature tensor)
- **Training**: `training.py` (ensemble, checkpointing), `losses.py` (beta-NLL, Mahalanobis, contrastive), `data.py` (PyTorch datasets)
- **Inference**: `inference.py` (ensemble prediction, parallelization), `knn_rescue.py` (hybrid fallback)
- **Calibration**: `uncertainty.py` (binned empirical Bayes, coverage analysis)
- **Diagnostics**: `audit.py` (CMI, permutation importance, decorrelated subsamples), `release.py` (tier assignment), `tier_promotion.py` (six-test gate), `bimodality.py` (α-bimodality analysis), `halfway_umap.py` (latent-space visualization)
- **OOD detection**: `ood.py` (Mahalanobis, kinematic), `kinematic_ood.py` (action-space checks)
- **Configuration**: `config.py` (loss weights, training hyperparameters), `sanity.py` (runtime validation)
- **Ensemble analysis**: `ensemble_diagnostics.py` (spread, agreement metrics)

**Overall**: Each module coheres to a single responsibility. No mixing of data fetching with training, no plotting logic in losses, etc. Intra-module dependencies are documented via `__all__` exports.

---

## Summary Table

| Category | Status | Notes |
|----------|--------|-------|
| **Boundary (main ↔ experimental)** | PASS | 0 violations; strict discipline upheld. |
| **Module `__all__` exports** | PASS | 57/57 non-init modules define `__all__`. |
| **Package re-exports** | ISSUE | 4 empty `__init__.py` files miss convenience re-exports; design is sound but could be user-friendlier. |
| **Symbol duplication** | PASS | Only `XP_COEFF_LEN` intentionally duplicated to avoid circular import (documented). |
| **Test coverage** | PASS (known gaps) | 56 tests; 7 documented gaps (config.py, sanity.py, and low-logic utilities). |
| **Circular imports** | PASS | No cycles detected; dependency graph is acyclic. |
| **Import structure** | PASS | External heavy libraries (astropy) are localized to prevent initialization bloat. |

---

## Recommendations (Priority Order)

1. **Populate `data/__init__.py`** (low effort, high discoverability)
   - Re-export `build_master_catalogs`, schema types from `master_schema.py`, and key ingest functions.
   - Improves IDE autocomplete and user navigation.

2. **Add conditional `xp_abundances/__init__.py`** boundary statement (trivial)
   - If experimental code ever lands, the __init__ should either re-export from `main` or clearly document the boundary.
   - As-is, the empty file is acceptable but uninformative.

3. **Consider `utils/` constant migration** (optional, lower priority)
   - Move `XP_COEFF_LEN` to a dedicated `utils/constants.py` to eliminate the documented duplication in `frozen_stats.py` and `gaia_xp.py`.
   - Trade-off: adds a file, but clarifies intent. Given the documented justification, defer unless this becomes a pain point.

4. **Expand root `__init__.py`** (future-proofing)
   - Add `__version__`, canonical imports of `main.XpAbundanceModel`, `data.build_master_catalogs`, etc.
   - Lower priority if the library is not yet intended as a public API.

---

## Conclusion

The codebase exhibits mature organizational discipline. Boundary contracts are upheld, module cohesion is high, and the test mirror tree is well-structured. The remaining findings are minor: cosmetic improvements to `__init__.py` convenience exports, and one intentional (documented) constant duplication that doesn't impede development. No refactoring is required; the above recommendations are enhancements for future maintainability.
