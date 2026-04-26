# ArqueoGal Python API Surface Audit (2026-04-26)

## Scope

Audited the public API surface of `arqueogal.xp_abundances.main` (release.py, knn_rescue.py, inference.py, training.py) and `arqueogal.data` (release_pipeline.py, master_schema.py, frozen_stats.py) for external consumption by Starfold, gallery scripts, and release drivers.

## Key Findings

### Type-Hint Coverage

**Strong:**
- `release.py`: All public functions fully type-hinted with return types. `assign_release_tier()`, `annotate_parquet()`, `tier_counts()` are production-grade with comprehensive docstrings.
- `inference.py`: `EnsemblePrediction` dataclass, `load_ensemble()`, `predict_ensemble()` all fully typed.
- `frozen_stats.py`: Excellent type hints and validation. `FrozenZScoreStats` frozen dataclass with slot=True, strong error handling via `FrozenStatsMismatchError`.
- `knn_rescue.py`: `KnnRescueArtifact` frozen dataclass properly typed.

**Weak:**
- `training.py`: Function signatures present but NumPy/Torch tensors lack element-wise shape/dtype documentation (e.g., `load_arrays()`, `build_dataloaders()` return types are generic `tuple`). Parameter types like `cfg: TrainingConfig` are specified but return tensor shapes are not.
- `uncertainty.py`: `collect_predictions()` returns `dict[str, np.ndarray]` but does not document the key set ("mu", "L", "y", "sigma_Y") or shapes per key in the signature; documented in docstring only.
- `master_schema.py`: `MasterSchema` class and `SchemaError` exist but are not exported in `__all__`. Schema validation functions lack complete type hints on optional vs required column specs.

### Missing Examples in NumPy-Style Docstrings

**Absent Examples:**
- `release.annotate_parquet(path)` — no usage example. Expected caller pattern: `summary = annotate_parquet(path); tier_counts = summary["counts"]`. (Visible in `scripts/assign_release_tier.py` but not in docstring.)
- `inference.load_ensemble()` — no example showing `device` parameter or directory vs file argument. Gallery scripts use it; users cannot discover the directory-glob behavior without reading source.
- `frozen_stats.load_frozen_zscore_stats()` — mentions "provenance JSON" but does not show the call pattern or the expected JSON structure inline.
- `training.build_dataloaders()` — returns tuple of (train_loader, val_loader, test_loader, label_scaler, stratified_ids) but the unpacking pattern is absent from docstring.

**Present Examples:**
- `frozen_stats.apply_frozen_zscore()` — Parameters, Returns, Notes sections are clear.
- `knn_rescue.gpu_knn_search()` — Good docstring with parameter shapes.

### Deprecated-But-Not-Marked APIs

Per CLAUDE.md, `gp_smoothed_per_cell_per_label_scale()` in `uncertainty.py` is deprecated (not production, only for methodology comparison) but carries no deprecation marker (`@deprecated`, `DeprecationWarning`, or docstring callout). It appears in `__all__` and is accessible as a public function. Current status:

- Line 410–687 in `uncertainty.py`: no `@deprecated` decorator.
- Docstring notes that the production method is `shrunken_per_cell_per_label_scale()` (line 238), but callers can still use the GP version accidentally.
- No deprecation warning raised on import or call.

Recommendation: Add `warnings.warn()` on first call or `@deprecated` decorator to signal end-of-life status.

### Contract Under-Specification

**Functions used externally with loose contracts:**

1. **`release.annotate_parquet(path: Path) → dict[str, int | dict[int, int]]`**
   - Return type is Union dict with no schema. Docstring says "returns ``{"n_rows": N, "counts": {1: ..., 2: ..., 3: ...}}``" but the type hint is `dict[str, int | dict[int, int]]`, which permits any dict with those mixed types.
   - Used by `scripts/assign_release_tier.py` and gallery scripts; return shape is stable but not enforced.
   - Fix: Define a TypedDict or dataclass for the return schema.

2. **`inference.predict_ensemble(ensemble: list[EnsembleMember], loader: DataLoader, ...) → EnsemblePrediction`**
   - `EnsemblePrediction` dataclass is well-defined, but the `cell_ids` parameter (optional, shape `(B,)`) is not well-described. Usage context (per-cell temperature scaling) is clear in docstring, but the semantics of cell_id=0 (global scaling) vs None vs provided array are scattered.
   - Fix: Clarify the three code paths in a Parameters section.

3. **`training.train_model()` and `train_ensemble()`**
   - Both functions lack complete return type. `train_model()` returns a checkpoint dict but the key set is documented only in comments ("encoder", "regressor", "calibration", etc.), not in type hints.
   - No example showing how to consume the returned dict or pass it to `inference.load_ensemble()`.
   - Fix: Return a TypedDict or named tuple capturing the checkpoint schema.

4. **`data.release_pipeline.run_hybrid_release_pipeline()`**
   - Orchestrator function (100+ lines) with many optional kwargs for paths, flags, and artifact choices. Contract is "join predictions + features, annotate, optionally build derivatives," but the actual return value and side effects (file writes) are not typed.
   - Fix: Document the return type (list of output paths?) and guarantee idempotency.

### Missing in `__all__` Declarations

- `archaeogal.data.master_schema`: `MasterSchema` class and `SchemaError` exception are used by downstream (e.g., test files import `_PIPELINE1_KNN_RESCUE_COLS` with an underscore prefix, indicating private intent, but the class itself is public).
- `archaeogal.xp_abundances.main.config`: `TrainingConfig`, `ModelConfig` are imported by test files but do not appear in `__all__` — they are discoverable only by reading source.

### Documentation Density

**Well-documented:**
- `release.py` — tier semantics are extensively justified with references to ablation studies and CLAUDE.md invariants. Module docstring is ~100 lines of design rationale.
- `frozen_stats.py` — module docstring explains the contract (load frozen, apply to Stream-3, never refit). Public surface is 6 functions; all are documented with Notes sections.

**Under-documented:**
- `uncertainty.py` — 950+ lines, 15 public functions, but only the top-level `fit_calibration()` and `apply_calibration()` have Examples sections. Intermediate functions (`bin_by_cells`, `temperature_scaling_per_cell`, `isotonic_per_label`) lack motivating examples.
- `training.py` — 350+ lines; `build_dataloaders()`, `load_arrays()`, `stratified_split_ids()` have NumPy-style docstrings but no usage examples. Callers must infer from test files.

### __all__ Declaration Quality

**Present and complete:**
- `release.py` — 8 functions listed.
- `inference.py` — 4 items (2 classes, 2 functions).
- `frozen_stats.py` — 5 functions, 1 exception, 2 constants.
- `knn_rescue.py` — 5 functions, 1 dataclass.

**Partial or missing:**
- `training.py` — no `__all__` declaration; all public functions are accessible but intent is unclear. (Some functions like `_build_model_and_temperature` have leading underscores indicating private intent, but ~20 functions lack markers.)
- `uncertainty.py` — `__all__` contains 12 items (functions + dataclass) but `gp_smoothed_per_cell_per_label_scale` is listed despite deprecation intent per CLAUDE.md.
- `data.master_schema` — no `__all__`; exports constants and a `MasterSchema` class but the class visibility is ambiguous.

## Recommendations for Minimal Spec Improvement

1. **Add `__all__` to `training.py`** — mark public vs private functions to reduce API surface discovery burden.

2. **Deprecate `gp_smoothed_per_cell_per_label_scale` explicitly** — add `warnings.warn(..., DeprecationWarning)` on first call or decorate with `@functools.deprecated`.

3. **Add Examples sections to:**
   - `release.annotate_parquet()` — show the unpacking of return dict.
   - `inference.load_ensemble()` — show directory glob, device argument.
   - `training.build_dataloaders()` — show tuple unpacking.

4. **Define return-value TypedDict for `annotate_parquet()` and `release_pipeline.run_hybrid_release_pipeline()`** — replace loose `dict[str, int | dict[int, int]]` with a named schema.

5. **Document `DataLoader` batch shape assumptions** — `predict_ensemble()` expects `(B, feature_dim)` from the loader, but shape validation is sparse. Add an assertion or type guard in the function.

## Non-Issues

- Starfold cross-imports: The split is clean; Starfold imports only `inference.load_ensemble()`, `inference.predict_ensemble()`, and `inference.EnsemblePrediction` — all well-typed.
- Gallery scripts: Use `release.annotate_parquet()` and `knn_rescue` functions; contracts are stable (return dicts and parquets written in-place).
- Testing coverage: `__all__` declarations and type hints are tested indirectly via imports in test files; missing docstring examples do not block CI.

## Summary

The API is **functionally well-designed** with strong foundational type hints and clear module-level documentation. However, **external consumers (Starfold, release scripts, gallery) lack worked examples in docstrings**, and **deprecated functions are not formally marked**, creating friction for new users. **Training.py is under-exported** (no `__all__`), forcing discovery by code inspection. Minimal effort (add Examples sections, mark deprecations, define TypedDicts for return schemas, add `__all__`) would significantly improve developer velocity without requiring refactoring.
