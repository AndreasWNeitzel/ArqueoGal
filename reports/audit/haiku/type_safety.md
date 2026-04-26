# Type-Hint Coverage Audit: ArqueoGal

**Scope**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/` (62 Python files)

## Overview

The codebase has mixed type-hint coverage: 189 of 371 public functions (51%) carry return-type annotations, all imports use `from __future__ import annotations` for forward compatibility, and dataclass usage is strong. However, three high-priority gaps exist: untyped pandas DataFrames masking column contracts, bare `Any` leaks in data-layer modules, and missing return-type hints on utility functions tied to external libraries.

## High-Priority Gaps

### 1. Untyped DataFrames Leak Column Contracts

**Pattern**: 90+ uses of `pd.DataFrame` with no TypedDict, Schema, or column-name Protocol.

**Examples**:
- `ingest_stream1.py:223` — `_write_parquet_atomic(df: pd.DataFrame, path: Path) -> None` accepts any DataFrame; callers cannot know which columns are required.
- `gaia_xp.py:182` — `normalise_xp(df: pd.DataFrame) -> pd.DataFrame` silently expects `["bp_coefficients", "bp_coefficient_errors", ...]` (the `_COEFF_COLS` tuple).
- `gaia_xp.py:240` — `zscore_c0(df: pd.DataFrame, ...) -> tuple[pd.DataFrame, XpC0Stats]` has no explicit contract on input/output column names or dtypes.
- `kinematics.py` — multiple functions accept and return DataFrames with fixed column structure (from `OUTPUT_COLS` and `REQUIRED_INPUT_COLS` tuples) but don't enforce it in the type system.

**Impact**: Data validation happens at runtime (key errors, dtype mismatches on downstream operations) instead of type-check time. Each DataFrame-returning function that feeds another requires manual column-ordering discipline across 15+ ingestion modules.

**Recommendation**: Replace bare `pd.DataFrame` with Protocols or column-name Literals at module boundaries. For the data layer, a lightweight pattern:

```python
class StreamSchema(Protocol):
    """Protocol for stream1_apogee_gaia output."""
    source_id: array
    teff_apogee: array
    ...  # match DESIGN.md column list
```

Enforce at I/O boundaries via `@dataclass` or `pyarrow` schema references. Low effort, high ROI for provenance tracking (CLAUDE.md invariant #14).

### 2. ANN202 (Missing Return Type) on Library-Wrapped Functions

**Examples**:
- `gaia_xp.py:130` — `_load_ye2024_model(model_dir: Path, device: str):  # noqa: ANN202` — returns torch.nn.Module, hidden behind noqa.
- `kinematics.py:XXX` — `_build_skycoord(df: pd.DataFrame):  # noqa: ANN202` — returns astropy SkyCoord.
- `dust_maps.py:XXX` — `_build_dust_coords(...)  # noqa: ANN202` — returns SkyCoord.
- `kinematics.py:XXX` — `_resolve_potential(name: str):  # noqa: ANN202` — returns galpy potential object.

**Impact**: Type checkers see these as returning `Any`. Inference drivers and audit paths that touch these functions cannot statically verify downstream operations (e.g., calling `potential.Rz_to_Lz()` on a potential-shaped return).

**Recommendation**: Import the relevant types explicitly and annotate. Cost is near-zero given the library imports are already present:

```python
import torch.nn as nn
def _load_ye2024_model(...) -> nn.Module:
```

### 3. Generic `dict[str, Any]` for Checkpoint State

**File**: `utils/io.py` (core I/O contract)

**Examples**:
- `load_checkpoint(...) -> dict[str, Any]` — callers must know to extract `state_dict`, `config`, etc. by string key.
- `save_checkpoint(path: str | Path, **state: Any)` — arbitrary kwargs with no schema.

**Impact**: Training.py and inference.py pass checkpoint dicts around without type hints; refactoring the checkpoint schema requires manual search-and-replace rather than type-checker assistance.

**Recommendation**: Define a `CheckpointDict` TypedDict once and reuse across `load_checkpoint`, `save_checkpoint`, and training entry points. Three-file effort (utils/io.py, training.py, inference.py).

### 4. Missing Return Types on Internal Helpers

**Scope**: ~180 functions lacking return-type hints (49% of public+internal functions).

**Examples**:
- `data/gaia_corrections.py` — `_load_zpt_module()` (likely returns a module/class, not annotated).
- `data/dust_maps.py` — `load_lallement2022_cube(cube_path)` (returns ndarray, not annotated).

**Impact**: These are internal helpers, so the risk is lower. But they propagate up: a caller function that wraps a non-annotated helper cannot infer its own return type, forcing a manual annotation throughout the call chain.

**Recommendation**: Start with public module entry points (50 functions across `ingest_stream*.py`, `enrich_*.py`, `fetch_*.py`) and backfill internal helpers. Incremental adoption: one module (e.g., `data/gaia_xp.py`) to full coverage.

## Summary of Type Annotation Coverage

| Category | Count | Gap |
|----------|-------|-----|
| Functions with return type | 189 | 51% (182 missing) |
| `Any` usages | 90 | mostly in data layer and checkpoint I/O |
| `np.ndarray[shape, dtype]` | 2 | 180+ bare `np.ndarray` (no shape/dtype) |
| `pd.DataFrame` with schema | 0 | 90+ bare `pd.DataFrame` (no columns/dtypes) |
| `type: ignore` comments | 5 | all in `model.py` JSON deserialisation (defensible) |

## Recommended Action Plan

1. **Immediate (1-2 hours)**: Annotate the 5 ANN202-noqa functions (library-wrapped returns). Resolves one-off type leaks in gaia_xp.py, kinematics.py, dust_maps.py.
2. **Short-term (one sprint)**: Define `CheckpointDict` TypedDict and propagate through training/inference boundary. Enables type-safe checkpoint schema evolution.
3. **Medium-term (ongoing)**: Introduce column-name Protocols at DataFrame-returning module boundaries (data layer). Start with stream ingestion (3 modules). Payoff: stronger provenance.
4. **Testing**: No mypy/pyright configured. Once annotations reach 80%+, enable `pyright --outputjson | jq '.generalDiagnostics'` in CI to catch regressions.

## Notes

- **Dataclass discipline is strong**: `CovarianceBlockLayout`, `FeatureLayout`, `LabelTiers`, `TrainingConfig` set a good precedent for rich types over dicts.
- **numpy.typing.NDArray underused**: Only 2 uses of structured ndarray types vs. 100+ bare `np.ndarray`. Low-lift pattern to adopt once adopted in one module.
- **Type guards in training.py are solid**: `isinstance`, `torch.isfinite`, etc. Type narrowing examples are present; the missing piece is the outer function signatures.
