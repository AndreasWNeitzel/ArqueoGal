# Python 3.12+ Idiom Audit: xp_abundances/main/

## Overview

Audit of six core modules (release.py, training.py, inference.py, model.py, knn_rescue.py, uncertainty.py) for Python 3.12+ modernization opportunities. Code is generally well-written; idiom gaps are minor and optional.

## Findings

### 1. PEP 695 Type Aliases

**Status**: Partly adopted. `Final` markers are consistent but no `TypeAlias` annotations on dict/tuple type definitions.

- **release.py:90**: `_PER_ELEMENT_CAVEAT_FLAGS: Final[dict[str, tuple[str, ...]]]` — could be `type Caveat = dict[str, tuple[str, ...]]` (PEP 695) for clarity at module level.
- **training.py:76**: `_AMP_DTYPES: dict[str, torch.dtype | None]` — missing `Final` marker; should be `_AMP_DTYPES: Final[dict[str, torch.dtype | None]]`.
- **inference.py**: No type aliases defined; uses inline tuple/dict types (acceptable for small surface).

### 2. Dataclass Slots

**Status**: Strong. `CovarianceBlockLayout` (model.py:54) correctly uses `@dataclass(frozen=True, slots=True)`. Others follow suit.

- **knn_rescue.py:77**: `KnnRescueArtifact` uses `@dataclass(frozen=True)` without `slots=True` — safe but could gain the memory/hash benefit with `slots=True`.
- **uncertainty.py:58**: `CalibrationArtifacts` uses `@dataclass` without `frozen` or `slots` — fields are mutable; the defensive choice is fine if intentional, but `frozen=True` would prevent accidental modifications.

### 3. F-Strings vs Format

**Status**: Excellent. Consistent f-string usage throughout. Zero instances of `.format()` or `%` formatting.

- No issues found.

### 4. Pattern Matching (match/case)

**Status**: No opportunities found. Code is procedural and linear; pattern matching would not simplify existing logic.

### 5. StrEnum / IntEnum Opportunities

**Status**: Minor. Tier values (1, 2, 3) and element names could be enums but literal values are acceptable here.

- **release.py:354**: Per-element tier return as `pd.Series(..., dtype="int8")` — could be `IntEnum` for type safety, but the int8 dtype requirement makes this less applicable.
- **knn_rescue.py:63**: `LABEL_NAMES` is a `tuple[str, ...]` constant; could be `StrEnum` if iteration and string coercion are needed, but current form is lightweight and correct.

### 6. TypeGuard and Defensive Checks

**Status**: Defensive checks present; no TypeGuard usage but not necessary here.

- **inference.py:113–114**: Type casts in `_build_model_from_blob` (e.g., `int(k)`) are explicit and safe; TypeGuard would add overhead without clarity gain.

### 7. Unused str.format / % Formatting

**Status**: Resolved. No legacy string formatting found.

### 8. Missed Final[] Module Constants

**Status**: Good coverage. Module-level constants are marked `Final` except one:

- **training.py:73**: `CHECKPOINT_VERSION: int = 2` — should be `CHECKPOINT_VERSION: Final[int] = 2`.

### 9. cast() Overuse

**Status**: Minimal and justified.

- **training.py:38**: `from typing import Any, cast` — used once at line 1039 in `load_checkpoint`: `blob = cast(dict[str, Any], torch.load(...))`. This is defensive given `weights_only=False`; appropriate.

### 10. Structural Patterns / Complex Typing

**Status**: Clean. No unnecessary `Protocol` or complex type hierarchies.

## Recommendations (Priority Order)

1. **training.py:73**: Add `Final[int]` to `CHECKPOINT_VERSION`.
2. **training.py:76–80**: Add `Final` marker to `_AMP_DTYPES`.
3. **knn_rescue.py:77**: Add `slots=True` to `KnnRescueArtifact` for consistency.
4. **uncertainty.py:58**: Consider `frozen=True` on `CalibrationArtifacts` if the intent is immutability; document if mutability is intentional.

## Conclusion

Code already follows modern Python idioms well. No breaking changes needed. Minor markers (`Final`, `slots`) would improve type clarity and memory usage but are optional. The codebase is production-ready as-is.
