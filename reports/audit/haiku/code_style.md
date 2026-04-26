# Code Style Audit — ArqueoGal src/ and scripts/

## Summary

The codebase demonstrates strong docstring hygiene and consistency overall. NumPy-style docstrings are well-formed with clear parameter and return sections. Naming is consistent with PEP 8 astronomical conventions (underscores for physical quantities like `M_sun`, `L_z`). Two minor pattern issues emerge: (1) descriptive docstrings occasionally omit Return sections for obvious returns (e.g., `provenance.py:sidecar_path`), and (2) a committed `print()` statement in `knn_rescue.py:237` violates the "no committed print" convention.

## Findings

### 1. Missing Return Sections (Low Impact)

Files: `provenance.py:144-155`, `gaia_enrich.py:102-129`, `credentials.py:82-89`

Simple functions with obvious returns sometimes omit the `Returns` section. Examples:

- `sidecar_path()` (provenance.py:144): Docstring describes the path transformation but omits a `Returns` section, despite returning a `Path` object. The logic is clear, so the omission is acceptable for a private/local utility, but inconsistent with public-API style in other modules.
- `resolve_path()` (credentials.py:82): One-liner docstring with no Returns section, again clear but inconsistent with sibling `load_credentials()` which includes a full Raises section.

**Recommendation**: Add single-line Returns sections to these simpler functions for consistency: "Resolves and returns the credentials file path."

### 2. Committed `print()` Statement

File: `knn_rescue.py:237`

The function `gpu_knn_search()` contains a `print()` for progress reporting during a long GPU loop. The CLAUDE.md convention is "no committed print"; this should be `logger.info()` instead. Context shows it's a verbose progress indicator (rate and ETA), not debug output — logging is the right choice.

**Fix**: Replace line 237:
```python
print(f"  knn {end}/{n_query} ({rate:.0f} qps, ETA {eta:.0f}s)")
```
with:
```python
logger.info("knn %d/%d (%.0f qps, ETA %.0f s)", end, n_query, rate, eta)
```

### 3. Documentation Quality — Strong

- **__all__ Coverage**: 49 of 62 modules export `__all__`. Missing exports in data-layer init files and internal credential modules are intentional (internal APIs).
- **Docstring Conventions**: Google/NumPy style is consistent across all public functions. Multi-line docstrings follow parameter → return → raises ordering.
- **Comment Hygiene**: Inline comments are explanatory, not decorative. Complex algorithms (Cholesky reordering, kinematics actions) have clear narrative comments preceding them.
- **Noqa Attribution**: Every `# noqa` includes a brief reason (e.g., "PLR0913 — Gaia measurement signature"). Meets the "no noqa without explanation" invariant.

### 4. Naming Consistency

Astronomical names follow CLAUDE.md rules: `M_sun`, `L_z`, `v_phi`, `r_med_photogeo`, `c0_z` are not snake_case (ruff N803/N806 ignored). Physical docstrings match (e.g., "UVW here is the heliocentric Galactic-Cartesian velocity" in coordinates.py:56).

### 5. Docstring Imperative vs. Descriptive

Module docstrings are consistently descriptive (e.g., "Level-5 enrichment — galpy actions..."). Function docstrings use imperative short forms (e.g., "Apply Mészáros+2025 Teff-trend corrections..."), which is correct. No inconsistent mixing detected.

## Non-Issues (Verified)

- No dead code blocks or commented-out functions.
- No broad `except:` clauses; exceptions are specific (`ImportError`, `KeyError`, `PermissionError`).
- Scripts (build_stream3_expansion_union.py, etc.) do use `print()` for user-facing CLI output; this is intentional and appropriate.
- Line length: all files conform to 100 character limit; no ruff violations present.

