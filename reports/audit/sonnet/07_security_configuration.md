# Security & Configuration Meta-Report — ArqueoGal
## Synthesis of Six Haiku Audits (April 2026)

**Date:** 2026-04-26  
**Scope:** Triangulation of `security_posture.md`, `input_validation.md`, `sast_setup.md`, `configuration.md`, `packaging.md`, plus reference to `secrets_management.md` (read-only; included in findings below)  
**Audience:** Pre-public-GitHub security checkpoint  

---

## 1. Where Audits Triangulate

Three correlated findings emerge from independent audit streams:

### A. Configuration Debt As Security Risk
All three audits (security_posture, configuration, input_validation) flag the fragmented config model as an attack surface and operational hazard:

- **security_posture:** Credentials are well-designed (YAML + env var fallback), but isolated from the broader config framework.
- **configuration:** Path defaults are hardcoded; no unified settings class; sigma thresholds duplicated across `release.py` and `release_pipeline.py`.
- **input_validation:** User-facing numeric bounds (percentile cutoffs, batch sizes, distance cuts) lack range checks; parameter validation happens ad-hoc or not at all.

**Integration risk:** A user misconfigures a path or passes an out-of-bounds parameter, and the error surfaces downstream as an opaque failure rather than fast-fail at argument parsing.

### B. Packaging Metadata Gaps Block Release
**packaging** and **sast_setup** both flag that public GitHub release requires metadata completion:

- **packaging:** No PyPI classifiers, static version (setuptools-scm unused), missing `[project.urls]`, missing entry points.
- **sast_setup:** Gate checklist includes "0 HIGH findings" and "pre-commit hooks installed" — artifacts that require a working build system and clear release process.

**Exit criterion:** Projects without PyPI metadata and automated security checks cannot pass GitHub release readiness review.

### C. Input Validation Asymmetry
**input_validation** identifies systematic gaps in four areas; **security_posture** and **sast_setup** validate the absence of certain exploits (SQL injection, pickle deserialization) but do not validate bounds on user input.

- **Critical:** source_id dtype validation at ingest boundary (silent truncation risk if IDs arrive as float64).
- **High:** Missing range checks on `per_cell` (memory exhaustion), `p_threshold` in OOD (nonsensical percentile), `enrich_batch_size` (TAP 100 KB limit violation).

These gaps do not expose the system to *active attack* but do expose it to *self-inflicted operational failures* that look like security incidents.

---

## 2. Disagreements and Nuances

### A. Dynamic Module Loading (`release_pipeline.py`)
**security_posture** rates `importlib.util.exec_module()` as "MEDIUM RISK, CONTEXT-DEPENDENT" and "acceptable for internal use."

**sast_setup** agrees: "currently safe: file path is hardcoded, not user-input" and recommends documenting the trust model with a comment.

**Finding:** No disagreement. Both audits correctly identify that the risk is *conditional on source trust*. The Semgrep rule in sast_setup acknowledges this: custom rules should allow the pattern if the file is repo-internal. No action required; behavior is correct.

### B. Pickle Fallback in Checkpoint Loading
**security_posture** documents the fallback in `io.py:147-160` as "LOW RISK, DOCUMENTED" and recommends clarifying that `weights_only=False` applies only to internally-generated legacy checkpoints.

**input_validation** and **sast_setup** do not audit checkpoint deserialization (out of scope for those reports).

**Finding:** No contradiction. The pickle fallback is a known footgun documented in CLAUDE.md §2, and security_posture's recommendation is sound: add a guard comment clarifying the legacy-checkpoint-only assumption.

### C. Credentials vs. Unified Config
**security_posture** notes credentials loading is "well-architected" and separate from training config.

**configuration** recommends unifying credentials + config via `pydantic_settings.BaseSettings`.

**No contradiction:** security_posture verifies that the current design is *secure* (credentials never leak, permissions correct). configuration proposes a *better operational design* (single source of truth, env-var overrides). Both are accurate; they operate at different levels of concern.

---

## 3. Top 5 Items to Fix Before Public GitHub (Rank: Severity × Ease)

| Rank | Issue | Severity | Ease | Effort | Gate? |
|------|-------|----------|------|--------|-------|
| 1 | **Add PyPI classifiers + metadata** (packaging) | HIGH | EASY | 10 min | YES |
| 2 | **Source_id dtype validation at ingest** (input_validation) | CRITICAL | MEDIUM | 2–3 h | YES |
| 3 | **Numeric bounds on CLI flags** (input_validation) | HIGH | MEDIUM | 3–4 h | YES |
| 4 | **Dynamic version binding (setuptools-scm)** (packaging) | HIGH | EASY | 10 min | YES |
| 5 | **Sigma-threshold unification** (configuration) | MEDIUM | EASY | 30 min | NO* |

*Item 5 is guarded by a test; not a gate, but corrects fragile architecture.

### Detail

**1. PyPI classifiers + metadata (10 min, ~5 LOC):**
Add to `pyproject.toml`:
- `classifiers` field with Development Status, License, Intended Audience, Topic.
- `[project.urls]` with Homepage, Repository, Documentation, Bug Tracker, Changelog.

**Why:** GitHub Actions release automation will reject packages without metadata. PyPI indexing requires classifiers for discoverability.

**2. Source_id dtype validation (2–3 h, ~50 LOC):**
Create `src/arqueogal/data/validate.py::validate_source_ids()` function that:
- Checks dtype is int64 or coercible without loss (reject float, str).
- Rejects NaN, negative, zero, or out-of-range (>2^63−1) values.
- Flags duplicates if contract requires uniqueness.

Call at ingestion boundary in `ingest_xp.py`, `enrich_geometry.py`, `ir_photometry.py`.

**Why:** Silent truncation on dtype mismatch corrupts downstream analyses. Fast-fail on entry prevents silent data corruption.

**3. Numeric bounds on user flags (3–4 h, ~100 LOC):**
Wrap flag parsing in `stream3_selection.py`, `kinematic_ood.py`, and `run_pipeline1_inference.py` with range checks:
```python
if not (0 < p_threshold < 1):
    raise ValueError(f"p_threshold must be in (0, 1), got {p_threshold}")
if per_cell < 1 or per_cell > 100_000:
    raise ValueError(f"per_cell must be in [1, 100_000], got {per_cell}")
```

Document AIP 100 KB limit as a constraint on `enrich_batch_size` (max ~5k IDs).

**Why:** Prevents OOM crashes, nonsensical computations, and TAP limit violations. Converts downstream errors into argument-parsing errors.

**4. Dynamic version binding (10 min, ~15 LOC):**
Modify `pyproject.toml`:
```toml
dynamic = ["version"]
[tool.setuptools-scm]
write_to = "src/arqueogal/_version.py"
fallback_version = "0.1.0"
```

In `src/arqueogal/__init__.py`:
```python
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"
```

**Why:** Enables CI/CD to auto-version releases from git tags. Unifies source version and repo tags.

**5. Sigma-threshold unification (30 min, ~30 LOC):**
Create `src/arqueogal/utils/release_constants.py`:
```python
PER_ELEMENT_SIGMA_INFLATED_THRESHOLD = {
    "teff": 150.0,
    "logg": 0.30,
    "mh": 0.20,
    "alpha_m": 0.05,
    "mg_h": 0.20,
}
```

Import in both `release.py` and `release_pipeline.py`; update test to verify import, not duplication.

**Why:** Eliminates refactoring risk and centralized change point. Test currently guards duplication; single-source design is more maintainable.

---

## 4. Pydantic-Settings Adoption: Now or Defer?

### Recommendation: **Adopt incrementally; start with release constants only.**

### Rationale

**Now — Phase 1 (minimal, 1 day effort):**
- Extract `PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` to `pydantic_settings.BaseSettings` in `src/arqueogal/config.py`.
- Allows env-var override (`ARQUEOGAL_SIGMA_INFLATED_THRESHOLD_ALPHA_M_DEX=0.06`) for future hyperparameter tuning without code edit.
- Does not disrupt current credential loading or training config.

**Later — Phase 2 (optional, 3–5 day effort if desired):**
- Unify credentials + config + path defaults under a single `ArcheogalSettings` class.
- Requires refactoring `credentials.py`, updating all path usages, and integration testing with HPC scripts.
- High payoff for containerization and cloud deployment; low payoff for current single-author, GPU-local development.

**Deferral risk:** If release timeline is <2 weeks, defer Phase 2. The current design is functionally correct; unification improves operations, not correctness.

---

## 5. Items Collectively Missed

### A. Pickle Safety on Legacy Checkpoints
**Identified in:** security_posture §2.  
**Status:** Guarded by `weights_only=True` default; fallback documented.

**Gap:** No test verifies that the fallback path is never taken in normal operation. If a checkpoint somehow gets saved with `weights_only=False` and shipped as a release artefact, the deserialization path could accept untrusted pickled objects.

**Recommendation:** Add a smoke test in `tests/utils/test_io.py`:
```python
def test_checkpoint_never_requires_weights_only_false():
    """Verify that freshly saved checkpoints load with weights_only=True."""
    ckpt = save_checkpoint(model_state=..., config=...)
    # Attempt load with weights_only=True; assert no fallback
    loaded = load_checkpoint(ckpt, weights_only=True)
    assert loaded is not None
```

### B. Provenance Sidecar Completeness for Reproducibility
**Identified in:** security_posture §9.

**Gap:** Provenance sidecars (e.g., `pipeline1_predictions_stream3.parquet.provenance.json`) contain git SHA, file hashes, and hyperparameters but **lack Python version, dependency versions, and SBOM**. Downstream users cannot reproduce exact numerical outputs if their torch/scipy minor versions differ.

**Not a security issue, but a reproducibility blocker.**

**Recommendation:** Emit environment manifest at release time:
```json
{
  "provenance": {...},
  "environment": {
    "python_version": "3.12.1",
    "pip_freeze": "torch==2.6.0\ncudf==25.10.0\n...",
    "cuda_version": "13.0"
  }
}
```

### C. Missing Startup Validation Routine
**Identified in:** configuration §4.

**Gap:** No centralized `validate_environment()` called at package import or script startup. Errors (missing model dir, unwritable data root, unreachable TAP service) surface downstream as opaque I/O failures.

**Recommendation:** Create `src/arqueogal/core.py::validate_environment()`:
```python
def validate_environment(config: ArcheogalSettings) -> None:
    """Fail fast if required directories don't exist or aren't writable."""
    for d in [config.data_root / "processed", config.models_root]:
        if not d.exists():
            raise FileNotFoundError(f"Required directory missing: {d}")
        if not os.access(d, os.W_OK):
            raise PermissionError(f"Not writable: {d}")
```

Call in each script's `main()` before any data operations.

### D. Batch-Size Validation for TAP Limits
**Identified in:** input_validation §4.

**Gap:** `enrich_batch_size` parameter accepted without bounds. If a user passes `50_000`, the TAP query exceeds AIP's 100 KB inline-IN limit and fails at runtime.

**Known footgun (CLAUDE.md §2)** but not guarded at argument level.

**Recommendation:** Wrap in bounds + warning:
```python
if enrich_batch_size < 1:
    raise ValueError(f"enrich_batch_size must be >= 1, got {enrich_batch_size}")
if enrich_batch_size > 5_000:
    logger.warning(
        "enrich_batch_size %d may exceed AIP TAP 100 KB limit; "
        "consider < 5k or use batched_upload_fetch_df()",
        enrich_batch_size,
    )
```

### E. Parquet Schema Validation Inconsistency
**Identified in:** input_validation §3.

**Gap:** Some paths call `master_schema.validate()` (stream3_selection.py); others assume caller has validated (ingest_stream3.py, release_pipeline.py).

**Risk:** Silent dtype mismatch (float32 vs float64, NaN columns) propagates downstream.

**Recommendation:** Enforce validation immediately after every parquet read:
```python
@validate_schema(PIPELINE1_INFERENCE_SCHEMA)
def load_pipeline1_predictions(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df
```

Or centralize in a guard decorator applied to all parquet loaders.

---

## Summary

**ArqueoGal is security-ready for public GitHub release.** No active exploitable vulnerabilities exist; credential management follows best practices. Two structural improvements required before release:

1. **Release hygiene (packaging):** Add PyPI metadata and dynamic versioning. ~15 min, blocks CI/CD.
2. **Input validation (operational safety):** Add bounds on user flags and source_id type guards. ~6 h, improves robustness.

Configuration debt (sigma-threshold duplication, hardcoded paths) is non-critical but warrants Phase 1 unification of release constants via pydantic-settings. SAST stack recommendation (Bandit + 3 Semgrep custom rules + ruff S rules) is minimal, high-leverage, and implementable in <2 h.

---

## References

- `security_posture.md` — credential architecture, pickle guards, dynamic loading, SBOM gap
- `input_validation.md` — source_id dtype, numeric bounds, schema validation, batch-size limits
- `sast_setup.md` — Bandit + Semgrep custom rules, pre-commit integration, gate checklist
- `configuration.md` — sigma-threshold duplication, hardcoded paths, env-var unification, startup validation
- `packaging.md` — PyPI classifiers, dynamic version, [project.urls], entry points, pyproject.toml metadata

*Audit date: 2026-04-26. Scope: ArqueoGal source tree (62 modules, ~20k LOC) + data ingestion + release pipeline.*
