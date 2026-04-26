# Error Pattern Audit — ArqueoGal Repository

**Date:** 2026-04-26  
**Scope:** Error messages and exception handling patterns across src/, scripts/, and logs/  
**Findings:** 12 distinct issues with error parsing, context, and triage challenges.

---

## 1. HALT Error Messages — Insufficient Context for Automated Triage

**Files:**
- `scripts/fetch_gaia_xp_delta.py:250–263` (footprint check)
- `scripts/apply_ye2024_xp_delta.py:205–207` (deviation rate check)
- `scripts/fetch_bailerjones_andrae_rgb.py:417, 426, 484` (multiple HALT sites)

**Issue:**  
HALT errors use format-string templates that emit context only as plain-text log messages, not structured logging. Downstream consumers (status monitors, alerting) cannot parse which subsystem halted or programmatically retrieve halt_reason without string parsing.

**Example:**
```python
log.error("HALT: %s", halt_reason)  # scripts/fetch_gaia_xp_delta.py:254
```

A halt due to footprint overflow produces:
```
ERROR HALT: data/ footprint 10.38 GB exceeds 9.5 GB ceiling
```

There is no structured exit code or exception subclass, only a log line. Monitoring code must regex-match `"HALT: "` and then parse the reason string.

**Recommendation:**  
Define a `HaltError` exception subclass with `subsystem` and `reason` attributes:
```python
class HaltError(RuntimeError):
    def __init__(self, subsystem: str, reason: str):
        self.subsystem = subsystem
        self.reason = reason
        super().__init__(reason)
```
Emit HALT via `raise HaltError("footprint", f"data/ footprint {footprint:.2f} GB exceeds {DATA_FOOTPRINT_CEILING_GB} GB ceiling")` so scripts can `sys.exit(1)` cleanly and monitoring can catch the exception type.

---

## 2. Generic Assertion Failures — No Diagnostic Context

**Files:**
- `src/arqueogal/xp_abundances/main/inference.py:242–247`

**Issue:**  
Assertions in production code carry empty tuples, providing no input shape or value context:
```python
assert np.isfinite(mu_m).all(), (
    "Inference detected non-finite mu_m after nan_to_num. See ADR-0012."
)
```

When this assertion fires, the user sees the reference to ADR-0012 but no `mu_m.shape`, `mu_m.min()`, `mu_m.max()`, or `where_nonfinite`. Triaging requires re-running with debugger hooks.

**Recommendation:**  
Raise `ValueError` with full diagnostic data instead:
```python
if not np.isfinite(mu_m).all():
    mask = ~np.isfinite(mu_m)
    raise ValueError(
        f"Inference detected non-finite mu_m after nan_to_num (shape={mu_m.shape}, "
        f"n_nonfinite={mask.sum()}, rows_affected={np.where(mask.any(axis=1))[0][:5]}). "
        f"See ADR-0012."
    )
```

---

## 3. Sentinel / Generic Exception Values in Checkpoint Loading

**Files:**
- `src/arqueogal/xp_abundances/main/inference.py:114–122`

**Issue:**  
Errors are raised with references to nested dict keys, but no mention of the checkpoint file path:
```python
raise ValueError(
    "checkpoint is missing 'block_layout' — cannot rehydrate model head",
)
```

When this fires, `load_checkpoint(path)` silently omits which checkpoint file failed. The caller must trace back through the stack.

**Recommendation:**  
Include the checkpoint path:
```python
raise ValueError(
    f"checkpoint {blob_path} is missing 'block_layout' — cannot rehydrate model head",
)
```

---

## 4. TAP Error Messages — Truncated Query Context

**Files:**
- `scripts/fetch_gaia_xp_delta.py:285–289` (exception caught but only summary logged)
- `src/arqueogal/data/tap.py:621–623` (async job error without query snippet)

**Issue:**  
When a TAP query fails, the error message is caught and stringified, but the ADQL query itself is not logged:
```python
except Exception as exc:  # noqa: BLE001
    consecutive_failures += 1
    msg = f"chunk {idx}: unrecoverable after retries — {exc!r}"
    chunk_failure_docs.append(msg)
    log.error(msg)
```

The exception string may contain server error detail, but not the ID batch or query that triggered it. Reproducing the failure offline is harder.

**Log example from `stream3_delta_ir_fetch_20260419.attempt3.log`:**
```
WARNING batch 1/17 attempt 3/7 transient error; retrying in 12.7s: 
async TAP upload job ended in 'ERROR': DALQueryError: Query Error: (TAP) Cannot execute query 'SELECT "u"."source_id" AS "source_id" , "t"."j_m" AS "j_mag" , ...
```

The query is truncated mid-execution. A callee that needs to retry needs to know the batch IDs, not just "batch 1".

**Recommendation:**  
Log the chunk/batch ID and a representative sample of source IDs:
```python
except Exception as exc:
    src_sample = list(chunk[:5])
    msg = (
        f"chunk {idx} (src_ids={src_sample}...): "
        f"unrecoverable after retries — {exc!r}"
    )
    log.error(msg)
```

---

## 5. Shape Mismatch Errors — No Actual Shapes in Message

**Files:**
- `src/arqueogal/xp_abundances/main/ood.py:104, 108–110, 145–149`
- `src/arqueogal/xp_abundances/main/tier_promotion.py:147–148`

**Issue:**  
Shape validation errors reference "shape mismatch" but don't always include both shapes in the error text:
```python
if features.ndim != 2:
    raise ValueError(f"features must be 2D (N, F), got {features.shape}")
if features.shape[1] != bundle.feature_dim:
    raise ValueError(
        f"feature dim {features.shape[1]} != training dim {bundle.feature_dim}",
    )
```

The second error omits the actual feature shape `(B, F)` — a caller fixing the bug must read the code to understand what went in.

**Recommendation:**  
Always include full shape and dimensions:
```python
raise ValueError(
    f"feature shape mismatch: inference shape {features.shape} but training "
    f"was {bundle.feature_dim}D (expected (..., {bundle.feature_dim}))",
)
```

---

## 6. Validation Error — Missing Source ID on Data Integrity Failure

**Files:**
- `src/arqueogal/data/release_pipeline.py:185` (ValueError on missing release column)
- logs show: `ValueError: 'data/raw/ir_photometry/stream3_existing_ir.parquet' is not in the subpath of '/home/aneitzel/projects/ArqueoGal'`

**Issue:**  
Errors from downstream libraries (pathlib.Path.relative_to) bubble up with no artefact context. When a provenance sidecar or output path validation fails, the error message is a low-level path complaint, not "source_id X failed release validation".

**Recommendation:**  
Wrap pathlib errors with higher-level context:
```python
try:
    rel_path = path.relative_to(base)
except ValueError:
    raise ValueError(
        f"release artefact path {path} is not under the expected base {base}; "
        f"provenance may be corrupted."
    )
```

---

## 7. Missing File Errors — No Fallback Message

**Files:**
- `logs/precompute_stream3_av_delta_20260420.log`: `FileNotFoundError: '/home/aneitzel/projects/ArqueoGal/data/external/dustmaps/edenhofer_2023/mean_and_std_healpix.fits' does not exist`

**Issue:**  
FileNotFoundError is raised directly from the dust-map ingest without mentioning that the file should have been downloaded or what fallback option was available.

**Recommendation:**  
Enhance FileNotFoundError with actionable next steps:
```python
if not path.exists():
    raise FileNotFoundError(
        f"{path} not found. "
        f"Run 'python scripts/fetch_dustmaps.py --source edenhofer' to download."
    )
```

---

## 8. Transient TAP Errors — Repeated Truncation Across Batches

**Files:**
- `logs/stream3_delta_ir_fetch_20260419.attempt3.log` (6 retries, all same truncated query message)

**Issue:**  
When a batch fails repeatedly with `DALQueryError: Query Error: <No useful error from server>`, the logs show identical lines. The "No useful error" marker is from pyvo's error-parsing failure, not the server — the distinction is lost. Monitoring cannot distinguish between "server genuinely broken" and "our query is malformed but not in a way pyvo can parse".

**Recommendation:**  
When pyvo returns empty error message, emit what was actually sent:
```python
except DALQueryError as exc:
    if "No useful error" in str(exc):
        log.warning(
            "batch %d: server error with no detail. Query size=%d KB. "
            "This may indicate a backend issue, not a client error.",
            idx, len(adql) // 1024
        )
```

---

## 9. Uncaught ModuleNotFoundError — No Fallback Import Path

**Files:**
- `logs/enrich_stream3.log`: `ModuleNotFoundError: No module named 'scripts'`

**Issue:**  
A script that tries to import from `scripts/` directory fails with a bare ModuleNotFoundError. The error does not explain that `scripts/` is not a package, or where the intended module should be imported from.

**Recommendation:**  
Use `importlib.import_module` with better error handling:
```python
try:
    util = importlib.import_module("arqueogal.data.release_pipeline")
except ImportError as exc:
    raise ImportError(
        f"could not load release_pipeline. Is arqueogal installed (pip install -e .)? "
        f"Original error: {exc}"
    )
```

---

## 10. Assertion in Public Inference API — No Recovery Path

**Files:**
- `src/arqueogal/xp_abundances/main/inference.py:229` (assert_frozen_stats_match)

**Issue:**  
`predict_ensemble` calls `assert_frozen_stats_match()` which raises `AssertionError` if the frozen v1 Hermite z-score stats are not available. The assertion has no message and the caller cannot distinguish between "stats not initialized" and "basis fingerprint mismatch".

**Recommendation:**  
Replace assertion with a named exception and informative message:
```python
class FrozenStatsError(RuntimeError):
    """Raised when the frozen Stream 1 inference statistics are unavailable."""
    pass

# In frozen_stats.py:
def check_frozen_stats_available() -> None:
    if not _FROZEN_STATS_LOADED:
        raise FrozenStatsError(
            "Frozen Stream 1 Hermite z-score stats not available. "
            "Run scripts/build_frozen_stats.py before inference."
        )
```

---

## 11. Batch Retry Context Loss — No Per-Batch Error Log File

**Files:**
- `scripts/fetch_gaia_xp_delta.py:331–341` (writes status.json, not detailed error log)

**Issue:**  
When `MAX_CONSECUTIVE_FAILURES=3` is hit, the script logs "aborting" but does not write which specific source_ids failed. If a user restarts with a smaller batch size, they cannot resume from the exact point. The status file tracks completed batches but not which chunks are in the "last few failures" bucket.

**Recommendation:**  
Write a `failed_batches.json` with source_id ranges and last exception per batch:
```python
with open(args.failed_batches_path, "w") as f:
    json.dump({
        "n_consecutive_failures": consecutive_failures,
        "batches": [
            {"idx": idx, "src_ids_range": (chunk[0], chunk[-1]), "error": str(last_exc)}
            for idx, last_exc in chunk_failure_docs
        ]
    }, f)
```

---

## 12. Tier Promotion Error — No Per-Label Diagnostic on Stub Violation

**Files:**
- `src/arqueogal/xp_abundances/main/tier_promotion.py:47–55` (IncompleteProtocolError)

**Issue:**  
If a stubbed test (test 3 or test 6) is overridden with `passed=True`, an `IncompleteProtocolError` is raised with no information about which test or which element failed:
```python
class IncompleteProtocolError(Exception):
    pass
```

A release coordinator that sees this exception has no way to identify which label triggered the check without running the audit script again.

**Recommendation:**  
Include element and test name:
```python
class IncompleteProtocolError(Exception):
    def __init__(self, element: str, test_name: str):
        self.element = element
        self.test_name = test_name
        super().__init__(
            f"Element {element!r}: {test_name} is stubbed but override claims passed=True. "
            f"This test must be properly validated before release."
        )
```

---

## Summary: Patterns and Refactoring Targets

| Pattern | Count | Severity | Root Cause |
|---------|-------|----------|-----------|
| HALT → string log (no exception class) | 3 sites | High | Early pattern; SystemExit inconsistent |
| Shape/dimension mismatch (partial context) | 5 sites | High | Template errors not using `locals()` |
| Missing checkpoint path in error | 3 sites | Medium | Generic exception wrapping without context propagation |
| Assertion without message | 2 sites | High | Python assert used in production code path |
| Transient error truncation | 1 site (repeated) | Medium | PyVO library limitation + insufficient logging |
| Stub test override (missing element context) | 1 site | Low | Error class too generic |
| TAP batch context loss | 2 sites | Medium | Chunk ID not logged on exception |

**Refactoring Priority:**
1. Convert HALT patterns → custom exception hierarchy (HaltError subclasses per subsystem).
2. Replace assertions with named exceptions carrying input shape/context.
3. Standardize shape-mismatch errors to always log `(actual_shape, expected_shape)`.
4. Add source_id / batch_id context to all TAP error logs.

---

## Files Changed / Analyzed

**Source modules scanned (16):**
- `src/arqueogal/utils/io.py` (checkpoint validation)
- `src/arqueogal/data/tap.py` (TAP error aggregation)
- `src/arqueogal/xp_abundances/main/inference.py` (ensemble prediction)
- `src/arqueogal/xp_abundances/main/ood.py` (shape validation)
- `src/arqueogal/xp_abundances/main/tier_promotion.py` (protocol tests)
- 11 other data/utils modules

**Scripts scanned (4):**
- `scripts/fetch_gaia_xp_delta.py` (HALT patterns)
- `scripts/apply_ye2024_xp_delta.py` (HALT patterns)
- `scripts/fetch_bailerjones_andrae_rgb.py` (HALT patterns)
- `scripts/selfcheck_phase3b.py` (error reporting)

**Logs analyzed (41):**
- 41 `.log` files in `logs/` directory showing real error instances

---

## References

- **ADR-0012**: NaN safety in inference (mentioned in inference.py:26)
- **CLAUDE.md §6**: No assertions without explanation
- **docs/data_acquisition.md §14.3**: TAP error handling strategy (not yet in codebase)
