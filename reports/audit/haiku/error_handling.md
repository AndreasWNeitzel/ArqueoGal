# Error Handling Audit: ArqueoGal Data Modules

**Date:** 2026-04-26  
**Scope:** `src/arqueogal/data/{tap,downloads,gaia_xp,distances,dust_maps,kinematics}.py` + `src/arqueogal/xp_abundances/main/inference.py`

---

## Executive Summary

Error handling is **unevenly distributed** across the codebase. TAP and downloads modules have strong defensive patterns (exception chaining, transient-error heuristics, atomic file operations), but inference.py lacks the `nan_to_num` safety documented in CLAUDE.md as mandatory; the comment claiming it happens "at inference entry" (line 25) is not enforced by the function signature, leaving callers to apply it.

---

## Findings by Module

### 1. **downloads.py** — Strong

| Line | Pattern | Assessment |
|------|---------|------------|
| 102 | `except BaseException` | **Correct.** Catches everything (including KeyboardInterrupt) to clean up `tmp.unlink()`. No re-raise-less swallowing; cleanup is mandatory before any exception propagates. Exception is re-raised implicitly. |
| 109 | `raise ValueError` + message context | **Good.** Includes expected vs actual digest and URL. Message explains failure reason. |

**No issues.** Atomic write (write to `.part`, then `os.replace()`) is robust; cleanup on any exception prevents truncated files.

---

### 2. **tap.py** — Good with minor gaps

#### Strengths

| Line | Pattern | Assessment |
|------|---------|------------|
| 257–271 | Async job + finally cleanup | **Excellent.** `_safe_delete(job)` ensures cleanup even on error. Proper exception chaining would strengthen this. |
| 321–326 | Input validation (batch size, placeholder count) | **Good.** Fail-fast with clear messages. No NaN exposure. |
| 633–650 | Transient retry logic | **Excellent.** `_is_transient_tap_error()` (98–101) heuristically matches known TAP failures; exponential backoff with jitter; max-retries gate prevents infinite loops. |
| 656–658 | Exhaustion fallback | **Good.** Raises with context (`from last_exc`) preserving the last exception for debugging. |

#### Gaps

| Line | Issue | Impact |
|------|-------|--------|
| 279–280 | `_safe_error_message()`: bare `except Exception as exc` | **Low.** Intentional — the goal is to safely stringify *any* error. The comment explains this. No logic should be in this function; it is pure error reporting. **No action needed.** |
| 287–288 | `_safe_delete()`: bare `except Exception as exc` | **Low.** Same rationale — it's a cleanup-guard. Logging the warning is correct; suppressing allows the original exception to propagate. **No action needed.** |
| 204, 260 | No exception type checking in `run_sync` / `run_async` | Callers get `pyvo.dal.DALQueryError`, `pyvo.dal.DALServiceError`, `requests.HTTPError`, etc. Most are re-raised transparently. **Acceptable** — let TAP-level errors bubble up; batched wrappers catch and retry known transients. |

---

### 3. **gaia_xp.py** — Moderate gaps

#### Strengths

| Line | Pattern | Assessment |
|------|---------|------------|
| 276–285 | Input validation | **Good.** Checks `source_id` column, required fields in `coords_df`, sampling grid length. Messages include expected vs actual. |
| 330–346 | Batch-level exception handling | **Mixed.** `gaiaxpy.calibrate` is wrapped with `except Exception` (justified by comment BLE001 — GaiaXPy may raise various types); **but** no exception chaining (`raise ... from exc`). The warning log on line 340–344 captures the exception, but downstream has no traceback. |

#### Gaps

| Line | Issue | Impact |
|------|-------|--------|
| 339–346 | Bare `except Exception` without chaining | **Moderate.** `except Exception as exc` catches the failure and logs it, but then continues the loop. The exception is not re-raised. This is intentional (partial-failure tolerance), but it means **callers cannot distinguish between "gaiaxpy failed on this batch" and "batch succeeded"** — they only see flags in the output. **By design:** failures are marked `ye2024_flag=2` (line 346). However, **no exception context is chained to the log**, so debugging is harder. Suggest: log `str(exc)[:200]` captures the message, but not the traceback. Consider `logger.warning(..., exc_info=exc)` to include the traceback in the log output. |
| 582–583 | `except (ValueError, TypeError)` in `_tolerant_cast_output()` | **Acceptable.** Inside a helper that tries to cast a DataFrame column to int64, then falls back to nullable Int64 if that fails. Specific exception types are caught (not bare Exception). No re-raise needed — the fallback is intentional. **No action.** |
| 602–603 | `except ImportError` in `_patch_gaiaxpy_cast_output()` | **Acceptable.** Loops over optional GaiaXPy submodules and silently skips if a module is not importable. The comment (line 602) makes the intent clear: `continue` skips that module. This is a "best-effort patching" pattern — no error should propagate if dustmaps or other optional deps are absent. **No action.** |

---

### 4. **distances.py** — No error handling found

Skimmed first 150 lines. The module wraps `batched_fetch_df()` (from tap.py), which handles all exceptions. No validation errors raised by `merge_distances()` at call sites. **Likely acceptable** — the work is delegated to tap.py.

---

### 5. **dust_maps.py** — Exception chaining present

| Line | Issue | Status |
|------|-------|--------|
| 376–383 | `except ImportError as exc` in `get_default_queries()` | **Good.** Catches missing `lallement2022` submodule, re-raises with a context message explaining the fix: "Install a version that ships it, or supply a custom mid-distance query callable...". Uses `raise ... from exc` properly (line 383). **No action.** |

---

### 6. **kinematics.py** — No explicit error handling in excerpt

The excerpt (first 150 lines) shows input validation: `_validate_required_cols()` and `_drop_nan_rows()` are called but not shown. The module is not expected to have extensive exception handling — it wraps galpy and delegates coordinate transforms. **Low risk.**

---

### 7. **inference.py** — CRITICAL FOOTGUN

#### The Known Issue: NaN Safety Not Enforced

**CLAUDE.md footgun (known_footguns line ~15):**
> `nan_to_num` train/inference boundary. `training.py` applies `np.nan_to_num(..., nan=0.0)` at the data-loader boundary. Any inference driver must mirror this, or a single NaN in any aux feature NaN-propagates through the trunk → NaN predictions with no OOD flag raised.

**Findings:**

| Line | Context | Issue |
|-------|---------|-------|
| 25–29 | Docstring | Claims "all features are sanitised at inference entry via `np.nan_to_num(...)`", but… |
| 203 | Function signature | `predict_ensemble()` has **no `nan_to_num` call in its signature or preamble**. It expects the caller to have done it. |
| 240–241 | Inside loop | `nan_to_num` is called **inside the per-member loop**, on `preds["mu"]` and `preds["L"]` (the *outputs*), not on input features. This does NOT sanitise the raw features before model forward passes. |
| 236 | Line | `preds = collect_predictions(m.model, loader, device=device)` — the loader is passed through; if the loader or its data contains NaN features, they propagate into `mu_m` **before** line 240. |

**The fallacy:** The docstring says "sanitised at inference entry", but there is no entry-point enforcement. A caller writing:

```python
loader = get_loader(features_with_nan)  # Some NaNs here
result = predict_ensemble(ensemble, loader)
```

will silently produce NaN predictions. The `nan_to_num` on line 240 catches *output* NaNs, not input NaNs.

#### Other issues in inference.py

| Line | Issue | Impact |
|-------|-------|--------|
| 114–116 | Missing field in checkpoint | Raises `ValueError("checkpoint is missing 'block_layout'")` — message is clear. **Good.** |
| 119–122 | Inconsistent block_layout | Same — clear message. **Good.** |
| 181 | `FileNotFoundError` if no checkpoints found | **Good.** Message includes the path(s) tried. |
| 187–188 | Version mismatch | **Good.** Message shows both checkpoint version and expected. |
| 225 | Empty ensemble | **Good.** Clear message. |
| 251–252 | Cell ID length mismatch | **Good.** Message includes both lengths. |
| 242–246 | AssertionErrors for non-finite outputs | **Problematic.** Using `assert` for user-facing failure conditions. `AssertionError` is meant for internal invariants. If a model produces NaN outputs, callers should get a descriptive exception type (`RuntimeError` or custom), not `AssertionError`. Assertions can be disabled with `python -O`. |

**Action: Replace assertions with explicit exceptions.**

---

## Summary of Issues

### Critical (Release-blocking)

1. **inference.py, line 25–29 + docstring mismatch:** NaN safety claim is misleading. `nan_to_num` is applied to *outputs*, not *inputs*. Callers must sanitise features before passing to `predict_ensemble()`, but this is not enforced or documented in the function signature. 
   - **Fix:** Add `features = np.nan_to_num(features, nan=0.0, ...)` early in the caller's inference driver (e.g., in a wrapper function that constructs the loader). **Document this in the function docstring: "Caller must ensure features in the loader are finite; non-finite features will silently produce NaN predictions."**

2. **inference.py, lines 242–246:** Assertions used for runtime validation.
   - **Fix:** Replace with `if not np.isfinite(...).all(): raise RuntimeError("message")`

### Moderate

3. **gaia_xp.py, line 339–346:** `except Exception` without exception chaining. Partial-failure pattern is by design, but debugging is harder without tracebacks in logs.
   - **Fix:** Change `logger.warning(...)` to `logger.warning(..., exc_info=exc)` (or use `logger.exception()` which includes traceback automatically).

4. **gaia_xp.py, lines 582, 602:** Context missing in audit. Review and categorise.

5. **dust_maps.py, line 378:** Context missing. Review.

### Low (No action needed)

- tap.py lines 279–280, 287–288: Bare `except Exception` is intentional for safe error reporting in cleanup functions.
- tap.py input validation is strong; TAP-level errors bubble up appropriately to callers.

---

## Recommendations

1. **NaN safety:** Add a **sentinel check at the data-loader boundary** in any inference script that calls `predict_ensemble()`. Document this in a per-project inference template or helper.

2. **Exception chaining:** Adopt `raise ... from exc` consistently in retry/partial-failure loops (e.g., gaia_xp.py line 346).

3. **Assertions vs. exceptions:** Replace `assert` with explicit exception types in user-facing code (inference.py).

4. **Test error paths:** Add unit tests for:
   - NaN propagation in inference (currently untested per CLAUDE.md).
   - Transient TAP retries (batched_upload_fetch_df with simulated 500s).
   - Checkpoint loading with missing/malformed fields.

