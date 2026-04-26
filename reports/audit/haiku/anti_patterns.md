# Python Anti-Patterns Audit — ArqueoGal Source

Sweep of `/home/aneitzel/projects/ArqueoGal/src/arqueogal/` (62 Python files, ~14k lines) for anti-patterns. CLAUDE.md exemptions (N803, N806, PLR2004 for astronomical names) respected.

## Findings

### Broad Exception Handlers (BLE001: acceptable with justification)

All bare-except or `except Exception` blocks carry `noqa: BLE001` comments explaining the reason. All are **justified and acceptable**:

1. **`src/arqueogal/utils/io.py:147`** — `except Exception` — Legacy torch checkpoint fallback. Catches `weights_only=True` safety-check failures on older pickled configs; retries with `weights_only=False` only on that specific error path. Well-scoped (re-raises if not the expected condition).

2. **`src/arqueogal/utils/config.py:153`** — `except Exception` — Dataclass type hint resolution via `get_type_hints()`. Falls back to raw `__annotations__` if PEP 563 string resolution fails (e.g., forward refs in user code). Specific and low-risk (only affects config instantiation, not data).

3. **`src/arqueogal/data/gaia_xp.py:339`** — `except Exception` — GaiaXPy `calibrate()` can raise various types (CalibrationError, ValueError, etc.). Catches all, logs warning, sets flag=2, continues batch processing. Proper batch-failure isolation.

4. **`src/arqueogal/xp_abundances/main/bimodality.py:210`** — `except Exception` — Scikit-learn GMM convergence failures. Catches convergence errors, returns False + stats dict with reason. Correct usage for model-fitting uncertainty.

5. **`src/arqueogal/data/tap.py:279, 287, 633`** — `except Exception` (3 instances) —
   - **279** (`_safe_error_message`): Catches DALQueryError from UWS `raise_if_error()` to extract error text. Low-risk (used only in error reporting).
   - **287** (`_safe_delete`): Catches all to safely delete an async job without crashing cleanup. Correct pattern.
   - **633** (`batched_upload_fetch_df`): Retryable TAP upload loop. Catches transient backend errors (PooledConnection, 5xx, timeout), distinguishes via `_is_transient_tap_error()`, re-raises non-transient. Well-scoped retry logic.

6. **`src/arqueogal/data/release_pipeline.py:406`** — `except Exception` — Partition build optional stage. Catches all, logs to manifest, continues release pipeline. Acceptable for optional derivative construction (not on critical path).

**Verdict:** All broad-exception handlers are **justified and carry explanatory comments**. No actionable issues.

---

### God Functions (>200 lines)

1. **`src/arqueogal/xp_abundances/main/uncertainty.py:410-684` — `gp_smoothed_per_cell_per_label_scale()` → 275 lines**
   - Fits Gaussian Process calibration with per-cell temperature scaling. High complexity but single responsibility.
   - **Status:** Noted as "deprecated but retained for methodology comparison" in CLAUDE.md §Known footguns. Not production.
   - **Concern:** 117 outgoing edges (function is the largest in the repo). Difficult to test and modify.
   - **Recommendation:** If kept, add reference in docstring to CLAUDE.md note and consider extracting GP helper functions (fit_gp_per_label, update_gp_predictions) as separate callables for testing.

2. **`src/arqueogal/data/gaia_xp.py:195-408` — `apply_ye2024_correction()` → 214 lines**
   - Vectorized XP preprocessing pipeline: Ye+2024 NN flux correction, per-coefficient z-scoring, batched I/O orchestration.
   - **Status:** Critical data-layer function (mandatory per CLAUDE.md §12).
   - **Concern:** High cyclometric complexity (multiple nested loops, nested try/except, conditional batch sizing).
   - **Recommendation:** Extract `_ye2024_batch_pass()` and `_post_calibration_rescale()` as helpers to reduce nesting and improve testability.

**Verdict:** Both are complex but justified. Neither is a refactoring blocker. The Ye2024 function would benefit from sub-function extraction for readability, but CLAUDE.md testing notes indicate production-size smoke tests exist for the data layer.

---

### Mutable Default Arguments

No mutable default arguments found (no `def foo(x=[])` or `def foo(x={})`). All defaults are immutable or None-checked.

**Verdict:** None. ✓

---

### String Concatenation in Loops

No string concatenation accumulation in loops. Strings are built via:
- `str.join()` (good): `",".join(str(i) for i in batch)` in `tap.py:339, 457`
- `.replace()` (good): Direct template substitution in `tap.py`
- Format strings or f-strings (good): No loop string accumulation found

**Verdict:** None. ✓

---

### List/Dict Comprehension vs Generator (Memory Impact)

Checked for large comprehensions where generators would be better. Found one **minor inefficiency**:

1. **`src/arqueogal/data/tap.py:431`** — `ids = [int(x) for x in source_ids]`
   - Materialises entire input iterable into a list to compute `n_batches` and slice it.
   - **Context:** `batched_fetch_df()` handles variable-size inputs; pre-materialising allows checkpoint logic (resume on rerun).
   - **Verdict:** Acceptable. The comment `Duplicates are kept as-is — dedupe upstream if needed` shows this is intentional. Materialization is necessary for checkpointing.

**Verdict:** None actionable. ✓

---

### Magic Numbers (non-exempt)

All reviewed magic numbers are either:
- **Physical constants** (exempt per CLAUDE.md: `M_sun`, `R_gal`, `v_phi`, etc.)
- **Astronomical conventions** (exempt: Gaia magnitude definitions, filter names)
- **Infrastructure tuning knobs with named constants**, e.g.:
  - `SYNC_ROW_THRESHOLD = 5_000` (tap.py:54)
  - `DEFAULT_ASYNC_TIMEOUT_SEC = 3600` (tap.py:62)
  - `_FEATURE_JOIN_COLS` tuple (release_pipeline.py:54)
  - `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` dict (release_pipeline.py:415)

No bare magic numbers found in critical code paths.

**Verdict:** None. ✓

---

### Timing: `time.time()` vs `time.perf_counter()`

1. **`src/arqueogal/data/release_pipeline.py:163, 205, 259, 344, 346`** — Five `time.time()` calls for pipeline wall-time tracking.
   - **Context:** Release pipeline reporting; absolute wall-clock is appropriate (humans want to know "this took 2 hours").
   - **Verdict:** Acceptable. `perf_counter()` is for relative benchmarking; `time.time()` is correct for absolute duration logging.

2. **`src/arqueogal/xp_abundances/main/knn_rescue.py:226, 235`** — Two `time.time()` calls in progress reporting.
   - **Line 226:** `t0 = time.time()` (start timer)
   - **Line 235:** `rate = (end + 1) / max(time.time() - t0, 1e-3)` (compute throughput)
   - **Context:** Wall-time progress/rate reporting for user feedback. Appropriate.
   - **Verdict:** Acceptable.

**Verdict:** No issues. All uses are for wall-clock reporting, not micro-benchmarking. ✓

---

### Dict-Stuffing Anti-Pattern

Checked for functions that return heterogeneous dicts used as pseudo-objects. Found **controlled and documented usage**:

1. **`src/arqueogal/data/release_pipeline.py`** — Manifest dictionaries (`manifest`, `summary`, `flag_distribution`)
   - **Context:** Pipeline orchestration returns structured dicts (join_summary, kin_ood_summary, annotate_summary) merged into a manifest.
   - **Justification:** Manifest is serialized to JSON (line 410), so dict is the correct container. Well-typed return hints in docstrings.
   - **Verdict:** Acceptable. Not a code smell here.

2. **`src/arqueogal/data/tap.py:584, 598`** — `submit_kwargs` dicts
   - **Context:** Conditional TAP submit parameters. Dict is appropriate for `service.submit_job(**kwargs)` unpacking.
   - **Verdict:** Acceptable. Standard pattern.

**Verdict:** No dict-stuffing anti-pattern. Used dict appropriately where JSON serialization or **kwargs unpacking is needed. ✓

---

### In-Band Sentinels (Masquerading as None)

Checked for use of special values (e.g., -1, "", 0 used as sentinel) instead of None or Optional. Found **one minor case**:

1. **`src/arqueogal/xp_abundances/main/uncertainty.py:918`** — `"edges_per_col": [[]]`
   - **Context:** `cell_definition = {"n_bins": [1], "edges_per_col": [[]]}` when no cell stratification is requested.
   - **Justification:** Represents "no edges per column" (empty list of empty lists). Matches schema of multi-column case.
   - **Verdict:** Acceptable. Not a sentinel; represents the actual empty-cell case.

**Verdict:** None. ✓

---

### Classes That Should Be Dataclasses

Scanned for classes with only `__init__` and no methods. Found **none**. Classes are either:
- Dataclasses already (e.g., `Config` in config.py)
- Have substantive methods (e.g., `CalibrationArtifacts` with calibration logic)
- Are Pydantic models or specialized classes

**Verdict:** None. ✓

---

## Summary

- **0 actionable issues.** All broad-exception handlers are justified with `noqa: BLE001` comments. Both god functions (275 and 214 lines) are acceptable given their domain complexity and documented status (one deprecated-but-retained, one critical data preprocessing).
- **2 improvement opportunities** (optional):
  1. `gp_smoothed_per_cell_per_label_scale()` could be sub-divided if further development is needed (currently deprecated for methodology comparison).
  2. `apply_ye2024_correction()` could extract batch-pass and rescaling helpers for clarity, but the function is well-tested per CLAUDE.md production-size smoke-test requirement.
- **Codebase hygiene is good:** mutable defaults, string concatenation loops, dict-stuffing, and timing patterns are all correct.

---

## Audit Scope

- 62 Python files scanned
- Patterns checked: bare except, broad exception handlers, mutable defaults, str loops, list/dict memory inefficiency, magic numbers, god functions, dict-stuffing, in-band sentinels, dataclass opportunities, timing practices
- CLAUDE.md exemptions (N803, N806, PLR2004) applied
