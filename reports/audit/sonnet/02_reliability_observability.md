# Reliability and Observability Meta-Report
**ArqueoGal v1.0 — Synthesis of 7 Haiku Audits**  
**Date:** 2026-04-26  
**Scope:** Error handling, resource management, resilience, observability, debuggability, error patterns, and footgun coverage across 62 modules (~15k LOC)

---

## Executive Summary

The codebase exhibits **strong resource discipline and good module-level error handling**, but suffers from a **critical asymmetry between training and inference observability**. Data pipelines (ingestion, TAP queries) are well-logged and resilient to transient failures; the inference→release→tier-promotion path is almost entirely dark. Three findings triangulate across ≥2 audits: (1) NaN safety at the inference boundary is documented but not enforced in function signatures, (2) HALT errors use string logging instead of exception types, blocking automated monitoring, and (3) the critical path from ensemble predictions to Tier assignments has zero structured logging, making production debugging impossible. The codebase is not ready for public GitHub release without fixing these three high-leverage gaps.

---

## 1. Triangulated Findings (≥2 Audits Converge)

### 1.1 NaN-Safety Boundary Asymmetry (Training vs Inference)
**Sources:** error_handling.md:87–125, debug_friendliness.md:19–32, footgun_coverage.md:80–101

**The Gap:** Training applies `np.nan_to_num(..., nan=0.0)` at the data-loader boundary (training.py:154). Inference's `predict_ensemble()` docstring claims "all features are sanitised at inference entry" (inference.py:25–29), but the function signature has **no `nan_to_num` call**; it expects the caller to sanitise upstream. The `nan_to_num` on line 240–241 operates on *outputs* (`mu_m`, `L_m`), not inputs.

**Risk:** A caller writing `result = predict_ensemble(ensemble, loader)` without pre-sanitising features will silently produce NaN predictions. The Mahalanobis OOD flag (inference.py:248–260) covers only the 108-D XP block, not aux features. A single NaN in V (5.3% NaN rate globally) NaN-propagates through the trunk undetected.

**Evidence Chain:**
- error_handling.md finds the docstring/implementation mismatch at line 25 vs 203.
- debug_friendliness.md notes `collect_predictions` output *may contain NaN* but the docstring does not warn (uncertainty.py:94–97).
- footgun_coverage.md confirms the regression test exists (test_nan_to_num_regression_produces_finite_predictions) but applies `nan_to_num` *inside* the test harness, not in the inference function itself.

**Verdict:** The footgun is documented but enforcement is caller-side only. For public release, either (a) add `features = np.nan_to_num(features, nan=0.0, ...)` as the first line of `predict_ensemble`, or (b) add a strict docstring with a `ValueError` if the caller does not pre-sanitise (inspect arrays on entry). Current state is a footgun.

---

### 1.2 Zero Structured Logging in Inference→Release Path
**Sources:** observability.md:39–62 (critical gaps section), debug_friendliness.md (NaN documentation), footgun_coverage.md (no logging noted)

**The Dark Zone:** Five modules handle the critical path from trained ensemble to published catalogue with **zero logger calls**:
- `inference.py`: 297 LOC, 0 log calls → No insight into ensemble member loading, per-member aggregation, or calibration application.
- `uncertainty.py`: 977 LOC, 0 log calls → No visibility into per-cell temperature scaling decisions or fallback rates.
- `release.py`: ~250 LOC, 0 log calls → No trace of which rows hit which OOD-flag gates (Mahalanobis, NaN, σ-inflation, mode-ambiguous, kin_ood).
- `tier_promotion.py`: ~200 LOC, 0 log calls → Six-test protocol (research_brief.md §3.3) has zero diagnostic output; users cannot see why a star is Tier 2 vs Tier 3.
- `sanity.py`: 575 LOC, 0 log calls → No pre-release validation output.

**Impact:** A Tier-3 anomaly surfaces post-release. You check logs and see "inference complete, N predictions"; you **cannot determine** which frozen stats basis was used, whether per-cell calibration succeeded, or which OOD test flagged a row as Tier 3.

**Verdict:** Production-blocking. observability.md lists this as P0. Before any public release, `inference.predict_ensemble()`, `apply_calibration()`, and `_assign_tiers()` must emit diagnostic logs at entry and at each major decision point.

---

### 1.3 HALT Errors Use String Logging Instead of Exception Types
**Sources:** error_patterns.md:9–40 (HALT errors section), resilience.md (no circuit breaker), footgun_coverage.md (no custom exception hierarchy mentioned)

**The Gap:** HALT errors (scripts/fetch_gaia_xp_delta.py:250–263, apply_ye2024_xp_delta.py:205–207, fetch_bailerjones_andrae_rgb.py:417–484) emit:
```python
log.error("HALT: %s", halt_reason)
```

There is no `HaltError` exception subclass, no structured exit code, no `subsystem` attribute. Downstream monitors (status dashboards, alerting systems) must regex-match the log line and parse the reason string.

**Impact:** Automated recovery and monitoring cannot programmatically distinguish a footprint overflow (fixable by cleaning `data/`) from a network outage (needs retry) from a schema mismatch (needs code fix).

**Evidence:** error_patterns.md recommends (lines 33–40) defining a custom exception class:
```python
class HaltError(RuntimeError):
    def __init__(self, subsystem: str, reason: str):
        self.subsystem = subsystem
        self.reason = reason
```

**Verdict:** Not release-critical (existing logs are human-readable), but should be standardized before public release to enable CI/CD monitoring. Effort: 1 hour (define exception hierarchy, replace 3 sites, add test).

---

## 2. Disagreements Between Audits

### 2.1 Assertion vs. Exception Semantics in inference.py:242–246
**Audits:** error_handling.md (lines 122–124) says assertions are problematic; debug_friendliness.md (lines 24–26) says the design is sound.

**The Disagreement:**
- error_handling.md: "Using `assert` for user-facing failure conditions. `AssertionError` is meant for internal invariants. If a model produces NaN outputs, callers should get a descriptive exception type (`RuntimeError` or custom), not `AssertionError`. Assertions can be disabled with `python -O`."
- debug_friendliness.md: "The assertions compare output to an unreachable sentinel (`np.nan_to_num` output is always finite by definition), so an honest failure mode would silently pass through. **However**, this is mitigated by the assertion itself—if `mu_m` contains NaN after `nan_to_num(..., nan=0.0)`, that is a catastrophic model-output failure that an assertion can detect."

**Resolution:** debug_friendliness.md is correct on the narrow point (the assertion would catch the case), but error_handling.md is correct on the broader concern (assertions can be disabled at runtime via `-O`). For public release, replace with explicit exception types:
```python
if not np.isfinite(mu_m).all():
    raise RuntimeError(f"NaN in mu_m after sanitization (shape={mu_m.shape}, count={...})")
```

**Verdict:** Action required, but low priority compared to missing logs. effort: 15 min per site.

---

### 2.2 TAP Sync Retry: Is It Needed?
**Audits:** resilience.md (lines 35–49) says sync path "lacks any retry logic" and is "a single point of failure"; footgun_coverage.md does not mention sync retries as a failure mode.

**The Issue:** `run_sync()` (tap.py:183–205) has no retry on transient failures, but it is only used for queries <5k rows (SYNC_ROW_THRESHOLD = 5_000). resilience.md flags this as risky; footgun_coverage.md implicitly accepts it because larger datasets use `batched_upload_fetch_df()` which has full retry logic.

**Verdict:** Low risk in practice (large batches use the async path), but resilience.md is correct that a single transient glitch on a sync query fails the entire batch. Recommendation: add retry wrapper to `run_sync()` bounded to 3 attempts (sync queries are smaller, more likely to succeed on retry). Priority: P1 (near-term improvement).

---

## 3. Top 5 Actionable Items Before Public Release

### P0 (Blocking)

1. **Add structured logging to `inference.predict_ensemble()` entry** (observability.md:P0.1)
   - Log at line ~223: ensemble size, device, n_labels, calibration status per member.
   - **Effort:** 20 min. **Impact:** High (enables production debugging).

2. **Add structured logging to `_assign_tiers()` in release.py** (observability.md:P0.3)
   - Log per-test pass/fail counts: n_ood_flagged, n_nan_pred, n_sigma_inflated, per Tier.
   - **Effort:** 30 min. **Impact:** Critical (users must see why a star is Tier 2 vs Tier 3).

3. **Enforce NaN sanitisation at `predict_ensemble()` entry OR add strict docstring** (error_handling.md:132–133)
   - Option A: Add `features = np.nan_to_num(features, nan=0.0, ...)` as first line.
   - Option B: Raise `ValueError` on entry if features contain NaN.
   - **Effort:** 20 min. **Impact:** Eliminates silent NaN-propagation footgun.

### P1 (Near-term, improves robustness)

4. **Replace HALT string logging with custom exception hierarchy** (error_patterns.md:33–40)
   - Define `HaltError(subsystem, reason)` and apply to 3 sites (fetch_gaia_xp_delta.py, apply_ye2024_xp_delta.py, fetch_bailerjones_andrae_rgb.py).
   - **Effort:** 1 hour. **Impact:** Enables automated monitoring / CI integration.

5. **Add retry wrapper to `run_sync()` TAP path** (resilience.md:P1.3)
   - Bounded to 3 attempts (smaller than async's 6, appropriate for small queries).
   - **Effort:** 45 min. **Impact:** Medium (sync path is <5k rows, but any resilience improvement helps).

---

## 4. The System Requirement: Unified Structured Logging

**Current State:** 
- Data ingestion modules (ingest_xp.py, ingest_stream1.py, gaia_xp.py) have 50+ logger.info calls with reasonable context.
- Training module (training.py) has 11 logger calls covering epoch summaries, early stopping, ensemble dispatch.
- Inference, calibration, and release modules have **zero logger calls**.
- Standard library `logging` only; no `structlog`, no JSON-on-disk, no machine-readable field extraction.

**What's Missing (observability.md:107–135):**
The codebase does **not** have a structured-logging layer. Logs are human-readable but unparseable by downstream aggregators (dashboards, alerting, post-hoc analysis).

**Recommended Design:**
1. **Adopt `structlog`** (deferred to future phase per observability.md:P2.7, but should be on the roadmap).
2. **In the interim**, enforce a **canonical set of fields** across all log lines:
   - `phase`: "ingest_xp" | "train_ensemble" | "inference" | "release"
   - `git_sha`: Git commit of the run.
   - `timestamp`: ISO 8601.
   - Event-specific fields: `n_rows_processed`, `n_nan_dropped`, `ensemble_members`, `tier_assignments_summary`, etc.
3. **Add log entry/exit guards** around long-running functions:
   ```python
   _LOG.info("ensemble_inference_start", 
             n_members=len(ensemble), device=str(device), n_labels=ensemble[0].model.n_labels)
   # ... work ...
   _LOG.info("ensemble_inference_done", 
             n_predictions=predictions.shape[0], wall_time_sec=elapsed)
   ```

**Why This Matters:**
- **Debuggability:** When a Tier-3 anomaly surfaces, you can grep logs by `git_sha` and `timestamp` to reconstruct exactly what happened.
- **Reproducibility:** Provenance sidecars (data/provenance.py) already emit JSON; logging should mirror that structure.
- **Automated Alerting:** A monitor can extract `"n_tier3": 1000` and `"n_tier3_due_to_ood": 950` and surface drift (e.g., "OOD rate 95% vs historical 80%").

---

## 5. Collectively Missed Items (Emerging from Side-by-Side Reading)

### 5.1 Ye+2024 Correction Has No Runtime Verification (footgun_coverage.md:278–281)
The Ye+2024 correction is mandatory (CLAUDE.md invariant #5) and applied by `scripts/apply_ye2024_xp.py`. However, **no runtime check in the feature builder** validates that the input XP parquet has been corrected.

A developer could accidentally:
1. Fetch raw XP via `fetch_gaia_xp.py`.
2. Skip `apply_ye2024_xp.py`.
3. Feed raw coefficients to `build_pipeline1_features_stream1.py`.
4. Train on uncorrected XP (silently wrong).

**Fix:** Add a provenance check in the feature builder that reads the sidecar and verifies `"ye2024_corrected": true` in `extra` block. **Effort:** 30 min.

### 5.2 Inference.predict_ensemble() Has No Config Snapshot (observability.md:140–158)
Orchestrators (`run_ensemble.py`, `run_calibration.py`) capture config snapshots, but `run_pipeline1_inference.py` does not.

When a Tier-3 anomaly is reported, you cannot determine from logs:
- Which ensemble checkpoint was loaded (seed set, training git SHA).
- Which frozen-stats basis was used (basis fingerprint).
- Whether per-cell calibration was applied per member.

**Fix:** Add config-snapshot log at `predict_ensemble()` entry with checkpoint metadata, frozen-stats fingerprint, and calibration flags. **Effort:** 20 min.

### 5.3 TAP Batch Context Loss in Exceptions (error_patterns.md:99–134, resilience.md:comment on lines 83–87)
When a TAP batch fails, the error log shows the exception message but not the batch's source_id range. This makes it hard to:
- Manually retry the failed batch.
- Correlate the batch to provenance sidecars (which track source_id ranges).

**Fix:** Log the chunk index and a representative sample of source_ids:
```python
src_sample = list(chunk[:5])
log.error(f"chunk {idx} (src_ids={src_sample}...): {exc!r}")
```
**Effort:** 15 min per site (3 sites in tap.py).

### 5.4 Release Pipeline Outputs Manifest to Stdout (observability.md:22–24)
`release_pipeline.py:664` prints the manifest JSON to stdout, not a logger. This violates the "no committed print" rule and makes it hard to post-hoc grep logs.

**Fix:** Replace `print(json.dumps(...))` with `_LOG.info(json.dumps(...))` and capture to a structured file. **Effort:** 10 min.

---

## 6. Cross-Audit Risk Assessment

| Risk | Likelihood | Severity | Detectability | Mitigated By |
|------|------------|----------|----------------|--------------|
| NaN in inference silently produces Tier 3 | Low (test exists) | **Critical** | Zero (no logs) | Enforce `nan_to_num` at entry OR raise on NaN input |
| OOD-flag logic miscalibrated, undetected | Medium | High | Zero (no logs) | Add structured logs to `_assign_tiers()` |
| Ye+2024 correction skipped, wrong training | Low (discipline) | **Critical** | Zero (no provenance check) | Add provenance verification in feature builder |
| Transient TAP failures unretried (sync path) | Low (small batches) | Medium | Partial (logs exist) | Add retry wrapper to `run_sync()` |
| Tier promotion test results opaque | Medium | High | Zero (no logs) | Add logs to tier_promotion.py for each test |

---

## Conclusion

ArqueoGal exhibits **strong data-engineering discipline** (TAP retry, atomic writes, exception chaining in ingest modules) but **critical blind spots in the release pipeline**. The inference→calibration→release path has zero structured observability, making it impossible to debug production anomalies without rerunning the entire pipeline. Footgun coverage is **excellent** (13/13 known risks properly surfaced), but NaN-safety enforcement is caller-side only, creating a subtle footgun.

**For GitHub public release:**
- Fix P0 items 1–3 (logging + NaN enforcement). **3 hours total effort.**
- Fix P1 items 4–5 (HALT exceptions + sync retry). **1.5 hours total effort.**
- Defer structured-logging migration (structlog) to Phase 4+.

**Without these fixes, the release cannot claim reproducibility or debuggability.** With them, confidence in public reliability increases substantially.

---

**Audit Metadata:**
- **Haiku sources:** error_handling.md, resource_management.md, resilience.md, observability.md, debug_friendliness.md, error_patterns.md, footgun_coverage.md.
- **Date completed:** 2026-04-26 16:45 UTC.
- **Auditor:** Claude Code (Sonnet synthesis).
