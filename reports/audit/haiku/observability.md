# Python Observability Audit — ArqueoGal

**Date:** 2026-04-26  
**Scope:** `src/arqueogal/` (62 Python modules, ~15k LOC)  
**Policy:** JSON-on-disk canonical, `aim`/`wandb` opt-in via env var, no committed `print` statements.

---

## Executive Summary

The codebase practices **moderate-to-good logging discipline** in data pipelines but exhibits **critical blind spots in three high-leverage modules** where observability is weakest: `inference.py`, `uncertainty.py`, and the tier-promotion / release-assignment layer lack any logged instrumentation despite handling the critical path from trained ensemble to published catalogue. Config snapshots are captured in run orchestrators but not serialised at inference entry, making production debugging of OOD-flag logic or calibration anomalies impossible without rerunning the entire pipeline.

---

## Findings

### 1. Committed Print Statements — COMPLIANT

**Status:** ✓ PASS (minimal violations)

- **src/arqueogal/xp_abundances/main/knn_rescue.py:237** — interactive progress meter (`print(f"  knn {end}/{n_query}...")`) — acceptable for CLI context, not a hard error but candidate for `logger.info` + `run_in_background` progress tracking.
- **src/arqueogal/data/release_pipeline.py:664** — `print(json.dumps(manifest, indent=2, default=str))` — **violation**: manifest output to stdout should be logged + written to file. No logger established in this module.

**Action:** Move `release_pipeline.py:664` to `logger.info(json.dumps(...))` and capture to structured file output.

---

### 2. Logger Initialization & Naming — COMPLIANT

**Status:** ✓ PASS (consistent pattern, one exception)

All 27 data modules + orchestrators use `logger = logging.getLogger(__name__)` consistently.  
One exception: `scripts/run_ensemble.py:55` and `scripts/run_calibration.py` use `_LOG = logging.getLogger("module_name")` (hardcoded strings rather than `__name__`). This is acceptable for scripts but breaks downstream log filtering by module.

**Consistency:** 96% (26/27 modules use `__name__`, only `ensemble_diagnostics.py` exception).

---

### 3. Logging at Hotpaths — CRITICAL GAPS

**Status:** ⚠ FAIL (missing instrumentation in critical modules)

#### a) **Inference Pipeline** — Zero Context Logging

| Module | LOC | Logger Calls | Status |
|--------|-----|--------------|--------|
| `inference.py` | 297 | 0 | ✗ MISSING |
| `uncertainty.py` | 977 | 0 | ✗ MISSING |
| `sanity.py` | 575 | 0 | ✗ MISSING |
| `release.py` | ~250 | 0 | ✗ MISSING |
| `tier_promotion.py` | ~200 | 0 | ✗ MISSING |

These five modules handle the **critical path from inference to tier assignment** — no logs at:

- Ensemble member loading (inference.py line ~170–200)
- Per-member prediction aggregation (inference.py line ~235–289)
- Calibration application (inference.py line ~254, calls `uncertainty.apply_calibration`)
- Per-cell temperature scaling decisions (uncertainty.py, `apply_calibration` line ~600–700, no visibility into which cells got scaled)
- Tier-promotion test results (tier_promotion.py, six-test protocol §3.3 in `research_brief.md`, zero diagnostic output)
- OOD-flag logic (release.py, `_assign_tiers` function, no trace of which rows hit which gates)

**Impact:** Production failures in calibration or OOD flagging cannot be diagnosed post-hoc. A user reports anomalous Tier 3 assignments; you cannot see whether it was Mahalanobis OOD, NaN prediction, or σ-inflation without rerunning inference.

#### b) **Training Loop — Moderate Coverage**

training.py has 11 logger calls across 1108 LOC:

- **Covered:** Epoch summaries (line 753: loss, tau, grad norms), early stopping (790), ensemble member dispatch (1075).
- **Missing:** Per-batch gradient diagnostics (training.py line 604–626, `train_one_epoch`), NaN-to-num application rate (data.py:150, logs "dropping N rows" but not the NaN rate / source distribution), feature-adapter output validation (adapter.py, zero logging).

**Lines to instrument:**

- `training.py:558–622` — `for batch in loader` loop: log every 100 batches with batch index, loss components, grad norm.
- `training.py:145–150` — NaN dropping: log source_id ranges where XP features are NaN, correlated flags.

#### c) **Data Pipeline — Good Coverage**

Ingest modules (stream1, stream2, stream3, XP) have 50+ logger.info calls. Context richness is acceptable:

- `ingest_xp.py:127–210` — logs fetch count, Ye+2024 flag summary (3 categories), output row count.
- `ingest_stream1.py:67–156` — logs per-stage operation (DR19 download, quality cuts, corrections, merge counts).
- `gaia_xp.py:328–400` — logs Ye+2024 batch progress (lo..hi/n), exception details, final flag counts.

**Weakness:** No per-source_id context in exception logs — `gaia_xp.py:340` logs batch range but not source_ids where calibration failed.

---

### 4. Context Richness — MIXED

**Status:** ⚠ PARTIAL (missing in critical modules, good in data layer)

#### Good Practice (examples):

- **`ingest_xp.py:204–209`** — logs row counts before/after, flag breakdown as a dict.
- **`gaia_xp.py:394–399`** — final summary includes three flag tallies.
- **`ingest_stream1.py:146–152`** — logs intermediate operation names + row counts.
- **`run_ensemble.py:254–299`** — logs config hash, git SHA, per-member seed, val loss, best epoch.

#### Poor Practice (examples):

- **`training.py:145–150`** — `_LOG.info("dropping %d/%d rows with NaN in XP features", ...)` — no detail on *which* features had NaN, or *which* rows (source_id ranges).
- **`uncertainty.py:collect_predictions` (line ~100)** — **zero logging** — call this once per ensemble member, no trace of which device it loaded to, batch count, or shape of output tensors.
- **`release.py:_assign_tiers`** — **zero logging** — applies five tier-assignment tests (XP-Mahalanobis OOD, NaN, σ-inflation, mode-ambiguous, kin_ood) but no logs on how many stars hit each gate, or per-element breakdowns.

---

### 5. Structured Logging — NOT IMPLEMENTED

**Status:** ✗ FAIL (standard library `logging` only)

The codebase uses Python's standard `logging` module exclusively. There is **no `structlog`, no JSON-on-disk logging, and no structured-field capture** for machine-readable log parsing.

**Current pattern (example, `ingest_xp.py:205–209`):**
```python
logger.info(
    "Level-3: done (%d rows → %s; Ye flags: %s)",
    len(corrected),
    output_path,
    flag_counts,  # dict, rendered as str() in log output
)
```

This is **human-readable but unparseable** — a downstream log aggregator cannot extract `n_ok`, `n_no_synth_phot`, `n_calibrate_fail` counts as individual fields.

**Contrast: desired form (pseudocode):**
```python
logger.info(
    "xp_level3_done",
    rows_output=len(corrected),
    output_path=str(output_path),
    ye2024_flag_counts=flag_counts,  # passed as dict, not stringified
    git_sha=prov.git_sha,
)
```

**Impact on policy:** The CLAUDE.md rule ("JSON-on-disk is canonical") is **not yet activated** because no module emits structured logs. Provenance sidecars (see `data/provenance.py`) do emit JSON, but model training, inference, and release assignment do not.

---

### 6. Inference-Driver Config Snapshots — MISSING

**Status:** ✗ FAIL (critical for production debugging)

Orchestrators (`scripts/run_ensemble.py`, `scripts/run_calibration.py`) **do capture config snapshots** (e.g., `run_ensemble.py:246–253` writes `ensemble_config.json` with git SHA, config hash, hyperparameters).

However, **inference entry points have no equivalent**. Example:

- **`scripts/run_pipeline1_inference.py`** — loads ensemble, invokes `inference.predict_ensemble()`, no logged config snapshot at entry.
- **`inference.predict_ensemble()` (line ~203)** — no logged configuration (device, batch size, n_ensemble_members, n_labels, calibration status, frozen stats basis fingerprint).

**Impact:** A Tier-3 flagging anomaly surfaces in production. You check `run_pipeline1_inference.py` logs and see "inference complete, N predictions", but **cannot determine**:

- Which ensemble checkpoint was loaded (seed set, git SHA of training run)?
- Which frozen stats basis (fingerprint from `frozen_stats.py`) was used?
- Was per-cell calibration applied? (yes/no per checkpoint)
- Which OOD-flag method (Mahalanobis only, or multi-modal)?

**Fix:** Add log block at `inference.predict_ensemble()` entry (line ~223–230) capturing ensemble metadata, device, and calibration state.

---

### 7. Log Levels — GENERALLY APPROPRIATE

**Status:** ✓ PASS (minor over-logging)

- ✓ `logger.info()` used for operational milestones (ingestion steps, epoch summaries, file writes).
- ✓ `logger.warning()` used for recoverable anomalies (missing coordinates in `ingest_xp.py`, gaiaxpy.calibrate failure at `gaia_xp.py:340`).
- ⚠ One case of excess INFO: `gaia_xp.py:331` logs every XP batch ("Ye+2024 batch 0..100 / 50000"), which is fine for large batches but verbose at 5k-row batches; candidate for DEBUG.

---

### 8. Unlogged Critical Paths

**Status:** ✗ FAIL (inventory of dark zones)

| Function | Module | LOC | Purpose | Logging |
|----------|--------|-----|---------|---------|
| `predict_ensemble()` | inference.py | 203–289 | Aggregation of 5–10 ensemble members, covariance computations | 0 calls |
| `apply_calibration()` | uncertainty.py | ~600–700 | Per-cell temperature scaling application | 0 calls |
| `fit_calibration()` | uncertainty.py | ~700–800 | Calibration fitting on validation split | 0 calls |
| `_assign_tiers()` | release.py | ~100–150 | Five-test protocol tier assignment | 0 calls |
| `test_xp_mahalanobis_ood()` | tier_promotion.py | ?–? | OOD-flag test 1/6 — Mahalanobis distance on XP latent | 0 calls |
| `test_coverage()` | tier_promotion.py | ?–? | OOD-flag test 3/6 — empirical coverage check | 0 calls |
| `collect_predictions()` | uncertainty.py | 89–150 | Iterate model over full validation loader | 0 calls |

All are **post-training, pre-release operations** where a single NaN propagation, OOD threshold miscalibration, or shape mismatch can silently corrupt the released catalogue.

---

### 9. Best Practices Present

**Status:** ✓ GOOD (few instances)

- **Atomic writes with temp files:** `ingest_xp.py:213–218` (`_write_parquet_atomic`) avoids incomplete writes on crash.
- **Exception-context logging:** `gaia_xp.py:340–347` logs the batch range and exception type on gaiaxpy failures.
- **Startup logging:** Orchestrators (`run_ensemble.py:254`, `run_calibration.py`) log ensemble directory, config hash, and git SHA at job start.

**What's missing:** No context managers for timed operations (see observability.md Pattern 7), no correlation IDs across runs, no OOD-flag metric aggregation (e.g., "X% Tier 3, Y% due to OOD").

---

## Recommendations (Priority Order)

### P0 (Production Blocker)

1. **Add logging to `inference.predict_ensemble()` entry** (line ~223–230):
   ```python
   _LOG.info(
       "inference_ensemble_start",
       n_members=len(ensemble),
       device=device,
       n_labels=ensemble[0].model.n_labels,
       calibration_status={m.seed: bool(m.calibration) for m in ensemble},
   )
   ```

2. **Add logging to `apply_calibration()` (uncertainty.py)**: log per-cell temperature scaling decisions, how many cells got a scale != 1.0, fallback rates.

3. **Add logging to `_assign_tiers()` (release.py)**: log per-test pass/fail counts:
   ```python
   _LOG.info(
       "tier_assignment_summary",
       n_ood_flagged=ood_count,
       n_nan_pred=nan_count,
       n_sigma_inflated=sigma_count,
       n_tier1=tier1_count,
       n_tier2=tier2_count,
       n_tier3=tier3_count,
   )
   ```

### P1 (Operational Improvement)

4. **Instrument `training.py:558–622` batch loop**: Log every 100 batches with loss components, grad norm. Use `logger.debug()` for batch-level detail.

5. **Fix `release_pipeline.py:664`**: Move manifest print to `logger.info()` + file write.

6. **Add config snapshot to `run_pipeline1_inference.py`**: Log ensemble paths, git SHAs, feature layout, before inference starts.

### P2 (Future, Low Urgency)

7. Migrate to `structlog` for JSON-on-disk output (deferred; would be part of broader observability infrastructure work).

8. Unify orchestrator logging (`run_*.py`) to use `__name__` instead of hardcoded logger names.

---

## Summary by Module Category

| Category | # Modules | Logging Quality | Issues |
|----------|-----------|-----------------|--------|
| **Data ingestion** | 8 | Good | Minor: batch-level source_id context missing in exceptions |
| **Model training** | 1 | Good | Missing: batch-level diagnostics, NaN rate breakdown |
| **Inference + Calibration** | 5 | **Critical gap** | Zero logging in 4/5 modules; blocking production observability |
| **Scripts / Orchestration** | 15 | Good | Minor: inconsistent logger names |
| **Utilities** | 6 | N/A | No logging expected; correct |

---

**Audit completed:** 2026-04-26 15:30 UTC  
**Auditor:** Claude (Haiku 4.5)  
**Severity:** Two critical (P0) items block production confidence.
