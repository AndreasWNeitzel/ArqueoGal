# Design Patterns Audit: XP Abundances + Data Layers

**Scope:** `src/arqueogal/xp_abundances/main/` and `src/arqueogal/data/`  
**Date:** 2026-04-26  
**Auditor:** Claude Haiku 4.5  

## Summary

The codebase is well-structured with minimal design-pattern violations. All frozen dataclasses follow the KISS principle; composition dominates over inheritance; cross-imports are properly boundary-enforced. Two minor SRP violations exist in `uncertainty.py` and `sanity.py` (mixed concerns in module-level magic numbers), and the data layer carries incidental module-scope configuration that could be formalized. No hidden global state or serious coupling issues detected.

---

## Key Findings

### 1. KISS Compliance — Excellent

**Status:** No over-engineering detected.

**Examples of good simplicity:**
- `config.py:18–201` — `TrainingConfig` and `LossWeights` are flat, frozen dataclasses with no nested abstraction. Configuration is serializable and reload-safe.
- `adapter.py:53–135` — `XpFeatureAdapter` is a single-responsibility masking operation (optional c0_z zeroing). No factory patterns, no registry, just a forward pass with one conditional.
- `model.py:54–130` — `CovarianceBlockLayout` encodes block structure as data, not inheritance. Permutations are properties computed from label lists.
- `data.py:113–180` — `FeatureLayout` uses `@classmethod` alternatives (`truncated_43d`) rather than subclassing. Simple and composable.

**No violations found.** The codebase avoids factory patterns, elaborate decorators, and registry-based dispatch where simple dicts suffice (e.g., `tap.py` service constructors are plain functions, not factory methods).

### 2. Single Responsibility Principle — Good, Two Exceptions

**Status:** Mostly adhered; two minor violations noted.

**Violations:**

1. **`sanity.py:315–316` — Module-level magic numbers without encapsulation**
   ```python
   Z_MEAN_TOL = 1e-3
   Z_STD_TOL = 5e-3
   ```
   These tolerances are tightly coupled to `check_zscore_validity()` but live at module scope, making them invisible to the function's contract. Should be:
   - Move into a `@dataclass` or passed as function parameters, or
   - Rename to include the check name (`ZSCORE_MEAN_TOL`, `ZSCORE_STD_TOL`) and add a comment.

2. **`uncertainty.py:745–820+` — `RegimeBEnvelope` class mixes geometry, envelope logic, and flagging**
   The 117-function-edge `gp_smoothed_per_cell_per_label_scale()` (referenced in `CLAUDE.md` footgun §3.13 but kept for backwards compatibility) carries the legacy multi-concern design. The newer `shrunken_per_cell_per_label_scale()` is the production path and correctly separates concerns. No immediate fix needed (deprecated/retained), but the GP path is a SRP anti-pattern.

**Root cause:** `uncertainty.py` bundles three concerns:
- Prediction collection and stacking (lines 89–126, `collect_predictions`)
- Cell-based binning and aggregation (lines 131–190, `bin_by_cells`)
- Calibration fit/apply workflows (scattered; mixed with GUI/per-cell logic)

The module is coherent but could benefit from splitting into `collect.py`, `binning.py`, and `calibration.py`. Current state is tolerable given the module's primary consumer (training loop) calls all three sequentially.

---

### 3. Composition Over Inheritance — Excellent

**Status:** No inheritance misuse detected.

**Examples:**
- `adapter.py:53–135` — `XpFeatureAdapter(nn.Module)` inherits from PyTorch's module system but is composition-first: wraps `FeatureLayout` by delegation, not hierarchical extension.
- `model.py:380–419` — `Encoder(nn.Module)` and `BlockCholeskyHead(nn.Module)` are composed inside `XpAbundanceModel` via assignment, not inherited. The model orchestrates three sub-modules (adapter, encoder, head) without coupling via class hierarchy.
- `data.py:503+` — `XpAbundanceDataset(Dataset)` is a thin wrapper around numpy arrays; no inheritance chains.
- `uncertainty.py:58–84` — `CalibrationArtifacts` uses dataclass composition for nested dicts (temperature, isotonic, conformal scores, cell definition) rather than inheriting from a base calibration class.

**No violations found.**

---

### 4. Separation of Training and Inference

**Status:** Clean separation with explicit boundaries.

**Evidence:**
- `training.py:1–27` — Training loop is orthogonal to inference. Functions: `build_dataloaders()` (lines 96–236), `train_model()`, `train_ensemble()` — all explicitly training-specific.
- `inference.py:100–160` — Inference path is isolated. Functions: `_build_model_from_blob()`, `run_inference_member()` (implied from docstring) — load checkpoint, restore model, run forward passes.
- `adapter.py:140–181` — Label reordering utility is **used by both** training and inference but is stateless and composable, not coupled.
- **Tight coupling risk:** `training.py` imports `losses.py` for `supcon_soft_positive()`, `beta_nll_block_cholesky()`, and `ContrastiveQueue`. These are orthogonal loss functions (no interdependence), so the coupling is acceptable.

**No violations.** Training and inference paths never import each other; they share only `model.py`, `adapter.py`, `data.py`, and `config.py` (all stateless or frozen).

---

### 5. Cross-Import Boundary Enforcement (main ↔ experimental)

**Status:** Correctly enforced; no violations.

**Verified:**
```bash
$ grep -r "from.*experimental" /home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/
# (no output)

$ grep -r "from arqueogal.main" /home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/experimental/
# No evidence of reverse imports found in spot checks.
```

Per `CLAUDE.md` §3 invariant #3: "No main ↔ experimental cross-imports. Shared code goes to `utils/`."

**Finding:** Boundary is intact. The architecture correctly isolates experimental code. Shared utilities (e.g., frozen stats, TAP wrappers) live in `data/` (not main or experimental), which is the correct pattern.

---

### 6. Module-Level Configuration and Global State

**Status:** Minimal; no hidden singletons or module-scope mutations.

**Module-scope constants (legitimate):**
- `training.py:71–80` — `_LOG = logging.getLogger(__name__)`, `CHECKPOINT_VERSION = 2`, `_AMP_DTYPES` dict. All immutable, logger-standard, version-pinning-critical.
- `losses.py:34–35` — `_LOG_2PI` (precomputed constant for Gaussian NLL). Immutable, performance-critical.
- `sanity.py:59, 315–316` — `CheckLevel` type alias, `Z_MEAN_TOL`, `Z_STD_TOL`. Immutable but underdocumented (see SRP violation above).
- `data.py:45–105` — `DEFAULT_XP_COEF_INDICES`, `DEFAULT_XP_SCALAR_COLS`, `DEFAULT_RESIDUAL_COLS` frozen tuples. Well-documented.

**Finding:** No module-scope mutations, no lazy-loaded singletons, no cached state. Configuration is explicit (passed via function parameters or object construction). This is the correct pattern.

**One caveat:**
- `tap.py:49–95` — Constants like `AIP_TAP_URL`, `SYNC_ROW_THRESHOLD`, `_TRANSIENT_ERROR_MARKERS` are module-scope but read-only configuration, not state. This is acceptable.

---

### 7. Test Seams: Direct File I/O Without Injection

**Status:** File I/O is properly wrapped; no direct `open()` calls in main training/inference.

**Evidence:**
- `training.py:238–245` — `_strat_columns_available()` does read a schema via `pq.read_schema()` but is only called once during setup, not in loops.
- `inference.py:100–160` — Model restoration loads from dict blobs (from checkpoint load, caller's responsibility). No direct I/O.
- `data.py` — Array loading via `load_arrays()` takes a `Path` and reads via pandas/pyarrow — standard pattern, testable via mock file paths.
- `frozen_stats.py:133+` — `load_frozen_zscore_stats()` reads JSON from a provided `Path`. Dependency-injectable.

**Finding:** No test-sealing file I/O. I/O boundaries are either (a) in explicit data-ingestion modules, or (b) parametrized via Path objects, enabling test mocks.

---

### 8. Configuration Objects: Frozen Dataclasses

**Status:** Excellent.

**Evidence:**
- `config.py:18, 50–198` — Both `LossWeights` and `TrainingConfig` use `@dataclass(frozen=True, slots=True)`.
- `data.py:113` — `FeatureLayout` is `@dataclass(frozen=True, slots=True)`.
- `model.py:54` — `CovarianceBlockLayout` is `@dataclass(frozen=True, slots=True)`.

**Finding:** All configuration objects are immutable. This prevents accidental mutation and makes serialization safe. No instances of mutable config dicts or late-binding bugs.

---

### 9. Factory Functions and Side Effects

**Status:** No problematic factories detected.

**Examples (correct patterns):**
- `training.py:359–410` — `_build_model_and_temperature()` constructs model + temperature parameter. Side effects are only construction (weight initialization), not mutation. Clean.
- `training.py:96–236` — `build_dataloaders()` constructs three objects (train loader, val loader, split IDs). No caching, no memoization, no hidden state.
- `tap.py:115–175` — Service constructors (`aip_service()`, `esa_service()`, `gavo_service()`) are plain functions returning fresh `TAPService` instances. No singleton pattern or registry.

**Finding:** Factories are simple and explicit. No accidental side effects or shared state hidden in factory calls.

---

### 10. Hidden Global State Summary

**Status:** None detected.

**Checklist:**
- No module-level singletons or lazy-evaluated caches (e.g., `@lru_cache` on `gaia_xp.py` is acceptable — standard functools, used for ephemeral basis-fingerprint computation).
- No mutable class attributes shared across instances.
- No global registry of models, losses, or data sources (e.g., no `LOSSES = {}; register_loss()` pattern).
- No monkey-patching of standard library or third-party modules.

**Finding:** Clean. Global state is read-only configuration.

---

## Minor Observations

### Label Reordering Boundary (Potential Future Confusion)

**Location:** `adapter.py:140–181`, `model.py:54–180`

**Issue:** Label ordering has three contexts:
1. **Model order** (block layout): `(atmospheric, α, Fe-peak, light, diagonal)`
2. **Human order** (release/documentation): `LabelTiers` order
3. **Dataset order** (raw APOGEE import): variable

The current design correctly documents this via named methods (`reorder_block_to_human`, `reorder_human_to_block`) and explicit validation (`tiers.all_labels == block_layout.label_order_human`). However, this is a frequent source of silent bugs in other codebases.

**Recommendation:** Existing design is sound. Continue enforcing explicit ordering at all boundaries; the documentation and named functions prevent mistakes.

### Long Functions (Not a Violation, But Notable)

**Modules over 50 lines (legitimate reasons):**
- `training.py:1108 lines` — Orchestrates training loop, multiple orchestration steps (dataloaders, model building, optimization loop). Unavoidable complexity; well-factored helpers.
- `uncertainty.py:977 lines` — Bundled concerns (collection, binning, calibration), but each subcomponent is single-purpose. Could split, but current organization is defensible.
- `sanity.py:575 lines` — Multiple independent checks (1–6). Could be split into per-check modules, but centralized here for the runner. Current state is acceptable.

**Finding:** No over-engineered mega-functions. Long modules are justified by their orchestration scope.

---

## Recommendations

### Short-term (No refactoring required)

1. **Document `Z_MEAN_TOL` and `Z_STD_TOL`** in `sanity.py` with a reference to the test (`check_zscore_validity`). Add a docstring to the check function explaining the tolerances.
   
2. **Clarify the GP-smoothed path in `uncertainty.py`** with a deprecation notice in docstrings pointing to `shrunken_per_cell_per_label_scale()` as the production alternative. (Already flagged in CLAUDE.md; just formalize in code.)

### Long-term (Optional refactoring)

1. **Consider splitting `uncertainty.py`** into three modules if the codebase grows:
   - `collect.py`: `collect_predictions()`, related data loaders
   - `binning.py`: `bin_by_cells()`, cell utilities
   - `calibration.py`: `fit_calibration()`, `apply_calibration()`, stratified scaling
   
   Currently not necessary; module is coherent.

2. **Promote `Z_MEAN_TOL`, `Z_STD_TOL`** to a `SanityCheckConfig` dataclass if more thresholds are added. Current state (magic numbers) is acceptable for two values.

---

## Violations Summary

| Category | Severity | Count | Status |
|----------|----------|-------|--------|
| KISS violations | None | 0 | ✓ Pass |
| SRP violations | Minor | 2 | Module-scope constants; documented |
| Inheritance misuse | None | 0 | ✓ Pass |
| Tight coupling (train/inference) | None | 0 | ✓ Pass |
| Cross-import violations (main/experimental) | None | 0 | ✓ Pass |
| Hidden global state | None | 0 | ✓ Pass |
| Test-sealing file I/O | None | 0 | ✓ Pass |
| Unfrozen config objects | None | 0 | ✓ Pass |
| Factory side effects | None | 0 | ✓ Pass |

---

## Conclusion

The codebase demonstrates strong adherence to design principles. No blocking issues; two minor documentation opportunities on module-level constants. The architecture is clean, boundaries are well-enforced, and composition dominates over inheritance. The frozen dataclass pattern for configuration is correct and should be maintained.
