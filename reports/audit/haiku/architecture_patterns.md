# Architecture Audit: Clean/Hexagonal/DDD Patterns

**Date**: 2026-04-26  
**Scope**: `/src/arqueogal/` — 62 Python modules  
**Focus**: Domain/IO entanglement, orchestration/computation mixing, configuration leaks, dependency discipline

## Summary

The codebase exhibits **strong separation of concerns** at the module level with clean dependency boundaries and minimal configuration leakage. Domain logic (feature contracts, model architecture) is properly isolated from orchestration (training loops, inference dispatch) and IO (file loading, network requests). The data layer is well-stratified, but one critical boundary discipline (`nan_to_num` sanitisation) is enforced asymmetrically between training and inference, creating a latent test seam.

---

## Findings

### 1. **Strengths: Clean Dependency Architecture**

#### No Circular Dependencies
Dependency graph inspection (62 modules, 67 edges) reveals **no cycles**. Layering is enforced:
- **Domain core** (`xp_abundances/main/data.py`, `model.py`, `losses.py`): pure dataclasses and tensor logic, zero framework state.
- **Orchestration** (`training.py`, `inference.py`): stateless coordinators importing domain.
- **Infrastructure** (`config.py`, checkpoint I/O, PyTorch wiring): imported only by orchestrators.

The most critical edge — `inference.py:49` imports `training.py:CHECKPOINT_VERSION` — is the sole backward reference within `xp_abundances.main`, used to validate checkpoint versioning at load time. This is acceptable (version validation is a cross-layer concern).

#### Configuration Not Leaking into Domain
- `data.py` (lines 44–109) declares feature/label contracts as frozen dataclasses with no config dependencies — `FeatureLayout` and `LabelTiers` are pure data structures.
- `model.py` (inferred from adapter usage) holds architecture knobs (`ModelConfig`) as a separate dataclass.
- Training config (`TrainingConfig` in `config.py:51`) is **not injected into** domain classes; it flows only to orchestration (`build_dataloaders`, `train_model`, `train_one_epoch`).
- Checkpoint schema validation (`training.py:1000–1004`) explicitly rejects misaligned label scalers before save, preventing silent downstream errors.

#### Test Seams Intact
Domain classes (`XpAbundanceModel`, `FeatureLayout`, `LabelTiers`) accept no file-system or config-file dependencies:
- `FeatureLayout.all_required_columns` is computed from frozen tuple indices — no disk I/O.
- `load_arrays` (lines 274–317) is a pure function, not a module-level import of data sources.
- `LabelScaler.fit()` is a class method on immutable value objects.

This permits in-memory unit tests without Parquet fixtures.

---

### 2. **Critical Issue: Asymmetric NaN Sanitisation Boundary**

**Severity**: Medium. Creates hidden coupling between `training.py` and `inference.py` data-loader contracts.

**The Problem**:
- **Training**: `build_dataloaders` (line 154) applies `np.nan_to_num(arrs["X"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)` on the feature matrix **before** constructing Datasets.
- **Inference**: `predict_ensemble` (lines 238–247) applies the same sanitisation **to predictions** (mu_m, L_m) *after* model forward pass, with an assertion guard.
- **XpFeatureAdapter** (adapter.py:117–128) is a **pass-through** for NaN; the docstring (lines 26–29 in `inference.py`) explicitly states it does not guard against NaN/Inf.

**Why This Is Fragile**:
1. The comment in `inference.py:26–29` documents that "NaN sanitisation must occur at the inference driver boundary before the first model forward pass" — but there is no active test of this contract.
2. Training applies `nan_to_num` at the Dataset level; inference applies it at the aggregation level (post-forward). A caller who writes a custom inference driver (not using `predict_ensemble`) and forgets to sanitise before model forward pass will silently produce NaN predictions with no OOD flag raised (Mahalanobis only covers the 108-D XP block per CLAUDE.md footnote).
3. The `XpFeatureAdapter` design intentionally skips NaN-guarding "to keep the adapter a shape-and-mask operation" (adapter.py:75–76), which is good for separation — but it shifts responsibility to callers without a visible contract check.

**Recommended Fix**: Add a shape-preserving `assert` in `XpFeatureAdapter.forward()`:
```python
# adapter.py:117–128
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ADR-0012 compliance: assert finite before masking.
    assert torch.isfinite(x).all(), "XpFeatureAdapter received non-finite input; call nan_to_num upstream (inference.py line 26–29)"
    if self.use_c0_scalars or self._c0_positions.numel() == 0:
        return x
    x_out = x.clone()
    x_out[..., self._c0_positions] = 0.0
    return x_out
```
This trades a micro-overhead for a visible fail-fast at the exact boundary where the contract matters most.

---

### 3. **Strong Orchestration/Computation Separation**

#### Training Loop is Properly Stratified

| Component | Responsibility | Imports |
|-----------|-----------------|---------|
| `train_one_epoch` (528–627) | Batch iteration, optimizer dispatch | `model`, `adapter`, `config` — no I/O |
| `_compute_losses` (432–526) | Loss calculation | Pure tensor ops + `model` forward |
| `build_dataloaders` (96–236) | Parquet read, split, scale fit | Data layer only; returns loaders + scaler |
| `train_model` (660–825) | Epoch loop orchestration, EarlyStopping | Calls `train_one_epoch`, `validate`, checkpoint save |

Each function has a single responsibility. Configuration knobs (`cfg.loss_weights.supcon`, `cfg.early_stop_patience`) are passed in, never captured in closures.

**Exception**: `_first_epoch_sanity_check` (827–898) reaches into the val loader to unscale predictions on epoch 0. This is a domain-specific gate (checking label-scaler alignment) that could live in an `uncertainty.py` validation helper, but it's small (71 lines) and well-documented, so the cost of extracting it exceeds the clarity gain.

#### Inference is Properly Decoupled

| Component | Responsibility |
|-----------|-----------------|
| `load_ensemble` (163–200) | Checkpoint file I/O → `EnsembleMember` list |
| `predict_ensemble` (203–289) | Model forward + calibration + aggregation |
| `_build_model_from_blob` (100–142) | Checkpoint dict → `XpAbundanceModel` rehydration |

`predict_ensemble` does not know about checkpoints — it accepts a ready-made ensemble. This permits testing with in-memory `EnsembleMember` mocks.

---

### 4. **Configuration Encapsulation: Mostly Clean**

**Good**: 
- `TrainingConfig` and `LossWeights` are immutable dataclasses (frozen=True, slots=True). Mutation-free.
- Config is serialised to JSON in checkpoints (`training.py:1023`) for round-trip reproducibility.
- `_config_to_jsonable` (1047–1054) handles Path coercion explicitly.

**Area for Tightening**:
- `TrainingConfig.output_prefix` (line 130) is a string label used only at checkpoint-save time (line 1077). It's a release-workflow dial, not a training hyperparameter. Could live in a separate `ReleaseConfig` to tighten the contract, but the cost of schema splitting exceeds the benefit for a single field.
- `stage_dataset_on_gpu` (line 148) and `checkpoint_every_n_epochs` (line 114) are operational tuning knobs (memory/disk trade-offs), not hyperparameters. They don't affect model behavior, just resource cost. Current placement is acceptable.

---

### 5. **Data Layer Stratification**

#### Feature Contract is Declared, Not Leaked
- `FeatureLayout` (data.py:113–167) declares the encoder input vector order in a single immutable dataclass.
- `all_required_columns` (159–166) is a computed property, not hard-coded.
- Truncation variant (`truncated_43d`) is a classmethod factory — changes are localized.

**Strength**: Any training run using `FeatureLayout` is auditable; the columns and order are visible in the instance.

#### Label Tier Gating is Explicit
- `LabelTiers` (data.py:210–269) defines release tiers with co-commit discipline documented in CLAUDE.md §15.
- `tier_promotion.py` imports `LabelTiers` and uses `tiers.all_labels` to validate consistency — no sneaking around it.

#### load_arrays is Data-Layer-Only
- `load_arrays` (274–317) reads only the columns it needs, no global state.
- Caller must pass `layout` and `tiers` explicitly — no hidden config file.

---

### 6. **Minor: Redundant Imports in __init__.py**

`xp_abundances/main/__init__.py` (lines 8–88) re-exports 48 names from submodules. This is good for discoverability but creates a maintenance burden: adding a new exported function requires edits to both the submodule and the `__all__` list.

**Recommendation** (low priority): Consider a protocol-based auto-export if this grows, but current size is manageable.

---

### 7. **Dependency on data.frozen_stats**

`inference.py:43` imports `from arqueogal.data.frozen_stats import assert_frozen_stats_match`, called at line 229 in `predict_ensemble`.

This is a **cross-cutting concern** (validation of preprocessing state) properly isolated as a precondition check. It belongs in the inference boundary because frozen stats are an inference-time contract (Stream 3 must load v1's per-coefficient z-score stats per CLAUDE.md §16).

---

## Recommendations

### Critical
1. **Add NaN guard in XpFeatureAdapter.forward()** (see Finding 2). Enforce the ADR-0012 contract at the exact boundary where it matters.

### Medium
2. **Explicit test for inference NaN-sanitisation contract**: Add a unit test in `tests/xp_abundances/main/` that verifies `predict_ensemble` handles a deliberately-injected NaN in the feature tensor (before adapter). This validates that the upstream `nan_to_num` requirement is not a silent assumption.

3. **Document adapter's pass-through design in the public __init__.py**: Add a one-line note to the XpFeatureAdapter docstring: "NaN/Inf sanitisation must occur upstream; see inference.py §NaN safety."

### Low
4. **Consider split release config**: Extract `output_prefix`, `checkpoint_every_n_epochs`, `stage_dataset_on_gpu` into a separate operational config if this repo grows to multiple pipelines. Today's 3-field split does not justify the overhead.

---

## Architectural Patterns Assessment

| Pattern | Applied? | Quality | Notes |
|---------|----------|---------|-------|
| **Clean Architecture** (dependencies inward) | Yes | Strong | No cycles, clear layer boundaries. Data/model logic isolated from orchestration and I/O. |
| **Hexagonal (Ports & Adapters)** | Partial | Good | Checkpoint versioning and calibration are clear ports. Feature preprocessing (adapter) is a pass-through. No full port/adapter hierarchy needed for a scientific codebase. |
| **DDD (Value Objects, Aggregates)** | Partial | Good | `FeatureLayout`, `LabelTiers`, `LabelScaler` are immutable value objects. `XpAbundanceModel` is the aggregate root. No domain events (appropriate for non-event-driven domain). |
| **Test Seams** | Yes | Strong | Domain classes accept no I/O; in-memory `LabelScaler.fit()`, `FeatureLayout` computation, and adapter logic are all testable without fixtures. |
| **Configuration as Data** | Yes | Good | Config is immutable dataclasses, serialized to JSON in checkpoints, never injected into domain classes. |

---

## Conclusion

The codebase demonstrates **disciplined architectural thinking**. The primary risk is not structural but operational: the NaN-sanitisation contract between training and inference is sound in design but invisible at the boundary where it matters most. Adding a single assertion in `XpFeatureAdapter.forward()` elevates this from an implicit documentation-only rule to a runtime check, improving robustness without increasing complexity.

No refactoring of the existing layering is needed. The separation of domain, orchestration, and infrastructure is clean and intentional.
