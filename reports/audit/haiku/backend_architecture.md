# Backend Architecture Audit — ArqueoGal Data Pipeline

Date: 2026-04-26
Scope: Service boundary definition, config/data flow, layer separation (data acquisition → feature engineering → training → inference → release)

---

## Executive Summary

The pipeline has clean conceptual boundaries (ingestion, feature build, training, inference, release) with well-documented responsibilities. However, three systematic leaks undermine modularity: (1) **script-layer orchestration logic duplicated in package code**, blurring responsibilities between CLI drivers and reusable modules; (2) **config objects created ad-hoc by scripts rather than injected**, forcing downstream modules to reconstruct context; (3) **data acquisition layer (TAP, downloads) mixed with transactional logic**, making it difficult to mock or integrate externally. The core data-layer modules are solid (ingest_stream1/2/3, gaia_corrections, provenance) but orchestration scripts bypass these in favor of inline pandas operations that could/should delegate. The training/inference boundary is cleanly separated. Release annotation layers have sound contracts (release.py §3.3 tier protocol) but rely on feature parquets not exposed through the data layer.

---

## Service Boundaries (Intended)

As documented in `docs/context/architecture.md`, the pipeline defines six logical services:

1. **Data Acquisition** (`src/arqueogal/data/tap.py`, `downloads.py`)
   - TAP/HTTPS fetches, credentials, async batching
   - Contract: URL + ADQL → Parquet + provenance sidecar

2. **Data Ingestion** (`ingest_stream{1,2,3}.py`)
   - Join multiple sources (APOGEE, Gaia, Andrae+2023), apply corrections
   - Contract: raw-data paths → stream-specific parquets + sidecars
   - Per-stream, resumable, checkpoint-aware

3. **Feature Engineering** (`build_pipeline1_features_*.py`, `emit_*_with_hermite.py`)
   - Photometry, astrometry, extinction, kinematics assembly
   - XP preprocessing: Ye+2024 → sampled flux → Hermite reproject → z-score
   - Contract: stream-ingested parquets → feature matrices

4. **Training Orchestration** (`run_contrastive_pretrain.py`, `run_ensemble.py`)
   - Model, loss, optimizer, ensemble seeds
   - Contract: feature matrix + config → checkpoint bundle

5. **Inference Driver** (`run_pipeline1_inference.py`)
   - Load ensemble, frozen stats, apply z-score, run batch prediction, flag OOD/Regime-B
   - Contract: ensemble checkpoint + feature matrix + frozen stats → predictions + OOD flags

6. **Release Annotation** (`release_pipeline.py`, `release.py`)
   - Join predictions × features, assign tier, emit FITS/VOTable
   - Contract: predictions + features → annotated catalog + derivatives

---

## Identified Boundary Leaks

### 1. Orchestration Logic Duplicated in Scripts vs. Package (Severity: Medium)

**Problem**: Each `build_pipeline1_features_stream*.py` script reimplements column selection, rename logic, and validation inline rather than delegating to reusable module functions.

**Examples**:

- **Columns and renames**: `build_pipeline1_features_stream1.py:62–100` defines `_APOGEE_RENAMES` dict (28 entries) as inline module-level constant. Identical or near-identical logic for Stream 2 and Stream 3.
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/build_pipeline1_features_stream1.py:62–100`
  - Should be: `src/arqueogal/data/master_schema.py` or `schemas.py` with accessor functions `get_apogee_renames()`, `get_feature_subset_columns()`.

- **Stream-specific filters**: Each build script has its own thresholds:
  - RGB window filters at `build_pipeline1_features_stream1.py:48–59` (Teff 4000–5500 K, log g 1.0–3.5).
  - Duplicated logic for Stream 3 with no shared constant.
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/build_pipeline1_features_stream3.py` (no explicit RGB window; inferred from Andrae+2023 pre-filtering).
  - Should be: `src/arqueogal/data/selection_function.py` or `src/arqueogal/utils/label_conventions.py` exporting `PIPELINE1_RGB_WINDOW = {"teff_min": 4000.0, ...}`.

- **Provenance sidecar emission**: Every script (build_stream1_apogee_gaia.py, build_pipeline1_features_*.py, emit_*.py) reimplements the same provenance-writing pattern.
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/build_stream1_apogee_gaia.py:104–143` vs. `build_pipeline1_features_stream1.py` (line 400+).
  - Should be: `src/arqueogal/data/release_pipeline.py` already has provenance logic; factor into a generic `write_pipeline_step_provenance(step_name, inputs, outputs)` helper.

**Impact**: Schema changes (e.g., renaming Mg/H to Mg_H) require edits in 3+ scripts. Typos in stream-specific filters are not caught until late (e.g., hardcoding RGB window in Stream 1 but forgetting to document for Stream 3).

**Recommendation**: 
- Move `_APOGEE_RENAMES` and other schema constants to `src/arqueogal/data/master_schema.py`.
- Expose accessor functions: `get_label_renames(stream: int) -> dict[str, str]`, `get_feature_subset_cols(stream: int) -> list[str]`.
- Create a helper `write_feature_provenance(step_id: str, ...)` that all build scripts call with minimal boilerplate.

---

### 2. Config Objects Created Ad-Hoc Rather Than Injected (Severity: Medium)

**Problem**: Training and inference drivers construct config objects inside scripts by assembling arguments, then passing to package functions. Downstream modules cannot easily use alternative configs or be called in different contexts (e.g., from notebooks or batch jobs).

**Examples**:

- **Training config**:
  - `run_ensemble.py:82–110` defines `build_ensemble_config()` which assembles `TrainingConfig` from CLI arguments.
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/run_ensemble.py:82–110`
  - The function is internal to the script; it's not in `src/arqueogal/xp_abundances/main/`.
  - If another tool (e.g., a hyperparameter sweep) wants to call `train_ensemble()` with varied configs, it must either duplicate `build_ensemble_config()` or manually construct `TrainingConfig`.

- **Inference setup**:
  - `run_pipeline1_inference.py` loads frozen stats from provenance JSON inline (lines ~40–60).
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/run_pipeline1_inference.py` (inspect at ~130–160 for stat loading).
  - The `load_ensemble()` and `predict_ensemble()` functions in `src/arqueogal/xp_abundances/main/inference.py` don't take frozen stats as a dependency; the script computes them outside and passes raw arrays.
  - Consequence: calling `inference.predict_ensemble()` from a test requires pre-loading stats in the same way as the script does, duplicating the logic.

- **Feature layout**:
  - `FeatureLayout` is instantiated fresh in each training script (e.g., `run_ensemble.py`, `run_calibration.py`).
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/run_calibration.py:73–74` (example).
  - The layout is a config-like object that should be versioned alongside the checkpoint; instead, scripts rely on the `FeatureLayout()` default or hardcode tweaks inline.

**Impact**: Every new entry point (notebook, batch job, remote inference service) must reimplement config assembly, risking inconsistencies. Tests cannot inject mock configs without importing from scripts.

**Recommendation**:
- Extract `build_ensemble_config()` from `run_ensemble.py` to `src/arqueogal/xp_abundances/main/config.py` and expose it.
- Create a `FrozenStatsLoader` class in `src/arqueogal/xp_abundances/main/inference.py` that encapsulates sidecar reading and validation.
- Add `from_checkpoint(ckpt_path: Path) -> FeatureLayout` factory method to `FeatureLayout` so downstream code can load the layout that was baked into a checkpoint.

---

### 3. Data Acquisition Layer Not Abstracted from Ingestion (Severity: Medium-High)

**Problem**: Raw TAP/HTTPS fetches and credential handling are entangled with business logic (corrections, joins, renames). New data sources or modified acquisition logic cannot be plugged in without editing existing ingestion modules.

**Examples**:

- **Gaia enrichment TAP call**:
  - `ingest_stream1.py:47` imports `enrich_source_ids()` from `gaia_enrich.py`.
  - File: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/ingest_stream1.py:47`
  - The enrichment function takes a `TAPService` parameter (default None, meaning it creates one internally via `aip_service()`).
  - File: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/gaia_enrich.py` (inspect for service creation).
  - If you want to swap AIP TAP for a different endpoint (e.g., ESA Archive for lower latency), you must edit `gaia_enrich.py`'s import or add conditional logic.

- **Ye+2024 NN model**:
  - `gaia_xp.py` loads the NN weights from a vendored Zenodo archive.
  - File: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/gaia_xp.py` (inspect for weight loading).
  - The path and checksum are hardcoded; no way to provide alternate weights (e.g., future improved NN) without code change.

- **Download resumption**:
  - Each ingestion script calls `download()` and `ingest_stream*()` sequentially, but if a download fails mid-stream, recovery is manual.
  - File: `/home/aneitzel/projects/ArqueoGal/scripts/build_stream1_apogee_gaia.py:57–65` (checks file existence, doesn't handle partial downloads).

**Impact**: Integrating new data sources (e.g., DR4 XP) or external APIs (e.g., another parallax correction) requires deep edits to ingestion logic. Testing with mock data is cumbersome because the acquisition layer isn't separated.

**Recommendation**:
- Define a `DataSourceContract` protocol/ABC in `src/arqueogal/data/sources.py` with methods: `fetch(ids: list[int]) -> DataFrame`, `get_schema() -> dict`, `validate()`.
- Implement concrete sources: `GaiaXpSource`, `ApogeeDr19Source`, `GaiaDr3Source`, each with pluggable endpoints.
- Refactor `ingest_stream1()` to accept a dict of sources (e.g., `{"apogee": ApogeeDr19Source(...), "gaia": GaiaDr3Source(...)}`).
- Move credential management to a separate `src/arqueogal/data/auth.py` module so scripts don't import from multiple places.

---

### 4. Release Annotation Relies on Implicit Feature Column Contract (Severity: Low-Medium)

**Problem**: `release_pipeline.py` joins predictions × features on `source_id` and hardcodes which feature columns to include (photometry, parallax, distance). If the feature matrix schema changes, the release pipeline must be edited.

**Examples**:

- **Feature join columns**:
  - `release_pipeline.py:54–78` defines `_FEATURE_JOIN_COLS` tuple hardcoded in the script.
  - File: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/release_pipeline.py:54–78`
  - These columns are not declared in `master_schema.py` or `FeatureLayout`; they're a separate contract.
  - If Stream 3 feature builds add a new column (e.g., `gaia_xp_flag`), release annotation won't include it until someone edits `_FEATURE_JOIN_COLS` by hand.

- **Tier logic**:
  - `release.py:56–88` hardcodes OOD flags and per-element caveat flags.
  - File: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/release.py:56–88`
  - The set of flags is well-documented but not linked to the feature matrix schema; if a new flag is added upstream (e.g., `new_ood_flag`), the tier-promotion logic must manually include it.

**Impact**: Schema evolution (adding photometric flags, new distance priors) requires edits in two places: the feature-build scripts and the release module. Risk of missed updates leading to silent data loss.

**Recommendation**:
- Add a `release_required_feature_cols() -> list[str]` function to `master_schema.py` that lists all columns the release pipeline needs.
- Replace `_FEATURE_JOIN_COLS` in `release_pipeline.py` with a call to this function.
- Add a `release_tier_flags() -> tuple[str, ...]` function exporting the definitive list of flag columns the tier logic checks.

---

## Layer Separation: Observations

### Data Layer (Ingestion, Correction, Provenance) — Sound

- `ingest_stream{1,2,3}.py` modules have single responsibility and clear return contracts.
- Corrections (Lindegren, Riello, Ye, Mészáros) are factored into dedicated modules (`gaia_corrections.py`, `gaia_xp.py`, `apogee_dr19.py`).
- Provenance is consistently written via `write_sidecar()`.
- Strengths: well-documented, checkpointed, auditable.

### Feature Engineering (Build, Emit) — Moderate Issues

- **Strength**: Hermite reproject → z-score pipeline is encapsulated in `gaia_xp.py`.
- **Weakness**: Column selections and renames live in scripts, not in reusable schema definitions.
- **Weakness**: XP preprocessing order (Ye → normalize c0 → log + z-score coef 0 → per-coef z-score) is enforced via inline comments and script order, not via a state machine or config.
- **Improvement**: Create a `XpPreprocessingPipeline` class in `data/gaia_xp.py` that enforces order and exposes each step.

### Training (Config, Model, Loss, Ensemble) — Clean

- Config objects are well-structured dataclasses (`TrainingConfig`, `LossWeights`).
- Model architecture and loss are separated: `model.py` vs. `losses.py`.
- Ensemble training is orchestrated in `training.py:train_ensemble()` with clear checkpointing.
- **Minor**: Config objects are created in scripts (see leak #2 above), but the package side is clean.

### Inference (Ensemble, OOD, Tier) — Clean

- `inference.py` has no torch imports outside the prediction loop; GPU/CPU agnostic.
- OOD flags (Mahalanobis, disagreement) are computed in dedicated `ood.py` module.
- Tier assignment logic (`release.py`) is independent of prediction code.
- **Minor**: Frozen z-score stats are loaded by the script, not provided as a dependency (see leak #2).

### Release (Pipeline, Artefacts, Gallery) — Adequate but Implicit

- `release_pipeline.py` is the canonical orchestrator for predictions + features join + tier assignment.
- `release_artefacts.py` handles FITS/VOTable export.
- **Weakness**: The feature columns needed are hardcoded, not schema-driven (see leak #4).

---

## Cross-Cutting Concerns

### Logging

- All data-layer modules use structured logging (`logging.getLogger(__name__)`).
- Scripts log to console via basicConfig.
- No correlation IDs across scripts; difficult to trace a full pipeline run.
- **Minor recommendation**: Add a `@log_provenance` decorator that emits a per-run UUID at the start of each script.

### Error Handling

- Data-layer modules raise specific exceptions (`ConfigValidationError`, `FrozenStatsMismatchError`).
- Scripts use `SystemExit` for missing files; not caught/retried.
- **Minor recommendation**: Expose a `RetryableDataError` for transient failures (network, TAP timeout); let orchestration layers decide retry strategy.

### Testing

- Unit tests mirror `src/` structure; integration tests are minimal.
- Scripts have no test entry points (e.g., no `--smoke-test` flag to validate on small data).
- **Minor recommendation**: Add a `--dry-run` or `--sample-n 100` option to each script for validation before full runs.

---

## Summary of Recommended Refactors (Priority Order)

1. **Extract schema constants** (`master_schema.py` gets column renames, RGB window, feature subsets, release flags).
   - Effort: Low (2–3h).
   - Payoff: Eliminates duplication across 5 scripts; enables schema-driven validation.

2. **Inject config into package functions** (move `build_ensemble_config()` to package; add `FrozenStatsLoader` class).
   - Effort: Medium (4–6h).
   - Payoff: Package functions become callable from tests/notebooks without script reimplementation.

3. **Abstract data acquisition layer** (sources.py with protocol; refactor `ingest_stream1()` to accept source dict).
   - Effort: High (8–12h).
   - Payoff: Enables swap-in data sources; facilitates mocking for tests; future-proofs against API changes.

4. **Link release feature columns to schema** (add `release_required_feature_cols()` accessor in `master_schema.py`).
   - Effort: Low (1–2h).
   - Payoff: Release pipeline resilient to feature-schema changes; single source of truth.

---

## Non-Issues (Design Trade-offs Respected)

- **No microservices abstraction**: Correct for a research pipeline. The orchestration scripts are the right layer for this.
- **No config files in git**: Correct; configs are baked into scripts and checkpoints. Parameterization via CLI args and env vars is sufficient.
- **Mixing provenance into data modules**: Correct; provenance is a first-class concern here, not an after-thought.
- **XP preprocessing order enforced by script sequence**: Acceptable given the documented constraints in `CLAUDE.md` §12 ("XP preprocessing order is fixed"). A state machine would be over-engineering.

---

## Conclusion

The pipeline is **well-designed at the module level** (clean layer separation for training/inference, sound provenance discipline). The **script layer** has systematic issues: column logic, config assembly, and data acquisition are duplicated or implicit rather than exposed as reusable contracts. These are not architectural flaws but **local refactoring opportunities** that would improve maintainability and testability without changing the overall system structure. The **recommendations above can be staged**: focus on leak #1 (schema constants) and #2 (config injection) for quick wins; defer #3 (data abstraction) to a post-release sprint if modularity becomes critical.
