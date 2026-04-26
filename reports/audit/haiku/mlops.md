# MLOps Audit Report: ArqueoGal — April 26, 2026

## Executive Summary

ArqueoGal implements a **strong foundation for production ML governance** with checkpoint versioning, comprehensive data provenance, and frozen preprocessing stats. However, **experiment tracking is opt-in-only (aim smoke test, not production pipeline), model cards and reproduction guides are absent, and there is no structured gating for experimental→main promotion**.

## Inventory by MLOps Layer

### Experiment Tracking

**Status: Partially implemented, non-canonical**

- **Aim installation:** Present (`.aim/` database initialized on 2026-04-24, smoke test script at `scripts/aim_smoke_test.py`)
- **Integration into training pipeline:** Missing. The smoke test is a toy 3-run example; `src/arqueogal/xp_abundances/main/training.py` does not call any aim APIs. Experiment tracking is not wired into the production training loop.
- **Policy alignment:** CLAUDE.md specifies "opt-in via env var, JSON-on-disk canonical, never aim/wandb as canonical source of truth." Current state: aim is not even opt-in; it exists but is unused.
- **Implication:** Training runs produce checkpoints and JSON provenance for data artifacts, but hyperparameter sweeps, loss curves, and per-epoch diagnostics are not logged to any central system (neither disk nor aim).

### Model Checkpoints & Registry

**Status: Robust structure, incomplete metadata**

**Checkpoint storage:**
- Location: `/models/main/xp_abundances/{date}_{git-sha7}_{variant}/member_seed{N}/`
- Format: PyTorch `.pt` files with embedded metadata (version, config YAML, training metrics, label scaler, tier map, git SHA, random seed)
- Validation: Checkpoints load under `torch.load(..., weights_only=False)` and pass version check
- Atomic writes: Implemented via temp-file → rename in `utils/io.py`

**Example from production:** `/models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label/member_seed0/`
- Contains: single `.pt` file (424 KB)
- Checkpoint contents verified: version 2, git SHA `6b96c06...`, 5 label names, training history with 30 epochs, best validation loss 0.888
- Missing: README describing what this ensemble is, how to load it, reproduction instructions

**Tier map and label scaler:** Embedded in checkpoint, no separate reference card.

**Provenance for data artifacts:**
- Implemented: `pipeline1_predictions_stream3.parquet.provenance.json` (comprehensive, 216 lines)
- Covers: input SHA-256, ensemble member hashes, frozen stats fingerprint, OOD criteria, tier assignments, label release notes
- Missing for checkpoints: equivalent sidecar `.provenance.json` file alongside the `.pt` file

### Model Promotion & Gating

**Status: Policy defined, gating not automated**

- **Separation:** Main vs experimental code paths exist in `src/arqueogal/{module}/main/` and `src/arqueogal/{module}/experimental/`
- **Documentation:** `CLAUDE.md §2` states promotion requires "beating main by a documented margin"; `docs/research_brief.md §11` defines the bar
- **Enforcement:** Manual review. No automated tests (e.g., baseline comparison, significance threshold, cross-catalog consistency) before promotion
- **Audit test 6 (tier_promotion):** Marked as stub in CLAUDE.md; when promoting, 5 of 6 tests are covered
- **Implication:** Promotion decisions are peer-reviewed but tracked informally (commit messages, release notes). No structured gate prevents a below-threshold model from entering main.

### Data Versioning & Frozen Preprocessing

**Status: Excellent**

- **Frozen stats:** Stream 1 per-coefficient Hermite z-score stats locked with basis fingerprint `0d34b565...` (referenced in CLAUDE.md §16)
- **Stream 3 inference:** Explicitly loads v1 frozen stats, never refits
- **Input versioning:** Data layer emits provenance JSON with input parquet SHA-256 (e.g., `a0b18888...` for `pipeline1_features_stream3.parquet`)
- **Corrections locked:** Ye+2024 XP correction, Lindegren+2021 parallax zpt, Riello+2021 G-mag correction all marked as mandatory at ingestion (CLAUDE.md §11–13)

### Reproducibility & Run Documentation

**Status: Partial**

**What's reproducible:**
- Git SHA embedded in checkpoint and data provenance
- Training config (hyperparameters, architecture, loss weights) serialized as JSON inside checkpoint blob
- Label scaler and calibration state preserved
- Training metrics (loss, metrics per epoch, best epoch) included

**What's missing:**
- **Model card:** No README in checkpoint directory describing:
  - What this model is (v1.0 ensemble? checkpoint #43 in a sweep?)
  - Data it was trained on (which parquet, row count, version)
  - Expected performance (RMSE by label, comparison to aux-only baseline)
  - Known limitations (Regime B exclusion, mode ambiguity bimodality caveat for α/M)
  - How to load and use it (torch.load() call, feature ordering, units)
  - Authors and funding attribution

- **Run config snapshot:** Checkpoint embeds `config_yaml` (good), but no separate `run_config.json` in the model directory capturing:
  - Full Python lock file (uv.lock or requirements-lock.txt) at training time
  - Dataset split seed and train/val row counts
  - GPU model, CUDA version
  - Aim or MLflow run ID (for link-back to experiment logs, once aim integration lands)

- **Reproduction instructions:** No `REPRODUCE.md` in `/models/main/xp_abundances/{run_id}/` showing:
  - How to reinstantiate the exact environment
  - How to load the checkpoint and apply it to new data
  - How to decode the covariance matrix (upper triangle ordering, label block order)

### Data-Model Traceability

**Status: Strong for inference, incomplete for training**

**Data→Model arrows:**
- Inference provenance (pipeline1_predictions_stream3.parquet.provenance.json) names the exact checkpoint directory and file hashes
- Frozen stats fingerprint cross-referenced
- OOD and regime-B flags documented

**Model→Data arrows (training):**
- Checkpoint embeds git SHA (can look up commit)
- Training config in checkpoint includes feature layout and label names
- No embedded link to the exact training parquet SHA-256 or row count
- Historical gap: Many training runs from 2026-04-19 named `20260419_nogit_*` — git SHA is missing, making reproduction impossible

### Infrastructure & Scaling

**Status: Single-machine, reproducible**

- **Hardware:** RTX 3060 6 GB VRAM (sequential ensemble training, gradient accumulation)
- **Training orchestration:** Not present. Each run invoked via shell script or manual `python training.py --config ...`
- **No Kubernetes, no DAG orchestration, no distributed training:** ArqueoGal trains ensembles sequentially on a single GPU
- **Logging:** JSON on disk (canonical), aim (non-canonical, not yet wired)
- **No secrets management, no CI/CD for model training, no automated retraining triggers**

## Gaps Relative to CLAUDE.md Policy

| Policy Statement | Status | Gap |
|---|---|---|
| "JSON-on-disk in models/main/… is canonical" | Partial | Data provenance JSON exists; checkpoint provenance sidecar missing |
| "env var opt-in for aim/wandb" | Incomplete | aim installed but not integrated; no env var gating |
| "Never aim/wandb as canonical source" | Compliant | Checkpoints are self-contained; aim would be supplementary |
| "Provenance sidecars are non-negotiable" | Partial | Data artifacts have sidecars; models do not |
| "DESIGN.md co-commit discipline" | Compliant | Schema updates tracked in design docs |
| "Frozen stats across runs" | Compliant | Stream 3 uses frozen v1 stats |

## Specific Findings

### Production Checkpoint Inspection
**Path:** `/models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label/member_seed0/xp_abundances_main_ensemble_5label_seed0_best.pt`

- **Checkpoint version:** 2 (correct)
- **Git SHA:** `6b96c06c8d6925939a3a102427a83d50afd97142` (full, embedded correctly)
- **Labels:** 5 (teff, logg, mh, alpha_m, mg_h) with correct names
- **Training metrics:** 30 epochs, best validation loss 0.8876 at epoch 29, full loss curve included
- **Tier map:** Embedded, all labels → Tier 1 or T1-caveat
- **Missing:** Any README or .provenance.json sidecar alongside the checkpoint

### Historical Runs Without Git SHA
- `20260419_nogit_*` naming pattern (9 checkpoints, 2026-04-19)
- Impossible to reproduce: git SHA replaced with `nogit`
- Pattern corrected by 2026-04-21 (runs named `20260421_38a993e_...`)
- Implication: Early April checkpoints are non-reproducible without external notes

### Experiment Tracking Setup
- Aim database initialized but unpopulated
- Smoke test exists (`aim_smoke_test.py`) with 3 toy runs
- No wiring into `src/arqueogal/xp_abundances/main/training.py`
- Conclusion: Scaffolding present, integration not done

## Recommendations (Priority Order)

### Immediate (Before Release)
1. **Add checkpoint provenance sidecars** (`.provenance.json` for each `.pt` file):
   - Input feature parquet SHA-256 and row count
   - Exact Python lock file (uv.lock snapshot)
   - Training hyperparameters (learning rate, batch size, epochs actually run)
   - Hardware used (GPU model, CUDA version)
   - Tie to inference provenance via git SHA

2. **Write model cards** (README in each checkpoint directory):
   - What the model is (Pipeline 1 v1 ensemble, trained 2026-04-25)
   - Expected performance (aux-only baseline 164 K vs. 67 K for Teff, etc.)
   - Caveat for Regime B and mode ambiguity
   - Label block order and tier assignments
   - Load/apply instructions with code snippet

3. **Add promotion gate automation:**
   - Define success criteria (aux-only baseline comparison, permutation importance, cross-catalog consistency)
   - Block promotion if criteria not met
   - Log gate results in the model directory

### Medium-term (Before Next Major Release)
4. **Integrate aim into training loop** (opt-in via env var):
   - Wire loss/metrics logging to `aim.Run()` if `AIM_ENABLED=1`
   - Store run ID in checkpoint for traceability
   - Document aim query patterns for post-hoc analysis

5. **Backfill missing training metadata:**
   - Retroactively link April 2026 `nogit` checkpoints to git SHA (via commit timestamps)
   - Or mark as non-reproducible in a deprecation note

6. **Add reproduction workflow:**
   - `.github/workflows/reproduce_checkpoint.yml` to reinstantiate environment from uv.lock
   - Test checkpoint loading on fresh environment monthly

### Long-term
7. **Consider DAG orchestration** if ensemble sizes grow or pipeline stages become more complex (Prefect or Dagster, not Airflow for academic use)
8. **Add automated retraining trigger** when new data arrives (e.g., via GitHub Actions on data commit)

## Compliance Assessment

| Dimension | Score | Notes |
|---|---|---|
| Checkpoint versioning | 9/10 | Embedded metadata robust; missing sidecar |
| Data provenance | 10/10 | Comprehensive, links to frozen stats |
| Reproducibility | 6/10 | Git SHA present; frozen stats locked; missing training lock file snapshot |
| Experiment tracking | 4/10 | Infrastructure present; integration incomplete |
| Promotion gates | 5/10 | Policy documented; not automated |
| Model cards | 2/10 | None present |
| Production readiness | 7/10 | Can reproduce checkpoint load; cannot reproduce training environment |

## Conclusion

ArqueoGal achieves **high reproducibility for inference** (frozen stats, checkpoint contents, data provenance) and **low operational overhead** (single-machine training, JSON storage). However, it lacks **human-readable documentation** (model cards, load instructions) and **training environment capture** (lock file snapshots, hardware specs). Before releasing D-Cat-b (August 2026), add checkpoint provenance sidecars and model card templates; before D5.1 (December 2026), integrate aim logging opt-in.

The policy in CLAUDE.md is sound; execution is 70% complete. Priority is bridging the documentation and training-metadata gaps rather than adopting new tools.
