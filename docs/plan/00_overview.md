# ArqueoGal plan — overview

Last updated: 2026-04-22. Pipeline 1 v1 shipped 2026-04-19 but exhibited an
[α/M]≈+0.11 attractor-stripe on metal-poor stars (chemistry bimodality
collapse). Active work is a joint-loss architecture port from TESS_ML —
single-stage `supcon + beta_nll + barlow_twins` replacing the two-stage
contrastive→supervised workflow. The port preserves the disc bimodality on
the Stream-1 validation split (attractor-stripe fraction 6.27% vs v2 ~72%).
Population classification (formerly "Pipeline 2") has been spun out into the
separate **Starfold** repository and is out of scope here; this repo's remit
ends at Pipeline 1 predictions (see `04_pipeline2_main.md`).

## Deliverables and dates

| Code | Title | Due | Status |
|---|---|---|---|
| D-Cat-b | XP abundance catalogue (supporting contribution) | Aug 2026 | Pipeline 1 v1 shipped; inference catalogue pending Stream 3 expansion |
| D5.1 | Open-source ML tool for stellar-population classification | Dec 2026 | Delivered by Starfold (separate repo); consumes this repo's Pipeline 1 predictions |
| D-Cat-d | Stellar-population membership probabilities | Feb 2027 | Delivered by Starfold; blocked on Pipeline 1 inference output here |

External dependency: Task 4 (Campante/Miglio team) asteroseismic ages, expected late 2026.
Starfold trains with `age=null` until Task 4 delivers.

## Phase status

| File | Phase | Status |
|---|---|---|
| `01_pipeline1_v1.md` | Pipeline 1 v1 production model | **Shipped but superseded** — tagged `pipeline1-v1-2026-04-19`; chemistry bimodality collapse motivated rebuild |
| `02_pipeline1_audit.md` | §9.2 information-content audit | **Partial** — 5/6 tests done + tier decisions ratified; test 6 (cross-catalogue) blocked on Stream 3 |
| `03_stream3_inference.md` | Stream 3 expansion + Pipeline 1 inference | **In progress** — Phase 2 prerequisites complete; Phase 3 gated on joint-loss ensemble |
| `04_pipeline2_main.md` | Population classification (D5.1 deliverable) | **Moved to Starfold** — separate repo; this file is now a pointer / integration-contract stub |
| `05_release_packaging.md` | D-Cat-b, D5.1, D-Cat-d packaging | **Underspecified** — release-format decisions deferred |
| `06_methods_paper.md` | Methods paper (parallel to deliverables) | **Tracked, not actively drafted** |

## Current work (joint-loss rebuild in flight as of 2026-04-22)

- Two-stage pipeline (run_contrastive_pretrain → run_ensemble) retired.
  Single-stage `run_joint_ensemble.py` is the new production driver
  (139-D FeatureLayout + use_c0_scalars=True + encoder_lr_ratio=1.0).
- 5-seed ensemble training in flight (~3.5h on RTX 3060) at
  `models/main/xp_abundances/20260422_38a993e_1727825_joint/`.
- Compat patches landed: `run_calibration.py` and
  `build_pipeline1_val_predictions.py` both handle
  `pretrained_encoder_ckpt=None` (joint checkpoints have no pretrain
  stage). `run_pipeline1_inference.py` was already joint-compatible
  (keys on checkpoint `input_dim`). §9.2 audit driver was already
  compatible (sets None explicitly).
- Retired scripts kept in-tree for git history but not on the critical
  path: `run_contrastive_pretrain.py`, `run_supervised_finetune.py`,
  `run_ensemble.py`. Old v1.x diagnostic scripts (10 files) reference
  `pretrained_encoder_ckpt` and will break on joint checkpoints —
  re-patch only when re-invoked for the methods paper.
- Gallery stages 10-14 cleared (READMEs preserved). Stage 10 is now
  intentionally empty (no pretrain stage in joint architecture).

## Known blocking/pending items

- **Test 6 (cross-catalogue consistency)** closes §9.2. Requires Pipeline 1 predictions
  on Stream 3 stars overlapping with AspGap / Guiglion+2024 / SHBoost. Gated on Phase 3.
- **Starfold feature matrix** cannot be built downstream until Stream 3 inference here
  produces per-star labels + covariance + OOD flags + selection_prob.
- **Kinematics (galpy actions)** is a small module (~3–5 days of work); per the
  integration contract in `04_pipeline2_main.md`, it may be exposed as a utility for
  Starfold or duplicated there. Choice deferred.
- **Task 4 age column** waits on Campante/Miglio team; no action needed here.

## Needs clarification

- D5.1 release logistics now live in Starfold's own repo planning, not here.
- Methods-paper venue, timeline, and scope. "Methods paper" appears in memory as
  deserved content but has no timeline or author list written down.
