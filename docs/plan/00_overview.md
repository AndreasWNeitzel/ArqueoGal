# ArqueoGal plan, overview

Last updated: **2026-04-29**

The 2026-04-29 architectural changes unified the preprocessing pipeline across all three streams, extended Stream 2 to fetch XP spectra (fixing a load-bearing bug), switched the production model from 5-label to 21-label output with a single-model (not ensemble) architecture, and added 4-way evolutionary-stage diagnostic and ARI contamination loss with per-element feature-noise marginalisation. Stream 2 (Hon+2021 TESS asteroseismic giants) is the D-Cat-b MVP; Stream 3 is a secondary follow-on.

Pipeline 1 v1 shipped 2026-04-19 but exhibited an [α/M] ≈ +0.11 attractor-stripe on metal-poor stars (bimodality collapse). The 2026-04-29 rebuild (v1.1, 21-label single model) is in active training.

Population classification (formerly "Pipeline 2") has moved to the separate **Starfold** repository; this repo's remit ends at Pipeline 1 predictions.

## Deliverables and dates

| Code | Title | Due | Status |
|---|---|---|---|
| D-Cat-b | XP abundance catalogue + Stream-2 kinematic catalogue (supporting contribution) | Aug 2026 | Pipeline 1 v1 shipped; Stream-2 builder shipped 2026-04-29; full Stream-2 chemistry inference + ablation pending |
| D5.1 | Open-source ML tool for stellar-population classification | Dec 2026 | Delivered by Starfold (separate repo); consumes this repo's Pipeline 1 predictions |
| D-Cat-d | Stellar-population membership probabilities | Feb 2027 | Delivered by Starfold; blocked on Pipeline 1 inference output here |

External dependency: Task 4 (Campante/Miglio team) asteroseismic ages,
expected late 2026. Starfold trains with ``age=null`` until Task 4 delivers.

## Phase ordering — the 2026-04-29 re-sequence

The previous order put Stream-3-scale inference (Andrae+2023 pool, ~650 k
stars) before the actual MVP. The brief re-read of 2026-04-29 corrected
this: D-Cat-b's primary requirement is **the Stream-2 (Hon+2021 TESS
asteroseismic-giant) catalogue with full astrometric / kinematic /
chemical content**, with Stream-3-scale generalisation a secondary
follow-on. The phase order is therefore:

| Phase | File | Goal | Status |
|---|---|---|---|
| **A (v1.1)** | `01_pipeline1_v1.md` | Pipeline 1 v1.1 production model: 21-label single model trained on Stream 1 (APOGEE DR19 × Gaia DR3). Includes 4-way evolutionary-stage diagnostic head, ARI contamination loss (weight 0.1), training-time feature-noise injection (100 epochs), inference-time analytical feature-noise marginalisation. | **Training active (2026-04-29)** |
| **A.audit** | `02_pipeline1_audit.md` | §9.2 information-content audit on Pipeline-1 predictions (21-label, v1.1). | **Framework ready; re-run pending v1.1 checkpoint** |
| **B** *(MVP target)* | `B_stream2_kinematic.md` | **Stream-2 kinematic catalogue.** Hon+2021 × Gaia DR3 → BJ21 distances + dust-map fusion + extinction correction (CCM89 R_V=3.1 + Yuan+2013) + galpy actions. XP coefficients now fetched (bug fix). Builder + tests shipped 2026-04-29. | **Code shipped, production run pending** |
| **C** | (Pipeline-1 inference on Stream 2) | Run the v1.1 21-label model on the Stream-2 kinematic catalogue. Emits 21 predictions + 21×21 block-Cholesky covariance + per-element release tiers + OOD flags. Feature-noise marginalisation applied at inference. | **Pending Phase A v1.1 checkpoint + Phase B kinematic parquet** |
| **D** | (Stream-2 cross-catalogue validation) | Run §3.3 Test 6 cross-catalogue framework on the Stream-2 v1.1 prediction parquet against AspGap / Guiglion+2024 / SHBoost / GALAH DR4 overlap. Generates methods-paper Figure-7. | **Pending Phase C** |
| **E** *(secondary)* | `03_stream3_inference.md` | Stream-3-scale generalisation: ~650 k Andrae+2023-pool stars (uniform-stratified + volume-limited). RGB+HeCB-filtered. Run v1.1 model on full pool. | **Pending Phase D acceptance** |
| **release** | `05_release_packaging.md` | D-Cat-b parquet packaging (Stream-2-primary, Stream-3-secondary) with v1.1 21-label + covariance schema. | **Underspecified**: release-format decisions deferred |
| **paper** | `06_methods_paper.md` | Methods paper (parallel to deliverables). Methods for v1.1: feature-noise marginalisation, diagnostic heads, loss design. | **Tracked, not actively drafted** |

The previous overview file's phase status entries 03 and 02 are
preserved but **demoted** in priority: §9.2 cross-catalogue Test 6
finishes after Phase D (when Stream-2 predictions land); Stream-3
generalisation runs in Phase E only after Stream-2 has cleared
validation.

## What changed today (2026-04-29)

**Preprocessing unification:**
- `src/arqueogal/data/preprocessing.apply_pipeline1_preprocessing()` is now the single source of truth for all three streams. All three call this identically; train and inference modes are byte-identical. Documented in `docs/protocols/preprocessing_pipeline.md`.

**Stream 2 XP bug fix:**
- `src/arqueogal/data/ingest_stream2.py` now fetches XP coefficients (was missing before). The Stream-2 builder (`build_stream2_kinematic_catalogue.py`) wires `apply_pipeline1_preprocessing()` with XP fetch enabled.

**Model architecture (v1.1):**
- **Single model** (not 5-seed ensemble). 21-label head (not 5-label).
- **Diagnostic heads:** 4-way evolutionary-stage classifier (RGB, HeCB, OOD_evolved, OOD_unevolved) for robustness tracking.
- **Loss design:** ARI contamination loss (weight 0.1) to preserve disc bimodality. Training-time feature-noise injection (100 epochs) + inference-time analytical marginalisation.
- **Output:** 21 posterior-mean predictions + 21×21 block-Cholesky covariance (per-element σ now separates aleatoric-intrinsic from feature-noise-propagated components).

**Extinction and kinematics:**
- `src/arqueogal/data/extinction.py` — frozen CCM89 R_V=3.1 + Yuan+2013 recipe (26 tests, documented in `docs/protocols/extinction_correction.md`).
- `src/arqueogal/data/build_stream2_kinematic_catalogue.py` — Stream-2 orchestrator (11 tests; CLI at `scripts/build_stream2_kinematic_catalogue.py`).

## Current work (v1.1 training)

Status as of 2026-04-29:

- Single-model architecture: 140-D FeatureLayout (post-extinction integration), 21-label head + 4-way diagnostic head.
- Loss: SupCon 0.3 + Barlow 0.8 + ARI 0.1 + KL (aleatoric) + feature-noise injection.
- Training on RTX 3060 (wall time TBD; v1 took ~3.5 h for 5-seed ensemble).
- Feature-noise marginalisation deployed at inference time: per-element σ_total = sqrt(σ_intrinsic^2 + σ_noise^2).
- Checkpoint will be tagged `pipeline1-v1.1-21label-YYYYMMDD` once convergence gates pass.

## Known blocking / pending items (2026-04-29)

1. **Pipeline 1 v1.1 checkpoint convergence** (Phase A). Single-model
   21-label training in flight. Validation gates: (i) 7-part stress
   battery pass; (ii) Stream-1 disc-bimodality preservation
   (ARI >= 0.55, α/M visual inspection); (iii) feature-noise
   marginalisation reduces σ-inflation spike >= 50 %. **Load-bearing
   blocker for all downstream phases.**
2. **Stream-2 kinematic catalogue production run** (Phase B). Builder is
   shipped; production run needs GAVO + AIP TAP to fetch BJ21 + dust-map
   columns for ~8,000 Hon+2021 stars. Wall time ~2 days. **Unblocked,
   can start now.**
3. **Stream-2 chemistry inference** (Phase C). Gated on items 1 and 2.
   Runs v1.1 model on Stream-2 kinematic catalogue, emits 21 predictions
   + 21×21 block-Cholesky covariance + per-element release tiers.
4. **Stream-2 cross-catalogue Test 6** (Phase D). Gated on item 3.
   Cross-match v1.1 Stream-2 predictions against AspGap, GALAH DR4,
   SHBoost for methods-paper validation. Framework ready; execution
   pending Phase C output.
5. **Promotion protocol re-run on 21-label model** (Phase A / Phase D).
   The §3.3 promotion protocol (per-element release-tier gating) must be
   re-run on v1.1 21-label output. 6 out of 21 elements are novel; their
   tier thresholds come from σ_train (provisional pending audit).
6. **Stream-3 generalisation** (Phase E). Demoted from primary. Execute
   only after Phase D clears, using v1.1 model on ~650 k Andrae+2023 pool
   with RGB+HeCB evolutionary-stage filter.
7. **Task 4 age column** (Campante/Miglio team). External; scheduled for
   late 2026. Merge into D-Cat-b parquet at release time.

## Needs clarification

- D5.1 release logistics now live in Starfold's own repo planning.
- Methods-paper venue, timeline, and scope. A&A Methods (~25-page
  scope) is the obvious target; deadline ~July 2026 to land before
  D-Cat-b ships in August.
- Whether to include the RGB / HeCB evolutionary-stage filter in
  Phase E (Stream 3) at inference time, or rely on the Andrae+2023
  `evolutionary_stage_andrae` column. Brief restricts inference to
  RGB + HeCB only; current code does not enforce.
