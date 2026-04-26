# Phase 03. Stream 3 expansion + Pipeline 1 inference

**Status: Phase 2 (prerequisites) complete 2026-04-19. Phase 3 (fetch + inference)
about to launch.**

## Goal

Run Pipeline 1 v1 ensemble inference on ~650 k Gaia XP stars (non-APOGEE) to produce
the D-Cat-b catalogue contribution and the downstream prediction parquets that
Starfold (separate repo; see `04_pipeline2_main.md`) consumes.

## Sampling strategy. Option C (dual samples)

- **~400 k uniform-stratified** (revised down from 800 k under 10 GB budget), for
  Pipeline 1 audit and §9.2 test 6 cross-catalogue consistency. Ensures equal
  statistical power across (Teff, log g, [M/H], G) cells.
- **~250 k volume-limited** (revised down from 500 k) at d ≤ 2.5 kpc, retained as
  a natural-density sample consumable by Starfold-side density-based clustering.
  Preserves natural density (Option A random sampling from the 4M-candidate pool,
  no oversampling. ADR-0013).

Union of source_ids after dedup: ~1.2–1.3 M unique; delta to existing 168k on disk
is ~1 M.

## Phase 2 prerequisites. DONE

| Deliverable | Status |
|---|---|
| BJ21 distance fetch for Andrae+2023 pool (10.48 M rows, 125 min, pure TAP UPLOAD) | Complete (`data/raw/bailer_jones_2021/andrae_pool_bj21.parquet`, 184 MB) |
| IR photometry module (`data/ir_photometry.py`, 2MASS + AllWISE via Gaia neighbour tables) | Complete; 99.46% IR-complete on existing 164k Ye-OK Stream 3 |
| Inference-driver NaN-safe rebuild (`scripts/run_pipeline1_inference.py` mirrors training's `nan_to_num` boundary) | Complete, 18/18 tests |
| Selection-function v1.1 (4-D \|b\|×G×Teff×log g grid + Ye retention + aux-missingness gates) | Complete, 23/23 tests |
| Aux-missingness flag system (IR, parallax, extinction) | Complete |

## Phase 3 execution plan

Launch order:

1. Define source_id union (~1.2–1.3 M unique after dedup against existing data).
2. **Parallel**: XP fetch + Ye correction for the ~1 M delta (AIP TAP UPLOAD,
   ~60–90 min); IR cross-match for the Ye-OK delta.
3. Feature-matrix emit: uniform-stratified ~400k → `pipeline1_inference_uniform.parquet`;
   volume-limited ~250k → `pipeline1_inference_volume.parquet`. Both carry
   `selection_prob` and `aux_missingness_*` flags. Apply frozen v1 per-coefficient
   z-score stats (basis fingerprint `0d34b565...`), do NOT refit.
4. Pipeline 1 v1 ensemble inference on both matrices. Outputs: 5-label μ, full Σ,
   OOD flags, Regime B flag, `selection_prob`, `aux_missingness_*`, tier marker
   (T1 or T1-caveat).
5. §9.2 test 6 (cross-catalogue consistency) on the uniform sample overlap with
   AspGap / Guiglion+2024 / SHBoost.

## Budgets

- Wall time: ~3 h total (most is the AIP TAP fetch).
- Disk: projected +~1.6 GB; current 5.1 GB + 1.6 GB = 6.7 GB vs 10 GB ceiling.

## Acceptance criteria

- Both feature matrices emit with provenance sidecars, including sampling-method
  documentation per matrix.
- Pipeline 1 inference succeeds with OOD flag rate in 2–20 % band (Stream 1 reference
  was ~0.5 %; Stream 3 expected higher due to coverage tails).
- Regime B exclusion captures ≤5 % of Stream 3 stars.
- selection_prob distribution is sensible (majority at 1.0, low-|b| faint stars
  substantially below).
- §9.2 test 6 report card produced.

## Halt conditions

- Ye NO_SYNTH_PHOT flag rate on Stream 3 substantially different from Stream 1
  baseline (distribution shift).
- OOD flag rate outside 2–20 % band.
- Regime B exclusion > 5 %.
- Storage footprint approaches 9.5 GB mid-phase.
- IR completeness rate on Stream 3 diverges materially from the 99.46 % Phase 2
  measurement.

## Needs clarification

- §9.2 test 6 acceptance criteria (same as in Phase 02 file).
- Whether failed-Ye Stream 3 stars (expected ~10% at |b|<5°) are included in
  inference with a flag, or excluded before inference. Current plan is flag-included
  for downstream users.
- Volume-limited parquet schema: kept as Pipeline-1-inference-output-shape. Starfold
  (downstream) is responsible for its own feature-matrix assembly (adding kinematics,
  [C/N], etc.) on top of whatever this repo ships.
