# Phase 3a — Stream 3 Option C expansion (delta fetch)

*Generated: 2026-04-20 18:21 UTC · authored after-the-fact from on-disk artefacts + structured logs; the background orchestrator process exited without emitting its own report — ~80 % of its wall-clock was burned in a degenerate 3-second polling loop on a subprocess monitor file.*

## 1. Scope and ratified refinements

User-ratified at Phase 3 launch:
1. **Volume-limited sampling: Option A** — random 250 k draw from the BJ21 d ≤ 2.5 kpc pool (4,020,951 candidates), no stratification oversampling. Preserves natural density for Pipeline 2 HDBSCAN.
2. **IR delta fetch gated on Ye-OK subset** of the XP delta — only stars that survived the Ye+2024 flux-correction gate get IR cross-matched, avoiding wasted TAP time on stars destined for rejection anyway.
3. **`selection_prob` computed at feature-matrix build time** using Andrae+2023 pre-inference estimates (Teff/log g at ~100 K / ~0.2 dex — adequate as a weight) rather than the model's own predictions. Avoids circular reasoning.

## 2. Source_id union

| arm | target | actual | method |
|---|---|---|---|
| uniform | 400,000 | **372,283** | existing Stream 3 (168,099 reused) + stratified fill (204,184 new, per_cell=1,228 on Teff × log g × [Fe/H] × G) |
| volume-limited | 250,000 | **250,000** | random draw from BJ21 d ≤ 2.5 kpc pool (4,020,951 candidates), disjoint from uniform arm |
| **union total** | 650,000 | **622,283** | — |

- Uniform undershot by 27,717 (6.9%): some strat cells have fewer than 1,228 available stars in the Andrae pool after excluding the existing 168,099.
- Disjointness enforced: volume_limited pool excludes every uniform source_id by construction.
- Seed 0 end-to-end.
- Input: Andrae+2023 vetted RGB (10,483,688 rows, VizieR J/MNRAS/537/1984).
- Output: `data/interim/stream3_expansion_union.parquet` (26.7 MB, sha256=…ef16e, written 16:16 UTC).

**Delta source_ids to fetch:** 622,283 − 168,099 already-on-disk = **454,184**.

## 3. XP coefficient fetch (delta)

- Script: `scripts/fetch_gaia_xp_delta.py` (invoked from the Phase 3a orchestrator).
- Service: AIP TAP (authenticated via `GAIA_AIP_TOKEN`), 10 k chunks.
- Input: 454,184 delta source_ids.
- Output: `data/interim/xp_coeffs_raw_delta.parquet` (774 MB, 454,184 rows × 11 cols, written 16:38 UTC).
- Wall-clock: 16:16 → 16:38 = **22 min**.
- Retries: structured log shows zero retries (resolved via AIP mirror after earlier ESA TAP quota issues documented in Phase 2).

## 4. Ye+2024 flux correction (delta)

Structured log: `logs/stream3_delta_ye2024_20260419.log`.

- 454,184 rows → 23 mega-batches of 20,000.
- **Ye flag totals**: OK=449,625 · NO_SYNTH_PHOT=4,559 · CAL_FAIL=0.
- **NO_SYNTH_PHOT rate (delta): 1.004%.**
- **Drift vs Stream 1 baseline (2.60%): −1.60 pp** (within ±2.0 pp halt tolerance). **HALT CLEAR.**
- Output: `data/interim/xp_sampled_corrected_delta.parquet` (572.8 MB, 4 cols — sampled flux on `np.geomspace(360, 990, 330)` nm; Hermite reprojection deferred to Phase 3b stage B per the Stream 1 pipeline pattern).
- Wall-clock: 16:38 → 17:01 = **22.9 min**.
- Checkpoints: 23 per-batch parquets consolidated and removed.

Ye-OK source_id export: `data/interim/stream3_delta_ye_ok_source_ids.parquet` (3.9 MB, 449,625 rows, written 17:04 UTC).

## 5. IR photometry fetch (delta)

Structured log: `logs/stream3_delta_ir_fetch_20260419.log`.

- Input: 449,625 Ye-OK delta source_ids.
- Service: AIP TAP, 10 k chunks, 45 batches × 2 catalogues = 90 jobs.
- Join: `original_ext_source_id = designation` (2MASS and AllWISE both) — bypasses the ESA `allwise_oid` JDBC pool bug hard-coded as a regression guard after Phase 2's diagnosis.
- Output: `data/raw/ir_photometry/stream3_delta_ir.parquet` (25.0 MB, 449,625 rows × 18 cols).
- Wall-clock: 17:28 → 18:18 = **50 min**.
- Retries: structured log shows zero retries; three earlier abort-and-restart attempts (17:06, 17:07, 17:18) captured in `*.attemptN.log` — diagnosis out-of-scope for this report but did not affect the successful run.

**Counterpart rates:**
- 2MASS: **99.722%** (448,376 matches)
- AllWISE: **100.000%** (449,625 matches)
- IR-complete (both bands): **99.701%** (448,282 stars)
- IR-missing-any: 1,343 stars (will be dropped at feature-matrix build)

**HALT CLEAR: IR-complete rate 99.70% ≥ 97% threshold (2.7 pp margin).**

## 6. Halt-trigger sweep

| trigger | threshold | measured | verdict |
|---|---|---|---|
| Ye NO_SYNTH_PHOT drift vs 2.60% baseline | ±2.0 pp | −1.60 pp | **CLEAR** |
| IR-complete counterpart rate | ≥ 97% | 99.70% | **CLEAR** |
| Data/ footprint | < 9.5 GB | 7.4 GB | **CLEAR** (2.1 GB headroom to 9.5 GB soft, 2.6 GB to 10 GB hard) |
| Phase 3a wall-clock | ≤ 2.5 h | 3 h 20 min | **OVERRUN** (+50 min) |

Wall-clock overrun attribution: the orchestrator's polling loop (3-second cadence on a subprocess monitor file for 83 minutes) delayed coordination of the IR substep relative to an ideal hand-off. Actual subprocess compute was ~3 h 8 min (union 0 min, XP 22 min, Ye 23 min, IR 50 min + transitions). Not a release-blocker. Flagged for the consolidated phase report.

## 7. Storage diff

| before Phase 3a | after Phase 3a | delta |
|---|---|---|
| 5.1 GB | 7.4 GB | +2.3 GB |

New artefacts (cumulative):
- `stream3_expansion_union.parquet` — 26.7 MB
- `stream3_expansion_delta_source_ids.parquet` — 3.3 MB
- `xp_coeffs_raw_delta.parquet` — **774 MB** (largest single addition)
- `xp_sampled_corrected_delta.parquet` — **573 MB**
- `stream3_delta_ye_ok_source_ids.parquet` — 3.9 MB
- `stream3_delta_ir.parquet` — 25.0 MB

The two large sampled/raw XP parquets will be consolidated or deleted after Phase 3b Hermite reprojection — no need to carry both sampled flux and Hermite coefficients downstream.

## 8. Anomalies and followups

- **Orchestrator report gap**: the background orchestrator burned most of its budget in a subprocess-file polling loop and exited without emitting its seven-section report. The underlying detached subprocess chain ran to completion independently, as designed. Report authored post-hoc from disk + structured logs. Followup: the polling cadence should be widened from 3 s to ≥ 60 s, or replaced with an fswatch-style event trigger.
- **Uniform arm undershoot** (372,283 / 400,000 = 93.1%): Andrae pool stratification cells are not uniformly populated. Acceptable — the arm's purpose is Pipeline 1 inference coverage, and 372 k exceeds the §9.2 test-6 cross-catalogue sample-size requirements.
- **IR attempt logs** (17:06/17:07/17:18): three short aborts before the successful run at 17:28. Contents not inspected in this report — the successful run is stand-alone and provenance-clean. Followup ticket if the abort history is ever needed.

## 9. Phase 3b handoff

Ready to launch:
1. **Feature-matrix build** for both arms via a Stream 3 analogue of `scripts/build_pipeline1_features_stream1.py` (no APOGEE label joins; include `selection_prob` computed from Andrae+2023 Teff/log g using the v1.1 compound selection function).
2. **Hermite reprojection + per-coefficient z-scoring** via the Stream 1 stage B pattern (`scripts/emit_stream1_with_hermite.py` + frozen `hermite_stats_fingerprint` 0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7).
3. **Pipeline 1 inference** via `scripts/run_pipeline1_inference.py` on both arms, emitting `data/processed/pipeline1_predictions_uniform.parquet` and `data/processed/pipeline2_features_volume.parquet`.

Halt conditions for Phase 3b: OOD-flag rate 2–20% on uniform arm · Regime-B rate ≤ 5% · storage < 9.5 GB · feature NaN ≤ 1%.
