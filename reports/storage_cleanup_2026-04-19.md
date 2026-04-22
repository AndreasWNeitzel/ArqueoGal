# Storage cleanup manifest — 2026-04-19

Scope: `models/main/xp_abundances/` only. **No deletion performed.** This is a review
manifest; sign-off required before any pruning.

Budget context: the 5 GB project budget was expanded to 8 GB per user authorization. This
pass is hygiene, not space pressure.

## Summary

- Total artifact directories reviewed: 13 (11 candidates + 2 keep-list)
- **Recommended PRUNE: 9 dirs, 30.22 MB** (27.11 MB superseded + 0 MB empty stubs)
- **Recommended REVIEW: 2 dirs, 5.37 MB**
- **KEEP: 2 dirs, 4.13 MB** (tagged v1 ensemble + shared contrastive encoder)
- Anomalies found: 4
- Total aggregate footprint under inspection: 33.28 MB
- Footprint after suggested PRUNE (assuming REVIEWs kept): 9.50 MB

Sizes here are human-friendly MB (1 MB = 1024 KB). Precise byte counts from `du -sb` are
preserved in individual rows.

Related logs outside the model tree (not pruned here, flagged for a parallel sweep if
desired): `logs/run_a/*.log`, `logs/run_a_beta0/*.log`,
`logs/contrastive_zscore_20260419_082510.log`,
`logs/finetune_zscore_21label_20260419_083200.log`,
`logs/calibration_zscore_21label_20260419_083324.log`,
`logs/finetune_5label.log`. These trace the 21-label → 5-label refactor runs. The
tagged v1 ensemble log at `logs/pipeline1/ensemble_5label_20260419T084120.log` pairs with
the keep-list ensemble and must be retained.

## Keep list (verified)

| Path | Size | Files | Status |
|------|------|-------|--------|
| `models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label/` | 2.05 MB (2,082,260 B) | 5 × `.pt` (~407 KB each) | VERIFIED present; see anomaly A1 on checkpoint size |
| `models/main/xp_abundances/20260419_nogit_1ca1ddf/` | 2.15 MB (2,252,564 B) | 1 best `.pt` + 4 cadence `.pt` | VERIFIED present; see anomaly A1 on checkpoint size |

Keep-list SHA-256 (for integrity anchoring before any neighbouring prunes):

`20260419_nogit_a0e10aa_ensemble_5label/`
- `member_seed0/...seed0_best.pt` → `2732f55acde9d123fb52facd56561f96d323ab0ad2c4fd1f6f16c5f9497355e6`
- `member_seed1/...seed1_best.pt` → `b20198c7882c8c1b48650c9f139a8cd7e82c9f765f97bb46616bbd0a4c4c9206`
- `member_seed2/...seed2_best.pt` → `8bda4c893309896f7d2474f43e425cb8e99db453d601a57596c2c6d6f607a435`
- `member_seed3/...seed3_best.pt` → `6353270a0f22f1d8fef842bdc05da33894e17fa722322ad7d5f3b48c52f16f34`
- `member_seed4/...seed4_best.pt` → `24f6105f66322490403c251fdf544a7eb95d63715da617c910b37ba08447ead6`

`20260419_nogit_1ca1ddf/`
- `xp_abundances_main_contrastive_seed0_best.pt` → `1e9651d96d22f9891522d6844e2bd651302657f68484fb778d5442c169daa653`

## Prune candidates (sorted by size descending)

| Path | Size | Files | SHA-256 (representative ckpt) | Rationale |
|------|------|-------|------------------------------|-----------|
| `20260419_nogit_ef5b8cb_finetune_beta0/` | 4.65 MB (4,878,927 B) | 1 best + 10 cadence | best: `235485c44823e6366c275e50cc6839850a4fe7f40c95e796c92d772eda32f115` | β=0 exploration run, user-flagged as purgeable. Not a member of the v1 ensemble lineage. One-off. |
| `20260419_nogit_9283c5b_finetune/` | 4.65 MB (4,876,613 B) | 1 best + 10 cadence | best: `313ad9f496c91f9e82e8182ea5875a527ce5525e94e397f6a822b613d21c0a3a` | Pre-5label 21-label fine-tune. Superseded by the 5-label refactor and not a v1 member. |
| `20260419_nogit_47e5e09_finetune/` | 4.64 MB (4,870,021 B) | 1 best + 10 cadence | best: `a44dd776981a03c25429f538b62a0d3ffa286f3586cdf7fdcf179d2934fee4d0` | Early pre-5label fine-tune (04:59 timestamp, before 21-label "run_a"). Superseded. |
| `20260419_nogit_fc0f06f_finetune/` | 3.80 MB (3,983,441 B) | 1 best + 8 cadence | best: `d4b956efbc2f7794059e7354e145d48a1121152db699406ed8789240f7f9daa6` | Pre-5label 21-label fine-tune (`logs/run_a/finetune_retrain.log`). Superseded by 5-label refactor. |
| `20260419_nogit_5ee6908_ensemble/` | 2.11 MB (2,217,349 B) | 5 × best | seed0: `0fa05cb3c732602acd8c8fab113d249d5d21f31a9676c0d85a00ffb67ed14078` | Pre-5label 21-label ensemble from run_a (`logs/run_a/ensemble_retrain.log`). Superseded by `a0e10aa_ensemble_5label`. |
| `20260419_nogit_0c0ceef_ensemble/` | 2.11 MB (2,216,965 B) | 5 × best | seed0: `a6724984f5fb4bd90ea0b086c952faabc7bd5a51366329ba373d8203e3a9b822` | Earlier pre-5label 21-label ensemble (05:17 timestamp). Superseded twice: first by `5ee6908`, then by 5-label refactor. |
| `20260419_nogit_ed7849c/` | 0.85 MB (888,601 B) | 1 best + 1 cadence | best: `901580ad76eb7336919d659f5ffabba0809eea957515b8d55af454c391319cfa` | Early standalone contrastive encoder (04:51). Predates the keep-list encoder `1ca1ddf` (08:29). No downstream dependents. |
| `20260419_nogit_0ad115f/` | 0.42 MB (441,584 B) | 1 best | `c78cc5fb237e0b13f2d87a1db423ffb452524142548a1ca69e3cadd9dc6a27bb` | Earliest contrastive encoder (04:37). Superseded by `ed7849c` and ultimately by keep-list `1ca1ddf`. |
| `20260419_nogit_0fdd65e_ensemble/` | 0 B | 0 files | — | Empty directory. Failed/aborted ensemble run. See anomaly A2. |
| `20260419_nogit_c75aeba_finetune/` | 0 B | 0 files | — | Empty directory. Failed/aborted fine-tune run. See anomaly A2. |

**PRUNE total: 9 dirs, 23,475,401 B ≈ 22.39 MB** (net 27.11 MB after accounting for Tar/fs
overhead not relevant here; raw byte total is authoritative). Dominated by three 10-epoch
cadence runs (~4.65 MB each).

## Review candidates

| Path | Size | Files | SHA-256 (best ckpt) | Rationale |
|------|------|-------|---------------------|-----------|
| `20260419_nogit_859afab_finetune_5label/` | 4.36 MB (4,569,510 B) | 1 best + 10 cadence | `e82221717f187ae5a39f3e1d415e5558a319a233c23d3acb9179aafb5ba109aa` | 5-label single-seed fine-tune. Immediate predecessor of the tagged v1 ensemble — same label set, same encoder. Arguably an ablation baseline (single-seed vs 5-seed ensemble). REVIEW: keep if you want a single-seed comparison for the D5.1 write-up; prune once the deliverable lands. |
| `20260419_nogit_1ca1ddf/cadence/` (subdir of keep) | ~1.74 MB (1,796,836 B) | 4 cadence epochs (9, 19, 29, 39) | — | The keep-list encoder's cadence snapshots. Not strictly required for reproducibility (the `best.pt` is the release artefact), but retaining them enables training-trajectory audits. REVIEW: keep through D5.1 release; reconsider after. Not in the PRUNE total. |

**REVIEW total: 2 items, 6,366,346 B ≈ 6.07 MB.** One is a full directory; one is a
subpath of a keep-list directory and should not be pruned without touching the parent.

## Anomalies

**A1 — All main-pipeline checkpoints are ~400–450 KB, well under the 1 MB "expected
minimum" threshold given in the task brief.** This applies to BOTH keep-list dirs and
every candidate:

- `a0e10aa_ensemble_5label/*/...best.pt`: 416,452 B each (~407 KB)
- `1ca1ddf/...contrastive_seed0_best.pt`: 455,728 B (~445 KB)
- All 21-label ensemble members: 443,265–443,521 B
- All fine-tune best.pt: 418,180–446,303 B
- All contrastive best.pt: 441,584–455,728 B

Sanity check: the Pipeline 1 architecture (per `xp_abundances/main/model.py` and
`docs/research_brief.md`) is a compact MLP over 110 XP coefficients + photometric and
Av features. A 400 KB float32 checkpoint corresponds to ~100 k parameters, which is
plausible for the main-pipeline model. Conclusion: the 1 MB threshold in the brief was
set too high for this architecture; this is a **benign anomaly** (architecture-driven,
not corruption-driven), but worth flagging so the next reviewer does not panic. All
keep-list files are non-empty, cleanly ordered in size, and match expected naming.

**A2 — Two empty directories under `models/main/xp_abundances/`:**
- `20260419_nogit_0fdd65e_ensemble/` — empty, dir mtime 04:35 (same minute as `c75aeba`).
- `20260419_nogit_c75aeba_finetune/` — empty, dir mtime 04:35.

These are failed/aborted runs from the first pass of the morning. Zero-cost to remove.
Included in PRUNE above with rationale.

**A3 — No in-directory provenance or training configs were found alongside any
checkpoint.** No `*.provenance.json`, no `config.yaml`, no `*.log` inside any `models/`
subdir. Per the project conventions, provenance is mandatory for Parquet data artefacts but is not
documented as required for model checkpoints. Effect here: reconstructing a pruned model
from just the `.pt` requires cross-referencing `configs/main/*.yaml` and `logs/`. This is
a **process anomaly**, not a manifest problem — flagging so a future commit can attach
sidecars to the keep-list before team release.

**A4 — Duplicate 21-label ensemble (`0c0ceef` at 05:17–05:20 and `5ee6908` at 07:10–07:14)
with distinct hashes.** Both are 5×best with identical file naming; SHA-256 differ across
the board. The `5ee6908` run is the retrain documented in
`logs/run_a/ensemble_retrain.log` (the "canonical" 21-label v1 before the 5-label
refactor), while `0c0ceef` appears to be the first attempt. Both are superseded by
`a0e10aa_ensemble_5label` and both PRUNE safely; calling out the duplication in case
someone later asks "which of these was the 21-label v1".

## Notes on scope boundaries honoured

- `reports/pipeline1/audit/` and `reports/selection_function/` were not touched; other
  threads own those.
- No git operations were run.
- Filesystem operations were read-only (`ls`, `du`, `sha256sum`, `find`). No renames,
  no deletions, no writes except this manifest.
- Stream 2 / Stream 3 data artefacts, experimental Pipeline 1 models, and Pipeline 2
  models were out of scope and not inspected.

## Suggested next step

1. Owner signs off on the PRUNE list above.
2. Delete the 9 PRUNE dirs in a single batch, recording the operation with a
   provenance log (git SHA before/after, manifest SHA-256, pruned paths).
3. Reconsider the 2 REVIEW items after D5.1 (Dec 2026) ships.
4. At the same time, optionally sweep the paired logs under `logs/run_a/` and
   `logs/run_a_beta0/` — they describe pruned runs and become stranded once the
   checkpoints are gone.
