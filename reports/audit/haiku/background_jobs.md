# Background Job & Batch Processing Audit — ArqueoGal

## Overview

This audit examines the ArqueoGal codebase for background-job and long-running batch patterns, with focus on per-batch checkpointing, graceful shutdown, job-state persistence, progress reporting, and parallelization opportunities.

## Summary Findings

1. **Checkpointing is solid in TAP/batch-fetch layers** (tap.py:batched_upload_fetch_df, batched_fetch_df) with atomic tmp→rename writes, but breaks down in ML training scripts (run_ensemble.py, run_pipeline1_inference.py) which offer no mid-training resumption.

2. **Graceful shutdown is nearly absent**: training scripts lack signal handlers for SIGTERM (Ctrl-C will leave corrupted model state); emit/fetch scripts serialize to disk successfully but provide no kill-safe checkpoints during long Hermite reprojection loops (emit_stream3_with_hermite.py:102–119).

3. **ML inference driver (run_pipeline1_inference.py) has good provenance sidecars but inadequate progress reporting for 1–2 hour runs** and no resumable-checkpoint pattern for the 6-phase workflow (ensemble load → z-score → assembly → inference → OOD scoring → output write).

## Detailed Findings

### A. Batch TAP Queries — Strengths

**File: src/arqueogal/data/tap.py**

- **Lines 371–486** (batched_fetch_df): Atomicity is enforced via temp-file pattern (line 480–482):
  ```python
  tmp = batch_file.with_suffix(batch_file.suffix + ".part")
  frame.to_parquet(tmp, index=False)
  tmp.replace(batch_file)
  ```
  On rerun, existing batches are detected (line 452) and skipped. This allows multi-hour ingestions to resume cleanly.

- **Lines 494–666** (batched_upload_fetch_df): Same checkpoint pattern (lines 660–663). Per-batch retry with exponential backoff (lines 636–649) covers transient TAP failures (ESA PooledConnection drops, AIP 504s).

- **Transient error detection** (lines 69–101): Hard-coded marker strings catch flaky TAP backends, enabling smart retry vs. permanent-failure detection.

### B. Batch TAP Queries — Weaknesses

- **No mid-batch resumption**: If a batch fails on attempt 4/6 before writing the checkpoint, the entire batch retries from scratch. Partial result sets within a batch are discarded.

- **No progress reporting during waits**: batched_upload_fetch_df logs only batch-completion (line 624–630). For large jobs (92 batches × 15 s ≈ 25 min as per fetch_gaia_xp.py:28), a user killing the job mid-run sees no indication which batch is in progress.

- **No SIGTERM handler**: Both scripts (fetch_gaia_xp.py, emit_stream3_with_hermite.py) run synchronously with no signal cleanup. Ctrl-C during a 15-second TAP query leaves partial parquet files or stale TAP jobs in the AIP queue.

### C. ML Training & Inference — Critical Gaps

**File: scripts/run_ensemble.py**

- **No per-epoch resumption** (lines 264–299): The ensemble trains 5 members sequentially. If the process dies at member 3/5 epoch 8/10, restart must retrain members 0–2 from scratch — the saved checkpoints (lines 269–284) are only best-val, not epoch-level cadence.

- **No SIGTERM handler**: No signal handler intercepts Ctrl-C. A SIGTERM during training (line 266: train_model call) will kill the model mid-backward pass, leaving `member_seed2_best.pt` incomplete or truncated.

- **No job-state log**: No "member 2 training in progress" marker. Concurrent reruns could silently clobber model_dir/member_seed2/member_seedN_best.pt.

**File: scripts/run_pipeline1_inference.py**

- **Monolithic inference graph** (lines 791–1189, run_inference function): 6 logical phases (ensemble load, z-score apply, feature assembly, ensemble forward, OOD scoring, output write) are not checkpointed individually. If killed at phase 4 (line 895, ensemble forward), phase 5 (OOD bundle fit, line 902) must refit the Mahalanobis bundle from scratch.

- **No progress reporting on multi-hour runs**: For 1–2 million rows × batch_size=4096, the DataLoader (line 893) runs silently for hours with only one log entry at start (line 894) and one at end (line 996).

- **Intermediate-result persistence**: OOD Mahalanobis bundle (line 902) is fit on the fly from training parquet. No cache or checkpoint exists if the job restarts.

- **No kill-safe handling** (line 331, atomic_write_parquet): The atomic write itself is safe (tempfile + rename), but 30+ minutes of prior work (ensemble inference, OOD fit) are lost on Ctrl-C immediately before the write.

**File: scripts/run_full_pipeline.sh**

- **Linear orchestration with no resumption** (lines 42–74): 6 bash stages are run sequentially. Stage 3 (latent-kNN, line 53) depends on stage 2 output (predictions parquet, line 30). If stage 4 kills the process, rerun must redo stages 3 & 4 even though stage 2's output is cached (line 43).

- **No per-stage health checks**: Stage 2 checks for existing predictions (line 43); stages 3–6 have no equivalent guards. Stage 5 (gallery rebuild) always runs, even if stage 2 succeeded hours ago.

### D. Hermite Reprojection Loop — Kill Vulnerability

**File: scripts/emit_stream3_with_hermite.py**

- **Uncheckpointed loop** (lines 102–119, _reproject_all function): 50k-row chunks are reprojected sequentially. No checkpoint is written mid-loop; a SIGTERM at chunk 15/18 (line 103) discards all 15 completed chunks and forces a full rerun.

- **No interim parquet**: The flux matrix (line 192–195) is loaded entirely into memory and reprojected in-place. A 500 MB parquet × 330 wavelengths = 150+ GB intermediate matrix (if dense) would OOM; the implementation avoids this via slicing, but provides no resumption.

### E. Parallelization Opportunities

**Not currently exploited**:

1. **Ensemble member training** (run_ensemble.py:264–299): 5 seeds trained sequentially (each ~1–2 hours). On multi-GPU or multi-node, seeds 0–4 could train in parallel via torch.distributed or Ray, reducing wall time from 5–10 hours to 1–2 hours.

2. **Batch XP fetch** (fetch_gaia_xp.py:120–130): 92 batches run sequentially via batched_upload_fetch_df. TAP jobs are submitted and awaited synchronously. Submitting 4–6 batches in parallel (bounded by AIP connection limits) could halve wall time.

3. **Per-label Mahalanobis scoring** (run_pipeline1_inference.py:901–908): The 108-D XP block is scored against a single Mahalanobis bundle. No per-label or per-cohort parallelization is attempted.

**Why not done**:

- **VRAM constraint**: RTX 3060 6 GB VRAM limits batch sizes and ensemble member count. Parallelizing ensemble training would require spilling to RAM or splitting across multiple GPUs (not available).

- **TAP rate limits**: AIP has undocumented per-connection and per-user throttling. Parallel batch submissions risk hitting limits and 429/503 responses. Current sequential approach is conservative and reliable.

- **State complexity**: run_pipeline1_inference.py's 6-phase graph has complex dependencies (OOD bundle fit depends on training parquet, not input parquet). Parallelizing individual phases risks subtle state bugs.

## Recommendations

### High Priority

1. **Add SIGTERM + SIGINT handlers to training scripts** (run_ensemble.py, run_contrastive_pretrain.py, run_supervised_finetune.py):
   - Save a `.in_progress` marker before starting member training.
   - On signal, save the current best-val checkpoint and delete `.in_progress`.
   - On restart, detect `.in_progress` (incomplete seed) and skip to the next seed.

2. **Implement per-phase checkpointing in run_pipeline1_inference.py**:
   - Save frozen stats after load (line 845) → checkpoint file.
   - Save assembled X matrix after assembly (line 869) → checkpoint file.
   - Save ensemble predictions after inference (line 895) → checkpoint file.
   - On restart, load the latest checkpoint and skip prior phases.

3. **Add progress reporting to long inference loops**:
   - Log batch progress in run_pipeline1_inference.py's DataLoader loop (line 895) every 10–50 batches.
   - Use tqdm or similar for visual feedback on multi-hour runs.

### Medium Priority

4. **Checkpoint Hermite reprojection chunks** (emit_stream3_with_hermite.py):
   - Write coefficients to a temporary per-chunk parquet after each chunk (line 112–117).
   - On rerun, detect existing chunk files and skip to the next uncomputed chunk.
   - Merge chunk results at the end.

5. **Add resumable-stage logic to run_full_pipeline.sh**:
   - Write a `.stage_N_done` marker file after each stage succeeds.
   - Check markers at start; skip completed stages.
   - Allow manual override with `--force-stage N` to recompute.

6. **Explore TAP batch parallelization** (fetch_gaia_xp.py):
   - Submit 3–4 batches in parallel to AIP (respecting rate limits).
   - Track job IDs and poll asynchronously for completion.
   - Requires refactoring batched_upload_fetch_df to support async submission.

### Low Priority (Exploratory)

7. **Distributed ensemble training** (run_ensemble.py):
   - Requires multi-GPU setup or multi-node cluster (not available on RTX 3060 single-GPU machine).
   - Consider for future if hardware changes.

8. **Per-label Mahalanobis scoring** (run_pipeline1_inference.py):
   - Profile to determine if per-label parallelization is worthwhile.
   - Current 108-D block scoring likely CPU/memory bound, not compute bound.

## Files Affected

- `src/arqueogal/data/tap.py` (batched_upload_fetch_df, batched_fetch_df) — **good baseline**, no changes needed
- `scripts/run_ensemble.py` — **needs SIGTERM handler + per-member resumption**
- `scripts/run_pipeline1_inference.py` — **needs per-phase checkpointing + progress reporting**
- `scripts/emit_stream3_with_hermite.py` — **needs chunk-level checkpointing**
- `scripts/run_full_pipeline.sh` — **needs stage-marker guards**
- `scripts/run_contrastive_pretrain.py`, `run_supervised_finetune.py` — **need SIGTERM handlers** (patterns identical to run_ensemble.py)

## References

- CLAUDE.md §4: Hard rule 4 — never cross-import between main and experimental.
- CLAUDE.md §5: Process and communicate departures from protocol before implementing.
- tap.py lines 69–95: Transient error markers for smart retry logic.
- run_pipeline1_inference.py lines 791–1189: Full inference orchestration (monolithic, no phases).
- run_full_pipeline.sh: 6-stage orchestration with partial guards.

---

**Audit date:** 2026-04-26  
**Auditor notes:** Background-job patterns are unevenly distributed. TAP batch layer is production-grade; ML training and inference scripts lack graceful-shutdown and resumption logic. No parallelization is currently attempted, partly due to VRAM constraints and TAP rate-limit conservatism. Recommend prioritizing SIGTERM handlers and per-phase checkpointing before scaling to multi-hour production runs.
