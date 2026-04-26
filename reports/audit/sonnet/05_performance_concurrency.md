# Performance & Concurrency Meta-Audit (Sonnet, 2026-04-26)

## Executive Summary

ArqueoGal's performance profile exhibits two distinct regimes: (1) *data preprocessing and model training*, where serial extinction loops and sequential ensemble training are the limiting factors, and (2) *large-scale inference*, where 614 k-star single-shot Parquet loads risk RAM spills and lack chunked processing. Concurrency is absent by design (TAP uses `time.sleep()` polling, ensemble training is sequential), acceptable for current RTX 3060 6 GB hardware but creating operational bottlenecks during restarts. GPU orchestration (AMP, streaming aggregation, inference-mode guards) is well-optimized; the bottleneck is I/O and serial preprocessing, not training/inference compute. Cross-stage pipeline checkpointing is fragmentary (TAP has atomic writes; ML scripts lack mid-run resumption), forcing re-runs of expensive downstream stages on failure.

---

## 1. Audit Triangulation: Where Findings Converge

### Common Threads Across Five Reports

#### (a) Per-Star Serial Loops are the Primary Data Bottleneck
- **performance.md §1** flags the Ye+2024 extinction loop (`gaia_xp.py:607-637`) as "data-blocking," with 10-50× vectorization potential.
- **async_tap.md** confirms no true async exists; TAP fetches block via `time.sleep()` in retry loops.
- **background_jobs.md §A** notes batched TAP fetches are checkpointed but lack mid-batch resumption; per-batch retry from scratch wastes 4+ minutes of sleep time on transient failures.
- **ml_pipeline_workflow.md §1** shows stages 3-6 always re-run, recomputing kNN (line 52-58) and Mahalanobis fits even when inputs are unchanged.

**Convergence**: Serial, non-resumable batch processing is both a performance loss (10-50× speedup available on extinction) and an operational reliability issue (4-6 min lost to retry sleeps; 30-60 min wasted re-running unmodified downstream stages).

#### (b) Single-Shot Large-Array Loads Risk RAM Collapse on 614 k Inference
- **performance.md §9** documents the issue: `pd.read_parquet()` loads all 614 k rows × 137 columns into RAM before column selection. On <32 GB hardware or under memory pressure, this triggers swap.
- **gpu_optimization.md** shows ensemble predictions correctly use streaming aggregation (no O(M·B·n²) explosion), but per-member stacking still costs 515 MB (acceptable but "worth noting if B grows").
- **background_jobs.md §C** flags the 6-phase inference graph as "monolithic" (lines 791-1189 in `run_pipeline1_inference.py`); no per-phase checkpointing means if phase 4 (ensemble forward) dies, phase 5 (OOD fit) must refit from scratch.

**Convergence**: Inference is memory-gated, not compute-gated. Chunked reads (Polars/Dask) and per-phase checkpointing are both needed; neither is implemented.

#### (c) Training Orchestration is Sequential by Design, Operational Gaps Remain
- **gpu_optimization.md** notes sequential ensemble training (5-10 seeds per CUDA session) is "memory-efficient on 6 GB" and explicitly recommends against gradient accumulation ("only if per-epoch time degrades below 2 min").
- **background_jobs.md §C** confirms no SIGTERM handler in `run_ensemble.py`; a Ctrl-C mid-training leaves `member_seed2_best.pt` truncated.
- **ml_pipeline_workflow.md §7** documents tests (stage 6) run unconditionally, even when inference (stage 2) output is unchanged (false idempotency).

**Convergence**: Sequential training is a sound hardware constraint, not a design flaw. But lack of SIGTERM handlers and intermediate-result checkpoints means crashes are unrecoverable without loss of work.

---

## 2. Disagreements and Scope Clarifications

### (a) GPU Optimization Status: "All Well-Optimized" vs. Inference Workload Reality

**gpu_optimization.md §Findings**:
> "All files employ AMP bfloat16 correctly and avoid common GPU anti-patterns. No blocking memory leaks or redundant copies."

**performance.md §9 and background_jobs.md §C**:
> Single-shot 614 k-row Parquet load risks RAM spike; no chunked reads; monolithic 6-phase inference graph offers no per-phase resumption.

**Clarification**: `gpu_optimization.md` audits *GPU code paths only* (training.py forward/backward, inference.py ensemble aggregation, model.py Cholesky assembly). It does **not** audit data I/O, which is CPU-resident. The 614 k inference dataset is never on GPU as a whole; it's staged in 4 k-row batches (line 893 in `run_pipeline1_inference.py`). GPU memory itself is well-managed. However, **CPU RAM during the Parquet load** is not GPU-local and is unaudited by gpu_optimization.md. Scope distinction: GPU ✓, CPU I/O ✗.

### (b) TAP Async: "No True Async" vs. Current Usage Pattern is Acceptable

**async_tap.md §1**:
> "Despite claiming 'async' mode, the module uses only synchronous (blocking) TAP submission + polling. This is safe but potentially inefficient."

**performance.md and background_jobs.md**:
> TAP fetches block via `time.sleep()` in retry loops; acceptable for single-threaded ingestion, not viable for multi-stream parallelism.

**Clarification**: `async_tap.md` is correct that true async/await is absent. However, "async" in TAP terminology means the *backend* (AIP) supports async job queues, not that the Python client is non-blocking. Current callers (fetch_gaia_xp.py, emit_stream3_with_hermite.py) run single-threaded, so blocking is fine. If multi-stream parallelism is needed (Stream 1 + Stream 2 concurrently), threading or native asyncio refactor would be required. **No current plans for this**; not a blocker for Pipeline 1 v1.

---

## 3. Top 5 Performance Improvements (Ranked by Leverage and Risk)

| Rank | Item | Leverage | Risk | Effort | Gate |
|------|------|----------|------|--------|------|
| **1** | Vectorize extinction loop (`_ccm89_deredden`, `_ccm89_redden`) | 10-50× on Ye+2024 preprocessing | Low (numpy/extinction only) | Medium (profile + iterate) | None; implement immediately |
| **2** | Chunked Parquet reads for Stream 3 inference (Polars/Dask) | 4-8× RAM savings; OOM prevention | Low (Polars is drop-in for pd) | Medium (refactor data.py:302 + test) | Profile 614 k load on test hardware |
| **3** | Per-phase checkpointing in inference (6-stage graph) | Eliminates 30-60 min waste on restart | Medium (state management) | Medium (add 4 checkpoint files) | Validate on next full run |
| **4** | Conditional execution for stages 3-6 (run_full_pipeline.sh) | 20-30 min saved per unchanged-input re-run | Low (shell scripting) | Low (--skip-existing flags) | Immediate; low risk |
| **5** | Batch Size tuning in kNN-rescue (gpu_knn_search, line 72) | 1-2% latency improvement; OOM fallback | Very low | Low (1-line change + test) | Include as defensive measure |

---

## 4. Concurrency Story End-to-End: TAP → Preprocessing → Training → Inference → Release

### Current Pipeline (Serial, No True Concurrency)

```
1. TAP Fetch (batched_upload_fetch_df)
   ├─ Synchronized: Submit batch N, poll, write checkpoint, repeat batch N+1
   └─ Retry: time.sleep() blocks on transients; no inter-batch parallelism
   
2. Preprocessing (apply_ye2024_correction, emit_stream3_with_hermite.py)
   ├─ Per-star extinction loop (10-50× too slow)
   ├─ Hermite reprojection loop (no checkpoints per chunk)
   └─ No parallelism attempt
   
3. Training (run_ensemble.py)
   ├─ Sequential ensemble: seed 0 → seed 1 → ... → seed 4
   ├─ No SIGTERM handler (crash loses work)
   ├─ Per-epoch best-val checkpoint only (no mid-epoch resumption)
   └─ GPU memory: 6 GB fits single seed + batch
   
4. Inference (run_pipeline1_inference.py)
   ├─ Monolithic 6-phase graph (no inter-phase checkpoints)
   ├─ Single-shot 614 k-row Parquet load (RAM spike risk)
   ├─ Streaming ensemble aggregation ✓ (covariance avoided)
   └─ No progress reporting on multi-hour runs
   
5. Release Assembly (run_knn_rescue.py, build_hybrid_release.py)
   ├─ kNN: Sequential K=50 search on 614 k queries (GPU-accelerated)
   ├─ Hybrid: Linear regressor assembly (CPU-only)
   └─ Tests & gallery: Always re-run (no skip-existing logic)
   
6. Output (atomic Parquet writes + sidecars)
   ├─ Comprehensive provenance in inference (good)
   ├─ Minimal provenance in kNN (missing SHA-256)
   └─ No end-to-end manifest (cannot audit cross-stage integrity)
```

### Parallelism Gaps

| Stage | Parallelism Opportunity | Why Not Done | Leverage |
|-------|--------------------------|--------------|----------|
| TAP fetch | Submit 4-6 batches in parallel; await asynchronously | AIP rate limits; risk 429/503 on flood; no async infrastructure | 2-4× wall time (100 batches) |
| Preprocessing | Vectorize extinction; parallelize reprojection chunks | Loop vectorization is primary (10-50×); chunking is secondary | 10-50× (extinction) |
| Training | Ensemble parallelism (torch.distributed, Ray) | RTX 3060 single-GPU; parallelism moves to pc127 for future | 4-5× (5 seeds in parallel) |
| Inference | Chunked batches; per-label Mahalanobis in parallel | Already chunked at DataLoader; Mahalanobis is CPU-fast relative to inference | ~5% (not compute-bound) |
| kNN + Hybrid | Independent: could start kNN before inference finishes | Dependency chain: kNN needs ensemble latents; hybrid needs both | Not available |
| Release tests | Parallel test execution | Tests are linear; ~10 min total; not a bottleneck | ~10% |

---

## 5. Items Collectively Missed

### (a) No Integrated Observability Stack
Five audits focus on code paths, not runtime behavior. **Missing**:
- Per-operation wall-time profiling (extinction loop, Parquet load, ensemble forward).
- Memory profiling (peak RSS, allocation rates, OOM precursors).
- Structured logging (JSON events indexed by timestamp, not stdout echoes).
- **Recommendation**: Add `cProfile` / `memory_profiler` runs to CI; emit timing sidecars.

### (b) No Graceful Degradation Strategy
- Batch fails: script re-runs from scratch (no partial-result recovery).
- GPU OOM in kNN: fallback to batch=1024 is documented (gpu_optimization.md) but not implemented.
- Missing input: fails after 1+ hour of GPU compute (background_jobs.md).
- **Recommendation**: Implement fallback modes (batch size auto-reduction, partial-batch checkpointing, early input validation).

### (c) No Production Readiness Checklist
- 614 k Stream 3 inference has not been run end-to-end on real hardware.
- Ensemble training (5-10 hours) has never been interrupted and resumed.
- Full pipeline (6-stage, 2-3 hours) has no staged failure test.
- **Recommendation**: Dry-run on pc127 (CPU-only, 32 GB RAM) with half-size data before production.

### (d) Frozen Hermite z-Score Stats Not Pinned to Basis Fingerprint
- Stream 3 inference loads v1 frozen stats (CLAUDE.md §16) but does not verify basis fingerprint matches.
- If frozen stats are accidentally deleted and regenerated, inference uses wrong statistics without warning.
- **Recommendation**: Emit `basis_fingerprint.txt` alongside frozen stats; validate on load.

### (e) No End-to-End Manifest (Cross-Stage Integrity Audit)
- TAP → Ye+2024 → Hermite: No checksum chain.
- Hermite → Inference: No SHA matching.
- Inference → kNN → Hybrid: No cross-stage validation.
- **Recommendation**: Emit `pipeline_run_manifest.json` at shell level with all stage SHAs.

---

## Key References

- **performance.md**: Extinction loop (10-50×), Parquet chunking (4-8×), ensemble aggregation ✓.
- **gpu_optimization.md**: AMP/scaler/pin_memory all correct; kNN batch=2048 near limit; GPU code is well-optimized.
- **async_tap.md**: No true async; checkpoint resumption is atomic and robust; AIP inline-IN ceiling properly gated.
- **background_jobs.md**: TAP batching is production-grade; ML scripts lack SIGTERM handlers and mid-phase checkpoints; parallelization blocked by hardware/rate limits.
- **ml_pipeline_workflow.md**: Restartability broken for stages 3-6; input validation out-of-order; staleness detection absent; no dry-run mode.

---

## Immediate Next Steps

1. **Profile extinction loop** on a 50 k-row batch; measure current time; implement vectorization (np.where + batch extinction call).
2. **Test chunked Parquet reads** with Polars on Stream 3 mock data (100 k rows); measure peak RAM vs. current approach.
3. **Add SIGTERM handlers** to `run_ensemble.py`, `run_pipeline1_inference.py` (3 files, standard pattern).
4. **Implement per-phase checkpointing** in inference (4 checkpoint files: after z-score load, after assembly, after ensemble forward, after OOD fit).
5. **Add `--skip-existing` flag** to shell script and all Python drivers (shell 1-line, Python ~3 lines per driver).

---

## Conclusion

Performance is not uniformly optimized. GPU paths (training, inference computation) are well-tuned for 6 GB; CPU I/O paths (Parquet loading, extinction correction) are serial and vulnerable to RAM spills at scale. Concurrency is absent (blocking TAP, sequential training, monolithic inference) by hardware constraint, acceptable for current single-run workloads but creating 30-60 min of waste on restarts due to missing checkpointing and conditional execution. The concurrency story is coherent (no false parallelism claims, honest about limitations), but observability (profiling, logging, manifests) is fragmentary. Top three improvements (extinction vectorization, chunked inference reads, per-phase checkpointing) would prevent OOM, reduce preprocessing time 10-50×, and eliminate restart waste, all with medium effort and low risk.
