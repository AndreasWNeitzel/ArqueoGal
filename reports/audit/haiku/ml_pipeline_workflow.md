# ML Pipeline Workflow Audit: run_full_pipeline.sh

## Executive Summary

The end-to-end ML pipeline exhibits **critical gaps in restartability, input validation ordering, and staleness detection**. While provenance sidecars are comprehensive, the orchestration cannot resume from arbitrary failure points and freely reuses intermediate files even when upstream inputs have changed. The shell script checks stage 2 output but skips stages 3-4-5-6, creating a false impression of idempotency.

---

## Findings by Category

### 1. Restartability and Resume Capability

**Status: BROKEN for stages 3+**

- **Stage 2 (inference)**: Conditional run with output check (line 43-50)
  - `if [[ -f "$PRED_PARQUET" ]]` — skip if exists, else run
  - Restartable: YES, but only if the file is deleted to force re-run
  
- **Stages 3-6**: No skip logic, always execute unconditionally
  - Line 52-58 (`run_knn_rescue.py`): Always runs, even if `$KNN_PARQUET` exists
  - Line 60-65 (`build_hybrid_release.py`): Always runs, even if output exists
  - Line 67-69 (gallery): Always runs diagnostic build
  - Line 71-74 (tests): Always re-runs test suite
  
- **Impact**: If stage 4 fails, restarting from line 1 will re-run kNN (expensive GPU operation on 74k stars), re-run hybrid (IO-heavy), re-run tests, and re-build gallery—wasting 30–60 min per restart
- **Root cause**: No `--resume` or `--skip-existing` flags in downstream Python drivers
- **Workaround in use**: Typically re-run stages individually by hand (not automated)

**Recommendation**: Add `--skip-if-exists` or `--force` flags to all four drivers (knn_rescue, hybrid, gallery, tests).

---

### 2. Input Availability Checks

**Status: INCOMPLETE and OUT-OF-ORDER**

#### Pre-flight checks in shell script:
- Line 35-40: Ensemble directory check ✓
  - `if [[ ! -d "$ENSEMBLE_DIR/member_seed0" ]]` — guards stage 1
  - Fail-fast with clear error message
  
- **Stage 2 inputs**: No check for `$S3_FEATURES` or `$FROZEN_STATS` before running
  - Missing check before line 46-50 (`run_pipeline1_inference.py`)
  - Would fail at `pd.read_parquet(input_parquet)` (line 861), wasting time loading model + ensemble first
  
- **Stage 3 inputs**: No validation at all in the shell script
  - Missing checks for `$TRAIN_PARQUET`, `$FROZEN_STATS`, `$S3_FEATURES` before line 52-58
  - Script loads `run_knn_rescue.py`, which only checks `--ensemble-dir` validity (line 115-119)
  - If `$TRAIN_PARQUET` missing, fails at `load_arrays(args.train_parquet)` (line 134 in knn_rescue), **after** loading ensemble and computing inference latents (line 153-154)—wasting GPU ops
  
- **Stage 4 inputs**: Partially checked in Python
  - Line 58 in `build_hybrid_release.py`: `knn_path = args.knn_rescue if args.knn_rescue.exists() else None`
  - Allows degraded run (regressor + caveat) if kNN missing; does not fail
  - But no check for `$PRED_PARQUET` or `$S3_FEATURES` before starting

#### Issue: Input validation happens downstream, not upfront
- Shell script should validate all inputs **before any stage runs**
- Current pattern: Load expensive resources (models, data) first, check inputs later
- Example: `run_knn_rescue.py` validates ensemble checkpoints (line 115-119) but validates input parquets only inside `load_arrays()` at runtime

**Recommendation**: Trap missing inputs at line 22-32 with `test -f` checks for all required parquets and JSON files.

---

### 3. Manifest and Provenance Emission

**Status: PARTIAL and FRAGMENTED**

#### Provenance sidecars implemented:
- **`run_pipeline1_inference.py`** (lines 998-1189): Comprehensive JSON sidecar
  - Emits: Git SHA, input file SHA-256, ensemble member SHAs, basis fingerprint, OOD flag rates, Regime B exclusion count, aux-missingness rates
  - File: `pipeline1_predictions_stream3.parquet.provenance.json`
  - Format: Clean, indexed by component (input, ensemble, frozen_stats, ood, regime_b, etc.)
  
- **`run_knn_rescue.py`** (lines 165-182): Minimal JSON sidecar
  - Emits: ensemble dir, member index, train/infer parquet paths (NOT SHA-256), K value, row counts, label order
  - File: `pipeline1_knn_rescue.parquet.knn_rescue.json`
  - **Missing**: Input/output file SHAs, Git SHA
  
- **`run_ensemble.py`** (lines 246-253, 313-314): Two separate files
  - `ensemble_config.json`: Config + git SHA + cfg hash (line 246-253)
  - `ensemble_history.json`: Val-loss summary + member paths (line 313-314)
  - **Missing**: Link back to training parquet SHA or data provenance
  
- **`build_hybrid_release.py`** (line 62-70): Manifest printed to stdout only
  - No persistent record of inputs + outputs at the script level
  - Delegates to `run_hybrid_release_pipeline()` in the library (line 62-69)
  - Library may emit `release_pipeline_manifest.json` but script does not verify

- **No end-to-end manifest**: The shell script does not emit a top-level `pipeline_run_manifest.json` tying all 6 stages
  - Cannot audit which inference + kNN output fed into which hybrid release
  - Cannot verify if stage 2 output SHA matches stage 3 input SHA
  
#### Missing cross-stage checksums:
- Stage 2 → Stage 3: No check that `$PRED_PARQUET` SHA matches kNN input expectations
- Stage 3 → Stage 4: No check that `$KNN_PARQUET` SHA matches hybrid input expectations
- Stage 4 → Tests: No check that hybrid output schema matches test expectations

**Recommendation**: 
- Add SHA-256 to `knn_rescue.py` sidecar (input + output files)
- Emit a root-level `full_pipeline_run_manifest.json` at script exit with stage entry + exit SHA-256s and timestamps
- Cross-check: verify stage 2 output path SHA matches stage 3 input reference before running stage 3

---

### 4. Staleness Detection

**Status: NONE**

The pipeline does not detect or warn when intermediate files are reused despite upstream changes:

#### Example 1: Hardcoded ensemble path
- Line 25: `ENSEMBLE_DIR="...20260425_6b96c06_cd1cbb9_ensemble_5label"`
- Dated tag + git SHA embedded in path; if v5 cycle retrains a new ensemble, the old path is stale
- No mechanism to auto-detect if `$ENSEMBLE_DIR` corresponds to the current codebase SHA
- Script only checks if the directory exists (line 35), not if it matches `$(git rev-parse HEAD)`

#### Example 2: Frozen stats not pinned
- Line 28: `FROZEN_STATS="...pipeline1_features_stream1.provenance.json"` (hardcoded path, not dated)
- Inference script loads and verifies basis fingerprint (line 844-845 in `run_pipeline1_inference.py`)
- But does not verify that the stats match the current training data provenance
- If Stream 1 is re-processed, the frozen stats become stale without warning

#### Example 3: No mtime comparison
- Stage 2 checks if `$PRED_PARQUET` exists but not if it's older than `$S3_FEATURES`
- If user updates Stream 3 features and re-runs, inference output is silently skipped (stale)
- No log warning: "predictions parquet (mtime X) is older than features (mtime Y); re-running stage 2"

#### Example 4: kNN always re-runs despite no change
- Stages 3-4-5 always execute even if all their inputs are unchanged
- Line 52-58: `run_knn_rescue.py` will recompute K=50 nearest neighbours even if `$TRAIN_PARQUET` and `$S3_FEATURES` are identical to the last run
- Wastes ~15 min of GPU time

**Recommendation**: 
- Emit `previous_run_manifest.json` and compare input SHAs at stage entry
- Log `WARN: stage 3 inputs unchanged from previous run; use --force to re-run`
- Add `--force` flag to force re-computation even if outputs exist

---

### 5. Dry-Run Mode

**Status: INCONSISTENT**

- **`run_ensemble.py`** (line 172): Has `--dry-run` flag (stops after config export, no training)
- **`run_pipeline1_inference.py`**: NO `--dry-run` flag (will always run inference)
- **`run_knn_rescue.py`**: NO `--dry-run` flag
- **`build_hybrid_release.py`**: NO `--dry-run` flag
- **`run_full_pipeline.sh`**: NO `--dry-run` mode for the entire orchestration

Users cannot validate the pipeline without waiting 2+ hours for full execution. Debugging a hardcoded path issue requires running all stages.

**Recommendation**: Add `--dry-run` flag to shell script that:
1. Validates all input paths exist
2. Counts rows in parquets without loading to GPU
3. Exports configs and provenance templates
4. Exits before any long-running operation

---

### 6. Progress Logging and Observability

**Status: ADEQUATE at stage level, MISSING at orchestration level**

Each Python script logs progress (e.g., line 254 in `run_ensemble.py`, line 825 in `run_pipeline1_inference.py`):
```
member seed=0: best_val_loss=0.1234 at epoch 5
loaded 5 ensemble members from ...
```

But the shell script lacks:
- Elapsed-time summaries per stage
- Total elapsed time at the end
- Start/end timestamps in the output
- A structured log file (not just stdout echo)
- Failure reason capture (if a stage fails, no summary of which stage + why)

**Recommendation**: Wrap each Python call with `time` and log to a structured JSON event stream or timestamp-indexed log file.

---

### 7. Error Handling and Failure Modes

**Status: PARTIAL**

#### What happens if stages fail:

- **Stage 1 fails** (ensemble missing): Script exits with code 1 (line 38)—caught correctly
  
- **Stage 2 fails** (inference panics): Script continues to stage 3 due to `set -euo pipefail` **not catching Python exit codes in the conditional** (line 43 checks only file existence, not whether stage 2 succeeded)
  - If inference crashes partway, `$PRED_PARQUET` is partially written (or not written at all)
  - Next run, line 43 check sees partial file and skips stage 2, then stage 3-4 fail with corrupted input
  
- **Stages 3-4-5-6 fail**: Script exits with the failing stage's exit code
  - `set -euo pipefail` correctly propagates (line 18)
  - But no cleanup of partial output files
  - Re-run must decide: delete stage N output or attempt to resume?

#### Missing failure recovery:
- No `trap` handler to clean up temp files on exit
- No rollback of partially-written Parquet files
- No status file to track which stages completed

**Recommendation**: Add a status checkpoint file (`pipeline_run.status.json`) that records stage entry/exit with timestamps and exit codes.

---

### 8. Circular and Cross-Stage Dependencies

**Status: LINEAR and EXPLICIT**

The DAG is simple: 1 → 2 → 3 → 4 → 5, 6 (tests).
No circular dependencies detected.
But: Tests (stage 6) depend on stage 4 output; tests are allowed to fail without blocking pipeline success (line 72-74 runs but does not exit on failure).

---

## Missing Artefacts and Examples

### Example: What a robust pipeline would look like

```bash
# At start:
PREV_MANIFEST="release/D-Cat-b/hybrid_pipeline_run/previous_run_manifest.json"
if [[ -f "$PREV_MANIFEST" ]]; then
    prev_s3_features_sha=$(jq -r '.inputs.s3_features.sha256' < "$PREV_MANIFEST")
    curr_s3_features_sha=$(sha256sum "$S3_FEATURES" | cut -d' ' -f1)
    if [[ "$prev_s3_features_sha" == "$curr_s3_features_sha" ]]; then
        echo "WARN: Stage 3 inputs unchanged; use --force to re-run"
        exit 0  # or read --force flag
    fi
fi

# Pre-flight validation:
for path in "$S3_FEATURES" "$FROZEN_STATS" "$TRAIN_PARQUET"; do
    if [[ ! -f "$path" ]]; then
        echo "FATAL: missing input $path"
        exit 1
    fi
done

# At end:
cat > "$HYBRID_OUT_DIR/run_manifest.json" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_sha": "$(git rev-parse HEAD)",
  "stages": [
    { "name": "ensemble", "status": "completed", ... },
    { "name": "inference", "status": "completed", "input_sha": "...", "output_sha": "..." },
    ...
  ],
  "elapsed_seconds": $((SECONDS))
}
EOF
```

---

## Severity and Impact Assessment

| Issue | Severity | Impact | Likelihood |
|-------|----------|--------|------------|
| No resume from stages 3-4 | HIGH | 30-60 min wasted per failed re-run | MEDIUM (happens after 2 hr run) |
| No input validation upfront | HIGH | Fails after 1+ hr of GPU compute | LOW (inputs usually present) |
| No staleness detection | HIGH | Silent reuse of stale intermediates | MEDIUM (happens with code/data updates) |
| No end-to-end manifest | MEDIUM | Cannot audit cross-stage data integrity | MEDIUM (matters for publication) |
| Missing dry-run mode | MEDIUM | Cannot validate paths without full run | HIGH (common during setup) |
| Partial provenance | MEDIUM | kNN sidecar lacks SHA-256 checksums | LOW (affects metadata, not science) |

---

## Recommendations (Priority Order)

1. **Add input validation at line 22-32** (before any computation)
   - Test all parquet and JSON paths: `test -f` or `python -c "import pandas; pandas.read_parquet(..., nrows=1)"`
   - Fail fast with clear error messages
   
2. **Add conditional execution for stages 3-4-5-6**
   - Accept `--force` / `--skip-existing` / `--resume` flags in shell script
   - Pass to Python drivers
   - Emit status checkpoint files after each stage
   
3. **Emit root-level end-to-end manifest**
   - Record all stage entry/exit with input/output SHA-256s
   - Save to `release/D-Cat-b/hybrid_pipeline_run/run_manifest.json`
   - Compare against previous run for staleness warning
   
4. **Add `--dry-run` mode to shell script**
   - Validates inputs without running long operations
   - Exports configs and dummy provenance
   
5. **Improve provenance in downstream scripts**
   - Add input SHA-256 to `run_knn_rescue.py` sidecar (line 165-182)
   - Track git SHA in `build_hybrid_release.py` output
   
6. **Add progress logging**
   - Use `time` builtin or `date` to measure stage duration
   - Log to structured file, not just stdout

---

## File Citations

- `/home/aneitzel/projects/ArqueoGal/scripts/run_full_pipeline.sh:43-50` — Stage 2 conditional (only existing check)
- `/home/aneitzel/projects/ArqueoGal/scripts/run_full_pipeline.sh:52-58` — Stage 3 always runs (no check)
- `/home/aneitzel/projects/ArqueoGal/scripts/run_knn_rescue.py:115-119` — Ensemble validation, but input parquets checked late (line 134)
- `/home/aneitzel/projects/ArqueoGal/scripts/run_pipeline1_inference.py:998-1189` — Comprehensive provenance (good model)
- `/home/aneitzel/projects/ArqueoGal/scripts/run_knn_rescue.py:165-182` — Minimal provenance (missing SHA-256)
- `/home/aneitzel/projects/ArqueoGal/scripts/build_hybrid_release.py:58` — Partial kNN check; no pred/feature check
- `/home/aneitzel/projects/ArqueoGal/scripts/run_ensemble.py:172, 256-258` — Only `--dry-run` flag in codebase
- `/home/aneitzel/projects/ArqueoGal/scripts/run_full_pipeline.sh:25, 28` — Hardcoded paths (no staleness detection)

---

## Conclusion

The pipeline **works end-to-end when inputs are present and correct**, but lacks the operational robustness required for reliable re-runs and debugging. The v5 ensemble retrain cycle will likely surface these issues (hardcoded ensemble path staleness). Adding checkpoint files, upfront validation, and conditional execution would reduce debugging time from hours to minutes.
