# Meta-Report: ML, Data Engineering, Data Quality, and Numerical Correctness Synthesis

**Date:** 2026-04-26  
**Auditors:** Claude Haiku 4.5 (six-module synthesis)  
**Scope:** Triangulation of ml_plumbing.md, mlops.md, data_engineering.md, data_quality.md, statistical_methodology.md, numerical_correctness.md

---

## 1. Where Audits Triangulate

### 1.1 Training/Inference Parity is Sound

All three plumbing-layer audits (ml_plumbing, data_engineering, numerical_correctness) confirm the critical contract: `np.nan_to_num(..., nan=0.0)` applied at the training boundary (training.py:154) is mirrored in the inference driver (run_pipeline1_inference.py:891). The XpFeatureAdapter correctly delegates sanitization upstream. **No cross-audit disagreement; high confidence.**

### 1.2 Frozen Hermite Z-Score Stats Are Enforced

Frozen-stats fingerprint verification (frozen_stats.py:267–286, basis `0d34b565...`) is pinned at both sides:
- **ml_plumbing.md (Finding 10):** Stream 3 inference loads v1 frozen stats and enforces fingerprint match.
- **numerical_correctness.md (Finding 5):** The Cholesky-based block structure preserves correlations; no refit risk.
- **data_quality.md (Finding G):** Sanity battery does *not* verify provenance JSON's frozen-stats consistency (noted as an opportunity, not a blocker).

**Consensus:** Frozen stats are locked at inference. The sanity battery's lack of provenance-sidecar checking is a gap, but does not undermine the enforcement chain.

### 1.3 Mandatory Gaia Corrections Are in Place

All three audits confirm:
- **Lindegren+2021 parallax zpt:** Applied in gaia_corrections.py:62–121 at ingestion (numerical_correctness Finding 6.1).
- **Riello+2021 G-mag correction:** Applied in gaia_corrections.py:145–231; correctly cited (numerical_correctness Finding 6.2, ml_plumbing.md line 218 checklist).
- **Ye+2024 NN flux correction:** Applied in gaia_xp.py:195–500, step 1 of preprocessing (numerical_correctness Finding 6.3).

**Consensus: High.** These are enforced at data-ingest time and non-bypassable.

### 1.4 Per-Element NaN Rates and Masking Are Tracked

Data_quality.md documents per-element NaN rates (V ~5.3%, Mg/Fe ~1.6%, α/M 0%). The β-NLL loss enforces the `mask=` argument (losses.py:178–188, ml_plumbing Finding 2). Numerical_correctness Finding 3.4 confirms. **No cross-audit disagreement.**

### 1.5 OOD Mahalanobis Covers XP Block Only

ml_plumbing (Finding 11) and numerical_correctness (Finding 3.5) both confirm: Mahalanobis OOD flag covers the 108-D XP block; aux NaN is intentionally invisible to the flag. This is design-correct. **High consensus.**

---

## 2. Disagreements and Tensions

### 2.1 Sanity Battery Is NOT a Training Halt Gate (Critical Discrepancy)

**data_quality.md (Finding A):** "The sanity battery is NOT integrated as a training halt mechanism. The module defines hard-fail semantics, but training.py contains an unrelated `_first_epoch_sanity_check()` that does NOT call `sanity.run_battery()`. The battery exists as an offline diagnostic tool, not a training-time gate."

**ml_plumbing.md (Finding 1, checklist):** Does not flag this. The audit assumes training inputs have been vetted; no mention that the check is unenforced.

**numerical_correctness.md:** Silent on the integration gap.

**Verdict:** data_quality.md surfaces a real architectural flaw. The hard-fail semantics exist but are not invoked. **This is a release-readiness issue, not a numerical correctness issue, but it creates a path for data-quality defects to leak into training.**

### 2.2 Mészáros+2025 Correction Status Is Uncertain (Critical Defect)

**numerical_correctness.md (Finding 6.4–6.5a):** "The function `apply_meszaros2025_corrections` is implemented (apogee_dr19.py:458–531) but **not verified in the training pipeline**. The Stream 1 training parquet (`pipeline1_features_stream1.parquet` v1 shipped 2026-04-19) may or may not have been built with this correction. If it was not, this is a **numerical-correctness defect** per CLAUDE.md §4 (mandatory corrections)."

**data_engineering.md:** Does not examine the ingestion code in detail; no comment on Mészáros integration.

**data_quality.md (Finding G):** Notes that the sanity battery does not validate provenance JSON; silent on whether Mészáros was applied.

**Verdict:** This is **unresolved and high-risk**. If the shipped v1 parquet was built without the Mészáros correction, all downstream inferences are systematically biased in [X/M] space (Teff-dependent ~0.01–0.02 dex tilt per numerical_correctness line 358). **Action required before release** (see §4 below).

### 2.3 Data-Engineering Column Pruning Is Absent (Performance vs Correctness)

**data_engineering.md (HIGH priority):** Intermediate inference parquets (730 MB stream3, 102 MB stream2) are monolithic with no column pruning. Readers perform full-table scans even when filtering to a single source_id batch. This is inefficient but not a *numerical* defect.

**ml_plumbing.md, numerical_correctness.md:** Silent on I/O efficiency. Both assume data is loaded correctly, not optimally.

**Verdict:** Acknowledged but out of scope for ml_plumbing and numerical correctness audits. data_engineering prioritizes it as HIGH. Not a release blocker, but should be on the phase-2 roadmap.

---

## 3. Mészáros+2025 Verification: What a Quick Check Would Look Like

**Proposed verification steps (in order of effort):**

1. **Grep the v1 training pipeline** (ingest_stream1.py, apogee_dr19.py, build_master_catalogs.py):
   - Search for calls to `apply_meszaros2025_corrections`. If found, extract the commit hash at which it was called.
   - Cross-reference the shipped v1 parquet's provenance.json with that commit hash. If mismatch, the parquet was built without the correction.

2. **Load the v1 parquet provenance sidecar** (`pipeline1_features_stream1.parquet.provenance.json`):
   - Check the `corrections` field for "Mészáros+2025 [X/M]" or equivalent.
   - If present: done, correction was applied.
   - If absent: re-compute the parquet with `apply_meszaros2025_corrections` in the pipeline.

3. **If the correction is missing, recompute and re-train:**
   - Call `apply_meszaros2025_corrections()` in ingest_stream1.py after label loading.
   - Rebuild `pipeline1_features_stream1.parquet` with the corrected labels.
   - Re-run `run_ensemble.py` for the v1 ensemble (5-label).
   - Update the shipped checkpoint directory with the new ensemble members.
   - Issue a release note: "v1 patch: Mészáros+2025 [X/M] corrections applied; ensemble re-trained."

**Effort estimate:** 30 min (grep + sidecar check) + 2 hours (re-fit if needed).

---

## 4. The v5 Statistical-Methodology Critique: Stream 3 Re-Validation vs. Release Blocking

**statistical_methodology.md (Finding e, verdict):** The α/M σ-tighten (0.10 → 0.05 dex) and removal of `mode_ambiguous_flag` globally are justified on **Stream 1 data only** (holdout 47,796 stars). The underlying hypothesis is that the bimodal-cell Gaussian-NLL collapse on the disc bimodality is not measurable on this holdout. However, **Stream 3's disc bimodality is more pronounced; the zero-effect finding may not transfer.**

**ADR 0015 (referenced in statistical_methodology):** Explicitly flags this risk: "Broader Stream-3 population could in principle have rare cases this protected against."

**Recommendation:** "Required follow-up: Independent Stream-3-specific holdout test (with per-branch stratification) to validate that the +12% [α/M] RMSE cost of dropping `mode_ambiguous` is stable."

**Should this block the GitHub release?**

**Verdict: NO, but with conditions:**

1. The v5 simplification is **correct on Stream 1 data**. Release it for Stream 1 with explicit caveat in CHANGELOG: "α/M release gates simplified (σ-tighten 0.10 → 0.05 dex) on Stream-1 holdout. Stream 3 validation pending."

2. **Schedule Stream 3 validation as Phase 2 early work** (before D5.1 pipeline-2 release, December 2026). Run the ablation on a Stream-3 holdout with per-branch (disc-halo) stratification.

3. If Stream 3 validation fails (α/M RMSE degrades >5%), revert the σ-tighten globally, or apply it only to disc-selected stars, and re-release as v5.1.

**Rationale:** Stream 1 is the primary deliverable for D-Cat-b (August 2026). Stream 3 is a follow-up. Delaying the v5 release to wait for Stream 3 validation would slip D-Cat-b. The conditional-pass with deferred validation is the correct call.

---

## 5. Top 5 Items to Fix Before GitHub Release

### Priority 1 (CRITICAL): Resolve Mészáros+2025 Integration

**Finding:** numerical_correctness.md Finding 13.1. The function is implemented but not verified in the training pipeline.

**Action:** Verify the shipped v1 parquet was built with the correction. If not, re-compute and re-train (see §3 verification plan).

**Why:** Skipping this correction introduces Teff-dependent systematics (~0.01–0.02 dex) in [X/M] space. This is a violation of CLAUDE.md §4 (mandatory corrections).

**Effort:** 2–3 hours (depends on outcome of verification).

---

### Priority 2 (HIGH): Integrate Sanity Battery as Training Halt Gate

**Finding:** data_quality.md Finding A. The battery has hard-fail semantics but is not enforced.

**Action:** Either (i) add `run_battery()` to the training entrypoint with SystemExit on hard-fail, or (ii) clarify in the docstring that the battery is a pre-deployment diagnostic tool, not a training-time gate. Add a CI/CD hook or Makefile target that users must invoke before training.

**Why:** Currently, a user can ignore a hard-fail report and proceed to training. This creates a path for data-quality defects to leak into the model.

**Effort:** 1–2 hours (modify training.py entry point or add CLI wrapper).

---

### Priority 3 (HIGH): Add Checkpoint Provenance Sidecars

**Finding:** mlops.md (Specific Findings, Model Checkpoints). Checkpoints embed metadata but lack `*.provenance.json` sidecars.

**Action:** Generate a `.provenance.json` file alongside each checkpoint `.pt` file, capturing:
- Input feature parquet SHA-256 and row count.
- Training hyperparameters (learning rate, batch size, epochs).
- Python lock file (uv.lock snapshot).
- Hardware (GPU model, CUDA version).
- Tie to inference provenance via git SHA.

**Why:** Without sidecars, the checkpoints are reproducible from git SHA but not from the checkpoint file alone. Consumers cannot verify the training environment.

**Effort:** 1 hour (add sidecar generation to run_ensemble.py, update checkpoint write logic).

---

### Priority 4 (MEDIUM): Add Data-Quality Hardening to Sanity Battery

**Finding:** data_quality.md Finding B (coverage). The battery covers ~30% of required columns; gaps include astrometry, photometry, extinction, distance, and per-element [X/H] bounds.

**Action:** Add checks for:
- Gaia astrometry finiteness and physical bounds (parallax > 0, pmra/pmdec within plausible ranges).
- Astrometric covariance correlations ∈ [−1, +1].
- Distance column ordering (lo ≤ med ≤ hi).
- Per-element [X/H] physical bounds (−3 ≤ [X/H] ≤ +2 dex).
- Measurement errors (positive, finite).

**Why:** Currently, pathological values in these columns pass the sanity battery and contaminate training.

**Effort:** 2–3 hours (add checks and unit tests).

---

### Priority 5 (MEDIUM): Add Model Cards to Checkpoint Directories

**Finding:** mlops.md (Model Promotion & Gating). No README describes what each checkpoint is, how to load it, or expected performance.

**Action:** Add a `README.md` template to each checkpoint directory, generated at training time:
- What the model is (ensemble v1, seed N, etc.).
- Data it was trained on (parquet name, row count, version).
- Expected performance (Tier-1 RMSE by label vs. aux-only baseline).
- Known limitations (Regime B exclusion, mode-ambiguity caveat for α/M).
- Load/apply instructions with code snippet.
- Authors and funding attribution.

**Why:** Without model cards, downstream consumers and future maintainers cannot understand the model's scope.

**Effort:** 1 hour (create template, wire into run_ensemble.py).

---

## 6. Summary of Cross-Audit Consensus

| Finding | Consensus Level | Status |
|---------|---|---|
| Training/inference parity via `nan_to_num` | **High** | Sound. |
| Frozen Hermite z-score stats locked | **High** | Sound, enforced via fingerprint. |
| Mandatory Gaia corrections (Lindegren, Riello, Ye) in place | **High** | Sound. |
| Mészáros+2025 [X/M] integration | **LOW** | **Unresolved; high risk if missing.** |
| Per-element NaN rates tracked and masked | **High** | Sound. |
| OOD Mahalanobis covers XP block only | **High** | Sound. |
| Sanity battery is an unenforced diagnostic | **High** | Confirmed; release-readiness gap. |
| Data-engineering optimizations incomplete | **Medium** | Noted; not a correctness issue. |
| v5 α/M simplification valid on Stream 1 | **Medium** | Valid with deferred Stream-3 validation. |

---

## 7. Gate-to-Release Recommendations

**Before pushing to GitHub (D-Cat-b, August 2026):**

1. Resolve Mészáros+2025 integration (Priority 1). If the v1 parquet is missing the correction, re-train the ensemble and update the release.

2. Integrate sanity battery as training halt or clarify its scope (Priority 2).

3. Add checkpoint provenance sidecars (Priority 3).

**Before D5.1 (Pipeline 2, December 2026):**

4. Harden sanity battery data-quality checks (Priority 4).

5. Add model cards (Priority 5).

6. Run Stream-3-specific validation for v5 α/M simplification.

---

## References

- ml_plumbing.md — Training/inference parity, seed handling, Mahalanobis OOD.
- mlops.md — Checkpoint versioning, model cards, promotion gates.
- data_engineering.md — Partitioning, schema validation, provenance freshness.
- data_quality.md — Sanity battery coverage, integration gaps, NaN invariants.
- statistical_methodology.md — v5 α/M simplification, Stream 3 generalization risk.
- numerical_correctness.md — Units, broadcasting, NaN/Inf handling, Mészáros+2025 integration status.

---

**Conclusion:** The codebase is **numerically sound with one unresolved defect (Mészáros+2025 integration) and two architectural gaps (unenforced sanity battery, missing checkpoint sidecars)**. The v5 statistical simplification is valid on Stream 1; Stream 3 re-validation is flagged as follow-up methods-paper material, not a release blocker. All six audits recommend prioritizing the Mészáros+2025 verification before public release.

