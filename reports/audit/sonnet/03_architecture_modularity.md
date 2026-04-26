# Architecture Modularity: Sonnet Meta-Report
## Synthesis of Four Haiku Audits on Cross-Stream Integrity

**Date:** 2026-04-26  
**Context:** Four independent Haiku audits (architect_review, architecture_patterns, adr_completeness, backend_architecture) have been cross-examined to produce synthesis of:
- Where audits triangulate on the same concern
- Disagreements or different framings
- Whether v5 release-tier simplification is a good refactoring moment for release.py
- Top 5 architectural items for public GitHub release
- Collectively missed items

---

## 1. Triangulation: Convergence Points

Three of the four audits independently identified the **same core modularity smell**: explicit responsibilities are well-defined and tested, but **reusable module responsibilities are not fully exposed to callers**.

### A. Constant Duplication (architect_review, architecture_patterns, backend_architecture)

**Finding**: Mirrored `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` in `release.py` and `release_pipeline.py` (architect_review, A; backend_architecture, leak #1).

**Audits agree**:
- architect_review: "The copy exists to avoid pulling `torch` into `release_pipeline`; guarded by a single test."
- architecture_patterns: No mention of this specific constant but notes "Configuration Encapsulation: Mostly Clean" with minor fields like `output_prefix` not justifying schema split.
- backend_architecture: Frames this as part of leak #1 — "orchestration logic duplicated in scripts vs. package" — proposing extraction to `master_schema.py`.

**Severity assigned**: architect_review calls it "non-critical"; backend_architecture calls it "medium" (part of broader duplication pattern).

**Consensus**: The test exists (`test_hybrid_thresholds_match_release`), so immediate risk is low. However, the existence of the constant in two places **is a symptom** that schema-driven validation (backend_architecture leak #1 mitigation) would eliminate it entirely.

---

### B. Schema/Config Not Exposed as Package-Level Contracts (backend_architecture, architecture_patterns)

**Finding**: `TrainingConfig`, `FeatureLayout`, and feature-column selections are created ad-hoc in scripts rather than injected into package functions (backend_architecture, leak #2; architecture_patterns, section 4).

**Audits agree**:
- backend_architecture: "Every new entry point (notebook, batch job, remote inference service) must reimplement config assembly, risking inconsistencies. Tests cannot inject mock configs without importing from scripts."
- architecture_patterns: "Config is serialised to JSON in checkpoints for round-trip reproducibility" — **but** the factory pattern (`from_checkpoint`) is missing.

**Consensus**: The problem is not that config is leaking into domain code (it isn't) but that it's not **re-hydrated** from checkpoints or provided as an injectable dependency. This forces downstream callers to reconstruct context.

---

### C. NaN-Sanitisation Contract Asymmetry (architecture_patterns, CLAUDE.md)

**Finding**: Training applies `np.nan_to_num` at the Dataset boundary; inference applies it post-aggregation. `XpFeatureAdapter` is a pass-through.

**arch_patterns audit**: "Add a shape-preserving `assert` in `XpFeatureAdapter.forward()`" to enforce the contract at the boundary where it matters.

**CLAUDE.md footgun §2**: Already documented as a known issue — but architect_review and backend_architecture did not flag this. This represents a gap in **architectural smell detection** across the four audits.

**Consensus**: architecture_patterns correctly identified this as a medium-severity boundary breach and proposed a one-liner fix (assertion). The lack of corroborating findings in the other three audits suggests it's a latent but not immediately urgent issue.

---

### D. Feature-Matrix Schema Not Linked to Release Annotation (backend_architecture)

**Finding**: `release_pipeline.py:54–78` hardcodes `_FEATURE_JOIN_COLS`; changes to feature builds don't automatically cascade (leak #4).

**Other audits**: No mention. architect_review and architecture_patterns focus on production-path logic (tier assignment, OOD flags) which are correctly isolated.

**Consensus**: This is a **low-to-medium risk item that only backend_architecture caught**. It's a dependency management issue, not an architectural flaw.

---

## 2. Disagreements and Different Framings

### A. Release.py Size Assessment

**architect_review** (lines 95–112): "The module is **well-documented and the v5 simplification removed 5 retired flags**, so the module is not bloating. However, the 682-line footprint approaches CLAUDE.md guidance on 'God modules'... **for now**, no blocking finding."

**Verdict**: Monitor for future growth; split only post-D-Cat-b if element promotion adds >10 new tier gates.

**architecture_patterns** (section 2, line 84): No size assessment. Focuses on responsibility separation: "training loop is properly stratified" and does not comment on release.py's 682 lines.

**backend_architecture**: No explicit size assessment but notes (line 198) "Release (Pipeline, Artefacts, Gallery) — Adequate but Implicit" with the weakness being implicit schema contracts, not size.

**Resolution**: architect_review is the only audit to perform a growth-trajectory analysis. The verdict (monitor, split post-D-Cat-b) is reasonable and not contradicted by the others.

---

### B. Orchestration Script vs. Package Boundary

**backend_architecture** (leak #2, lines 79–105): Frames this as a **severity: medium-high** issue — "Config objects created ad-hoc rather than injected" — proposes extracting `build_ensemble_config()` from scripts to package.

**architecture_patterns** (section 4, line 95): Does not flag this as a problem. Notes that "Config is serialised to JSON in checkpoints" and "schema splitting [does] not justify the benefit for a single field" — accepting the current script-driven approach as pragmatic.

**architect_review**: Does not address this.

**Disagreement**: backend_architecture sees the script-as-entry-point pattern as a modularity problem; architecture_patterns views it as acceptable for a scientific codebase where configs are checkpoint-embedded.

**Resolution**: Both are partially right. The config objects **are** well-designed (frozen dataclasses, serializable); the issue is that they're not **injectable** into package functions without the script layer. For public release (GitHub, external users), this is a legitimate modularity concern (backend_architecture is correct). For internal research workflows, the current pragmatism is acceptable (architecture_patterns is correct). **Recommendation for v5 release**: Refactor `build_ensemble_config()` into `src/arqueogal/xp_abundances/main/config.py` to improve modularity without imposing new constraints.

---

### C. Data Acquisition Layer Abstraction

**backend_architecture** (leak #3, lines 109–138): "**Severity: medium-high**. Raw TAP/HTTPS fetches and credential handling are entangled with business logic. New data sources cannot be plugged in without editing existing ingestion modules."

Proposes: `DataSourceContract` protocol, concrete sources (`GaiaXpSource`, etc.), refactor `ingest_stream1()` to accept source dict.

**architect_review, architecture_patterns**: Silent on this.

**Disagreement**: backend_architecture identifies a real but **not urgent** limitation (integrating new data sources requires code edits). The current design is **sufficient for production** because Gaia DR3 + APOGEE DR19 is the stable ingestion pair; DR4 or future data sources would trigger this refactor.

**Resolution**: Defer to post-release. backend_architecture's recommendation is correct in principle but premature for v5 release scope.

---

## 3. Should release.py Be Refactored During v5 Simplification?

**Evidence**:

- **architect_review verdict**: "No refactoring required for correctness; monitoring deferred to the next major element-promotion cycle." Split tier logic into `tier_logic.py` only if post-D-Cat-b element promotion adds >10 new gates.

- **v5 simplification context**: 5 retired flags removed (latent_support, ood_aux_mahalanobis, ood_disagreement, regime_b, mode_ambiguous). The module **got simpler**, not more complex.

- **Current size**: 682 lines for 3 orthogonal tasks (tier logic, auxiliary columns, parquet orchestration) is **not excessive** for a production-critical module. Compare: `uncertainty.py` at 117 outgoing edges (retained for methodology-only work).

**Recommendation: NO, do not refactor release.py during v5 release.**

Reasoning:
1. v5 simplification **reduced complexity** (5 flags removed); the module is stable.
2. Splitting now (to `tier_logic.py` + `release_annotations.py`) creates two new inter-module boundaries that must be tested.
3. Tier promotion is the next major event that would justify split (post-D-Cat-b, when new element gates are added).
4. The module has excellent test coverage; refactoring introduces risk with no immediate payload.

**Instead**: Document the 3 responsibilities clearly in the module docstring (already done per architect_review line 100) and monitor for >10 new tier gates.

---

## 4. Top 5 Architectural Items for Public GitHub Release

### 1. **Schema Constants Extraction** (backend_architecture, leak #1)

**Status**: Not done.  
**Effort**: Low (2–3h).  
**Payoff**: Eliminates duplication across 5 scripts; enables schema-driven validation.  
**Action**: Move `_APOGEE_RENAMES`, `PIPELINE1_RGB_WINDOW`, feature subsets, release flags to `src/arqueogal/data/master_schema.py` with accessor functions.  
**Why public release needs this**: External users integrating ArqueoGal data will need stable schema contracts. Hardcoded constants in scripts are not stable.

### 2. **Config Dependency Injection** (backend_architecture, leak #2; architecture_patterns, section 4)

**Status**: Partially done (config is well-structured but not injectable).  
**Effort**: Medium (4–6h).  
**Payoff**: Package functions callable from tests/notebooks/external drivers without script re-implementation.  
**Action**: Extract `build_ensemble_config()` to `src/arqueogal/xp_abundances/main/config.py`; add `FrozenStatsLoader` class; add `from_checkpoint()` factory to `FeatureLayout`.  
**Why public release needs this**: Users deploying the inference pipeline (e.g., to new data) need to programmatically construct configs. Script-only pattern is not accessible.

### 3. **NaN-Sanitisation Assertion** (architecture_patterns, finding 2)

**Status**: Not done.  
**Effort**: Minimal (1 assertion, 2 lines).  
**Payoff**: Catches NaN-propagation bugs at the boundary where they occur.  
**Action**: Add `assert torch.isfinite(x).all()` in `XpFeatureAdapter.forward()` with a clear error message.  
**Why public release needs this**: Inference is exposed to external data. A user's feature matrix with NaN in an auxiliary column will silently produce NaN predictions unless the boundary is guarded.

### 4. **Release Feature-Column Schema Link** (backend_architecture, leak #4)

**Status**: Not done.  
**Effort**: Low (1–2h).  
**Payoff**: Release pipeline resilient to feature-schema changes; single source of truth.  
**Action**: Add `release_required_feature_cols()` and `release_tier_flags()` accessors to `master_schema.py`; replace hardcoded `_FEATURE_JOIN_COLS` in `release_pipeline.py`.  
**Why public release needs this**: Release annotation is the final output stage. External users adding new features (e.g., photometric quality flags) need to know the contract.

### 5. **ADR-0009 Implementation Status Clarification** (adr_completeness, gap 1)

**Status**: Outstanding (section "Needs clarification" in ADR-0009).  
**Effort**: Minimal (verify code, update ADR status).  
**Payoff**: Removes ambiguity about whether PCA-CMI change landed.  
**Action**: Verify `audit.py` has PCA-CMI summary code; if present, remove "Needs clarification" flag and mark ADR-0009 as "Accepted, in production"; if deferred, update to "Accepted, pending implementation" with target date.  
**Why public release needs this**: The ADR corpus is a decision log for external users. Unresolved status flags reduce trust.

---

## 5. Items Collectively Missed

### A. XpFeatureAdapter NaN-Guarding in architect_review

architect_review does not flag the NaN-sanitisation asymmetry identified by architecture_patterns. This is a **coverage gap in architect_review's scope** (it focused on invariants 3, 14, 15, 16; NaN safety is not covered by a hard invariant but is documented in CLAUDE.md as a footgun §2). **Implication**: The nightly invariant audits should include a sweep of CLAUDE.md footguns to catch this class of issue.

### B. Monitoring Strategy for Constant Duplication (architect_review)

architect_review recommends keeping the test `test_hybrid_thresholds_match_release` in CI but does not propose a **structural refactor to eliminate the duplication**. This is pragmatic (test-only protection is sufficient); however, it leaves a **low-grade hygiene smell** that would be cleaned up by leak #1 refactor (schema constants to master_schema.py). **Implication**: The v5 release should bundled schema-extraction with constant consolidation.

### C. Cross-Stream Data Source Coupling

None of the audits flag the **implicit coupling of Stream 1, 2, 3 ingestion to the AIP TAP endpoint**. The invariant "all TAP queries use pyvo over astroquery" (CLAUDE.md invariant 10) is not ADR-formalized (adr_completeness, undocumented decision 1). While this is low-risk (AIP is stable and heavily tested), **public release should include an ADR justifying the pyvo-over-astroquery choice** to help external users understand why they must configure pyvo credentials, not astroquery's brittle auto-fallback.

### D. Checkpoint Format Stability

architect_review notes (lines 23–24) that `inference.py:49` imports `training.py:CHECKPOINT_VERSION` for backward-reference validation. However, none of the audits flag the need for a **checkpoint versioning policy** (e.g., "we support reading checkpoints from the last 3 major releases"). For public release, this should be documented.

---

## Summary

**Strengths** (all four audits agree):
- Invariants #3, #14, #15, #16 are upheld with clear test coverage.
- No circular dependencies; clean layering of domain, orchestration, infrastructure.
- Provenance discipline is systematic (write_sidecar called 43 times across src/).
- ADR corpus is 95% complete and internally consistent.

**Weaknesses requiring fix before public release**:
1. Schema constants duplicated across scripts (leak #1).
2. Config not injectable into package functions (leak #2).
3. NaN-sanitisation contract not enforced at boundary (finding from architecture_patterns).
4. Release feature-column contract implicit (leak #4).
5. ADR-0009 status ambiguous (adr_completeness, gap 1).

**Weaknesses acceptable for v5 but address in v6**:
- Data acquisition layer not abstracted (backend_architecture, leak #3; premature to refactor).
- release.py size (monitor post-D-Cat-b, only split if >10 new gates added).
- Orchestration script layer pattern (pragmatic for research; acceptable if config dependency injection is solved).

**v5 Refactoring Recommendation**: Do not split release.py. Instead, prioritize items 1–5 above (schema constants, config injection, NaN assertion, release-feature contract, ADR-0009 status). These are higher-payoff and lower-risk refactors that improve public-release readiness without destabilizing production code.

---

**Cited Audits**:
- architect_review.md — Invariant compliance, constant duplication, release.py size assessment.
- architecture_patterns.md — Dependency discipline, NaN-sanitisation asymmetry, orchestration/computation separation.
- adr_completeness.md — ADR corpus status, evidence traceability, undocumented decisions.
- backend_architecture.md — Service boundary leaks, config/data flow, schema-driven validation gaps.
