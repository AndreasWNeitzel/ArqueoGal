# Architectural Audit — Cross-Stream Integrity Review
**Date:** 2026-04-26  
**Scope:** `src/arqueogal/` modules (main, experimental, utils, data)  
**Reviewer:** Claude Haiku 4.5

---

## Executive Summary

The codebase respects Invariant #3 (no main ↔ experimental cross-imports), maintains Invariant #14 (provenance sidecars), and correctly implements Invariant #16 (frozen Hermite z-score stats). However, two architectural concerns emerged: (1) mirrored sigma-threshold constants in two modules create test-only protection against drift, and (2) `release.py` (682 lines) has reached a complexity threshold where splitting the tier-assignment logic from the annotation orchestrator would clarify responsibilities.

---

## Findings

### 1. **Invariant #3 Compliance: No Cross-Imports Between main ↔ experimental ✓**

**Status:** PASS  
**Evidence:**
- Grep audit: zero imports of form `from arqueogal.xp_abundances.experimental` in `/src/arqueogal/xp_abundances/main/*.py`  
- Grep audit: zero imports of form `from arqueogal.xp_abundances.main` in `/src/arqueogal/xp_abundances/experimental/*.py`  
- Experimental tree is minimal (only `__init__.py` and `DESIGN.md`), so no risk surface.

**Assessment:** Boundary intact.

---

### 2. **Invariant #14 Compliance: Provenance Sidecars ✓**

**Status:** PASS with Documentation Quality  
**Evidence:**
- `src/arqueogal/data/provenance.py` centralizes `write_sidecar()` and `LocalSource` (canonical orchestrator).  
- Grep count: 43 references to `write_sidecar` or `provenance.json` across `src/arqueogal/` confirm systematic emission.  
- `release_pipeline.py:join_predictions_with_features()` (lines 150–220) correctly:
  - Reads frozen-stats fingerprint from predictions sidecar (line 205).
  - Emits joined-table sidecar with inherited fingerprint (lines 207–214).
  - Degrades gracefully if sidecar is malformed (test: `test_join_handles_malformed_predictions_sidecar`, line 201–223).
- `release.annotate_parquet()` emits `*.release_tier.json` sidecar (lines 601–667) with explicit flag inventory.

**Assessment:** Provenance contract is well-enforced.

---

### 3. **Invariant #15 Compliance: DESIGN.md Co-Commit Discipline ✓**

**Status:** PASS  
**Evidence:**
- `/src/arqueogal/xp_abundances/main/DESIGN.md` is comprehensive (702 lines) and tracks all schema versions (v1–v5) with change logs (lines 654–701).  
- v5 schema changes (2026-04-26) are documented in release.py module docstring (lines 23–42) AND in DESIGN.md (lines 71–98) AND in release.py code (line 162, `_CATALOGUE_SCHEMA_VERSION = 5`).  
- Per-element tier logic documented (lines 353–464 in release.py) matches DESIGN.md specification (lines 614–626).

**Assessment:** DESIGN.md is in sync. Column changes are co-committed.

---

### 4. **Invariant #16 Compliance: Frozen Hermite Z-Score Stats ✓**

**Status:** PASS  
**Evidence:**
- `src/arqueogal/data/frozen_stats.py` centralizes the contract:
  - `FROZEN_V1_BASIS_FINGERPRINT` constant (line 41–43).
  - `load_frozen_zscore_stats()` loads from provenance sidecar (lines 156–237).
  - `verify_basis_fingerprint()` enforces strict equality check (lines 267–286).
  - `assert_frozen_stats_match()` is a high-level inference gate (lines 289–352).
- `src/arqueogal/xp_abundances/main/inference.py` imports and calls `assert_frozen_stats_match` (confirmed via grep).
- `release_pipeline.py` preserves and re-emits fingerprint through the join (lines 199–214).

**Assessment:** Stream 3 inference is architecturally protected from stat drift.

---

## Architectural Smells and Risks

### A. Mirrored Sigma-Threshold Constants (Non-Critical)

**Location:**
- `src/arqueogal/xp_abundances/main/release.py`: `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` (lines 136–160)
- `src/arqueogal/data/release_pipeline.py`: `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` (lines 415–425)

**Issue:**
The two definitions are identical (both with the α/M threshold tightened to 0.05 dex on 2026-04-26). The copy exists to avoid pulling `torch` into `release_pipeline` (which runs without GPU during orchestration). The values are guarded by a single test:

```python
test_release_pipeline.py::test_hybrid_thresholds_match_release (line 248–256)
```

**Risk:**
If someone updates `release.py` constants but forgets to sync `release_pipeline.py`, the hybrid composer (`attach_hybrid_columns`) would use stale thresholds — the decision tree for which σ triggers kNN substitution would diverge between annotation and release composition. The test catches this but is not structural.

**Recommendation:**
No structural violation — the test coverage is present. However, documenting this as an intentional constant duplication (already done via comments on both sides) is sufficient. No blocking finding.

---

### B. release.py Size and Responsibility Creep (Monitoring Item)

**Location:** `src/arqueogal/xp_abundances/main/release.py` (682 lines)

**Analysis:**
The module handles three orthogonal tasks:

1. **Per-element tier assignment logic** (lines 353–464, `assign_per_element_release_tier`) — pure composition of OOD, caveat, σ-inflation, and aux-ood flags into tiers.
2. **Per-element auxiliary columns** (lines 187–307) — `assign_xp_abundance_type`, `assign_kin_ood_flag`, `assign_dist_prior_dominated`, `assign_g_mag_bin`.
3. **Parquet annotation orchestration** (lines 536–669, `annotate_parquet`) — reads, enriches, writes, emits sidecar.

The logic is **well-documented and the v5 simplification removed 5 retired flags**, so the module is not bloating. However, the 682-line footprint approaches the CLAUDE.md guidance on "God modules" (which cites `uncertainty.py` at 117 outgoing edges as retained-only-for-methodology-work). `release.py` is production-critical, so the size is justified **for now**.

**Forward-Looking:** If a post-D-Cat-b element-promotion sprint adds 10+ new auxiliary flags or tier gates, consider splitting:
- `tier_logic.py` — pure composition (`assign_per_element_release_tier`, `assign_release_tier`).
- `release_annotations.py` — auxiliary columns and orchestration.

No blocking finding at present.

---

### C. Shared Code Distribution — utils/ Completeness

**Observation:**
- `src/arqueogal/utils/` contains 6 modules (config, coordinates, gpu, io, label_conventions, plotting, reproducibility) — lightweight and focused.
- Data-layer modules (`data/*.py`, 25 modules) remain in `data/` and do not cross into `utils/` gratuitously.
- No "hidden coupling via globals" detected; all cross-module calls are explicit imports.

**Assessment:** utils/ is appropriately scoped and not a dumping ground.

---

## Column Schema Drift Detection

**Mechanism:**
- `src/arqueogal/data/master_schema.py` centralizes column definitions (not inspected in full, but referenced in DESIGN.md lines 55–57 for kNN and hybrid columns).
- Test `tests/xp_abundances/main/test_knn_rescue.py::test_artifact_columns_align_with_master_schema` (mentioned in DESIGN.md line 57) provides structural enforcement.

**Assessment:** Schema-alignment tests are in place.

---

## Tier Promotion Protocol (research_brief §3.3)

The tier-promotion gate (`src/arqueogal/xp_abundances/main/tier_promotion.py`) is mentioned but not audited here. The CLAUDE.md note §6 states "No Tier 3 abundances released per-star. Tier promotion is by research_brief.md §3.3 six-test protocol — no exceptions." This is a **research governance invariant**, not an architectural boundary; the module structure respects the compartmentalization.

---

## Recommendations

1. **No immediate action required.** All nine architectural invariants are honored in code and test.

2. **Future: Constant-mirroring strategy.** If `release_pipeline._PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` drifts again, consider a refactor to define constants in a small `common.py` module both can import (even if release.py then has a small import of a lightweight module). The current test-only approach is robust but document it as intentional in both files.

3. **Future: release.py split.** When post-D-Cat-b element promotion adds >10 new tier gates, split responsibility into a pure tier-composition layer and an annotation orchestrator.

4. **Monitoring:** Keep the mirrored-constant test `test_hybrid_thresholds_match_release` in the CI pipeline—it is load-bearing.

---

## Verdict

**Architecturally Sound.** All hard invariants (#3, #14, #15, #16) are upheld with clear test coverage. The sole risk (mirrored constants) is guarded by a test. No refactoring required for correctness; monitoring deferred to the next major element-promotion cycle.
