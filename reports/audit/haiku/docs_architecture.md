# ArqueoGal Documentation Architecture Audit

**Audit date:** 2026-04-26  
**Scope:** `/home/aneitzel/projects/ArqueoGal/docs/` + `README.md` + module-level `DESIGN.md` files  
**Auditor:** Claude (Haiku 4.5)

---

## Executive Summary

The ArqueoGal documentation is well-structured and comprehensive at the specialist level, with clear separation of concerns (research rationale, data contracts, design decisions, phase planning). However, three critical gaps prevent new users from adopting the repository: (1) no Getting Started or Quick Start guide for first-time setup and a minimal reproduction run, (2) no "Reproducing the Catalog" walkthrough that chains together ingestion → training → inference as a single narrative, and (3) primary entry points are scattered across `README.md`, `docs/plan/00_overview.md`, and `docs/data_overview.md` with no single canonical onboarding path. Additionally, three documents show staleness relative to the April 2026 joint-loss ensemble rebuild (v1→v2→v5), and module DESIGN.md files exist but carry zero internal cross-links from parent docs.

---

## 1. Navigation and Entry Points

### 1.1 Current structure

The README correctly identifies three "start here" links:
- `docs/data_overview.md` — visual tour (8 panels)
- `docs/research_brief.md` — scientific rationale, tier protocol, audit protocol
- `docs/data_acquisition.md` — TAP queries, preprocessing, 5 GB budget

Supporting structure:
- `docs/plan/00_overview.md` — current phase status (last updated 2026-04-26)
- `docs/plan/{01–06}_*.md` — per-phase narrative (01: v1 shipped and superseded; 02: 5/6 audit tests; 03: Stream 3 Phase 3; 04: Starfold integration pointer; 05: release packaging underspecified; 06: methods paper tracked)
- `docs/decisions/` — 15 ADRs (0001–0015), fully cross-linked within decisions
- `docs/context/` — architecture.md (diagram-free prose), conventions.md (enforceable rules), open_questions.md (user decisions needed)
- `docs/protocols/` — 4 validation recipes (cross_catalogue_test6, fisher_crlb, open_cluster_benchmark, tau_sweep)
- Per-module `src/arqueogal/{data,utils,xp_abundances}/DESIGN.md` — 5 design docs at module level, plus subdirectory `xp_abundances/{main,experimental}/DESIGN.md`
- `docs/CATALOG_SCHEMA.md` — release parquet column contract (v4, dated 2026-04-25)
- `docs/HARDWARE_SETUP.md` — WSL2 memory, filesystem, monitoring recipes (audit 2026-04-24)

### 1.2 The gap: no onboarding narrative

A new user landing on the GitHub repo sees:
1. A 3-link README "start here" that assumes familiarity with scientific context.
2. No "I want to run a smoke test to verify the environment works" document.
3. No "I want to reproduce the Pipeline 1 v1 catalogue from scratch" checklist.
4. No "What if I want to extend this to a new element?" guide (though research_brief.md §3.3 has the protocol, it is embedded in methodology, not as a standalone task).

The README links to research_brief.md as the scientific entry point, but that document's §0 Executive Summary immediately disambiguates three mechanisms (genuine spectral info vs training-set priors vs survey leakage) and says *"our pipeline will be the first XP-abundance catalogue to require an information-content audit"* — expert framing. First-time readers trying to understand "what do I need to do to run this code?" will get lost.

### 1.3 Module-level DESIGN.md files exist but are orphaned

Five DESIGN.md files exist in the codebase:
- `/src/arqueogal/data/DESIGN.md` — data ingestion contracts
- `/src/arqueogal/utils/DESIGN.md` — utilities
- `/src/arqueogal/xp_abundances/DESIGN.md` — Pipeline 1 top-level architecture
- `/src/arqueogal/xp_abundances/main/DESIGN.md` — production model
- `/src/arqueogal/xp_abundances/experimental/DESIGN.md` — experimental branches

None of these files are referenced from the primary documentation tree (research_brief, data_acquisition, plan/, decisions/, context/). A user reading `docs/context/architecture.md` (which describes Pipeline 1 data flow) has no pointer to `src/arqueogal/xp_abundances/DESIGN.md`. The current prose says "Durable prose reference to the system. Diagram-free to survive refactors" but does not link outward to the code-level design contracts.

---

## 2. Staleness Relative to April 2026 Rebuild

### 2.1 Pipeline 1 v1 is superseded; docs track v1 lineage but v2/v5 are under-documented

The April 19 v1 ensemble collapsed [α/M] bimodality (72% attractor-stripe on metal-poor stars). The April 25 v2 rebuild (`strong-contrastive-v2`) reduced the prior-collapse spike from 18.32% → 0.20% with SupCon=1.0, Barlow=0.5. The April 26 v5 schema simplified release-tier gating (retired 6 columns, tightened [α/M] σ-threshold 0.10 → 0.05 dex).

Where these appear in docs:
- `docs/plan/00_overview.md` — **fully updated** (2026-04-26): v1 tagged, v2 shipping 2026-04-25, v5 ablation study complete.
- `docs/plan/01_pipeline1_v1.md` — **marked "shipped but superseded"**, correctly framed, but the file itself is largely unchanged; a reader wanting to know "why did v1 fail?" must cross-reference back to 00_overview.md.
- `docs/research_brief.md` — **v2 revision date April 2026**, but the text itself does not mention the attractor-stripe failure, the April-25 rebuild, or the April-26 v5 schema changes. Section 3.3.2 on XP preprocessing is correctly cited as §6.4 of data_acquisition.md, but the "information-content audit" protocol in §9 references "Tier 1/2/3 tagging" without explicitly saying "Tier decisions ratified in May 2026 per ADR-0015" or "prior Tier definitions were more complex; see archive/ for v1 tier gates."
- `docs/data_overview.md` — **static visual tour**, no versioning. Eight PNG panels generated by `scripts/plot_data_overview.py`. Panel titles reference Stream 1, Stream 2, Stream 3 but never mention which model version the figures illustrate (though the data streams themselves are version-agnostic, the "label availability matrix" and "magnitude distribution" panels implicitly assume Stream 1 v1 composition).
- `docs/CATALOG_SCHEMA.md` — **v4 dated 2026-04-25**, correctly tracks that v2–v4 columns were added sequentially. However, it does not list which columns were removed between v1 and v5 (see ADR-0015 for the list). A user reading the schema has no easy way to know "this column `latent_support_flag` was in v1 but was retired in v5."
- `docs/decisions/0015_v5_release_tier_simplification.md` — **authored**, correctly documents retired columns and new σ-thresholds. However, CATALOG_SCHEMA.md does not reference ADR-0015 (the reader must cross-reference manually).

### 2.2 Inference and release procedures under-specified

`docs/plan/05_release_packaging.md` is openly marked as **"underspecified"** (section "Phase status"). The document lists three deliverables (D-Cat-b, D5.1, D-Cat-d) but does not provide the concrete sequence:

1. How to invoke `run_pipeline1_inference.py` for Stream 3 (the command-line args, the expected output schema, where the parquet lands).
2. How to invoke `release.py` (or equivalent) to convert raw predictions to release tiers.
3. How to validate release tiers (which tests must pass, which columns are mandatory, what the `*.release_tier.json` sidecar must contain).
4. Where D-Cat-b catalogue outputs land in `/data/processed/` and what the filename convention is (version tag, date, git SHA).

The implied workflow exists (evidenced by the April-25 Stream 3 inference run and the April-26 ablation study), but the standard operating procedure is not written down for reproducibility by a new user or a future maintainer.

### 2.3 Missing link: protocols/ folder exists but is not referenced from the plan

Four validation recipes exist in `docs/protocols/`:
- `cross_catalogue_test6.md` — compares against AspGap, Guiglion+2024, SHBoost, GALAH DR4
- `fisher_crlb.md` — Fisher information / Cramér-Rao lower bound analysis
- `open_cluster_benchmark.md` — open-cluster precision floor (research_brief.md §3.3 test 3)
- `tau_sweep.md` — shrinkage hyperparameter sweep (related to calibration, ADR-0003)

These files are not linked from `docs/plan/02_pipeline1_audit.md` (which describes the §9.2 six-test audit protocol). A user reading plan/02 to understand "what does the audit entail?" will not find a direct path to the protocol implementations.

---

## 3. Overlap and Potential Drift

### 3.1 data_acquisition.md vs CATALOG_SCHEMA.md

Both documents describe data provenance and column definitions:

- `data_acquisition.md` — **5 GB budget accounting (§12), TAP query recipes (§3), Gaia corrections (§4), APOGEE DR19 HDU 2 pre-baked columns (§5), external priors (§8), XP preprocessing order (§6.4), three-stream ingestion logic (§7).**
- `CATALOG_SCHEMA.md` — **column-by-column reference for released parquet: identifiers, astrometry, photometry, XP coefficients, distance, extinction, diagnostics, quality flags, predicted abundances, uncertainty covariance, OOD flags, release tiers.**

Potential drift points:
1. `data_acquisition.md` §5 says DR19 HDU 2 pre-bakes "Gaia astrometry, 2MASS/WISE photometry, four dust maps, Bailer-Jones distances for APOGEE stars" but does not enumerate which four dust maps (only Edenhofer, Lallement, SFD are spelled out in §8.5). CATALOG_SCHEMA.md §Extinction clarifies the four-map fusion (Edenhofer < 1.25 kpc, Lallement 1.25–3 kpc, SFD > 3 kpc, neighborhood-median Av) but says "essential for G-magnitude color-excess correction when per-map A_V unavailable" — this rationale is not explained in data_acquisition.md.
2. `data_acquisition.md` does not mention Mészáros+2025 [X/M] corrections (mandatory per CLAUDE.md invariant 13). The corrections are cited in the README and in research_brief.md §2.1 but not sequenced into the data pipeline steps in data_acquisition.md.
3. CATALOG_SCHEMA.md lists `evolutionary_stage_andrae` as a diagnostic column (optional), but data_acquisition.md does not explain when/why this column is fetched or how it relates to the Pipeline 1 regime-B envelope (ADR-0004).

### 3.2 research_brief.md vs decisions/ ADRs

`research_brief.md` is framed as the "primary scientific reference" (CLAUDE.md) but does not link to the 15 ADRs, even though many architectural decisions originate in that reference:

- §2.1 table "What exists and what each paper delivers" motivates specific methodological choices but does not reference ADR-0001 (five-label production head) or ADR-0007 (IR photometry mandatory).
- §3.2 "Tiering adopted throughout this project" (Tier 1/2/3 definitions) does not reference ADR-0005 or ADR-0015.
- §3.3.2 Ye+2024 flux correction is framed as "mandatory preprocessing" but does not reference the frozen Hermite basis contract or ADR-0002 (per-coefficient z-scoring).
- §9.2 "Information-content audit" protocol describes six tests but does not reference ADR-0009 (PCA-CMI replaces 2-D KSG summary), which is a methodological linchpin.

The asymmetry creates a credibility gap: a reader citing research_brief.md in a paper needs to know whether the design rationale is current or has been superseded by a later ADR. The solution is bidirectional links (research_brief.md points to ADRs; ADRs reference research_brief.md sections).

---

## 4. Missing "Reproducing the Catalog" Walkthrough

No single document chains together the end-to-end workflow. A new user asking "I want to reproduce the D-Cat-b catalogue from scratch, from Gaia DR3 → final release parquet" must synthesize information from:

1. `README.md` (credentials, environment, references)
2. `docs/data_acquisition.md` (TAP queries, batch sizes, checkpointing)
3. `docs/plan/03_stream3_inference.md` (Stream 3 sampling strategy, Phase 3 prerequisites)
4. `scripts/ingest_stream1.py`, `scripts/ingest_stream3.py` (not documented; user must read code)
5. `scripts/run_joint_ensemble.py` or `scripts/run_pipeline1_inference.py` (entry points, CLI args, expected outputs)
6. `docs/plan/05_release_packaging.md` (underspecified release procedure)
7. `src/arqueogal/xp_abundances/main/release.py` (tier assignment logic, sidecar schema)

A structured "Reproducing the Catalog" document would:
- List prerequisites (environment, credentials, disk budget, hardware).
- Walk through each major step with expected runtimes and intermediate outputs.
- Show concrete command-line invocations.
- Clarify when to use `main` vs `experimental` code branches.
- Explain how to validate intermediate outputs (row counts, column dtypes, sidecar checksums).
- Point to audit checkpoints (does the v5 schema match expected columns? do Tier 1/2/3 counts match the ablation study?).

---

## 5. Missing "Getting Started" for Local Development

No document describes the minimal setup and smoke-test workflow for a new contributor:

1. **Environment activation:** The README says `rapidsenv` is a shell alias, but does not document how to create it or what it does (activates `~/.venvs/rapids25.10_python3.12_cuda13/` presumably).
2. **Smoke tests:** `src/arqueogal/data/` mentions "production-size stratified smoke test" as a gate (CLAUDE.md conventions.md), but no README or GETTING_STARTED.md points a new user to where these live or how to run them.
3. **First inference run:** A user wanting to validate the environment might run a single-star inference on a tiny subset. No document explains how to extract 100 rows from Stream 3, create a minimal feature matrix, and run the inference driver.
4. **Debugging memory/storage issues:** The HARDWARE_SETUP.md is excellent (covers WSL2 ceiling, sysctl tuning, OOM recovery) but is not linked from README.md or plan/00_overview.md. A new user hitting an OOM error would have to discover this document by search, not from the entry-point tree.

---

## 6. Per-Module DESIGN.md Cross-Reference Gap

Five DESIGN.md files (data, utils, xp_abundances top-level, xp_abundances/main, xp_abundances/experimental) are the source of truth for column shapes, dtype contracts, and module-level invariants. However:

- `docs/context/architecture.md` describes "The three data streams" and "Data flow" in prose but does not say "see `src/arqueogal/data/DESIGN.md` for the ingestion contract" or "see `src/arqueogal/xp_abundances/DESIGN.md` for the model input/output schema."
- `docs/plan/00_overview.md` mentions the current ensemble pointer (`20260425_6b96c06_cd1cbb9_ensemble_5label`) but does not link to `src/arqueogal/xp_abundances/main/DESIGN.md` where the model's input feature layout is defined.
- No README or index document lists all five DESIGN.md files or explains their relationship.

The consequence: a user reading `docs/context/architecture.md` has no automated way to find the code-level contracts. They must either grep for DESIGN.md or rely on a team member's institutional knowledge.

---

## 7. Archive and Obsolete Documentation

`reports/archive/` and `docs/decisions/` contain legacy material (e.g., old v1 tier gates, superseded two-stage pretraining scripts), but:

- No "What Changed Between Versions" document summarizes breaking changes (v1 → v2 → v5) for a user upgrading an older inference run or retraining on new Stream 1 data.
- No guidance on when to consult archive/ vs current docs.
- The "legacy v1.x diagnostic scripts" mentioned in plan/00_overview.md are not explicitly listed or deprecated (a user might accidentally run an old script expecting it to work on v5 checkpoints).

---

## 8. Specific Staleness Issues

### 8.1 `docs/plan/01_pipeline1_v1.md` is informative but reads as historical

The file correctly documents v1's scope and rationale (December 2025 → April 2026 development). However, it opens with "Pipeline 1 v1 production model" without a callout that v1 is **no longer recommended**. A reader skimming the title and first paragraph might assume v1 is current. A bold warning box ("This page describes the superseded v1 ensemble shipped 2026-04-19. For the current v2/v5 ensemble, see plan/00_overview.md.") would clarify.

### 8.2 `docs/research_brief.md` §2.1 table does not mention Ye+2024 updates or Bach & Schwarz follow-up

The table of "What exists and what each paper actually delivers" was authored in April 2026 but does not account for post-April research:
- Ye et al. 2024 is listed, but the "valuable flux-correction" note does not cite a follow-up on NN robustness or failure modes.
- Buck & Schwarz 2024 is marked "workshop proof-of-concept" and "no all-sky catalogue yet." If published or extended between April and now (July 2026), the document is stale.

A "Last updated" timestamp at the top of research_brief.md would signal to readers that they should cross-check recent arXiv uploads.

---

## 9. Summary of Recommendations

### Critical gaps (block new-user adoption):
1. **Write a "Getting Started" document** (~2–3 pages): environment activation, smoke-test suite location, minimal inference run, expected outputs.
2. **Write a "Reproducing the Catalog" checklist** (~3–5 pages): end-to-end workflow from Gaia DR3 → D-Cat-b release parquet, with concrete CLI examples and intermediate validation checkpoints.
3. **Create a top-level index of module DESIGN.md files** in `docs/context/architecture.md` or a new `docs/CODEBASE_MAP.md`, with direct pointers to each module's design contract.

### Medium priority (reduce expert barrier):
4. Link `docs/plan/02_pipeline1_audit.md` → `docs/protocols/*.md` explicitly.
5. Add bidirectional cross-links between `research_brief.md` and `docs/decisions/*.md` (especially ADRs that justify tier decisions, preprocessing order, loss functions).
6. Update `CATALOG_SCHEMA.md` to reference ADR-0015 when listing retired columns; add a "Column History" section mapping v1 → v5 changes.
7. Expand `docs/plan/05_release_packaging.md` from "underspecified" to a full SOP (Step 1: invoke inference driver with args X/Y/Z; Step 2: run release.py; Step 3: validate sidecar; Step 4: archive to external storage).

### Lower priority (polish):
8. Add "Last updated" timestamps and version metadata to `research_brief.md`, `data_overview.md`, `data_acquisition.md` so readers can assess staleness.
9. Mark `docs/plan/01_pipeline1_v1.md` with a bold callout: "This page is historical. See plan/00_overview.md for the current v2/v5 ensemble."
10. Create a `docs/VERSIONING.md` document explaining the v1 → v2 → v5 rebuild history, attractor-stripe failure, and schema breakage. This supports methodology-paper readers and future maintainers.

---

## Appendix: Document Inventory

### Primary decision and methodology documents
- `research_brief.md` (v2, April 2026) — scientific rationale, tier protocol, audit protocol, landscape review
- `data_acquisition.md` — TAP recipes, 5 GB budget, preprocessing order, dust-map composition
- `data_overview.md` — visual tour (8 panels), row-count waterfall, label availability
- `CATALOG_SCHEMA.md` (v4, 2026-04-25) — column contract, release-tier definitions

### Planning and phase documentation
- `docs/plan/00_overview.md` (2026-04-26, current) — v1/v2/v5 rebuild history, deliverables, phase status
- `docs/plan/01_pipeline1_v1.md` — v1 scope and rationale (superseded)
- `docs/plan/02_pipeline1_audit.md` — audit status (5/6 tests done, test 6 deferred)
- `docs/plan/03_stream3_inference.md` — Stream 3 Phase 3 workflow
- `docs/plan/04_pipeline2_main.md` — Starfold integration contract (out of scope)
- `docs/plan/05_release_packaging.md` — release procedures (underspecified)
- `docs/plan/06_methods_paper.md` — methods-paper tracking (venue/timeline TBD)

### Architectural and code-level documentation
- `docs/context/architecture.md` — system diagram-free prose, three-stream data flow
- `docs/context/conventions.md` — enforceable naming, file layout, testing, provenance rules
- `docs/context/open_questions.md` — specific user decisions (test 6 thresholds, CMI deprecation, methods-paper venue)
- `src/arqueogal/{data,utils,xp_abundances}/DESIGN.md` (5 files) — module-level contracts (orphaned from primary docs)

### Decision records and protocols
- `docs/decisions/` (15 ADRs, 0001–0015) — all major architecture and methodology decisions with rationale
- `docs/protocols/` (4 files) — validation recipes for open-cluster benchmarking, cross-catalogue consistency, Fisher CRLB, τ-sweep

### Operational documentation
- `docs/HARDWARE_SETUP.md` — WSL2 memory ceiling, sysctl tuning, OOM recovery (not linked from entry points)
- `README.md` — project overview, deliverables, start-here links, environment, credentials

### Inventory counts
- **Primary decision docs:** 4 (research_brief, data_acquisition, data_overview, CATALOG_SCHEMA)
- **Phase planning docs:** 7 (plan/00–06 + PHASE_B_KIN_OOD_INTEGRATION)
- **Architectural docs:** 3 (context/{architecture, conventions, open_questions})
- **Module-level DESIGN.md:** 5 (data, utils, xp_abundances, xp_abundances/main, xp_abundances/experimental)
- **Decision records:** 15 (decisions/0001–0015)
- **Validation protocols:** 4 (protocols/*.md)
- **Total docs in /docs tree:** 37 files (including duplicates in archive/)

---

## Conclusion

The ArqueoGal documentation suite is internally coherent and well-suited for specialist readers (methodologists, collaborators familiar with the project history). However, three systemic gaps prevent new-user onboarding and reduce discoverability of operational procedures:

1. **No unified entry point for "I want to run the code" workflows.** Getting Started and Reproducing the Catalog are missing.
2. **Module-level design contracts exist in code (5 DESIGN.md files) but are orphaned from the primary documentation tree.** A reader of `docs/context/architecture.md` has no clear path to `src/arqueogal/*/DESIGN.md`.
3. **The v1 → v2 → v5 rebuild history is documented in `plan/00_overview.md` but is not explicitly reflected in other primary documents** (research_brief.md, CATALOG_SCHEMA.md), creating potential confusion about whether cited version numbers are current.

Addressing these gaps with three documents ("Getting Started", "Reproducing the Catalog", and a "Module DESIGN Index") would unlock the current high-quality content for new users while maintaining the specialist depth that existing collaborators rely on.
