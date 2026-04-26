# Mermaid Diagram Opportunities in ArqueoGal Documentation

**Audit Date:** 2026-04-26  
**Auditor:** Haiku  
**Scope:** ArqueoGal docs/ for complex flows amenable to visual representation

---

## (a) Data Acquisition Graph — Three-Stream Ingestion & Preprocessing Flow

**Location:** `docs/data_acquisition.md` (new subsection, §0.1 after "Overview and constraints")

**Rationale:** The document describes three independent data streams (APOGEE DR19×Gaia, TESS Hon+2021×Gaia, Gaia RGB+RC), cross-matching, corrections, and enrichment in prose. A flowchart showing source URLs → raw downloads → per-stream ingestion → cross-match logic → corrections (Lindegren+2021 parallax, Riello+2021 G-mag, Ye+2024 XP flux, Mészáros+2025 [X/M]) → enrichment stages → frozen Hermite z-score stats → output parquets would clarify dependencies and prevent readers from losing track of the stream-specific branches.

**Diagram Type:** `flowchart TD` (top-down)

**Structure outline:**
- Three parallel input nodes: APOGEE FITS | Gaia TAP | Andrae+2023 catalogue + TESS Hon+2021
- Each stream flows through: raw download → interim ingestion → enrichment fork
- Common enrichment tasks: Gaia astrometry fetch, Bailer-Jones distance, dust-map composition, galpy kinematics
- Frozen stats checkpoint (basis fingerprint) before Stream 1 → `pipeline1_features_stream1.parquet`
- Streams 2, 3 → respective interim/processed parquets
- Side nodes for correction layers (tap into the enrichment stages)
- Decision node: "Ye+2024 flag = 0?" for XP retention

**Value:** Eliminates paragraph-heavy narrative on stream divergence; exposes order dependencies (e.g., Ye correction must precede Hermite reproject) and identifies which corrections block each stream.

---

## (b) Release-Tier Decision Flowchart — Per-Element Promotion & OOD/Caveat Composition

**Location:** `docs/research_brief.md` (new subsection, after §3.3 "Six-test protocol")

**Rationale:** Section §3.3 outlines a six-test protocol (physical gate, hold-out RMSE, open-cluster precision, information-content audit, conditional MI, cross-catalogue consistency) with a promotion decision tree. The current prose ("Passes 1–3 only → Tier 3", "Passes 1–4 but fails 5 → Tier 3, internal use", etc.) is correct but requires re-reading. A decision-tree diagram (flowchart or state diagram) would show: element candidate → Test 1 (physical gate?) → Test 2 (hold-out RMSE gate?) → ... → Tier 3 / Tier 2 with caveat / Tier 1 release decision node with justification. Additionally, the later protocol (Option 2, "three-question diagnostic" for Test 4 null-skill-ratio failure cases) creates a secondary branching tree (aux-ratio check, PCA-CMI check, XP permutation check) that feeds into a tiering decision. A flowchart integrating both the main gate sequence and the Option 2 diagnostic fork would make the tier protocol transparent and auditable.

**Diagram Type:** `flowchart TD` with decision diamonds

**Structure outline:**
- Entry: Element X candidate
- Diamond: Test 1 (physical spectral feature in XP window?) → No: Tier 3 (terminal)
- Diamond: Test 2 (hold-out RMSE & bias < threshold?) → No: Tier 3 (terminal)
- Diamond: Test 3 (open-cluster σ_intra < 1.5 σ_APOGEE?) → No: Tier 3 (terminal)
- Diamond: Test 4a (audit suite 1/2/4/6 pass?) → No: Tier 3 (terminal)
- Diamond: Test 5 (conditional MI bootstrap CI excludes zero?) → No, but Test 4a passed: Secondary fork (Option 2 diagnostics)
  - Sub-diamond: aux-ratio > 1.10? Check
  - Sub-diamond: PCA-CMI > 0.02 nats? Check
  - Sub-diamond: ≥3 XP features in top-10 permutation? Check
  - Re-join: if ≥2 pass → Tier 1 with prior-augmented caveat; else → Tier 2 population-only
- If Test 5 passes: Diamond Test 6 (cross-catalogue consistency; currently stub) → Pass all → Tier 1 clean
- Terminal outputs: Tier 1 clean, Tier 1 with caveat, Tier 2 population-only, Tier 3 internal, Tier 3 rejected

**Value:** Clarifies the **conditional logic** of tier promotion (the Option 2 branch only applies if Test 5 MI fails but Test 4a audit partially passes) and exposes that v1 operates at 5/6 test coverage with three audit subtests stubbed. Readers immediately see which decisions are deferred and the decision-point entry criteria.

---

## (c) Schema Migration Flow — v4 (PIPELINE2) → v5 (Hybrid Release) Column Transformations

**Location:** `docs/plan/05_release_packaging.md` (new subsection, before "What's decided")

**Rationale:** The project transitioned from a 21-label "v4" schema (retained for methods-paper comparison) to the 5-label v5 release (Teff, log g, [M/H], [α/M], [Mg/H]) plus metadata (covariance, OOD flags, Regime B flag, selection_prob, aux_missingness flags, release_tier). The master_schema.py module defines `PIPELINE1_TRAINING_SCHEMA` and `PIPELINE1_INFERENCE_SCHEMA` but is silent on what columns were dropped, renamed, or added relative to the prior layout. A diagram showing v4 columns (21 labels + cov) → filtering (drop 16 labels, retain 5) → augmentation (add OOD/Regime B/tier metadata) → v5 parquet would help downstream users understand the contract change and allow teams integrating this repo (e.g., Starfold) to track which columns persist and which are new.

**Diagram Type:** `flowchart LR` (left-to-right) or ERD (entity relationship for column inheritance)

**Structure outline:**
- Left column (v4 schema): identifiers (source_id, apogee_id, sdss_id) | 5 retained labels (Teff, log g, [M/H], [α/M], [Mg/H]) | 16 dropped labels (Na, Al, Ni, Ti, O, etc.) | v4 covariance (5×5)
- Middle fork:
  - Retained path: 5 labels + full 5×5 covariance → v5 core
  - Dropped path: 16 labels → archive/methods-paper only (with link to v1 checkpoint)
- Right column (v5 schema): identifiers | 5 labels + covariance | OOD (Mahalanobis distance, ensemble disagreement flag) | Regime B exclusion flag | selection_prob | aux_missingness_{g,bp,rp,av,dist} | release_tier {T1, T1-caveat, T2}
- Annotations on edges: reason for drop (e.g., "Tier 3, no XP signal"), reason for add (e.g., "calibration gate")

**Value:** Serves downstream integrators (Starfold, team catalogue); clarifies which v4 columns are archived vs. v5 production; makes the metadata-augmentation surface explicit.

---

## (d) Hybrid Composer Decision Logic — Regressor vs. kNN Substitution Rule

**Location:** `scripts/gallery/plot_19_hybrid_composer.py` comments + new `docs/plan/01a_hybrid_composer_logic.md` (before §4.5 release pipeline mention)

**Rationale:** The `run_full_pipeline.sh` script (Stage 4) runs `build_hybrid_release.py`, which orchestrates a regressor → kNN rescue → composition step. The current code and the gallery stage 19 plot show results, but no flow diagram explains *when* the hybrid composer invokes kNN-substitution vs. regressor-retention. Logic likely depends on: (1) Mahalanobis OOD flag; (2) ensemble disagreement; (3) nearest-neighbor quality metric in latent space; (4) training-data sparsity in the target cell. A decision tree showing the substitution rule (e.g., "if Mahalanobis > 3σ AND ensemble std > 0.15 dex AND kNN-distance < 0.05, use kNN mean; else use regressor") would allow users to reproduce the composition rule and modify it if needed. This is the interface between Pipeline 1 (regressor-dominant) and Stream 3 inference (hybrid-dominant).

**Diagram Type:** `flowchart TD` with decision nodes

**Structure outline:**
- Entry: per-star prediction from ensemble regressor
- Diamond: Mahalanobis OOD distance > threshold? → Yes: collect kNN candidates
- Diamond: ensemble disagreement std > threshold? → Yes: kNN quality check
- Diamond: kNN distance in latent space < threshold? → Yes: substitute kNN mean
- Diamond: training-set sparsity (cell hit count) < threshold? → Yes: kNN fallback
- Terminal: output = regressor | hybrid (regressor + kNN weight blend) | kNN-dominant

**Value:** Exposes the hybrid composition rule to reproducibility audits; clarifies when kNN is invoked and how regressor-vs-kNN weights are decided; informs Stream 3 users of which rows are regressor-only vs. hybrid.

---

## (e) Orchestration DAG — Pipeline Stages in `run_full_pipeline.sh`

**Location:** `docs/plan/00_overview.md` (new subsection after "Current phase status")

**Rationale:** The bash script `run_full_pipeline.sh` sequences six stages: (1) ensemble checkpoint validation, (2) Stream 3 inference, (3) latent-kNN rescue, (4) hybrid composition, (5) gallery diagnostics, (6) test suite. The script is linear, but in principle stages 2 & 3 could be parallelized, and stage 5 depends on outputs from 2–4. A DAG (directed acyclic graph) showing stage dependencies, input/output parquets, and checkpoint gates would allow users to understand: (a) which stages can be run independently (e.g., re-running gallery on frozen predictions); (b) which outputs are intermediates vs. release artefacts; (c) where resumption points exist if the script fails. Currently all six are sequential; a DAG would clarify whether this is a hard requirement or a convenience.

**Diagram Type:** `graph TD` or `gitGraph` (showing commits/stages in sequence)

**Structure outline:**
- Node: "ensemble checkpoint check" → if missing, halt
- Node: "Stream 3 inference (regressor)" → input: ensemble dir, S3 features → output: pred_parquet
- Node: "latent-kNN rescue" → input: ensemble dir, train parquet, S3 features → output: knn_parquet
- Node: "hybrid composer" → input: predictions, features, knn_parquet → output: hybrid release dir
- Nodes: "gallery stages 18–20", "unit/integration tests" → input: hybrid output
- Edges: show dependencies (e.g., composer depends on both Stage 2 & Stage 3)

**Value:** Non-expert users see the full release pipeline at a glance; developers can identify parallelization opportunities; failure modes (e.g., missing ensemble) are immediately visible.

---

## Summary

These five diagrams address the following pain points in the documentation:

1. **Data stream confusion** — three parallel ingestion pipelines with different corrections and enrichment paths are hard to track in prose alone.
2. **Tier-promotion ambiguity** — the six-test protocol and Option 2 diagnostic fork create nested conditionals that are clearer as a flowchart than as a paragraph.
3. **Schema change invisibility** — downstream teams integrating this repo don't see which columns survived v4 → v5 without reading master_schema.py directly.
4. **Hybrid-composition opacity** — the regressor ↔ kNN decision rule is buried in code; a decision diagram makes the substitution logic auditable.
5. **Pipeline-orchestration linearity** — the bash script is sequential but doesn't clarify whether this is by necessity or convention; a DAG exposes parallelization potential.

All five diagrams are **non-essential for correctness** (the underlying documentation and code are self-contained) but **high-value for onboarding and auditability**. They should be treated as *reference aids* that travel alongside the prose, not replacements for it.

---

**Estimated placement priority for implementation:**
1. **(a) Data acquisition** — highest value for onboarding new team members on Stream 1/2/3 divergence
2. **(b) Tier-promotion flowchart** — essential for any future tier-promotion audits or extensions to new elements
3. **(e) Pipeline DAG** — helpful for users re-running subsets of the release pipeline
4. **(c) Schema migration** — moderate value; mainly for Starfold/team integration
5. **(d) Hybrid composer** — lower priority; the decision rule is already documented in code comments; refactor after (b) is stable
