# Testing & QA Meta-Report (Sonnet Synthesis)
**Date:** 2026-04-26  
**Scope:** Synthesis of five Haiku audits on test coverage, pytest patterns, integration tests, TDD discipline, and parallel debug infrastructure  
**Audience:** Release readiness assessment

---

## 1. Triangulation Points: Where Audits Agree

### (a) Production-Size Smoke Test Gap is Critical

All five audits converge on the same finding: **100-300 row toy fixtures are integration placebos** (test_coverage.md, pytest_patterns.md, integration_tests.md). The CLAUDE.md hard invariant ("Production-size stratified smoke test required") is violated across all data-layer modules:

- test_gaia_xp.py: 1-5 rows for Hermite reprojection, 10 rows for Ye+2024 correction.
- test_ingest_stream1.py: 5-row fixture.
- test_ingest_stream3.py: 300 rows (better, but no stratification).
- test_crossmatch.py, test_dedup.py: toy scale.

**Impact:** NaN propagation at scale, regime-cell edge cases, and DR2→DR3 many-to-one tie-breaking under load are undetected until production runs.

### (b) Four to Five Modules Are Untested

Both test_coverage.md and tdd_discipline.md list identical gaps:
- config.py (34 lines, dataclass but no serialization contract tests)
- sanity.py (575 lines, 7 check functions, critical training gate but untested)
- kinematic_ood.py (287 lines, OOD detector in placeholder status)
- ensemble_diagnostics.py (194 lines, mode_ambiguous_flag computation untested)
- release_artefacts.py (provenance sidecar contract not verified post-write)

### (c) Integration Tests Are Underspecified

integration_tests.md and parallel_debug_support.md both note that the single integration test (test_hybrid_stress_battery.py) exercises kNN validation but **misses the end-to-end prediction → annotate_parquet → attach_hybrid_columns chain**. No v5 schema validation, no kNN rescue integration test, no cross-stream invariant checks.

### (d) Fixture Consolidation is Duplicative

pytest_patterns.md identifies repeated monkeypatch patterns in test_gaia_enrich.py, test_ingest_stream1.py, and test_ingest_stream3.py. test_coverage.md extends this: no centralized `tests/fixtures/` directory (CLAUDE.md invariant #4 requires small realistic parquets for schema verification without mocking entire pipelines).

### (e) Test-First Discipline is Strongest Where Release Logic is Contractual

Both pytest_patterns.md and tdd_discipline.md agree that release.py (41 tests, v5 changeset shipped with tests) demonstrates exemplary TDD: parameterized boundary conditions, per-element caveat testing, and version-history comments. The same discipline is weaker in configuration and ensemble agreement logic.

---

## 2. Disagreements and Framings

### Disagreement 1: Research-Stage Asymmetry

**test_coverage.md framing:** "This is a release blocker. CLAUDE.md §3.3 gates require Tier 1 release to have zero gaps; untested config.py and sanity.py are hard stops."

**tdd_discipline.md framing:** "Research projects have asymmetric coverage. Script-level integration tests (test_run_pipeline1_inference.py) are strong; config gaps are known and acceptable during research phase, not after publication."

**Resolution:** The asymmetry is acceptable **during research** (2026 Q1–Q2), but **not at publication** (2026 Q4 release). Both assertions are true in sequence. A GitHub-ready release requires the gaps filled; a draft thesis does not.

### Disagreement 2: 5k-Row Smoke Tests vs. 300-Row Production Gradient

**test_coverage.md:** "Demands 5k-row stratified tests with regime cells, magnitude bins, |b| bins."

**parallel_debug_support.md:** "test_ablations/run_per_cell_ablations.py works at scale but uses pre-computed predictions; hypothesis testing is post-hoc and doesn't re-inference. Production drivers use tiny 14-D layouts in unit tests to avoid loading the full ensemble."

**Resolution:** Both are correct. Unit tests should use small synthetic data (14-D layouts, 100 rows) for fast iteration. Integration tests should use production-scale stratified smoke tests (5k rows) to catch edge cases that don't appear until batch size hits a threshold. The gap is the **absence of the integration-scale tests**, not the presence of small unit tests.

---

## 3. Minimum Test-Coverage Uplift for GitHub-Ready Release

To ship without compromising research-stage velocity, the following 6 test files are required:

### Tier 1 (Blocking for Tier 1 Release)

1. **tests/data/test_release_artefacts.py** (~80 lines)
   - Verify provenance sidecar exists, parses, contains required fields (source URL, query, row counts, corrections, git SHA, timestamp, input SHA-256).
   - Test each data-layer emitter (stream1, stream3, xp, etc.) actually writes sidecars post-run.
   - This is non-negotiable per CLAUDE.md invariant #14.

2. **tests/xp_abundances/main/test_sanity.py** (~150 lines)
   - Each of 7 check functions with parameterized boundary tests (exactly-at-bounds, per-element NaN edge cases, duplicate source_ids).
   - Use a production-sized (5k-row) fixture from tests/fixtures/ (see below).
   - Sanity gates training; untested gates are release-blocking.

3. **tests/integration/test_release_pipeline_e2e.py** (~100 lines)
   - Loads 10k-row slice of Stream 1 predictions + features.
   - Runs `join_predictions_with_features`, `annotate_parquet`, `attach_hybrid_columns` in sequence.
   - Asserts output schema, row counts, tier distributions match contract.

### Tier 2 (Blocking for Methods Paper Claims)

4. **tests/xp_abundances/main/test_ensemble_diagnostics.py** (~80 lines)
   - Parameterized agreement/disagreement tests on synthetic multi-member ensembles.
   - Verifies mode_ambiguous_flag logic (per-element, v5 caveat).
   - If kinematic claims appear in paper, kinematic_ood.py must have tests too.

5. **tests/xp_abundances/main/test_config.py** (~60 lines)
   - Serialization round-trips (YAML encoding/decoding).
   - Weight-constraint validation (__post_init__ guards).
   - Default-value stability across runs.

### Tier 3 (Blocking for GitHub Release Package)

6. **tests/integration/test_v5_schema.py** (~50 lines)
   - Load 1k-row production predictions parquet.
   - Validate column presence, dtypes, cardinality match v5 contract.
   - Cross-check hardcoded thresholds in release.py (σ-inflation per element, mode_ambiguous caveat).

---

## 4. Top 5 Testing Investments by ROI

| Rank | Investment | Effort (days) | Payoff | Velocity Impact |
|------|-----------|--------------|--------|---|
| **1** | Production-size stratified smoke tests (5k rows, regime/mag/|b| bins) for data layer | 3–4 | Catches NaN propagation, regime-cell edge cases, many-to-one tie-breaking at scale. Blocks silent data corruption at release time. | **High** — unit tests catch errors in development; smoke tests catch errors at scale. |
| **2** | Provenance sidecar full-stack tests (each ingest emits correct row counts, corrections, SHA-256). | 2–3 | CLAUDE.md invariant #14 is non-negotiable. Without this, artefacts are not reproducible. | **High** — one test per emitter prevents silent regression on sidecar schema. |
| **3** | Extract OOD/Regime-B/Bimodality logic into post-hoc ablation harnesses (test_ablations/*_hypotheses.py). | 2–3 | Hypothesis-driven debugging without re-inference. Cuts debug cycle from 30+ min to <1 min for tier-promotion hypotheses. | **Medium** — accelerates research iteration on gate tuning; not blocking for release. |
| **4** | DESIGN.md drift detection tests (assert emitted columns/dtypes match contract for each module with DESIGN.md). | 1–2 | Co-commit discipline: DESIGN.md drift = test failure. Prevents silent schema mutations. | **High** — low effort, high confidence payoff. |
| **5** | Consolidate TAP/fixture patterns to tests/fixtures/ + tests/data/conftest.py. | 1–2 | Reduce maintenance burden on repeated monkeypatch setup. Enables shared test infrastructure. | **Low** — hygiene improvement; no correctness payoff. |

---

## 5. Items Collectively Missed

### (a) Experimental Test Tree is Empty

`tests/xp_abundances/experimental/` is only `__init__.py`. If `src/arqueogal/xp_abundances/experimental/` has real modules (e.g., kin_ood_detector.py, novel_loss_term.py), they are untested. CLAUDE.md §3 (hard invariant on main ↔ experimental separation) requires test mirroring. If experimental code will be promoted to main or included in papers, tests are required before promotion.

**Impact:** Kinematic OOD claims in methods papers are unverifiable; placeholder status of `kinematic_ood.py` (noted in test_release.py line 1 docstring) blocks publication claims about kinetic diagnostics.

### (b) No Centralized Fixture Directory

CLAUDE.md invariant #4 requires `tests/fixtures/` with small realistic parquets (100–200 rows) for schema verification. This directory does not exist. All tests either use `_fake_*()` helper factories (synthetic, not realistic) or monkeypatch TAP (no disk footprint). The consequence: no test can validate a schema contract without mocking the entire pipeline or assuming production-scale inputs are available.

**Recommendation:** Create 5 fixture parquets:
- `stream1_sample_100rows.parquet` (representative APOGEE subset with all labels, flags, Gaia data)
- `stream3_sample_100rows.parquet` (Andrae+2023 subset with extinction, kinematics)
- `xp_raw_100rows.parquet` (XP raw coefficients, unbaked Ye+2024 correction)
- `predictions_10rows.parquet` (output of run_pipeline1_inference on tiny subset)
- `hybrid_10rows.parquet` (after attach_hybrid_columns; tests schema contract)

### (c) Parametrization Coverage is Sparse

Only 3 parametrized tests in the entire suite (all in test_release.py). Missing parametrization:
- NaN masks per element (V, Mg, Fe, α/M) in loss tests.
- Preprocessing order violations (flip Ye+2024 and normalise; should break).
- DR2→DR3 tie-break edge cases at scale.
- Regime-B envelope boundary conditions (|b|, Teff, logg edge cases).

### (d) No CI Pipeline Tests

No test validates that `conftest.py` marker filtering (`@pytest.mark.stress` skipped by default, run with `--run-stress`) works as intended. If a developer forgets `--run-stress`, the test silently skips without warning.

### (e) Integration Test Paths are Hardcoded; Silent Skips in CI

`test_hybrid_stress_battery.py` references production-size artefacts (1.2 GB of parquets, frozen checkpoint). The paths are relative-resolved (safe for local runs), but in CI without these artefacts staged, tests skip gracefully. No explicit documentation of what must exist for stress battery to run; no CI stage that stages the artefacts.

---

## 6. Recommendations Snapshot

### Ship-Ready Checklist (by Deadline)

- [ ] Add 6 Tier 1–2 test files (section 3) — 10 days effort, blocks release.
- [ ] Build 5 fixture parquets (tests/fixtures/) and document stratification — 3 days, enables schema-contract tests.
- [ ] Add DESIGN.md drift detection to all modules with column contracts — 2 days, prevents schema mutations.
- [ ] Consolidate TAP mocking to tests/data/conftest.py — 1 day, hygiene improvement.
- [ ] Document integration test prerequisites in tests/integration/README.md — 1 day, clarity for CI engineers.

### Research-Stage Accelerators (deferred after release)

- [ ] Extract OOD/Regime-B/Bimodality ablation harnesses (test_ablations/*_hypotheses.py) — 3 days, cuts debug cycles by 30x.
- [ ] Fill experimental test tree (tests/xp_abundances/experimental/) if experimental code will ship — 5+ days, depends on experimental module scope.
- [ ] Unit tests for run_calibration.py and run_knn_rescue.py script drivers — 3 days, enables hypothesis-driven debugging.

---

## 7. Cost–Benefit for GitHub-Ready Release

**Blocking gaps (cannot ship without):**
- Provenance sidecars must be verified post-write (release_artefacts.py tests).
- Sanity gates must be tested (sanity.py tests).
- End-to-end chain must be validated (test_release_pipeline_e2e.py).

**Non-blocking, high-confidence improvements:**
- DESIGN.md drift detection (catches schema mutations, low effort).
- Production-size smoke tests (catches edge cases, high effort, high payoff).
- Fixture directory (enables robust schema tests, medium effort).

**Total effort for minimum release readiness:** ~15–18 days.  
**Payoff:** Zero silent data-corruption regressions, reproducible provenance, verifiable tier-promotion logic for published catalogues.

---

## 8. Sources Cited

- `test_coverage.md:1–185` — missing test files, toy fixture gaps, parametrization coverage.
- `pytest_patterns.md:1–353` — v5 release.py design quality, fixture consolidation, marker usage (sparse).
- `integration_tests.md:1–300` — missing end-to-end tests, v5 schema validation gaps, checkpoint dependencies.
- `tdd_discipline.md:1–66` — config.py/sanity.py untested, release.py test-first discipline, 4 blocking modules for GitHub release.
- `parallel_debug_support.md:1–337` — tight coupling in production drivers, post-hoc ablation pattern, hypothesis-driven debug gaps.

**Cross-references to CLAUDE.md hard invariants:**
- Invariant #1 (production-size smoke test required): violated across all data-layer tests.
- Invariant #4 (no mocked database; use tests/fixtures/): fixtures directory does not exist.
- Invariant #14 (provenance sidecars non-negotiable): full-stack tests missing.
- Invariant #15 (DESIGN.md co-commit discipline): drift-detection tests exist for only 1 module (knn_rescue.py).
