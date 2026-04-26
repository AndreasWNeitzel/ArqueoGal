# TDD Discipline Audit — ArqueoGal

## Snapshot

As of 2026-04-26, ArqueoGal maintains disciplined but selective test coverage across the production Pipeline 1 library code. The codebase mirrors the stated TDD asymmetry of a research project: tight test-first discipline where release logic is contractual (release.py: 41 tests, all passing, with explicit tier-promotion decision tests); smoke-level coverage of configuration and data-prep modules (gaps in config.py and sanity.py); and strong integration-level validation for script-driven inference pipelines. The v5 release.py changeset (commit 3ddd9b4, 2026-04-22) exhibits test-first discipline: tests and code shipped simultaneously in the same commit, with parameterized boundary-condition tests pinning σ-threshold logic to strict `>` inequalities and per-element caveat precedence.

## Finding 1: Production code outpaced tests in config.py and sanity.py

**Gap**: Two high-leverage modules lack any unit test coverage.

- **config.py** (200 lines): Dataclass-based training configuration knobs (`LossWeights`, `TrainingConfig`). The module is frozen (CLAUDE.md §12.16 notes field names as stable commitments for checkpoint round-tripping). Untested: serialization contracts (YAML encoding/decoding), field-value bounds validation (e.g., `supcon` / `beta` weights must be non-negative and sum meaningfully), and default-value stability across runs.
  
- **sanity.py** (575 lines): Seven check functions that gate training. Although the checks are wrapped in runner code (scripts/run_pretraining_sanity.py), the individual check functions (`check_tier1_label_completeness`, `check_parameter_bounds`, `check_zscore_validity`, etc.) are public API and lack unit tests. Edge cases that trigger in 5k-row feature matrices (misaligned flag columns, NaN propagation in per-element rates, APOGEE dynamic-range boundaries) are not validated in isolation.

**Recommendation**: Add test suites matching the stated contract for each module:
- `tests/xp_abundances/main/test_config.py`: Serialize / deserialize round-trips; validate weight constraints and defaults.
- `tests/xp_abundances/main/test_sanity.py`: Parameterized boundary tests on each check function (e.g., exactly-at-bounds parameter values, per-element NaN edge cases, dedup idempotence with duplicate source_ids).

**Impact**: Both modules are actively used in training pipelines (config drives every run, sanity gates every training start). A typo in config field names or a missed NaN in sanity checks would propagate downstream undetected until model-training notebooks fail.

## Finding 2: Test-first discipline on release.py is strong; test suite is behavioural spec

**Evidence**: The v5 release changeset (commit 3ddd9b4) introduces 136 lines of production code and 144 lines of test code in the same commit. The test suite is not smoke-level; it is a detailed behavioural specification:
- 41 passing tests covering tier gating, hard-kill precedence, per-element demotion logic, σ-inflated caveat introduction (v4 feature), per-element tier columns (v3 feature), and annotation round-trips.
- Parameterized tests on boundary conditions: σ exactly at threshold (must NOT fire), v5 diagnostic-only flags (regime_b, aux_missing, ood_disagreement, latent_support) that were retired from gating logic (tests explicitly assert they do not demote).
- Idempotence and schema-version tests that bind sidecar JSON to tier-gating decisions.
- The test comments document version history: "v5 (2026-04-26): mode_ambiguous_flag demotes ONLY [α/M] now" — this is a research decision (per-element caveat narrow-down) that would have been invisible without the test assertion.

**Assessment**: This is test-driven in the classical red-green-refactor sense. The tests were written to specify the tier-promotion logic before (or alongside) implementation; the codebase treats the test suite as the authoritative definition of what "release-tier assignment" means.

## Finding 3: Integration and smoke-test coverage is strong; script drivers well-validated

**Structure**: Test suites split cleanly between unit (`tests/xp_abundances/main/`) and integration (`tests/scripts/`).

- `test_run_pipeline1_inference.py` (47 KB, 1000+ lines): Production-size stratified smoke tests on the inference pipeline with tiny ensembles (2 members, 14-D layout) and synthetic data. Tests exercise schema detection, atomic-write semantics (partial failure recovery), basis-fingerprint validation (frozen v1 stats contract), and OOD-bundle round-trips. This is not 100-row placebo; it validates the full glue.

- `test_hybrid_stress_battery.py`: Stress tests for data-layer modules under realistic load (5k-row parquets, strided access patterns).

**Assessment**: The codebase explicitly rejected "integration placebo" per CLAUDE.md, and the test suite bears this out. Script drivers have high-fidelity validation.

## Finding 4: Test coverage gaps that would block GitHub-ready release

**Missing or partial coverage**:

1. **ensemble_diagnostics.py** (194 lines) — no unit tests. This module computes ensemble agreement / disagreement metrics (used for the `mode_ambiguous_flag` in release.py). Its public functions are untested; a bug in agreement computation would silently corrupt Tier-2 demotion decisions.

2. **kinematic_ood.py** (287 lines) — no unit tests. Implements kinematic OOD detection; the placeholder in release.py (test_assign_kin_ood_flag mocks all False) acknowledges that the real logic is not yet tested. This is future work, but it blocks publication of any kinematic OOD claim.

3. **config.py field validation** — no runtime checks. The module is a bare dataclass; invalid weight combinations (e.g., `supcon=0.0 and beta_nll=0.0` simultaneously disables both terms) are not caught at instantiation. Serialization of non-JSON-safe types (e.g., Path objects in the frozen config checkpoint) is untested.

4. **sanity.py per-element NaN rates** — soft-fail gate (check 7) is report-only with no threshold validation. Downstream code assumes per-element rates are stable; a silent drift in APOGEE label availability would be invisible.

**Release-blocking**:
- The release tier system depends on `ood_joint_flag` being set correctly. If `ensemble_diagnostics` silently bugs the agreement logic, Tier 1 releases will include mode-ambiguous stars without warning.
- The kinematic OOD placeholder will trigger questions from reviewers if any kinematic claims appear in the methods paper without backing tests.

**Recommendation for GitHub-ready**:
- Add `test_ensemble_diagnostics.py` with parameterized agreement / disagreement tests on synthetic multi-member ensembles (e.g., identical predictions → full agreement; one member offset → disagreement detected).
- Add `test_kinematic_ood.py` with boundary-condition tests on the kinematic OOD envelope (when released from placeholder status).
- Add validation guards to `config.py`: dataclass `__post_init__` that rejects `supcon=0 and beta_nll=0`.
- Add `test_sanity.py` with per-element NaN-rate assertions on known APOGEE DR19 baseline (capture baseline in the audit markdown as the comment in sanity.py already notes, then test drift).

## Summary

Test-first discipline is strongest in contractual release-tier logic (release.py) and script-level integration tests; weakest in configuration validation and per-element ensemble diagnostics. The 4 untested modules (config, sanity, ensemble_diagnostics, kinematic_ood) total 1,256 lines and represent the highest-risk gap for a GitHub-ready release. The v5 release changeset demonstrates that when tests are written (release.py), they are written as behavioural specifications with attention to version history and edge cases; the pattern should extend to configuration and ensemble agreement logic before catalogues are published.
