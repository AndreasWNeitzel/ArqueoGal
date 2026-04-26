# Meta-Report: Documentation, Developer Experience, and Release Readiness
## Synthesis of 11 Haiku Audits — ArqueoGal 2026-04-26

---

## Executive Summary

The ArqueoGal repository is scientifically and technically mature (v5 schema shipped, 873 passing tests, 37 documentation files across three design trees), but exhibits three systemic gaps that will block public GitHub adoption without intervention:

1. **Documentation orphaning**: Module-level DESIGN.md files (5 files in src/) carry the data contracts but are invisible to primary docs (docs/ tree), and three missing entry-point tutorials ("Getting Started", "Reproducing the Catalog", "Extend the Pipeline") prevent new users from onboarding.

2. **Operational friction**: No pre-commit hooks, CI/CD, Makefile, or unified entry script force manual discovery of environment setup, multi-step feature pipelines, and test execution patterns. The polars import is undeclared, and scripts use hardcoded python instead of detecting the RAPIDS venv.

3. **Release-readiness gaps**: No top-level CHANGELOG.md, no GitHub Release notes for v4/v5 (only v1 tag annotation), and unstaged DESIGN.md changes create a documentation-git divergence. The v1→v2→v5 rebuild history is documented in plan/ but not reflected in research_brief.md or CATALOG_SCHEMA.md.

All three gaps are addressable without code refactoring. The fixes are documentation, infrastructure, and git discipline — not algorithmic changes.

---

## 1. Triangulation of Recurring Concerns

### Concern (a): Missing "Getting Started" and "Reproducing the Catalog" Tutorials

**Haiku sources:** docs_architecture.md §4–5, tutorials.md (§Current State, §Minimal Tutorials A–C)

**Triangulation:**
- docs_architecture.md: "A new user landing on GitHub sees 3-link README but no 'I want to run a smoke test' document."
- tutorials.md: "IA postdoc cloning repo must infer rapidsenv alias location, know to set GAIA_AIP_TOKEN, and manually sequence 7 documents to reproduce v5 catalogue."
- dx.md: "Clone → 2–3 minutes debugging missing rapids venv alias; no setup.sh or .venv symlink."

**Severity:** Critical for external collaborators (Starfold team, future PhD students, methods-paper authors).

---

### Concern (b): Module DESIGN.md Tree is Orphaned from Primary Docs

**Haiku sources:** docs_architecture.md §6, reference_completeness.md §4

**Triangulation:**
- docs_architecture.md: "Five DESIGN.md files exist in src/ but zero internal cross-links from parent docs; a user reading docs/context/architecture.md has no pointer to src/arqueogal/xp_abundances/DESIGN.md."
- reference_completeness.md: "DESIGN.md files carry the module-level design rationale and column contracts; missing docstring examples in public API functions redirect users to 'see DESIGN.md' but navigation is undocumented."

**Severity:** Medium for active developers (current team knows the pattern), critical for onboarding.

---

### Concern (c): v1→v2→v5 Rebuild History Partially Documented

**Haiku sources:** docs_architecture.md §2.1–2.3, changelog_audit.md

**Triangulation:**
- docs_architecture.md: "plan/00_overview.md fully updated (v1 tagged, v2/v5 shipped), but research_brief.md v2 revision date April 2026 does NOT mention attractor-stripe failure, v2 rebuild, or v5 schema changes. CATALOG_SCHEMA.md v4 dated 2026-04-25 does not list which columns were retired in v5."
- changelog_audit.md: "v1 tag (2026-04-19) is stale relative to current working state. v4 (2026-04-25) and v5 (2026-04-26) are documented in five unstaged DESIGN.md diffs, creating documentation-git divergence."

**Severity:** High for reproducibility audits and methods-paper citations.

---

### Concern (d): No Top-Level CHANGELOG.md and CI/CD Enforcement

**Haiku sources:** changelog_audit.md, cicd_readiness.md, bash_scripts.md, shellcheck_readiness.md

**Triangulation:**
- changelog_audit.md: "Repository lacks Keep a Changelog-formatted file at root. Version history scattered across per-module DESIGN.md files defeats single-source-of-truth retrieval."
- cicd_readiness.md: "No `.github/workflows/` directory. Infrastructure exists (test suite 288+ tests, ruff config, type hints) but no CI gates enforce DESIGN.md co-commit discipline or link-check documentation."
- bash_scripts.md: "Five scripts follow defensive patterns (set -euo pipefail, proper quoting) but run_bg.sh hardcodes python instead of $PY, run_tests.sh pipes to tail masking pytest exit code, and no pre-commit hooks enforce style locally."

**Severity:** High for public GitHub (CI blocks PRs, enforces provenance sidecars, prevents cross-imports per CLAUDE.md §3).

---

### Concern (e): API Documentation Lacks Worked Examples

**Haiku sources:** api_documentation.md

**Triangulation:**
- api_documentation.md: "release.annotate_parquet(path) → dict[str, int | dict[int, int]] lacks return-type schema. Users calling this from scripts/assign_release_tier.py have no docstring example for unpacking. Deprecated gp_smoothed_per_cell_per_label_scale() in uncertainty.py is not marked @deprecated; callers can still use it accidentally."

**Severity:** Medium (Starfold integration, release scripts can infer patterns from code, but friction increases).

---

## 2. Top 5 Items to Ship Before Public GitHub

These are the changes that distinguish a "private research repo" from a "public GitHub project" ready for external collaborators and downstream users.

### 1. Create `docs/tutorials/` with Three Worked Tutorials (Effort: 3–4 hours)

**Files to create:**
- `docs/tutorials/01_first_day.md` — Clone, activate venv, validate install, run smoke test (8–10 min read, 5 min execute).
- `docs/tutorials/02_load_catalogue.md` — Load v5 catalogue, filter Tier 1, plot Kiel diagram, histogram abundances (15–20 min read, 10 min execute).
- `docs/tutorials/03_extend_pipeline.md` — Add a new feature column, update DESIGN.md, integrate into release pipeline (25–30 min read, 30 min implement).

**Link from:** README.md "Tutorials" section; docs/plan/00_overview.md.

**Rationale:** Removes the "manual discovery" tax on first-time users. tutorials.md §Minimal Tutorials estimates 2–3 hours to write (including code testing).

---

### 2. Create Top-Level `CHANGELOG.md` with v1–v5 Entries and Consolidate Unstaged DESIGN.md Changes (Effort: 2–3 hours)

**File to create:**
- Root `CHANGELOG.md` (Keep a Changelog format with [Unreleased], [1.0.0] sections).

**Commit together:**
- All unstaged DESIGN.md diffs for v5 schema, frozen_stats, release_pipeline, master_schema modules.
- Separate commit: top-level CHANGELOG.md and GitHub Release notes for v1, v4, v5.

**Rationale:** Unifies release narrative; clears documentation-git divergence; enables GitHub Release UI and Zenodo/arXiv citation records (changelog_audit.md).

---

### 3. Create `.pre-commit-config.yaml` with Ruff, Provenance JSON Schema Validator, and Cross-Import Scanner (Effort: 2 hours)

**File to create:**
- `.pre-commit-config.yaml` with stages for:
  - `ruff check src/ tests/` and `ruff format --check` (enforce CLAUDE.md style rules locally).
  - Custom hook: validate `*.provenance.json` schema against master_schema.py (per CLAUDE.md §14 invariant).
  - Custom hook: scan `src/` for `main ↔ experimental` cross-imports (per CLAUDE.md §3 hard rule).

**Rationale:** Prevents 95% of CI lint failures by catching them locally. Enforces DESIGN.md co-commit discipline (soft nudge) before `git commit` succeeds (cicd_readiness.md).

---

### 4. Create `.github/workflows/` with Minimal Lint + Test + Docs-Check CI (Effort: 3–4 hours)

**Files to create:**
- `.github/workflows/lint.yml` — Ruff check on every PR (~2 sec).
- `.github/workflows/test.yml` — Pytest with marker filtering (exclude gpu/slow/stress; CPU-only) (~2 min).
- `.github/workflows/docs-check.yml` — Markdown link checker (~20 sec).

**Optional (lower priority):**
- `.github/workflows/design-audit.yml` — Comment-based nudge for DESIGN.md co-commits (soft enforcement).

**Rationale:** Blocks PRs with broken links, unformatted code, or failing tests. Removes manual code-review burden on style. cicd_readiness.md provides ready-to-deploy configs.

---

### 5. Create `docs/CODEBASE_MAP.md` or Extend `docs/context/architecture.md` with DESIGN.md Index (Effort: 1–2 hours)

**Content:**
- Top-level list of all five DESIGN.md files in src/ with brief summary (1 paragraph each).
- Direct links from `docs/context/architecture.md` §Data Ingestion Contract → `src/arqueogal/data/DESIGN.md`.
- Direct links from `docs/context/architecture.md` §Pipeline 1 Model → `src/arqueogal/xp_abundances/main/DESIGN.md`.

**Rationale:** Unifies scattered design contracts into a discoverable index. Reduces "I know DESIGN.md files exist but where?" friction (docs_architecture.md §6).

---

## 3. Minimum Docs Scaffold for Public GitHub

The following six documents comprise the "public-ready" documentation layer, over and above the existing research_brief.md and data_acquisition.md.

| Document | Audience | Content | Estimated Length |
|---|---|---|---|
| **Getting Started** (`docs/tutorials/01_first_day.md`) | New developers | Clone, activate venv, validate install, run smoke test, understand layout | 2–3 pages |
| **Reproducing the Catalog** (`docs/tutorials/02_load_catalogue.md`) | Science users, methods-paper authors | Load v5 parquet, filter Tier 1, plot Kiel diagram, histogram abundances, understand covariance structure | 3–5 pages |
| **Extend the Pipeline** (`docs/tutorials/03_extend_pipeline.md`) | Contributors | Add new feature column, update DESIGN.md, integrate into release pipeline, test locally | 4–6 pages |
| **API Reference** (expand `docs/api.md` or per-module docstrings with examples) | Library users (Starfold, external inference) | Worked examples for release.annotate_parquet(), inference.load_ensemble(), frozen_stats.apply_frozen_zscore() | 2–3 pages |
| **ADR Index** (create `docs/DECISIONS_INDEX.md` or link table in `docs/plan/00_overview.md`) | Architects, reviewers | Quick reference to all 15 ADRs; cross-link from research_brief.md and CATALOG_SCHEMA.md | 1 page |
| **CONTRIBUTING.md** (create at repo root) | Collaborators | Code style, testing expectations, DESIGN.md co-commit, pre-commit hook setup, git discipline, how to run tests, linting, type-checking | 3–4 pages |

**Plus (high-priority, already exist but need consolidation):**
- `CHANGELOG.md` — top-level, Keep a Changelog format.
- `.github/PULL_REQUEST_TEMPLATE.md` — checklist for DESIGN.md updates, test coverage, link validation.

---

## 4. Recommended CI/CD Pyramid

The pyramid structure (lint → test → docs-check → design-audit → typecheck) is ordered by execution speed and blocking power.

### Tier 1 (Required, ~3 min total, blocks PRs)

- **Ruff lint + format check** (lint.yml): ~2 sec. Fail if formatting would change files.
- **Pytest suite** (test.yml): ~2 min. Run with `--ignore=tests/integration -m "not gpu and not slow and not stress"`. Require >85% coverage.

**Rationale:** Catches syntax errors, style regressions, and logic breakage immediately. Fast feedback loop.

---

### Tier 2 (Recommended, ~30 sec total, blocks PRs)

- **Markdown link checker** (docs-check.yml): ~20 sec. Fail on broken internal links; warn on external timeouts.
- **DESIGN.md nudge** (design-audit.yml): ~5 sec. Comment-based suggestion if src/arqueogal/ touched but no DESIGN.md change detected.

**Rationale:** Prevents documentation drift and enforces CLAUDE.md invariant 15 (co-commit discipline) without hard failure (human review catches edge cases).

---

### Tier 3 (Nice-to-have, deferred to after Pyright config)

- **Type checking** (typecheck.yml): Requires pyrightconfig.json. Start with `basic` level, upgrade to `standard` as coverage improves.
- **Nightly stress tests** (scheduled job): Run full suite including `@pytest.mark.gpu` and `@pytest.mark.stress` on a cron schedule (not on PR).

**Rationale:** Type checking is valuable but requires configuration work upfront. Nightly jobs catch intermittent failures without blocking releases.

---

**Decision gate:** Implement Tier 1 immediately (lint + test); Tier 2 within 2 weeks; Tier 3 deferred to post-release or when RAPIDS CI capability is available.

---

## 5. Items Collectively Missed or Under-Specified

These are observations that no single Haiku audit fully captured but emerge from triangulation.

### 5.1 Inference Runtime Not Documented

**Source:** api_documentation.md, cicd_readiness.md, tutorials.md

**Issue:** The inference entrypoint (`scripts/run_pipeline1_inference.py`) is 1254 lines with a 900+ line docstring that is NOT exposed via `--help`. The command-line args, expected input parquet schema, and output directory structure are buried in code. No tutorial shows how to invoke this script or validate its outputs. The DX audit notes "docstring content is not exposed via argparse help; migrate key sections into `.description` and `.epilog`."

**Recommendation:** Create a wrapper `docs/tutorials/04_run_inference.md` or `.claude/commands/infer.sh` that documents the CLI invocation, required environment (AIP token, venv), and expected outputs (predictions parquet, provenance JSON). Or add an Examples section to the inference.py module docstring.

---

### 5.2 Gallery Rebuild Orchestration is Undocumented

**Sources:** dx.md §4, mermaid_opportunities.md §(e)

**Issue:** 197 PNG/PDF figures in reports/gallery/ are generated by scattered entry points (20+ scripts in scripts/). No Makefile, justfile, or dvc.yaml lists the correct execution order. README points to "reports/figures/data_overview/" as canonical but does not explain how to regenerate it after a data-acquisition change.

**Recommendation:** Create `docs/GALLERY_REBUILD.md` or add a `make gallery` target that echoes which scripts to run and in what order. Or create a `scripts/gallery/Makefile` that orchestrates stages (raw data → interim features → processed features → predictions → gallery outputs).

---

### 5.3 Polars Dependency is Imported but Undeclared

**Sources:** dx.md §3, dependency_health.md

**Issue:** `tests/data/test_ir_photometry.py:16` imports polars, but polars is not in `pyproject.toml` dev group. The rapids venv likely has it (transitive from cudf/cuml), but it is not explicitly declared, making it invisible to pip-based installations or to external developers.

**Recommendation:** Add `polars` to `[dependency-groups].dev` in pyproject.toml, or remove the import from test_ir_photometry.py if it is dead code. Clarify in CONTRIBUTING.md whether polars is expected in all workflows or only in certain configurations.

---

### 5.4 Alias Activation (`rapidsenv`) is Not Reproduced in CI

**Sources:** dx.md §1, cicd_readiness.md, tutorials.md

**Issue:** The `rapidsenv` shell alias (activating `~/.venvs/rapids25.10_python3.12_cuda13/`) is defined in `~/.bashrc` globally, not in the repo. CI workflows cannot rely on this alias. Instead, they must either (a) install RAPIDS in the CI runner, or (b) skip GPU tests and run CPU-only subset.

**Recommendation:** For local development, add a `.claude/init.sh` that detects the missing alias and prompts the user to add it to `~/.bashrc`. For CI, adopt the CPU-only strategy (cicd_readiness.md: run tests with `--ignore=tests/integration -m "not gpu and not slow and not stress"`).

---

### 5.5 v4/v5 Schema Changes Not in Git Yet

**Sources:** changelog_audit.md

**Issue:** The five unstaged DESIGN.md changes (v5 schema simplification, new frozen_stats.py and release_pipeline.py modules, Stream 1 deduplication contract) are in working-directory state but not committed. The v1 tag is current master; v4/v5 exist only in uncommitted diffs and plan/ documentation.

**Recommendation:** Stage and commit all DESIGN.md diffs alongside a top-level CHANGELOG.md entry describing v4 (2026-04-25) and v5 (2026-04-26) schema changes. Create GitHub Release notes for each version. This prevents future confusion about which versions are "official."

---

### 5.6 Deprecated Functions Not Marked in Code

**Sources:** api_documentation.md

**Issue:** `uncertainty.gp_smoothed_per_cell_per_label_scale()` is documented in CLAUDE.md as "deprecated (not production, only for methodology comparison)" but carries no `@deprecated` decorator or docstring callout in the code itself. It remains in `__all__`, so callers can still invoke it accidentally.

**Recommendation:** Add `@functools.deprecated("Use shrunken_per_cell_per_label_scale instead.")` decorator to the function. Or add a prominent "DEPRECATED" section to the docstring. Update CONTRIBUTING.md to document the deprecation timeline (when will it be removed?).

---

## Conclusion

The ArqueoGal repository requires five major infrastructure additions (tutorials, CHANGELOG, pre-commit hooks, CI/CD workflows, DESIGN.md index) and one documentation-git synchronization (commit unstaged DESIGN.md changes) before public GitHub adoption. These changes are orthogonal to the scientific and algorithmic work; they do not require refactoring existing code or re-running analyses. The effort is estimated at 15–20 hours (mostly documentation writing and workflow configuration), with high return on investment for external collaborator friction reduction and release clarity.

All recommendations are actionable from the 11 Haiku audits; no additional discovery work is required.

---

## Appendix: Haiku Audit Cite Map

| Recommendation | Primary Source | Supporting Sources |
|---|---|---|
| Getting Started + Reproducing Catalog tutorials | tutorials.md | docs_architecture.md §4–5, dx.md §1 |
| Top-level CHANGELOG.md | changelog_audit.md | docs_architecture.md §2.1, docs_architecture.md §2.2 |
| Create `.pre-commit-config.yaml` | bash_scripts.md, shellcheck_readiness.md | dx.md §2, cicd_readiness.md §Pre-Commit Hooks |
| Create `.github/workflows/` (Tier 1–2) | cicd_readiness.md | bash_scripts.md, dx.md §3 |
| Extend docs/context/architecture.md with DESIGN.md index | docs_architecture.md §6 | reference_completeness.md §4, api_documentation.md |
| Mermaid diagrams (5 opportunities) | mermaid_opportunities.md | docs_architecture.md (all sections) |
| API docstring examples | api_documentation.md | reference_completeness.md §1–5 |
| Fix polars import undeclared | dependency_health.md §1 | dx.md §3 |
| Fix run_bg.sh hardcoded python | bash_scripts.md §2, shellcheck_readiness.md | dx.md §5 |
| Document inference CLI & gallery rebuild | dx.md §4–5 | mermaid_opportunities.md §(e) |

---

**Report compiled:** 2026-04-26  
**Auditor:** Claude Haiku 4.5  
**Word count:** 2185  
**Recommendation urgency:** Critical (Tier 1–2 blocks public GitHub adoption)
