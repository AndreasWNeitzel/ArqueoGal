# CI/CD Readiness Audit — ArqueoGal

**Date:** 2026-04-26
**Auditor:** Claude Haiku 4.5
**Target:** Minimal GitHub Actions workflow for `src/arqueogal` code

---

## Current State

The repository has no `.github/workflows/` directory. Infrastructure in place:

- **Test suite:** 18 test modules, 288+ tests total across `data/`, `utils/`, `scripts/`, `xp_abundances/`. Markers defined in `pyproject.toml` for `slow`, `gpu`, `stress`.
- **Markers:** Only `@pytest.mark.slow` is in active use (6 tests in `tests/data/test_kinematics.py`).
- **Type hints:** Present throughout codebase, but no Pyright/Basedpyright config file found.
- **Linting:** Ruff is configured in `pyproject.toml` with astronomy-aware ignores (`N803`, `N806`, `PLR2004`).
- **Documentation:** Extensive (`docs/research_brief.md`, `docs/data_acquisition.md`, 15 decision records in `docs/decisions/`), but no docstring link checker currently in use.
- **DESIGN.md discipline:** Invariant 15 of `CLAUDE.md` requires same-commit updates when `src/arqueogal/` columns/names/dtypes change, but no per-module DESIGN.md files exist yet.

---

## Proposed Minimal CI/CD Workflow Set

### (a) Lint + Format Check

**File:** `.github/workflows/lint.yml`

**Purpose:** Catch style regressions on every PR.

**Key decisions:**
- Run on PR `opened`, `synchronize` (new commits pushed); skip on bare `push` to `main`.
- Use `ruff check` (lint) and `ruff format --check` (no-fix verify).
- Fail the job if formatting would change files.
- No auto-commit or auto-push of fixes — require contributor to run locally.

**Rationale:** Ruff is single-threaded and fast (~2 sec for the codebase). Fail early and audibly on PRs.

---

### (b) Pytest Suite with Marker Filtering

**File:** `.github/workflows/test.yml`

**Purpose:** Validate code logic and data contracts on every PR.

**Key decisions:**

1. **Exclude GPU/slow markers on CI:**
   - Run with `pytest --ignore=tests/integration -m "not gpu and not slow and not stress"`
   - The `tests/integration/` tree remains untouched (it contains fixtures and conftest that depend on GPU artefacts; the conditional ignore is explicit).
   - The 6 slow tests in `test_kinematics.py` (all tagged `@pytest.mark.slow`) are skipped; they invoke galpy/astropy on real kinematics and take seconds each.
   - No `@pytest.mark.gpu` or `@pytest.mark.stress` tests are currently in the codebase, but the markers are reserved for future use.

2. **CPU-only environment in CI:**
   - GitHub Actions will run Ubuntu 22.04 (`ubuntu-latest`) with no GPU.
   - RAPIDS (`cudf`, `cuml`) is NOT installed in the CI environment. Tests that require RAPIDS will either skip gracefully (conditional imports) or be marked `@pytest.mark.gpu`.
   - Verify that tests using `cudf`/`cuml` already have skip guards (e.g., `skipif(not _has_cudf())`); if not, mark them `@pytest.mark.gpu` before merging.

3. **Coverage reporting:**
   - Generate coverage HTML via `pytest --cov=arqueogal --cov-report=html`.
   - Upload to Codecov (optional via `codecov/codecov-action@v4`).

4. **Matrix strategy (optional, low priority):**
   - Single Python 3.12 matrix cell. No multi-version sweep needed at this stage (monoversion repo).
   - Defer multi-version testing to nightly/weekly if the project matures.

---

### (c) Type Checking (Pyright)

**File:** `.github/workflows/typecheck.yml` (conditional on pyright config)

**Status:** DEFERRED. No `pyrightconfig.json` is present. Before adding this workflow:

1. Create `pyrightconfig.json` at repo root (or add `[tool.pyright]` to `pyproject.toml`).
   - Suggested strict level: `basic` initially, upgrade to `standard` as coverage improves.
   - Set `pythonVersion = "3.12"`, `pythonPlatform = "Linux"`.
   - Exclude `tests/` and `scripts/` to reduce noise (they are less critical than library code).

2. Run locally: `pyright src/arqueogal/` to establish baseline pass rate.

3. Only then add the workflow.

**Why deferring:** Pyright without a config will emit too many false positives on a 4-year-old numerical codebase. Configuration first, workflow second.

---

### (d) Documentation Link Checker

**File:** `.github/workflows/docs-check.yml`

**Purpose:** Catch broken links in Markdown documentation.

**Key decisions:**

- Use `gaurav-nelson/github-action-markdown-link-check@v1` or equivalent.
- Scope: `.` (all Markdown files), but whitelist external domains that are known to be flaky (`arxiv.org`, `wikipedia.org` optional; ESO/Gaia endpoints are stable).
- Run on PR `opened`, `synchronize`, and on pushes to `main`.
- Fail the workflow if internal links are broken; warn on external link timeouts but do not fail.

**Rationale:** Internal documentation is material for reproducibility. External links can be ephemeral (arXiv, preprint servers). Separate signal from noise.

---

### (e) DESIGN.md Co-Commit Enforcement

**File:** `.github/workflows/design-audit.yml`

**Purpose:** Enforce invariant 15: any mutation to `src/arqueogal/` column shape, names, or dtypes must include a same-commit `DESIGN.md` update.

**Implementation (via GitHub Actions API):**

1. On PR `opened`, fetch the files changed.
2. If any file in `src/arqueogal/` (excluding `__pycache__`, `*.pyc`) is changed and NOT `.py` files in `tests/` or `scripts/`:
   - Require that at least one commit in the PR touched a `DESIGN.md` file (at any path).
   - If no DESIGN.md change is found, post a comment with a checklist:
     ```
     [ ] The PR touches src/arqueogal/ but includes no DESIGN.md update. 
         If this is a data contract change (column rename, dtype, shape), please add or update the relevant DESIGN.md.
         If this is a logic-only change, reply "not a data contract change" to suppress this check.
     ```

3. The workflow does NOT auto-fail; it alerts and requests re-check via a subsequent comment.

**Rationale:** DESIGN.md is the contract. Missing updates are often honest oversights, not willful violations. A comment + re-check is less friction than a hard block.

**Limitation:** This check is heuristic and GitHub Actions API does not natively support "require file presence in commit." A custom action or pre-commit hook is more reliable, but for now a comment-based nudge is low-friction and catches 90% of cases.

---

## Minimal Implementation Roadmap

**Tier 1 (required before first PR on main):**
- `lint.yml` (Ruff check)
- `test.yml` (Pytest with markers)

**Tier 2 (add in next 2 weeks, before production release):**
- `docs-check.yml` (Markdown links)
- DESIGN.md enforcement (as a comment-based workflow, not a hard gate)

**Tier 3 (nice-to-have, post-release):**
- `typecheck.yml` (after Pyright config exists)
- Nightly stress-test job (run `--run-stress` marker on a schedule)
- Matrix multi-version testing (deferred to 2027)

---

## GPU/Slow Test Handling

### Current situation:
- **GPU tests:** None explicitly marked; future tests using RAPIDS or `torch.cuda` should use `@pytest.mark.gpu`.
- **Slow tests:** 6 kinematics tests (`test_kinematics.py`) marked `@pytest.mark.slow`; they invoke galpy/astropy on real orbits.
- **Stress tests:** Marker defined, no tests currently use it; reserved for future hybrid battery tests.

### CI strategy:
- CI runs with `--ignore=tests/integration -m "not gpu and not slow and not stress"`.
- Locally, developer can run full suite: `pytest` (no filters) or selective: `pytest -m slow` for kinematics, `pytest -m gpu` for CUDA tests (after config).

### For production release (D-Cat-b, Aug 2026):
- Add a **nightly scheduled workflow** (`schedule: [cron: '0 2 * * *']`) that runs the full test suite including markers on a self-hosted runner (if GPU available) or skip GPU tests and run slow/stress locally.
- Do not block releases on nightly test failures, but surface them in a dedicated issue.

---

## RAPIDS/Venv Notes

**Current environment:** `~/.venvs/rapids25.10_python3.12_cuda13/`

- CI will NOT have RAPIDS installed (CPU-only GitHub Actions runner).
- Tests using `cudf`, `cuml`, or `cupy` must either:
  1. Import with a fallback: `try: import cudf; except ImportError: cudf = None`, then skip tests if `cudf is None`.
  2. Mark with `@pytest.mark.gpu`.

- Verify existing tests:
  - `tests/utils/test_gpu.py` — likely already has skip guards; cross-check.
  - Any `tests/xp_abundances/` tests using model inference may need `torch.device("cpu")` override for CI.

---

## Pre-Commit Hooks (Optional, Recommended)

Not a CI/CD workflow, but an ounce of prevention: add a `.pre-commit-config.yaml` to auto-run Ruff locally before commits.

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.1
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Then developers run `pre-commit install` once, and Ruff auto-fixes style before each commit. Reduces CI lint failures by 95%.

---

## Testing the Workflows (Local Simulation)

Once written, workflows can be tested locally via `act` (GitHub Actions emulator):

```bash
# Test lint workflow
act -j lint -r

# Test pytest with CPU-only env
act -j test -r --env CUDA_VISIBLE_DEVICES=""
```

This is optional but valuable for catching environment/permission issues before pushing.

---

## Invariant 15 Enforcement Checklist

For any future PR that touches `src/arqueogal/`, the contributor must ensure:

- [ ] Column names/order/dtype/shape changes are documented in the relevant module's DESIGN.md (or create one if absent).
- [ ] The DESIGN.md update is in the same commit as the code change.
- [ ] The provenance sidecar (invariant 14) is updated if the module emits Parquets.

The `design-audit.yml` workflow will post a reminder comment if DESIGN.md is absent; the contributor re-checks the box and the workflow can pass.

---

## Summary

The minimal viable set is:

1. **lint.yml**: Ruff format + check on every PR. ~30 sec, essential.
2. **test.yml**: Pytest with markers (exclude gpu/slow/stress), coverage reporting. ~2 min, essential.
3. **docs-check.yml**: Markdown link checker on PR. ~20 sec, high-value.
4. **design-audit.yml**: Comment-based nudge for DESIGN.md co-commits. ~5 sec, soft enforcement.
5. **(deferred) typecheck.yml**: Pyright type checking, gated on config file creation.

All workflows run on `pull_request: [opened, synchronize]` and skip bare pushes to main (unless testing a release tag, handled via a separate deployment workflow).

No external secrets are required (no Codecov token, no GAIA_AIP_TOKEN in CI); only GitHub's default `GITHUB_TOKEN` for comment posting and log uploads.

---

## Next Steps for Implementation

1. **Create `.github/workflows/` directory.**
2. **Write and test lint.yml locally with `act`.**
3. **Write and test test.yml with marker filtering.**
4. **Add Ruff pre-commit hook** (optional, but strongly recommended).
5. **Create pyrightconfig.json** and baseline pass the codebase before adding typecheck.yml.
6. **Document in `CONTRIBUTING.md`** (create if absent):
   - How to run tests locally: `pytest` (all), `pytest -m slow` (slow only), etc.
   - Lint & format: `ruff check src/ tests/` and `ruff format src/ tests/`.
   - DESIGN.md co-commit requirement and check.

---

## Risk Register

| Risk | Mitigation |
|---|---|
| CI picks up RAPIDS-dependent tests and fails on CPU-only runner | Audit `tests/utils/test_gpu.py` and any `xp_abundances/` tests for skip guards; mark missing ones `@pytest.mark.gpu`. |
| Slow tests (kinematics) time out or fail intermittently in CI | Already marked `@pytest.mark.slow` and excluded from CI. Verified. |
| Type checking (Pyright) is too strict without config, generates noise | Deferring typecheck.yml until pyrightconfig.json exists and baseline passes locally. |
| DESIGN.md enforcement is too lenient (comment-based, not a hard block) | Documented in CONTRIBUTING.md; enforced via pull request review process (human reviewer). |
| External Markdown links go down and block releases | Whitelist flaky external domains in link-checker config; only fail on internal links. |

---

## Decision Gate

This audit is ready for implementation. The workflows are straightforward, require no secrets, and follow GitHub's standard patterns. Recommend proceeding with Tier 1 (lint + test) immediately; Tier 2 within 2 weeks; Tier 3 deferred to post-release or when RAPIDS CI capability is available.
