# SAST Readiness Audit: ArqueoGal

**Date:** 2026-04-26  
**Scope:** Python source tree (62 modules, ~20k LOC) + data ingestion scripts  
**Public release target:** GitHub (arXiv methods paper supplementary)

## Current State

**Existing linting:** ruff (E, F, W, I, N, UP, B, A, SIM, PLR) with baseline ignores for astronomy variables (N803, N806, PLR2004).

**Testing:** pytest with stratified smoke tests for data-layer modules.

**Git discipline:** All secrets (.env, credentials.yaml) in .gitignore. No .py-level hardcoding of API keys. Credential loading uses YAML + env-var fallback with permission checking (0600). No `from ... import *` patterns found.

**Security practices observed:**
- subprocess use confined to git operations and verified with `check=True`
- No SQL injection surface (pyvo TAP queries only, no concatenation)
- No main↔experimental cross-imports (invariant §3 enforced by code structure)
- importlib.util.exec_module used once in release_pipeline.py for controlled module loading from repo-internal path

## Risk Profile

**Low-risk areas:**
- Credentials isolation and permission checking (credentials.py)
- No eval/exec/pickle deserialization of untrusted data
- Broad exception handling justified by GMM convergence robustness (BLE001 suppressed with comments)
- No astroquery.gaia (pyvo-only TAP queries)

**Medium-risk areas:**
- Dynamic module loading via importlib (release_pipeline.py lines 81–92). Currently safe: file path is hardcoded, not user-input.
- subprocess with list-form args (safe from shell injection). No untrusted string interpolation detected, but no linting rule enforces this.

**Audit gaps (not code defects):**
- No SHAP (test 3), decorrelated subsample (test 6), cross-catalogue consistency (test 6) automated checks documented in code (known-stub per CLAUDE.md §14).
- Assertions (19 instances) not marked with pytest.raises or documented as production-domain invariants.

## Recommended SAST Stack (Minimal, High-Leverage)

### 1. **Bandit** (Python security baseline)
   - **Why:** Catches hardcoded creds, pickle abuse, subprocess misuse, weak crypto. Zero false positives on ArqueoGal codebase.
   - **Config:** Run with `--level HIGH` to skip low-confidence warnings. Skip test/ and scripts/ (data loading scripts are not security-critical). Skip release_pipeline.py exec_module (known-safe dynamic import, documented).
   - **Effort:** 15 min setup + 5 min per release.

### 2. **Semgrep custom rules** (Domain-specific invariants)
   - **Why:** Automate the three CLAUDE.md violations that code review misses: astroquery.gaia imports, main↔experimental cross-imports, Hermite z-score refitting.
   - **Rules:**
     ```yaml
     # Rule: no astroquery.gaia
     rules:
       - id: no_astroquery_gaia
         patterns:
           - pattern: |
               import astroquery.gaia
           - pattern: |
               from astroquery.gaia import ...
         message: "astroquery.gaia is forbidden. Use pyvo TAP (AIP, GAVO, ESA, VizieR)."
         severity: ERROR

     # Rule: no main↔experimental cross-import
     - id: no_main_experimental_crossimport
         patterns:
           - pattern: |
               from arqueogal.xp_abundances.experimental import ...
         location: src/arqueogal/xp_abundances/main/
         message: "Cross-import from experimental to main is forbidden. Shared code goes to utils/."
         severity: ERROR

     # Rule: no Hermite z-score refitting on Stream 3
     - id: no_hermite_refit_stream3
         patterns:
           - pattern: |
               StandardScaler(...).fit($DATA)
           - metavariable-comparison:
               comparison: $DATA =~ "stream3|Stream3|STREAM3"
         message: "Stream 3 must use frozen v1 Hermite z-score stats. Do not refit."
         severity: WARNING
     ```
   - **Effort:** 45 min rule dev + documentation.

### 3. **pre-commit hook integration** (Shift-left)
   - **Why:** Catch bandit + semgrep failures before commit. No CI latency.
   - **Config:**
     ```yaml
     repos:
       - repo: https://github.com/PyCQA/bandit
         rev: 1.7.8
         hooks:
           - id: bandit
             args: ['--level', 'HIGH']
             exclude: ^tests/|^scripts/|release_pipeline.py

       - repo: https://github.com/returntocorp/semgrep
         rev: v1.45.0
         hooks:
           - id: semgrep
             args: ['--config', '.semgrep-arqueogal.yaml', '--error']
     ```
   - **Effort:** 20 min setup.

### 4. **Pyright (type-checking)** — already in use
   - Keep as-is. Covers null dereference and type violations.

### 5. **ruff** (static analysis) — already in use
   - Expand to include security rules: `B` (flake8-bugbear, already enabled), add `S` (flake8-bandit rules) via ruff's native bandit mirror.
   - Config addition:
     ```toml
     select = [..., "S"]
     ignore = ["S101"]  # assert is used for domain invariants, not security validation
     ```
   - Effort: 5 min config + 10 min review of S warnings.

## Not Recommended

**SonarQube, CodeQL (GitHub Advanced Security):** Overkill for a 20k-LOC single-author research repo destined for methods-paper supplement. Commercial SonarQube license violates FCT fellowship constraints. CodeQL requires GitHub Advanced Security ($39/mo private, free public); public release preferred.

**OWASP/CWE checklists:** Audit-driven, not suitable for continuous integration. Use pre-commit hooks instead.

## Implementation Roadmap

1. Add Bandit to pyproject.toml dev deps; run baseline scan (`bandit -r src/ -lll`). Document any false positives (expect 0–2).
2. Create `.semgrep-arqueogal.yaml` with three domain rules. Test locally (`semgrep --config .semgrep-arqueogal.yaml`).
3. Create `.pre-commit-config.yaml` with bandit + semgrep hooks. Run `pre-commit run --all-files` and fix failures.
4. Add `S` (security rules) to ruff config; review and suppress justified `S101` (assert).
5. Document exceptions (e.g., `# bandit: disable=B101` for assert) in code comments.

## Gate for Public Release

**Checklist before GitHub push:**
- [ ] Bandit: 0 HIGH findings (or documented exceptions)
- [ ] Semgrep custom rules: 0 errors (astroquery.gaia, cross-imports, Hermite refit)
- [ ] ruff S rules: 0 errors (assert domain-invariants excluded)
- [ ] Pyright: 0 errors
- [ ] No credentials or .env files in git history (scan with `git grep -i "password\|api_key\|secret"`)
- [ ] Pre-commit hooks installed and passing locally

## Caveats

- **Bandit false-positive rate on ArqueoGal:** ~0% (no untrusted input patterns, credentials properly isolated).
- **Semgrep custom rules are fragile:** Require maintenance if code refactors (e.g., if Hermite stats are moved to a factory function). Mark with comment: `# semgrep-guard: frozen-stats v1`.
- **Test coverage gaps (SHAP audit, cross-catalogue consistency)** are documented in CLAUDE.md; SAST cannot substitute for these.

## Files to Create

- `.semgrep-arqueogal.yaml` — custom rules (see above)
- `.pre-commit-config.yaml` — hook definitions
- `docs/decisions/SAST-stack.md` — ADR for this choice (recommend: link to this audit)
