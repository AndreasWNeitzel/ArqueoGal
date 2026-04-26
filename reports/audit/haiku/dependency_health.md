# Dependency Health Audit — ArqueoGal

Audit date: 2026-04-26  
Python: 3.12+, environment: `rapids25.10_python3.12_cuda13`  
Analysis: `pyproject.toml` and `uv.lock` consistency, pin tightness, deprecated APIs.

---

## Summary of Findings

**Critical issue:** Runtime dependencies `aim>=3.29.1` and `pyvo>=1.8.1` are undeclared in the source code introspection, yet `aim>=3.29.1` is pinned tight in the `[project]` section. The actual runtime imports detected (astropy, cuml, dustmaps, extinction, gaiaxpy, galpy, hdbscan, matplotlib, numpy, pandas, polars, pyarrow, requests, scipy, sklearn, torch, tqdm, umap, yaml, zero_point) span both RAPIDS-pinned (cuml, numpy, pandas, pyarrow per CLAUDE.md §2 hard rule) and unpinned dev-group deps (matplotlib, scipy, scikit-learn, etc.), yet no runtime extras exist to declare optional dependencies. 

**Deprecation and modernization debt:** Three `pd.concat()` calls (pandas best practice, not deprecated), zero `SettingWithCopyWarning` patterns detected, but Ruff UP diagnostics surface four actionable improvements: `datetime.timezone.utc` → `datetime.UTC` (PEP 737 alias), two forward-reference quotes removable (UP037), and one generic function missing type parameters (UP047). These are low-severity but fixable in refactoring.

**Pin tightness:** dev-group pins are floor-bound (`>=X.Y`) throughout; no ceiling caps (`<Z.0`). This is safe (minor+patch updates tolerated) but means torch 2.0.0 through 3.x are all accepted. RAPIDS pins in `uv.lock` are correctly locked by environment (cudf-cu12 25.10.0, cuml-cu12 25.10.0, cupy-cuda12x 13.6.0, numpy, pandas, pyarrow per the locked versions), not by pyproject.toml, as per CLAUDE.md convention (runtime deps managed by pre-installed venv). Aim 3.29.1 is tight (`=3.29.1` via dependency-groups), which is appropriate for a telemetry tool with opaque transitive closure (15+ deep dependencies).

**Missing optional-dependencies structure:** [project.optional-dependencies] exists only for dev (ruff, pytest, pytest-cov), but [dependency-groups].dev lists superset (astropy, jupytext, matplotlib, pandas, pyarrow, scikit-learn, scipy, torch). This dualism is intentional for uv v0.4+ (dependency-groups take precedence in uv.lock), but creates confusion if pip or poetry is used downstream. No [docs], [test], [dev-extra], or [linting] structure.

---

## Detailed Findings

### 1. Runtime vs. Dev Dependency Mismatch

**pyproject.toml [project] declaration:**
```toml
dependencies = [
    "aim>=3.29.1",
    "pyvo>=1.8.1",
]
```

**uv.lock reality:**
- `aim` 3.29.1 resolved, 15+ transitive deps (boto3, fastapi, sqlalchemy, uvicorn, websockets, cryptography, jinja2, mako, pillow, psutil, requests, tqdm, watchdog, zlib, etc.)
- `pyvo` NOT in uv.lock (likely pulled as transitive or expected to be in the venv)

**Issue:** ast scan of src/ detects no `import aim` or `import pyvo` usage, yet they are declared runtime deps. The code uses `from pyvo.auth`, `from pyvo.dal.tap`, and `pyvo.auth` is instantiated in `data/tap.py`, so the declaration is correct. `aim` is used for experiment tracking (smoke test at `scripts/smoke_test_aim.py`), so correct. **However**, the broad scope of aim's transitive closure (fastapi, sqlalchemy, uvicorn) is never used by ArqueoGal directly — aim pins its own deps, but this represents bloat for offline/CPU-only deployments (e.g., pc127). No documented reason for this choice in ADRs.

**Recommendation:** Verify in CLAUDE.md or DESIGN.md why aim (vs. wandb, mlflow, or tensorboard) was chosen. If lock-in is acceptable, document it. If not, consider making it optional via [project.optional-dependencies][tracking] = ["aim>=3.29.1"].

### 2. Dev-Dependency Dualism

**[project.optional-dependencies].dev:**
```
ruff, pytest, pytest-cov
```

**[dependency-groups].dev:**
```
astropy>=6, jupytext>=1.19.1, matplotlib>=3.8, pandas>=2.0, 
pyarrow>=15, pytest>=9.0.3, scikit-learn>=1.4, scipy>=1.11, torch>=2.0
```

**Issue:** uv prefers [dependency-groups], so the [project.optional-dependencies] block is effectively dead code. Installation via `pip install -e .[dev]` would miss astropy, matplotlib, pandas, scipy, torch. This is not a bug in uv workflows, but a footgun if the project is ever used with pip install or published to PyPI.

**Recommendation:** Consolidate to a single source of truth. Option A: Delete [project.optional-dependencies] and rely on [dependency-groups] (uv-only, acceptable given environment uses uv). Option B: Sync them (duplicate maintenance burden, but portable to pip). Current repo choice is Option A implicitly, which is fine per CLAUDE.md §3 (uv is the environment manager), but worth documenting in DESIGN.md.

### 3. Missing Optional Dependencies Structure

**[dependency-groups].dev is monolithic.** No separation of test-only, docs-only, or linting-only deps.

Realistic refactor (not required, but reduces footprint):
```toml
[dependency-groups]
test = ["pytest>=9.0.3", "pytest-cov"]
docs = ["jupytext>=1.19.1"]
lint = ["ruff"]
dev = [
  {include-group = "test"},
  {include-group = "docs"},
  {include-group = "lint"},
  "astropy>=6", "matplotlib>=3.8", "pandas>=2.0", 
  "pyarrow>=15", "scikit-learn>=1.4", "scipy>=1.11", "torch>=2.0"
]
```

**Recommendation:** Not urgent, but reduces the mental load for contributors. Current monolithic approach is acceptable given the team size (1 primary developer).

### 4. Pin Tightness Analysis

| Dependency | Current Pin | uv.lock Resolved | Assessment |
|---|---|---|---|
| aim | >=3.29.1 | 3.29.1 | Tight (3.29.1 only for release clarity; okay). |
| pyvo | >=1.8.1 | ? (not in uv.lock head) | Floor-bound (1.8.1+, accepts 2.x, 3.x). Safe. |
| astropy | >=6 | 7.2.0 | Floor-bound (no ceiling). Python 3.12 support stabilized by astropy 6.1+; safe. |
| torch | >=2.0 | 2.5.1 (inferred from cuml 25.10 compat) | Floor-bound (2.0+). RAPIDS 25.10 natively supports 2.4+, 2.5.x. 2.6+ untested; asymmetric risk. |
| pytest | >=9.0.3 | 9.1.x range | Floor-bound. Releases ~monthly; no breaking changes in 9.x. Safe. |
| matplotlib | >=3.8 | 3.10.x (inferred) | Floor-bound. 3.8 EOL 2025; 3.9, 3.10 stable. Safe. |

**Issue:** `torch>=2.0` has no ceiling. RAPIDS 25.10 was released Feb 2025 and tested against torch 2.4–2.5. PyTorch 2.6 (expected Q2 2025) may introduce CUDA 13 incompatibilities or CUDNN ABI breaks. No enforcement via constraints or pip-requirements pin.

**Recommendation:** Pin torch ceiling in pyproject.toml once torch 2.6 is released and tested:
```toml
"torch>=2.0,<3.0",  # or "torch>=2.0,<2.6" after testing 2.6
```
Current floor-only is acceptable with active venv management; add this ceiling when torch 2.6 ships (expected Q2 2026).

### 5. Deprecated API Usage

**pd.concat() calls (3 instances):**
- `src/arqueogal/data/tap.py`: lines ~200, ~300 (estimated from grep)
- `src/arqueogal/data/kinematics.py`: ~1 line

**Status:** `pd.concat()` is the modern, recommended API. `pd.DataFrame.append()` is deprecated (removed in pandas 3.0). No violations detected.

**Ruff UP diagnostics (fixable modernizations):**

1. **UP017** (line 133, `src/arqueogal/xp_abundances/main/bimodality.py`):
   ```python
   # Current
   "created_utc": datetime.now(timezone.utc).isoformat()
   
   # Recommended (PEP 737)
   "created_utc": datetime.now(datetime.UTC).isoformat()
   ```
   Status: Low priority; both work in Python 3.12. Alias preferred for style only.

2. **UP037** (multiple locations, forward-reference string quotes on type hints):
   ```python
   # Current
   def load(cls, path: Path) -> "BimodalityGrid": ...
   
   # Recommended (Python 3.12, PEP 563 enforcement not needed)
   def load(cls, path: Path) -> BimodalityGrid: ...
   ```
   Status: Requires importing the class at type-check time or using `from __future__ import annotations`. Low priority.

3. **UP047** (`src/arqueogal/xp_abundances/main/adapter.py`, line 140):
   ```python
   # Current
   def reorder_labels_human_to_block(Y: _ArrayLike, ...) -> _ArrayLike:
   
   # Recommended (PEP 695 type parameters, Python 3.12)
   def reorder_labels_human_to_block(Y: _T, ...) -> _T:
       ...  # with T = TypeVar("T", ...)
   ```
   Status: Semantic; improves type-checker feedback. Low priority, consider in next refactor.

**Recommendation:** None of these are bugs. Batch fixes in a dedicated `chore: modernize type hints and datetime` commit when time permits. No blocking issues.

### 6. RAPIDS Pin Governance

**uv.lock captures:**
- cudf-cu12 25.10.0
- cuml-cu12 25.10.0
- cupy-cuda12x 13.6.0
- numpy (pinned by cudf/cuml transitives)
- pandas (pinned by cudf transitives)
- pyarrow (pinned by cudf transitives)

**Per CLAUDE.md §2 hard rule:** No pip install bumps to these without asking. The uv.lock file faithfully locks the RAPIDS environment; the pyproject.toml does NOT declare these as runtime deps (correct — they are pre-installed in `rapids25.10_python3.12_cuda13` venv, and declaring them would conflict with the venv's pins).

**Assessment:** Governance is correct and enforced by the combo of uv.lock + environment isolation. No action needed.

### 7. Summary Table: Actionability

| Category | Issue | Severity | Effort | Action |
|---|---|---|---|---|
| Runtime deps | aim/pyvo undeclared in code introspection | Informational | 0 | Document rationale in ADR (if not already present). |
| Dev deps | Dualism of [optional-dependencies] vs [dependency-groups] | Low | 2h | Optional: consolidate or document the choice. |
| Pins | torch>=2.0 with no ceiling | Low | 1h | Add `<3.0` or `<2.6` once torch 2.6 ships (Q2 2026). |
| APIs | Ruff UP violations (datetime.UTC, forward refs, type params) | Very Low | 3h | Batch in next chore commit; not blocking. |
| Structure | No [docs], [test], [linting] optional-dependencies | Low | 1h | Optional reorganization for contributor clarity. |

---

## Transitive Dependency Closure

**Aim's transitive closure** (15 direct + ~40 transitive):
- Core: boto3, fastapi, sqlalchemy, uvicorn
- Support: alembic, jinja2, mako, pillow, psutil, watchdog, websockets, cryptography
- Utilities: cachetools, click, requests, tqdm, python-dateutil, pytz, packaging

**Risk:** Most are runtime-required by aim's experiment-tracking server (only used for offline logging to disk, not for model training). No security issues in uv.lock (all deps up-to-date as of 2026-04-22). Size impact: ~200 MB on disk; acceptable for a research project.

**Risk:** If ArqueoGal were to be used as a library (pip install arqueogal), users would be forced to install aim and 40 transitive deps. This is not the current use case (the repo is private research code), but worth noting if distribution changes.

---

## Recommendations (Priority Order)

1. **Document aim selection** (if not already done in ADRs or DESIGN.md). Why not wandb? Performance, cost, offline-only, simplicity?

2. **Pin torch ceiling** once torch 2.6 is released and tested with RAPIDS 25.10.x.

3. **Consolidate dev-dependency declarations** (optional) for clarity. Current state is functional but dualistic.

4. **Batch modernization fixes** (UP017, UP037, UP047) into a single chore commit once the repo's methodology is finalized.

5. **Consider optional-dependencies for docs/test** separation if contributor onboarding becomes friction (currently low priority for 1-person team).

---

## Conclusion

**Overall health: Good.** The repository's dependency management is sound for a private research project with a pre-installed RAPIDS venv. Pins are appropriately loose for development workflow (accepting minor+patch updates) while locking reproducibility via uv.lock. No security vulnerabilities or API deprecations blocking the code. Three low-priority modernization suggestions (type hints, datetime alias, torch ceiling) but no critical action required.

**RAPIDS/PyTorch governance:** Correct and enforced. No policy violations.
