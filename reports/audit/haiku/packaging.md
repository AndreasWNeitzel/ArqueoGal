# Packaging Audit: ArqueoGal pyproject.toml

**Date:** 2026-04-26  
**Scope:** `pyproject.toml` (65 lines, 9 sections), src/ layout, build configuration, dist-readiness

---

## Findings

### 1. Missing PyPI Classifiers (CRITICAL for gh release readiness)

**File:** `pyproject.toml` lines 5–21 (project metadata block)

**Issue:** No `classifiers` field. PyPI recommends Development Status, License, Intended Audience, Programming Language, and Topic classifiers for discoverability. Current state has zero classifiers.

**Impact:** Package appears incomplete to PyPI indexers and lacks signals for users (stable vs beta, license clarity, Python version coverage).

**Recommendation:**
```toml
[project]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

---

### 2. Version at 0.1.0 (GATE for GitHub release automation)

**File:** `pyproject.toml` line 7

**Issue:** Static version 0.1.0 with setuptools-scm in `requires` (line 2) but no dynamic version binding.

**Context:** 
- `setuptools-scm>=8.0` is listed as a build requirement (git-aware versioning ready).
- Current config does not use it; version is hardcoded.
- For GitHub Action automation (releases tied to git tags), this must be dynamic.

**Impact:** 
- Manual version bumps required for each release.
- Release CI/CD cannot auto-increment from git metadata.
- Inconsistency between source version and git tags.

**Recommendation:**
```toml
[project]
dynamic = ["version"]

[tool.setuptools-scm]
write_to = "src/arqueogal/_version.py"
fallback_version = "0.1.0"
```

Then in `src/arqueogal/__init__.py`:
```python
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"
```

---

### 3. Missing project.urls (GitHub/docs/ReadTheDocs links)

**File:** `pyproject.toml` — absent block

**Issue:** No `[project.urls]` section. PyPI template requires Homepage, Repository, Documentation, "Bug Tracker", Changelog.

**Impact:** 
- Users landing on PyPI have no direct links to source, docs, or issues.
- Fragments like "See GitHub" must appear in README or description.

**Recommendation:**
```toml
[project.urls]
Homepage = "https://github.com/AndreasWNeitzel/ArqueoGal"
Repository = "https://github.com/AndreasWNeitzel/ArqueoGal"
Documentation = "https://github.com/AndreasWNeitzel/ArqueoGal/tree/main/docs"
"Bug Tracker" = "https://github.com/AndreasWNeitzel/ArqueoGal/issues"
Changelog = "https://github.com/AndreasWNeitzel/ArqueoGal/releases"
```

---

### 4. Minimal `dependencies` block (intentional but undocumented)

**File:** `pyproject.toml` lines 18–21

**Issue:** Only `aim>=3.29.1` and `pyvo>=1.8.1` are listed. Comment on line 16 states runtime deps are "managed by the monolithic venv — this list is for documentation only" and warns against `pip install -e .`.

**Assessment:** 
- **Correct design:** ArqueoGal runs in a fixed `rapids25.10_python3.12_cuda13` venv where all deps (torch, scipy, cuml, etc.) are pre-installed. Declaring them here would create redundancy and version conflicts.
- **Missing:** metadata explaining why the list is minimal. Users cloning without the venv will face import errors with no clear guidance.

**Recommendation:** Expand the comment and add a note:
```toml
# Runtime dependencies are pre-installed in the monolithic rapids25.10_python3.12_cuda13 venv.
# Do NOT use `pip install -e .` — the venv is required for GPU acceleration and CUDA 13 compatibility.
# See docs/data_acquisition.md § 1 for setup instructions.
dependencies = [
    "aim>=3.29.1",
    "pyvo>=1.8.1",
]
```

---

### 5. `optional-dependencies` under-specified

**File:** `pyproject.toml` lines 23–24

**Issue:** Single `dev` extra with only `["ruff", "pytest", "pytest-cov"]`. Real dev environment includes 10+ packages (`torch`, `matplotlib`, `scipy`, `scikit-learn`, etc.) declared in `[dependency-groups]` (lines 54–65).

**Problem:** 
- `[project.optional-dependencies]` is PEP 508 (PyPI, `pip install -e ".[dev]"`).
- `[dependency-groups]` is a uv-specific extension, not recognized by pip.
- Users installing via pip see only ruff/pytest/pytest-cov as "dev".
- mismatch: full dev environment requires the venv anyway, so the `[project.optional-dependencies]` declaration is misleading.

**Recommendation:** Either (1) align them:
```toml
[project.optional-dependencies]
dev = [
    "astropy>=6",
    "jupytext>=1.19.1",
    "matplotlib>=3.8",
    "pandas>=2.0",
    "pyarrow>=15",
    "pytest>=9.0.3",
    "scikit-learn>=1.4",
    "scipy>=1.11",
]
```

Or (2) keep both in sync and note the uv-only [dependency-groups] is the canonical source.

---

### 6. Dependency pin tightness not specified

**File:** `pyproject.toml` lines 18–21, 54–65

**Assessment:**
- Declared deps (`aim>=3.29.1`, `pyvo>=1.8.1`) use lower-bound-only pins (flexible).
- `[dependency-groups]` dev pins (e.g., `torch>=2.0`, `scipy>=1.11`) also lower-bound-only.
- **Rationale:** Broad pins allow env manager (uv) to resolve newer compatible releases.
- **Risk:** No upper bound means new major versions (scipy 2.0, torch 3.0) auto-upgrade if available.

**Recommendation:** Document the pin strategy:
```toml
# Dependency versions: lower-bound only. Rationale: GPU deps (torch, cudf, cuml)
# are managed by uv and the RAPIDS 25.10 env pinning (cudf>=25.10.0, cuml>=25.10.0).
# Upper bounds are enforced at the venv level, not here.
```

---

### 7. Missing entry_points / [project.scripts]

**File:** `pyproject.toml` — absent block

**Issue:** 11+ CLI scripts in `scripts/` (apply_gaia_corrections.py, build_pipeline1_features_stream3.py, etc.) are not registered as entry points.

**Current state:** Scripts must be run via `python -m` or direct invocation; no CLI shortcuts like `arqueogal-apply-corrections` exist.

**Impact:** 
- Users installing the package cannot call scripts by name from shell.
- Scripts are discovery-unfriendly.

**Recommendation:** Add entry points if these should be public CLI tools:
```toml
[project.scripts]
arqueogal-apply-gaia-corrections = "arqueogal.scripts.apply_gaia_corrections:main"
arqueogal-build-features-stream3 = "arqueogal.scripts.build_pipeline1_features_stream3:main"
```

**Note:** This requires refactoring scripts into proper `arqueogal/scripts/` modules with `def main():` entry functions. Current scripts are ad-hoc; wrapping them is a separate effort.

---

### 8. Build-system configuration is sound

**File:** `pyproject.toml` lines 1–3

**Assessment:**
- `setuptools>=68.0` (modern, PEP 517/518 compliant).
- `setuptools-scm>=8.0` declared but not used (version is static, not dynamic).
- No build metadata issues.
- `[tool.setuptools.packages.find]` correctly points to `src/`.

**Verdict:** Build backend is ready; only needs dynamic version binding (item 2).

---

### 9. Src/ layout is clean

**File:** `src/arqueogal/` structure

**Assessment:**
- Correct layout: `src/arqueogal/{data/, xp_abundances/, utils/}`.
- `__init__.py` present (currently empty, see item 2 for version binding).
- No flat-layout drift detected.

**Verdict:** Layout is sound.

---

### 10. Ruff and pytest configuration is production-ready

**Files:** `pyproject.toml` lines 29–52

**Assessment:**
- Ruff config (line-length 100, py312, sensible rule set) matches CLAUDE.md conventions.
- Ruff ignores `N803`, `N806`, `PLR2004` (astronomical variables, magic constants) — correct.
- Pytest config includes testpaths, pythonpath, markers for slow/gpu/stress tests.

**Verdict:** Tooling is well-configured.

---

## Summary Table

| Issue | Severity | Type | File:Line |
|-------|----------|------|-----------|
| No PyPI classifiers | HIGH | Metadata | 5–21 |
| Version not dynamic (setuptools-scm unused) | HIGH | Config | 7 |
| Missing [project.urls] | MEDIUM | Metadata | absent |
| optional-dependencies ≠ dependency-groups | MEDIUM | Clarity | 23–24, 54–65 |
| Comment on deps minimal but unclear | LOW | Documentation | 16 |
| Pins strategy not documented | LOW | Documentation | 18–21 |
| Scripts not registered as entry_points | MEDIUM | Discoverability | absent |
| Build-system is sound | — | Positive | 1–3 |
| Ruff/pytest config is correct | — | Positive | 29–52 |
| Src/ layout is clean | — | Positive | src/ |

---

## Prioritized Action List

### For "now" (pre-GitHub release):

1. **Add PyPI classifiers** (line 5) — 5 min.
2. **Bind dynamic version** (line 7 + __init__.py) — 10 min.
3. **Add [project.urls]** — 5 min.
4. **Align optional-dependencies or document the split** (uv-only) — 10 min.

### For "later" (if public CLI is desired):

5. Refactor scripts into `arqueogal.scripts.*` modules with `def main()` entry points; register them.

### For "documentation":

6. Update README to clarify venv requirement and no `pip install -e .`.
7. Add inline comments explaining pin strategy and why minimal runtime deps.

---

## Notes

- Repo is designed for **internal-use research code**, not public PyPI consumption. The monolithic venv, uv-specific dependency-groups, and minimal declared deps reflect this.
- Shifts toward public GitHub release require metadata completeness (classifiers, URLs, dynamic versioning) and documentation clarity (why minimal deps, why `[dependency-groups]` not `[project.optional-dependencies]`).
- The build backend and lint/test tooling are solid; the gaps are in release-hygiene metadata and script discoverability.
