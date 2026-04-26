# Configuration Audit — ArqueoGal

Date: 2026-04-26  
Scope: `src/arqueogal/`, `src/arqueogal/xp_abundances/main/`, `scripts/`, configuration infrastructure

---

## Critical Finding: Per-Element Sigma Thresholds Duplicated

**Status:** Guarded by a test, but architecture is fragile.

### Locations

| File | Line(s) | Definition | Context |
|------|---------|-----------|---------|
| `/src/arqueogal/xp_abundances/main/release.py` | 136–145 | `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` (canonical) | Release-tier gate; heavily documented |
| `/src/arqueogal/data/release_pipeline.py` | 416–424 | `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` (duplicate) | Orchestrator; docstring says "Duplicated rather than imported to avoid coupling…to the heavy release.py module" |

### Values

Both dictionaries hold:
- `teff`: 150.0 K
- `logg`: 0.30 dex
- `mh`: 0.20 dex
- `alpha_m`: 0.05 dex (tightened 2026-04-26)
- `mg_h`: 0.20 dex

### Guard

Test `/tests/data/test_release_pipeline.py::test_sigma_threshold_sync` (implicit import via `from archeogal.xp_abundances.main.release import _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD as canon`) validates equivalence at test time. **Test is sound, but cannot prevent at-runtime divergence if only one is updated during refactoring.**

### Secondary Duplicates

The same thresholds are **hardcoded as literal values** in histogram binning logic:
- `/src/arqueogal/xp_abundances/main/bimodality.py`:268–278 uses `0.30`, `0.20` for logg, mh edges (in context of a different algorithm, but numericaly identical)
- `/src/arqueogal/data/master_schema.py`:269 documents "0.30 dex; [M/H], [Mg/H]: 0.20 dex" in prose without a constant

---

## Hardcoded Paths (Relative)

**Status:** CLI-friendly but not environment-variable externalized.

### Training Configuration

File: `/src/arqueogal/xp_abundances/main/config.py` (frozen dataclass, `TrainingConfig`)

```python
train_parquet: Path = Path("data/processed/pipeline1_training.parquet")  # line 60
output_dir: Path = Path("models/main/xp_abundances")  # line 61
```

These are defaults in a frozen dataclass; CLI scripts override via argparse. **No env-var fallback; changes require code edits or CLI args on every invocation.**

### Release Pipeline Defaults

File: `/src/arqueogal/data/release_pipeline.py` (function defaults, lines 406–410)

```python
@click.command()
@click.option("--predictions-parquet", type=click.Path(), default=Path("data/processed/pipeline1_predictions_stream3.parquet"))
@click.option("--features-parquet", type=click.Path(), default=Path("data/processed/pipeline1_features_stream3.parquet"))
```

Hardcoded relative paths in function signature. **No environment variable option; users must pass CLI flags or edit code.**

### Inference Defaults

File: `/scripts/run_pipeline1_inference.py` (module-level constants)

```python
DEFAULT_ENSEMBLE_DIR = Path("models/main/xp_abundances")  # line ~55
DEFAULT_FROZEN_STATS = Path("data/processed/pipeline1_features_stream1.provenance.json")  # line ~56
DEFAULT_OOD_TRAIN_PARQUET = Path("data/processed/pipeline1_training.parquet")  # line ~57
```

These are overridable by `--ensemble-dir`, `--frozen-stats`, `--ood-training-parquet` flags, but **no env-var support** (e.g., `ARQUEOGAL_ENSEMBLE_DIR`). In HPC and container environments, users must either export env vars that code does not read, or pass explicit CLI flags every time.

---

## Credentials & Secrets Management

**Status:** Well-designed, but could be unified with config framework.

### Strengths

- Central module: `/src/arqueogal/data/credentials.py`
- YAML-based (`~/.arqueogal/credentials.yaml` with `0600` permission checking)
- Environment-variable fallback: `GAIA_AIP_TOKEN` (token auth) or `ARQUEOGAL_CREDENTIALS_PATH` (file override)
- Never hardcoded in source; never dumped in logs (no `print(credentials)` in codebase)

### Weakness

Credentials are loaded via custom YAML parser (lines 92–123) separate from the unified config framework elsewhere in the codebase. Two loading patterns:

1. **Credentials:** custom YAML + pydantic-free dataclasses (`ServiceCredentials`, `TokenCredentials`)
2. **Training/model config:** generic `load_config()` (utils/config.py) with dataclass schema

**Unification opportunity:** migrate credentials to `BaseSettings` from `pydantic_settings`, coexisting with training config — single source of truth for all externalized values.

---

## TAP Configuration Constants

**Status:** Well-organized in module scope, but duplicated in docstrings.

### Defined Constants

File: `/src/arqueogal/data/tap.py`

```python
SYNC_ROW_THRESHOLD = 5_000  # line 54
DEFAULT_ASYNC_TIMEOUT_SEC = 3600  # line 62
_TRANSIENT_ERROR_MARKERS = (...)  # lines 69–95
```

**Used by:** `fetch_ir_photometry.py`, `tess_hon2021.py`, `gaia_enrich.py`, and multiple ingestion scripts.

**Docstring duplication:** value 5000 is documented in `data_acquisition.md §3.6`, §13.11 **and** in inline comments in multiple modules. **Change risk:** if threshold changes, docs must be updated in three places.

---

## Environment Variables

### Documented

| Variable | Purpose | Default | Locations |
|----------|---------|---------|-----------|
| `GAIA_AIP_TOKEN` | AIP TAP bearer token | None (optional) | `data/credentials.py`:46, `data/tap.py`:41 |
| `ARQUEOGAL_CREDENTIALS_PATH` | Path override for credentials YAML | `~/.arqueogal/credentials.yaml` | `data/credentials.py`:45 |

### Not Documented as Env Vars (But Could Be)

- Data root directory (relative paths in code; no `ARQUEOGAL_DATA_ROOT`)
- Model checkpoint directory (hardcoded `models/main/xp_abundances`)
- Log output directory (relative `reports/`, no `ARQUEOGAL_REPORTS_ROOT`)
- Ephemeral cache for TAP jobs, AIM experiment tracking (no env vars; scripts write to cwd)

---

## Validation & Fail-Fast

### Strengths

- Credentials file existence check (line 106–110, `credentials.py`)
- Permission bit validation (line 125–131, `credentials.py`)
- Config-file unknown-key warnings (line 160–163, `utils/config.py`)
- Type coercion and validation in `load_config()` (lines 54–136, `utils/config.py`)

### Gaps

1. **No single startup validation routine.** Scripts validate on demand (e.g., `run_pipeline1_inference.py` checks ensemble dir existence in `main()`), but there is no unified `validate_environment()` called at package import.
2. **Path defaults not validated at definition time.** The `TrainingConfig` dataclass holds relative paths; code assumes they are resolvable from cwd. No check that `data/processed/` or `models/main/` directories exist.
3. **Missing config:** No mechanism to audit whether all required directories are writable, or whether external services (AIP, GAVO) are reachable at startup.

---

## Recommendations

### Priority 1: Unify Sigma Thresholds

**Current:** Two identical dicts in `release.py` and `release_pipeline.py`, guarded by a test.

**Option A (minimal):** Add a module-level constant in a shared location (`src/arqueogal/utils/release_constants.py`) and import both places. Update test to verify import not duplication.

**Option B (best):** Migrate to a pydantic settings class:
```python
from pydantic_settings import BaseSettings

class ReleaseConfig(BaseSettings):
    sigma_inflated_threshold_teff_k: float = 150.0
    sigma_inflated_threshold_logg_dex: float = 0.30
    sigma_inflated_threshold_mh_dex: float = 0.20
    sigma_inflated_threshold_alpha_m_dex: float = 0.05
    sigma_inflated_threshold_mg_h_dex: float = 0.20
    
    @computed_field
    @property
    def sigma_inflated_threshold_dict(self) -> dict[str, float]:
        return {
            "teff": self.sigma_inflated_threshold_teff_k,
            ...
        }
```
Allows env-var override (`ARQUEOGAL_SIGMA_INFLATED_THRESHOLD_ALPHA_M_DEX=0.06`) without code edit for future tuning.

### Priority 2: Externalize Path Defaults

Add environment variable support to all path arguments in scripts and module defaults:

```python
# Before
DEFAULT_ENSEMBLE_DIR = Path("models/main/xp_abundances")

# After
import os
from pathlib import Path

_ENSEMBLE_DIR_DEFAULT = os.getenv("ARQUEOGAL_ENSEMBLE_DIR", "models/main/xp_abundances")
DEFAULT_ENSEMBLE_DIR = Path(_ENSEMBLE_DIR_DEFAULT).expanduser()
```

Rationale: enables HPC job scripts and containers to set a single env var instead of passing CLI flags on every invocation.

### Priority 3: Unify Credentials & Config Loading

Migrate `credentials.py` to use `pydantic_settings.BaseSettings` alongside training config:

```python
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class ArcheogalSettings(BaseSettings):
    # Credentials (secrets)
    aip_user: str | None = Field(None, validation_alias="ARQUEOGAL_AIP_USER")
    aip_password: str | None = Field(None, validation_alias="ARQUEOGAL_AIP_PASSWORD")
    aip_token: str | None = Field(None, validation_alias="GAIA_AIP_TOKEN")
    
    # Paths
    data_root: Path = Field(Path("data"), validation_alias="ARQUEOGAL_DATA_ROOT")
    models_root: Path = Field(Path("models"), validation_alias="ARQUEOGAL_MODELS_ROOT")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
```

Benefits: single env-file (`.env.local` for dev, `.env.prod` for prod), type validation, secrets never logged.

### Priority 4: Add Startup Validation

Create a `arqueogal.core.validate_environment()` function called in each script's `main()`:

```python
def validate_environment(config: ArcheogalSettings) -> None:
    """Fail fast if required directories don't exist or aren't writable."""
    required_dirs = [
        config.data_root / "processed",
        config.models_root,
    ]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Required directory missing: {d}")
        if not os.access(d, os.W_OK):
            raise PermissionError(f"Not writable: {d}")
```

### Priority 5: Document TAP Constants

Add a single source of truth for `SYNC_ROW_THRESHOLD`:

```python
# src/arqueogal/data/tap.py
SYNC_ROW_THRESHOLD = 5_000
"""Queries expected to return ≤ this many rows use sync; larger queries go async.

Rationale: AIP sync queries time out at ~90 s (per data_acquisition.md §13.11).
When syncing, aim for 5k–10k rows per batch; when async, support larger batches.
See data_acquisition.md §3.6 and §13.11 for usage guidelines."""
```

Then reference this constant in docs (`data_acquisition.md`), not duplicate the number.

---

## Configuration Debt Summary

| Issue | Impact | Fix Difficulty | Recommendation |
|-------|--------|-----------------|-----------------|
| Sigma thresholds duplicated (two dicts + three indirect refs) | Medium — test guards, but fragile refactoring risk | Low | Unify via constants module or settings class |
| Hardcoded relative paths (no env-var fallback) | Medium — requires CLI flags in scripts; blocks containerization | Low | Add `os.getenv()` fallback to defaults |
| Credentials + config use different loading patterns | Low — both work, but inconsistent | Medium | Migrate to unified `pydantic_settings` |
| No startup validation routine | Low — errors caught downstream | Low | Add `validate_environment()` helper |
| TAP thresholds documented in multiple places | Low — low-frequency change | Trivial | Consolidate; reference single source |

---

## Files to Monitor

- `/src/arqueogal/xp_abundances/main/release.py` — sigma thresholds (canonical)
- `/src/arqueogal/data/release_pipeline.py` — sigma thresholds (duplicate)
- `/src/arqueogal/xp_abundances/main/config.py` — hardcoded paths in `TrainingConfig`
- `/scripts/run_pipeline1_inference.py` — hardcoded path defaults
- `/src/arqueogal/data/credentials.py` — credentials loading (separate from config framework)

---

## Test Coverage

**Passing:** `/tests/data/test_release_pipeline.py::test_sigma_threshold_sync` validates sigma dict equivalence.

**Missing:** No integration test for path resolution, credential file validation, or startup environment checks. Consider adding `tests/test_environment.py` with fixtures for checking writable dirs, credential file access, and TAP service connectivity.
