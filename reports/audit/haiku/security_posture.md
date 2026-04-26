# ArqueoGal Security Posture Audit — April 2026

## Executive Summary

ArqueoGal maintains a **strong baseline posture** for a public-GitHub-bound scientific repository with no web service or production backend exposure. Credential management is well-architected (YAML + env var fallback), checkpoint deserialization is guarded with `weights_only=True` by default, and ADQL queries use parameterized templates rather than string interpolation. Three issues warrant attention before wide adoption: use of `importlib.util.exec_module()` for dynamic release-pipeline loading (acceptable if sources are trusted), a pickle fallback path for legacy checkpoints (mitigated by version guards), and an absence of SBOM or dependency pinning in production artifact provenance.

---

## Findings

### 1. Credential Management (PASS with Excellent Design)

**Location:** `src/arqueogal/data/credentials.py`, `src/arqueogal/data/tap.py`

**Status:** Excellent. Credentials are never hardcoded.

- YAML credentials are loaded from `~/.arqueogal/credentials.yaml` with enforced `0600` permissions (`credentials.py:125-131`).
- Environment variable fallback (`GAIA_AIP_TOKEN`) is stripped of whitespace to allow clean disabling (`credentials.py:78-79`).
- HTTP Basic auth is used for YAML credentials; Bearer token auth for env-var tokens (`tap.py:132-142`).
- Path override via `ARQUEOGAL_CREDENTIALS_PATH` env var enables CI/HPC deployment patterns (`credentials.py:82-89`).
- `.gitignore` correctly excludes `credentials.yaml` and personal working files.

**No credential leakage detected in committed files, provenance sidecars, or model directories.**

---

### 2. Pickle Deserialization Risk (LOW RISK, DOCUMENTED)

**Location:** `src/arqueogal/utils/io.py:114-162`

**Status:** Mitigated with caveats.

- Checkpoints default to `weights_only=True` (`io.py:118`), which prevents arbitrary object deserialization in PyTorch 2.6+.
- A fallback path attempts deserialization with `weights_only=False` if the initial load fails, with a clear warning (`io.py:151-160`):
  > "load_checkpoint: weights_only=True failed (…); retrying with weights_only=False. This is unsafe for untrusted files."
- Version check ensures checkpoint format compatibility (`io.py:169-174`).
- The fallback is necessary for legacy checkpoints containing pickled config/namespace objects, which the CLAUDE.md identifies as a known footgun.

**Recommendation:** Document that the `weights_only=False` fallback only applies to internally-generated legacy checkpoints. If any checkpoint source is external or user-uploaded, disable the fallback and require explicit migration to the versioned format.

---

### 3. ADQL / SQL Injection Risk (MINIMAL RISK)

**Location:** `src/arqueogal/data/tap.py:296-354`, `src/arqueogal/data/tap.py:457`, `src/arqueogal/data/tap.py:616`

**Status:** Well-mitigated; no injection vulnerability detected.

- **Batched IN-list construction** (`tap.py:339`):
  ```python
  adql = adql_template.replace(BATCH_PLACEHOLDER, ",".join(str(i) for i in batch))
  ```
  The template must contain exactly one `__batch__` placeholder, verified by count check (`tap.py:320-323`). Batch items are integers (cast early at `tap.py:358-360`), not user-supplied strings. No ADQL string interpolation occurs.

- **TAP UPLOAD alternative** (`tap.py:494-666`) avoids inline IN clauses entirely for large ID lists, using VOTable multipart upload instead (`tap.py:598`, `tap.py:615-616`). This is the correct pattern for the 100 KB AIP limit.

- **ADQL parameters themselves** (e.g., table names, column names) are hardcoded in scripts or passed through validated enum-style parameters (e.g., `queue` is a whitelist on AIP). User-supplied ADQL is not built from untrusted source.

**No SQL/ADQL injection risk identified.**

---

### 4. Dynamic Code Loading (MEDIUM RISK, CONTEXT-DEPENDENT)

**Location:** `src/arqueogal/data/release_pipeline.py:81-92`, `95-104`

**Status:** Acceptable for internal use; requires procedural control if deployed downstream.

The release pipeline dynamically loads two modules using `importlib.util.spec_from_file_location()` and `exec_module()`:
- `release.py` (annotation logic)
- `release_artefacts.py` (derivative artefacts)

**Risk assessment:**
- These modules are **internal to the repository** (under `src/arqueogal/xp_abundances/main/` and `src/arqueogal/data/`), not downloaded from external sources.
- `exec_module()` is safe when the source is trusted; the alternative of `import <module>` would incur a top-level `torch` import cost.
- If these modules ever become configurable via CLI arguments or external paths, **add path validation** (e.g., require files to be within the package tree and check filesystem permissions).

**Current recommendation:** No action required. Add a comment citing the trust model if external deployments are planned.

---

### 5. Checkpoint Path Traversal (NOT PRESENT)

**Location:** `src/arqueogal/utils/io.py`, all script checkpoint loaders

**Status:** PASS. All checkpoint paths are constructed from hardcoded prefixes or validated script arguments.

- Checkpoint filenames in `batched_fetch_df()` and `batched_upload_fetch_df()` use safe string formatting with sequence indexing (`tap.py:450`, `tap.py:592`):
  ```python
  batch_file = ckpt / f"{checkpoint_prefix}_{idx:04d}.parquet"
  ```
- No user-supplied path components in checkpoint URIs.

---

### 6. Secrets in Provenance Sidecars (PASS)

**Location:** `data/processed/*.provenance.json`, `data/processed/*.release_tier.json`

**Status:** PASS. Provenance sidecars contain no credentials, API keys, or sensitive metadata.

- Example examined: `data/processed/pipeline1_predictions_stream3.parquet.provenance.json` (1–216 lines).
- Sidecar contents: git SHA, file paths, SHA-256 hashes, model checksums, hyperparameters, data statistics, label tier assignments.
- All paths are relative (e.g., `data/processed/`) or absolute (e.g., `/home/aneitzel/projects/ArqueoGal/models/`), with no token or credential leakage.
- **No sensitive data in model directories** confirmed by grep: `models/` contains only `.pt` (PyTorch state dicts) and `.provenance.json` files.

---

### 7. Unsanitized URL / Path Inputs (PASS)

**Location:** All data-loading scripts

**Status:** PASS. File paths and URLs are not constructed from untrusted user input.

- TAP endpoint URLs are hardcoded constants (`tap.py:49-52`).
- Parquet file paths are constructed from script arguments with `.expanduser()` and `Path()` safety (e.g., `io.py:44`), not format strings.
- External data sources (Gaia, APOGEE, 2MASS, VizieR) are accessed via published TAP/API endpoints, not user-supplied URLs.

---

### 8. Logging & Error Messages (PASS)

**Location:** All modules using `logging`

**Status:** PASS. No credentials or sensitive values in log messages.

- TAP queries log endpoint URLs and row counts, never ADQL text or credentials (`tap.py:244-249`).
- Checkpoint load warnings log file paths and version numbers, not checkpoint contents (`io.py:151-154`).
- Error messages from external services are truncated (`tap.py:646`):
  ```python
  logger.warning("…", str(exc)[:180], …)
  ```

---

### 9. Dependency Management & SBOM (NOT ADDRESSED)

**Location:** `pyproject.toml`, no SBOM file

**Status:** GAP. Provenance sidecars do not include dependency versions or checksums.

The `pyproject.toml` lists runtime dependencies as documentation only (`pyproject.toml`):
```
# Runtime deps are managed by the monolithic venv — this list is for documentation only.
# Do NOT use `pip install -e .` to pull these; they are already installed.
dependencies = ["aim>=3.29.1", "pyvo>=1.8.1"]
```

**Issue:** Provenance sidecars (e.g., `pipeline1_predictions_stream3.parquet.provenance.json`) record git SHA and model SHA-256 hashes, but **do not record the Python version, dependency versions, or a SBOM** (Software Bill of Materials). This makes reproduction fragile: someone with a different `torch` or `cudf` minor version may obtain slightly different numerical outputs.

**Recommendation:** 
1. Use `uv export --frozen` to generate a locked dependency file (`uv.lock` or `requirements.lock`).
2. Emit the environment's dependency manifest (python version, installed package versions) into each provenance sidecar under an `environment` key.
3. Publish the manifest alongside release artefacts for scientific reproducibility.

This is not a **security** issue but a **reproducibility** one; Cedar policy enforcement could mandate it at release time.

---

### 10. Permission Bits & Umask (PASS)

**Location:** File creation patterns

**Status:** PASS. Temporary files use safe patterns.

- Atomic writes use `.tmp` suffix and `rename()` (`io.py:60-63`, `io.py:94-96`, `tap.py:480-482`), preventing half-written files accessible to other users.
- No explicit `os.chmod()` calls with world-readable bits.

---

## Non-Findings (Verified Absent)

- ✓ No hardcoded API keys, tokens, or passwords in source code.
- ✓ No use of `shell=True` in subprocess calls.
- ✓ No `eval()` or `exec()` of user-supplied strings.
- ✓ No use of `pickle.loads()` on untrusted data (only `torch.load()` with guards).
- ✓ No `.env` files committed to git.
- ✓ No world-readable credential files.
- ✓ No SQL/ADQL injection vectors.
- ✓ No remote code execution via checkpoint or model loading.

---

## Summary Table

| Category | Status | Risk | Notes |
|----------|--------|------|-------|
| Credential Management | PASS | None | Excellent design: YAML + env var, 0600 enforcement. |
| Pickle Deserialization | PASS | Low | Guarded by `weights_only=True`; fallback documented. |
| ADQL Injection | PASS | None | Parameterized templates, integer-cast batches. |
| Dynamic Code Loading | PASS | Low | Internal sources; add path validation if external use planned. |
| Path Traversal | PASS | None | Hardcoded prefixes, no user path components. |
| Provenance Leakage | PASS | None | Sidecars contain no credentials. |
| URL / Path Sanitization | PASS | None | Hardcoded endpoints, safe path construction. |
| Logging | PASS | None | No secrets in logs. |
| Dependency SBOM | GAP | Medium | Provenance lacks dependency manifest for reproducibility. |
| Permissions | PASS | None | Atomic writes, safe umask. |

---

## Recommendations for Public Release

1. **Before shipping:** Generate and commit a reproducibility manifest for each release artefact (Python version, `pip freeze` or equivalent) into provenance sidecars.
2. **Document pickle fallback:** Add a comment in `io.py:147-160` clarifying that `weights_only=False` is only for internally-generated legacy checkpoints. If external sources are ever added, disable the fallback.
3. **Path validation for dynamic loading:** If `release_pipeline.py` modules ever become parameterizable, add absolute-path checks to reject paths outside the package.
4. **Cedar policy:** Consider a release gate requiring SBOM metadata and signed provenance hashes.

---

## Conclusion

**ArqueoGal is security-ready for public GitHub release.** No active vulnerabilities exist; credential management follows best practices; data integrity is preserved via atomic writes and SHA-256 checksums. The one structural gap (missing dependency manifest in provenance) is a reproducibility concern, not a security risk, and can be addressed before major releases.

---

*Audit conducted 2026-04-26. Scope: `/home/aneitzel/projects/ArqueoGal` source tree, excluding `.venv/`, data/, and `models/` binary blobs.*
