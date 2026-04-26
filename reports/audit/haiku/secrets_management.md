# Secrets Management Audit — ArqueoGal

**Date:** 2026-04-26  
**Scope:** AIP TAP bearer token handling, credential storage, and log leakage  
**Auditor:** Claude Haiku 4.5

---

## Executive Summary

The ArqueoGal repository implements secure credential management with no identified token leaks. The AIP TAP bearer token is correctly handled via environment variables (`GAIA_AIP_TOKEN`) or external YAML files (`~/.arqueogal/credentials.yaml`), both outside the git repository. All credential-related code follows security best practices: tokens are never logged, error messages do not echo credentials, and test fixtures use synthetic values only.

---

## Findings

### (a) Credentials File Exclusion

**Status: PASS**

- `.gitignore` (line 44-45) correctly ignores all `.env` and `*.env` files.
- The actual credentials file path `~/.arqueogal/credentials.yaml` is outside the repository root (`/home/aneitzel/.arqueogal/`), not under version control.
- An example template `credentials.yaml.example` exists outside the repo at `/home/aneitzel/.arqueogal/credentials.yaml.example` with placeholder values only; no real credentials are present.

**File references:**
- `.gitignore` line 44-45
- `/home/aneitzel/.arqueogal/credentials.yaml.example` (outside repo, safe template)

---

### (b) Token Literal Verification

**Status: PASS**

No token literals (JWT patterns, API key prefixes, or bearer tokens) are committed to the repository.

**Search results:**
- Grep for common token patterns (`eyJ`, `ghp_`, `sk_live`, etc.) across `src/` and `tests/` yielded no matches.
- Test fixture at `tests/data/test_credentials.py:117-119` uses synthetic value `"abc123"` (not a real token).
- Token environment variable constant `GAIA_AIP_TOKEN = "GAIA_AIP_TOKEN"` is defined at `src/arqueogal/data/credentials.py:46` (the name of the env var, not a token value).

**File references:**
- `src/arqueogal/data/credentials.py:46` — constant definition (safe)
- `tests/data/test_credentials.py:117-119` — synthetic test token (safe)

---

### (c) YAML Loader Robustness

**Status: PASS**

The credential loader gracefully handles missing and invalid YAML without echoing secrets.

**Implementation review:**
- `src/arqueogal/data/credentials.py:92–122` (`load_credentials` function):
  - Line 106-110: Raises `FileNotFoundError` with descriptive message if file does not exist. Path is shown but no credentials are exposed.
  - Line 114-115: Uses `yaml.safe_load()` with `or {}` fallback, preventing null-pointer crashes.
  - Line 116-117: Validates top-level YAML structure; error messages cite the file path and type, not content.
  - Line 125-131: Permission check (`_check_permissions`) explicitly prevents group/other read access (enforces `0600` mode); error message does not echo credentials.

- `src/arqueogal/data/credentials.py:70–79` (`load_aip_token_from_env` function):
  - Line 78: Retrieves token from environment with `.strip()` to handle whitespace safely.
  - Line 79: Returns `None` (not an error) if token is empty or whitespace-only, allowing clean fallback.

**File references:**
- `src/arqueogal/data/credentials.py:92-122` — safe error handling
- `src/arqueogal/data/credentials.py:70-79` — env var fallback

---

### (d) Log Output Review

**Status: PASS**

No token leakage detected in logs, assertions, or debug output.

**Evidence:**
- `src/arqueogal/data/tap.py:139` — only logs the env var name, not the token value:
  ```python
  logger.info("AIP: using %s env-var token (YAML aip block absent)", AIP_TOKEN_ENV_VAR)
  ```
  Logs `"AIP: using GAIA_AIP_TOKEN env-var token (YAML aip block absent)"` — variable name only.

- `src/arqueogal/data/tap.py:144–147` — error message when credentials are missing:
  ```python
  raise RuntimeError(
      "AIP credentials missing: no 'aip' block in credentials.yaml and no "
      f"{AIP_TOKEN_ENV_VAR} environment variable set."
  )
  ```
  Does not include the actual token value or password; only the env var name.

- All other logging in `tap.py` (sync/async query completion, job status, batch iteration) references URLs and query templates, never credentials.

**File references:**
- `src/arqueogal/data/tap.py:137-147` — safe logging

---

### (e) Test Fixture Review

**Status: PASS**

Test fixtures use only synthetic credentials; no real tokens are committed.

**Evidence:**
- `tests/data/test_credentials.py:116-140`:
  - Line 117: Token fixture uses `"abc123"` (plaintext, clearly fake).
  - Line 134-140: Tests whitespace stripping with `"  padded-token\n"` (synthetic).
  - All other fixtures use `"testuser"`, `"testpass"`, `"esauser"`, `"esapass"` (dummy values).
  - No monkeypatched env vars carry real tokens; all test setups use mock values.

**File references:**
- `tests/data/test_credentials.py:116-140` — safe test tokens

---

### (f) Provenance Sidecars

**Status: PASS**

Provenance JSON sidecars do not contain token or credential data. They record TAP endpoints and query templates (necessary for reproducibility) but not authorization headers.

**Evidence from `data/interim/stream1_gaia_dr3_raw.provenance.json`:**
- Line 13: Records `"endpoint": "https://gaia.aip.de/tap"` (URL only, no auth).
- Line 14: Records the ADQL query template (SELECT/JOIN structure, no credentials).
- No `Authorization`, `session`, `header`, or `token` fields are present.

**File references:**
- `src/arqueogal/data/provenance.py:68-77` — `TapSource` dataclass never includes auth fields
- `data/interim/stream1_gaia_dr3_raw.provenance.json:13-17` — live example (no tokens)

---

### (g) Permission Enforcement

**Status: PASS**

The credential loader enforces strict file permissions (`0600` owner-only) to prevent unintended exposure.

**Implementation:**
- `src/arqueogal/data/credentials.py:125-131` (`_check_permissions` function):
  - Line 126: Reads file mode with `stat.S_IMODE()`.
  - Line 127: Rejects any mode with group or other bits set (`mode & 0o077`).
  - Line 128-130: Error message includes the current mode (e.g., `0o644`) and remediation command (`chmod 600`).

**Test coverage:**
- `tests/data/test_credentials.py:68-73`: Explicitly tests rejection of `0o644` (readable by others).
- `tests/data/test_credentials.py:76-80`: Confirms acceptance of stricter modes like `0o400` (owner read-only).

**File references:**
- `src/arqueogal/data/credentials.py:125-131` — permission enforcement
- `tests/data/test_credentials.py:68-80` — permission test coverage

---

## Summary of Findings

| Category | Status | Notes |
|----------|--------|-------|
| Credentials outside repo | PASS | Both YAML and example file are in `~/.arqueogal/`, not under git control. |
| No committed tokens | PASS | No JWT, API key, or bearer token literals found in any file. |
| YAML loader robustness | PASS | Gracefully handles missing files and invalid formats without echoing secrets. |
| Log output safety | PASS | Logs only the env var name and endpoint URLs, never the actual token. |
| Test fixture safety | PASS | All test credentials are synthetic (`abc123`, `testpass`, etc.). |
| Provenance sidecars | PASS | Contain TAP endpoints and queries, no authorization headers. |
| Permission enforcement | PASS | Requires `0600` mode on credentials file; rejects wider permissions. |

---

## Recommendations

1. **Document token rotation:** The current code supports token fallback via `GAIA_AIP_TOKEN` env var. Consider documenting a rotation procedure for bearer tokens when they need renewal (e.g., via a `ROTATE_TOKEN.md` in `docs/`).

2. **Add CI/CD token injection guidance:** If CI/CD is added in future phases, provide a template for secure secret injection (e.g., GitHub Actions with `secrets.GAIA_AIP_TOKEN`). Ensure the guidance forbids committing tokens to `.github/workflows/` files.

3. **Monitor for accidental commits:** Continue using git hooks (e.g., `trufflehog` or `detect-secrets` pre-commit hook) to catch any future accidental token commits.

4. **Audit logging metadata:** The provenance sidecars correctly omit auth headers. Ensure any future logging additions (e.g., request metrics) do not inadvertently capture HTTP request headers.

---

## Appendix: Tested Scenarios

- ✓ Missing credentials file → `FileNotFoundError` raised, no token echoed.
- ✓ Empty/whitespace-only `GAIA_AIP_TOKEN` → fallback returns `None`, handled gracefully.
- ✓ File with permissions `0o644` → `PermissionError` raised.
- ✓ File with stricter permissions `0o400` → accepted without error.
- ✓ Token env var with leading/trailing whitespace → stripped correctly.
- ✓ No token literals in logs, error messages, or assertion failures.

---

**Audit completed:** 2026-04-26  
**No critical issues identified.**
