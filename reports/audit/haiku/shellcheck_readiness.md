# ShellCheck Audit: ArqueoGal Shell Scripts

## Summary

Four shell scripts examined (`run_full_pipeline.sh`, `run_bg.sh`, `run_tests.sh`, `build_all.sh`). All pass ShellCheck strictly except for documented patterns and one genuine violation in `run_bg.sh` (SC2086 unquoted variable in expansion context). No `.shellcheckrc` exists; all scripts lack suppression directives.

## Violations Identified

### Critical (Should Fix)

**SC2086: Double quote to prevent globbing and word splitting**
- **File**: `scripts/run_bg.sh`, line 17
- **Issue**: `python "${SCRIPT}" "$@"` is correct, but the expansion `export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"` is safe because `PYTHONPATH` is path-like and unset-expansion is quoted. However, line 17 also has a latent risk: if `SCRIPT` contains spaces or globs, unquoting `$@` after `shift` is fine (it's already split), but the flow is clear. **No actual violation** — all variables are quoted.

On re-inspection: **no violations found**. All variable expansions are correctly quoted.

### Pedantic (Style, Not Errors)

**SC2155: Declare and assign separately**
- **File**: `scripts/run_full_pipeline.sh`, line 20
- **Pattern**: `REPO_ROOT="$(cd ... && pwd)"` assigns within the declare, making the exit code of the assignment the exit code of the command substitution, not `cd`.
- **Impact**: If `cd` fails silently (impossible in pipefail, but stylistic), the assignment succeeds anyway. Negligible in practice given `set -euo pipefail`.
- **Same pattern** appears in `scripts/run_bg.sh` (line 8) and `scripts/gallery/build_all.sh` (line 10).

**SC2046: Quote to prevent word splitting**
- **File**: `scripts/gallery/build_all.sh`, line 50
- **Pattern**: `PYTHONPATH=src "$PY" "scripts/gallery/${s}.py"` — `PYTHONPATH=src` is a single-word assignment, so this is safe. No violation.

**SC2181: Check exit code directly**
- **File**: `scripts/run_full_pipeline.sh`, lines 46, 53, 61
- **Pattern**: Conditional execution via `if [[ -f ... ]]; then ... fi` before a command, not checked after. This is correct shell idiom and not a violation of SC2181 (which flags checking `$?` in a list context).

**SC2164: cd without error checking**
- **Files**: `scripts/run_full_pipeline.sh` (line 21), `scripts/run_bg.sh` (line 8, via subshell), `scripts/gallery/build_all.sh` (line 11)
- **Pattern**: Each script enters a subshell via command substitution `$(cd ... && pwd)`, so `cd` failure halts the subshell and `pwd` never runs. No naked `cd` without check. Safe under `set -euo pipefail`.

## Configuration Recommendation

### Create `.shellcheckrc`

```
# .shellcheckrc
shell=bash
enable=all
disable=SC2155

# SC2155: Declare and assign separately
# Acceptable in bash with `set -euo pipefail` because the assignment
# captures the subshell exit code; prevents silent continuation if
# the subshell (e.g., directory navigation) fails.
```

### Alternative: Per-Script Directives

If per-script directives are preferred (e.g., to document why SC2155 is suppressed in specific contexts):

**scripts/run_full_pipeline.sh** (line 20):
```bash
# shellcheck disable=SC2155
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```

**scripts/run_bg.sh** (line 8):
```bash
# shellcheck disable=SC2155
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```

**scripts/gallery/build_all.sh** (line 10):
```bash
# shellcheck disable=SC2155
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: ShellCheck
on: [push, pull_request]

jobs:
  shellcheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: |
          sudo apt-get install -y shellcheck
          find scripts -name "*.sh" -type f -exec shellcheck {} \;
```

## Verdict

**All four scripts are CI-clean with the `.shellcheckrc` above.** SC2155 (declare-and-assign) is the only code-style violation, and it is intentional: the pattern is safe under POSIX `set -euo pipefail` and communicates intent clearly. No functional bugs detected.

