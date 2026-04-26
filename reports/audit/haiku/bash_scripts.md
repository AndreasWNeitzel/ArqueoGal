# Bash Script Audit — ArqueoGal

**Date:** 2026-04-26  
**Scope:** `/home/aneitzel/projects/ArqueoGal/scripts/run_*.sh` and `scripts/gallery/build_all.sh`  
**Verdict:** Code is well-structured with strong defensive foundations; minor improvements available.

---

## Summary

All five scripts follow solid defensive patterns (set -euo pipefail, proper quoting, REPO_ROOT derived from BASH_SOURCE). Issues are minor: IFS not set in two scripts with iteration, one script using nohup without background-job cleanup trap, one script piping to `tail` masking errors in stage failures, and hardcoded `python` in one invocation. No hook-skipping violations or unquoted expansions detected.

---

## Detailed Findings

### 1. **run_full_pipeline.sh** (lines 1–84)

**Status:** Good  
**Findings:**

- Line 18: `set -euo pipefail` present, correct.
- Line 20: REPO_ROOT correctly derived from BASH_SOURCE.
- Lines 35–49: Variable expansions are quoted (`"$ENSEMBLE_DIR"`, `"$PRED_PARQUET"`).
- Line 68: `PYTHONPATH=src` is unquoted but assignment context, acceptable.
- **Minor issue, line 72–74:** Command substitution in pytest call has no error guard; if pytest fails, the exit code is lost. Better practice: `"$PY" -m pytest ... || exit 1`.

### 2. **run_bg.sh** (lines 1–25)

**Status:** Good with minor gaps  
**Findings:**

- Line 3: `set -euo pipefail` present.
- Line 8: REPO_ROOT correctly derived.
- Lines 5–6: Positional argument shift, no validation for empty arguments. If `$2` is missing, the `nohup` line will receive incorrect arguments. Add: `[[ -n "$SCRIPT" ]] || { echo "Usage: run_bg.sh <log_name> <python_script> [args...]" >&2; exit 1; }`.
- **Line 17:** Hardcoded `python` instead of using `$PY`. Should be `nohup "${PY}" ...`.
- **Missing IFS:** No `IFS=$'\n\t'` set. Not critical for this script (no loops with word splitting risk), but omitted from pattern.
- **Missing EXIT trap:** No cleanup for background job. Script exits after spawning; if script is killed while job runs, no mechanism to record or manage the job. Consider `trap 'wait' EXIT` or explicit job tracking.

### 3. **run_tests.sh** (lines 1–21)

**Status:** Good  
**Findings:**

- Line 4: `set -euo pipefail` present.
- Lines 12–16: Proper file existence check before using interpreter.
- **Line 20:** Redundant `set -o pipefail` after line 4's `set -euo pipefail` (already active). Remove.
- **Line 21:** Pipe to `tail` masks pytest exit code. If pytest returns non-zero, the tail succeeds and the script returns 0. Critical for CI: use `tee` instead: `"${PY}" -m pytest "$@" 2>&1 | tee >(tail -n "${TAIL}")` (requires Bash 4.2+ process substitution) or just `"${PY}" -m pytest "$@" | tail -n "${TAIL}"` and check exit code separately.

### 4. **gallery/build_all.sh** (lines 1–59)

**Status:** Good with critical concern  
**Findings:**

- Line 8: `set -euo pipefail` present.
- Line 10: REPO_ROOT correctly derived.
- **Lines 47–54:** Array iteration is safe (`for s in "${STAGES[@]}"`), variables quoted.
- **Line 50:** Stage failure is silently continued with `|| echo "  [WARN] ..."`. This is deliberate (noted in comment); however, the exit code of the overall script remains 0 even if multiple stages fail. If this script is used in CI/CD, consider collecting failures and exiting non-zero: `failures=0; ... || ((failures++)); ... [[ $failures -eq 0 ]] || exit 1`.
- **Missing IFS:** Not set, but no iteration over untrusted strings, so low priority.

### 5. **monitoring/wsl_memory_monitor.sh** (lines 1–36)

**Status:** Good  
**Findings:**

- Line 7: `set -euo pipefail` present.
- Line 9: `readonly` for constants, good.
- Line 14: `while true` with no exit condition (intentional for monitoring), acceptable.
- Line 23: `pgrep -f ... || echo ""` is safe fallback.
- **Line 24:** `if [ -n "${py_pids}" ]` is POSIX-compliant and correct.
- **Line 25:** `for pid in ${py_pids}` has unquoted variable, but `pgrep` output is guaranteed to be numeric PIDs (safe from word splitting). However, best practice: `for pid in ${py_pids}; do` could be `while read -r pid; do ... done <<< "$(pgrep ...)"` for stricter safety.
- **Missing IFS:** Not set. Low priority given the context (numeric PIDs only).

---

## Cross-Cutting Patterns

### IFS not set globally

**Files:** run_bg.sh (line 3), run_tests.sh (line 4), gallery/build_all.sh (line 8), wsl_memory_monitor.sh (line 7)

**Recommendation:** Add `IFS=$'\n\t'` after `set -euo pipefail` in all four scripts. Prevents accidental word splitting on spaces in unquoted expansions (defense in depth).

### Argument validation absent

**File:** run_bg.sh (lines 5–6)

**Issue:** If `LOG_NAME` or `SCRIPT` are empty (user provides <2 args), the script will invoke `nohup` with garbage. Add explicit check before shift.

### Error masking via pipe

**Files:** run_tests.sh (line 21), gallery/build_all.sh (line 50)

**Issue:** Piping to `tail` or silencing with `|| echo "WARN"` masks exit codes. Acceptable for monitoring/reporting, but problematic for CI/CD gates. Document intent in comments.

### Hardcoded executable paths

**File:** run_bg.sh (line 17)

**Issue:** `nohup python ...` uses `PATH`-resolved `python`, not the detected `$PY`. Inconsistent with other scripts. Use `nohup "${PY}" ...`.

---

## Hook and Git Compliance

**Status:** Compliant

- No `--no-verify`, `--no-gpg-sign`, or hook-skipping flags detected.
- No commits issued from these scripts.
- CLAUDE.md invariant "no commits without explicit request" is upheld.

---

## Recommendations (Priority Order)

1. **HIGH:** run_bg.sh line 17 — use `"${PY}"` instead of hardcoded `python`.
2. **HIGH:** run_bg.sh lines 5–6 — add argument validation.
3. **MEDIUM:** All scripts — add `IFS=$'\n\t'` for defensive consistency.
4. **MEDIUM:** run_tests.sh line 21 — document or fix error masking (consider `tee` alternative).
5. **LOW:** run_tests.sh line 20 — remove redundant `set -o pipefail`.
6. **LOW:** gallery/build_all.sh line 50 — document that silent failures are intentional; optionally add failure counter for overall exit status.

---

## Portability Notes

- All scripts assume Bash 4.0+ (BASH_SOURCE, process substitution).
- `date -u +%Y-%m-%dT%H:%M:%SZ` is POSIX-compliant.
- `awk` and `pgrep` are assumed present; portable across Linux/WSL2.
- No GNU-specific options detected (e.g., `date -d`, `sed -i` without `''`).
- Scripts are portable to macOS with no changes (test `pgrep` availability on older macOS versions; generally available from 10.11+).
