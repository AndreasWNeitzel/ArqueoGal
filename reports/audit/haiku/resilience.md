# Resilience Audit: TAP and Downloads Modules

**Date:** 2026-04-26  
**Scope:** `/src/arqueogal/data/tap.py`, `/src/arqueogal/data/downloads.py`  
**Auditor:** Claude Haiku 4.5

---

## Executive Summary

The **TAP module** (`tap.py`) implements production-grade resilience for async TAP queries: exponential backoff with jitter on transient errors, appropriate retry budgets, proper exception classification, and comprehensive logging (lines 609–650). However, the **sync query path** (`run_sync`, line 183–205) lacks any retry logic, and the **downloads module** (`downloads.py`) has no retry or timeout handling for HTTP failures — both are single-shot with hard-coded 60s timeout regardless of file size or network conditions.

---

## Findings

### 1. TAP Async Path: Production-Ready Retry (Lines 609–650)

**Status:** ✓ Compliant

The `batched_upload_fetch_df()` function implements proper exponential backoff with jitter:

- **Transient detection:** Uses `_is_transient_tap_error()` (lines 98–101) with heuristic string matching on documented transient markers (PooledConnection, gateway timeouts, 5xx codes, connection reset) — matches CLAUDE.md footgun list.
- **Exponential backoff:** `delay * 2**attempt` (line 636), capped implicitly by max_retries loop.
- **Jitter:** ±25% random jitter (line 637), prevents thundering herd.
- **Bounded retries:** Default `max_retries=6` (line 506), yielding `~95% P(success)` on ESA's observed ~40% per-attempt transient rate (line 544 comment).
- **Sleep enforcement:** Via `time.sleep(sleep_for)` (line 648), monotonic delay.
- **Logging:** Warning log on each retry (lines 639–647) with exception snippet (180 char limit), attempt count, and sleep duration.
- **Timeout:** Forwarded `timeout_sec` to `job.wait()` (line 619), respecting the documented AIP `queue="2h"` recommendation (line 540 comment).

**Exception handling:** Lines 633–650 distinguish transient (retry) from permanent (re-raise). Non-transient exceptions exit immediately.

---

### 2. TAP Sync Path: Zero Retry Logic (Lines 183–205)

**Status:** ✗ **Missing**

`run_sync()` is a single-shot call to `service.search()` with no retry on transient failures:

```python
# line 204–205
result = service.search(adql, **kwargs)
return result.to_table()
```

**Risk:** Any transient TAP backend glitch (PooledConnection drop, 502 gateway hiccup) fails the entire batch. With SYNC_ROW_THRESHOLD = 5k rows (line 54–58), sync queries are bounded to smaller datasets, but they still occur in production (`mode="sync"` is used explicitly in test_batched_fetch_df_writes_checkpoints_atomically, line 320).

**Recommended pattern:** Wrap with `@retry(retry=retry_if_exception_type(TRANSIENT_ERRORS), stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=10))` or inline retry loop matching async path.

---

### 3. Downloads: Missing Timeout Scaling and Retries (Lines 50–113)

**Status:** ✗ **Critical gaps**

The `download()` function has two resilience failures:

#### 3a. Hard-Coded 60s Timeout (Line 37, 90)

```python
DEFAULT_TIMEOUT_SEC = 60
# ... line 90:
with requests.get(url, stream=True, timeout=timeout_sec) as r:
```

**Problem:** 60 seconds is insufficient for multi-gigabyte files over slow connections. FITS summaries (data_acquisition.md §14.2 design rationale) can be 100–500 MB. A 500 MB file at 10 MB/s requires 50 s just for transfer, plus read latency. A 60s timeout will fail on typical network conditions.

**Impact:** Large APOGEE DR19 FITS files are at risk (e.g., astraAllStarASPCAP-0.6.0.fits.gz, ~1.5 GB). A timeout mid-stream leaves a `.part` file (line 103 cleanup), but the calling code has no visibility into transient network vs genuine permanent failure.

#### 3b. No Retry on Network Transients (Lines 90–104)

```python
try:
    with requests.get(url, stream=True, timeout=timeout_sec) as r:
        r.raise_for_status()
        # ... stream to disk
except BaseException:
    tmp.unlink(missing_ok=True)
    raise
```

**Problem:** `ConnectionError`, `ReadTimeout`, and HTTP 5xx (502/503/504) are transient but not retried. A single transient network glitch aborts the entire download without any backoff. The broad `except BaseException` mask masks transient from permanent (404, 403, etc.).

**Observed impact:** Multi-hour download ingestions (per checkpoint design) fail mid-stream on any transient, forcing manual restart from scratch or manual checkpoint recovery.

---

### 4. Lack of Rate-Limiting Awareness

**Status:** Note (not critical in current scope)

Neither module proactively handles rate-limit backoff. `batched_upload_fetch_df()` logs transient errors but does not distinguish HTTP 429 (rate-limited) from 504 (gateway timeout) — both are retried equally. For VizieR TAP (VIZIER_TAP_URL, line 52) which has stricter polite-use limits, this could cause rapid retry-storm if client quota is exhausted.

**Mitigation:** `_TRANSIENT_ERROR_MARKERS` does not include "429" explicitly, but it may be caught by pyvo's error message string. No explicit Retry-After header parsing.

---

### 5. Circuit Breaker: Absent

**Status:** Not implemented

Neither module implements a circuit breaker for repeated failures. If a TAP backend is down for 10 minutes, `batched_upload_fetch_df()` will burn 6 retries per batch × N batches, each waiting `4 * 2^5 = 128 s` before the final timeout. For 100 batches, this is hours of wasted compute.

**Recommendation:** Implement a simple circuit breaker that trip-wires (fast-fail) after 3+ consecutive batch failures to different services (ESA vs AIP failover), with a reset timeout (5 min cold start).

---

### 6. Exception Masking in Downloads

**Status:** ✗ Design smell

Line 102: `except BaseException:` catches `KeyboardInterrupt`, `SystemExit`, and memory errors. It will attempt cleanup, but re-raise. This is correct for atomicity, but the bare `BaseException` hides the distinction between transient (retry) and permanent (fail). Callers cannot distinguish a 404 (invalid URL, don't retry) from a 502 (backend hiccup, retry).

**Better pattern:** Separate exception types:
```python
except (ConnectionError, TimeoutError, requests.RequestException) as e:
    # transient
except requests.HTTPError as e:
    if e.response.status_code >= 500 or e.response.status_code == 429:
        # transient
except BaseException:
    # permanent — cleanup and re-raise
```

---

## Summary by Module

| Module | Sync Retry | Async Retry | HTTP Timeout | Rate Limit | Circuit Breaker | Notes |
|--------|-----------|-----------|--------------|-----------|----------------|-------|
| **tap.py** | ✗ | ✓ | N/A (async) | N/A | ✗ | Async is solid; sync path is dark. No 504 queue remediation without explicit `queue=` param. |
| **downloads.py** | ✗ | N/A | ✗ (60s fixed) | ✗ | ✗ | Single-shot, no backoff. File-size-aware timeout needed. |

---

## Detailed Recommendations

### Immediate (P0 — blocks production ingestion)

1. **Downloads retry wrapper (downloads.py, line 50):** Add `@retry(retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS), stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=2, max=60))` around the `requests.get()` block. Set `timeout_sec` as a function of expected file size (e.g., 10 + size_mb / 5 seconds).

2. **Async timeout tuning (tap.py, line 263):** Document that `timeout_sec` must exceed the server-side queue duration. For AIP `queue="2h"`, use `timeout_sec=8000` (2h + 10 min margin), not the default 3600. Add a warning log if `timeout_sec < 4000` and `queue is not None and "2h" in queue`.

### Near-term (P1 — improves resilience)

3. **TAP sync retry (tap.py, line 183):** Wrap `run_sync()` with same `@retry` decorator as async, bounded to 3 attempts (sync queries are smaller, more likely to succeed on retry).

4. **Exception classification (downloads.py, line 90):** Distinguish transient (`ConnectionError`, `TimeoutError`) from permanent (`requests.HTTPError` with 4xx status). Log and re-raise permanent errors without cleanup.

5. **File-size-aware timeout (downloads.py):** Accept `content-length` header, estimate transfer time (assume 1 MB/s baseline), and set timeout to `max(60, content_length_bytes / 1e6 + 30)` seconds.

### Future (P2 — architectural)

6. **Circuit breaker (tap.py, ~line 590):** Implement a per-service circuit breaker in `batched_upload_fetch_df()`. After 3 consecutive batch failures (across all retries), fast-fail for 5 minutes (log every attempt, emit metrics).

7. **Rate-limit backoff (tap.py, line 636):** If `_is_transient_tap_error()` detects HTTP 429 in the error message, increase base delay to 30 s and cap retries at 3 (polite backoff, avoid amplification).

---

## Code References

- **tap.py lines 609–650:** `batched_upload_fetch_df()` retry loop — exemplary exponential backoff.
- **tap.py lines 69–95:** `_TRANSIENT_ERROR_MARKERS` — comprehensive list of known transient error patterns.
- **tap.py lines 183–205:** `run_sync()` — lacks retry, single point of failure for small queries.
- **downloads.py lines 50–113:** `download()` — single-shot HTTP, no retry, 60s hard timeout.
- **downloads.py line 102:** `except BaseException:` — overly broad exception catching.

---

## Testing Notes

- **tap.py** has comprehensive offline tests (test_tap.py) for batching and placeholder validation, but no network smoke tests for retry behavior. Add a mock-based retry test with simulated transient failures.
- **downloads.py** has offline tests (test_downloads.py) for atomic writes and checksum validation, but no timeout or retry coverage. Add tests for timeout exceeded and transient HTTP errors.

---

**Verdict:** The async TAP path is production-ready for long-running enrichment jobs. The sync path and downloads path require hardening before production use on large datasets.
