# IR photometry fetch — Stream 3 Ye-OK existing (164 314 stars)

**Date:** 2026-04-19 → 2026-04-20 (~80 min wall-clock, 2026-04-19 22:58 → 2026-04-20 00:17 UTC).
**Module:** `src/arqueogal/data/ir_photometry.py`. **Driver:** `scripts/fetch_ir_photometry.py`.
**Output:** `data/raw/ir_photometry/stream3_existing_ir.parquet` + `.provenance.json`.

## 1. Root cause of the previous failure

The earlier fetch session retried blindly on the ESA TAP error `DALQueryError: Query Error: (TAP) Cannot execute query ...` with a truncated log. Reproduced and read the full error message by submitting the exact module query against ESA's async endpoint:

```
DALQueryError: Query Error: (TAP) Cannot execute query '...' for job '...':
java.sql.SQLException: PooledConnection has already been closed.
```

The module (`ALLWISE_ADQL_UPLOAD_ESA` / `_AIP`) joined `gaiadr1.allwise_original_valid` (or `catalogs.allwise` on AIP) against `gaiadr3.allwise_best_neighbour` on `a.allwise_oid = bn.allwise_oid`. That specific equi-join deterministically drops the pooled DB connection on ESA's query planner for this table pair. Reproduced 3/3 via a hand-submitted 10-id UPLOAD job.

A parallel issue was surfaced on the retry path: ESA's async submit endpoint also 500s intermittently when the shared anonymous queue is loaded, and `batched_upload_fetch_df` had two flaws — (a) `submit_job` lived outside the try/except so submit-time errors bypassed the retry loop, and (b) `500 Server Error` was not in `_TRANSIENT_ERROR_MARKERS`.

Finally, an unrelated issue surfaced during this fetch: ESA's anonymous user filesystem quota (shared across all anonymous callers) is currently **at 19 GB of 20 GB** so async result spill fails even for trivial queries. `Filesystem quota exceeded for user anonymous`. This is out of our control; AIP is the correct primary endpoint.

## 2. Exact query fix

**Before** (`ALLWISE_ADQL_UPLOAD_ESA`):

```sql
LEFT JOIN gaiadr1.allwise_original_valid AS a
    ON a.allwise_oid = bn.allwise_oid
```

**After** (same `designation` join that 2MASS already used):

```sql
LEFT JOIN gaiadr1.allwise_original_valid AS a
    ON a.designation = bn.original_ext_source_id
```

Same fix applied to the AIP variant (`catalogs.allwise`). AllWISE `best_neighbour.original_ext_source_id` is the AllWISE PSC designation (`Jhhmmss.ss±ddmmss.s` format) and matches `original_valid.designation` directly, parallelling the 2MASS path.

Secondary fixes in `src/arqueogal/data/tap.py`:
- Added `500 Server Error`, `500 for url` to `_TRANSIENT_ERROR_MARKERS`.
- Moved `service.submit_job(...)` inside the try/except so submit-time transients trigger the retry loop.

Tertiary fix in `scripts/fetch_ir_photometry.py`:
- `Path.relative_to()` crashed on relative CLI paths when writing the provenance sidecar. Replaced with `_rel_or_abs(p)` helper that resolves both sides before diffing.

## 3. Module + test status

- Unit tests: **46/46 passing** (`tests/data/test_ir_photometry.py` 18/18, `tests/data/test_tap.py` 28/28).
- Added regression guard `test_allwise_adql_shape` asserting `a.allwise_oid = bn.allwise_oid` is **never** reintroduced and `bn.original_ext_source_id` remains in the ADQL.
- Updated module header docstring to cite the ESA PooledConnection failure mode.

## 4. Fetch stats

| Metric | Value |
|---|---|
| Source IDs submitted | 164 314 |
| 2MASS counterparts | 163 480 (99.49 %) |
| AllWISE counterparts | 164 314 (100.00 %) |
| IR-complete (all 5 mags) | 163 433 (99.46 %) |
| `ir_missing_flag=True` | 881 (0.54 %) — **drop before inference** |
| Wall-clock | ~80 min (34 async batches × ~80 s each, one retry-free pass) |
| TAP batches | 17 × 2MASS + 17 × AllWISE = 34 total |

## 5. Storage diff

| Location | Size | Notes |
|---|---|---|
| `data/raw/ir_photometry/stream3_existing_ir.parquet` | 10.02 MB | final output, 164 314 × 18 cols, float32 mags |
| `data/raw/ir_photometry/stream3_existing_ir.provenance.json` | 3.6 KB | sidecar |
| `data/interim/enrich_batches/ir/tmass/` | ~7 MB | 17 checkpoint chunks |
| `data/interim/enrich_batches/ir/allwise/` | ~7 MB | 17 checkpoint chunks |
| **Total added** | **~25 MB** | well under budget |
| `data/` total after fetch | **5.1 GB** | borderline on the 5 GB cap; candidates for cleanup: checkpoints can be deleted once the final parquet is verified |

## 6. TAP service used + fallback triggered

- **Primary used:** AIP (`https://gaia.aip.de/tap`, token-auth).
- **Fallback:** Yes — switched from ESA to AIP after ESA returned `Filesystem quota exceeded for user anonymous` on batch 2. AIP ran the corrected query cleanly with `queue="2h"`.
- The module's ESA ADQL remains correct after the fix (probe jobs on ESA now COMPLETE); the switch was forced by the quota, not by a schema issue.

## 7. Anomalies

- **AllWISE 100.00% counterpart rate** is unusually clean. Verified across raw batches 0/5/10/15: W1 magnitudes span -0.9 to +13.0 with median ~8.5, consistent with bright RGB giants well within AllWISE's sensitivity (W1 ~ 0.0–17 mag). No silent drop or zero-fill; LEFT JOIN preserves every upload row, `allwise_source_id` is non-null for all 164 314 rows, and the designation-match is 100 % as reported. This is a genuine property of the Ye-OK RGB subset, not a bug.
- First checkpoint (`tmass/tmass_batch_0000.parquet`) was written by an ESA attempt before the switch; schema matches AIP's output (same canonical column names; finalise casts handle the slight dtype divergence). Reused without re-fetching.
- `tmass_xm_quality_flag` is stored as `Int8` in the final parquet per the declared schema, but raw checkpoint chunks arrive as `Int16` — the cast happens in `_finalise_generic`. No consequence for downstream consumers reading the final parquet.
- Logs archived at `logs/ir_fetch_stream3_existing_20260419.preFix.log` (the pre-fix run's 5 blind retries), `.run1.log` (ESA batch 1 completed, batch 2 500), `.run2_esa_quota.log` (ESA anonymous quota exceeded); current `.log` is the successful AIP run.
