# BJ21 photogeometric distances — Andrae+2023 vetted RGB

*Generated: 2026-04-19 22:47:13Z — script `scripts/fetch_bailerjones_andrae_rgb.py`*

Thread-3 prerequisite for the volume-limited arm of Pipeline 2's feature basis.

## 1. Strategy

- Primary strategy: **TAP UPLOAD** (VOTable join against `TAP_UPLOAD.sidlist`).
  Validated against GAVO live service with a 10 k smoke test before the run.
- Actual strategy used: **`upload`** (no fallback needed)
  > UPLOAD keeps request body small and sidesteps the ~100 KB IN-list ceiling that would otherwise push 10k-ID lists into HTTP-header territory on some TAP gateways.

## 2. Source_id counts

- Andrae+2023 total rows: **10,483,688**
- Andrae+2023 unique source_ids: **10,483,688**
- Already-on-disk BJ21 reused (Stream 1 + Stream 3 union ∩ Andrae): **410,628**
- Newly fetched from GAVO (delta): **10,073,060**
- Final combined rows in output: **10,483,688**
- BJ21-missing source_ids (no photogeometric solution on GAVO): **0** (0.000% — BJ21 coverage is ~99.9% of Gaia DR3; a small residual is expected and not a halt condition)

## 3. Wall-clock

- Total: **7532 s** (2.09 h)
- Batches completed: **1008**
- Mean chunk wall-clock: **7.4 s**
- Median chunk wall-clock: **6.8 s**

## 4. Retries + failures

- Total retry count (across all batches): **0**
- No failures; all batches completed on first pass.

## 5. Distance distribution (`r_med_photogeo`, pc)

- n finite: **10,482,529**
- min: 27.33540916442871
- Q25: 1978.0128173828125
- median: 2972.13818359375
- Q75: 4232.36328125
- max: 22587.54296875

- Count with `r_med_photogeo ≤ 2500` pc: **4,020,951** (38.36%)

  > Phase 3 target (revised under 10 GB budget): > **250,000** volume-limited stars.
  > **PASSED**

## 6. Storage diff

- `data/` footprint before: **4.65 GB**
- `data/` footprint after:  **5.06 GB**
- Final Parquet size: **184.6 MB**
- Per-chunk checkpoints consolidated + removed: **False**

## 7. Anomalies

- None.
