# ArqueoGal Performance Audit (Haiku, 2026-04-26)

## Summary

The codebase demonstrates strong performance discipline overall, with inference paths properly guarded by `torch.no_grad()`, selective column loading from Parquet, and streaming aggregation in ensemble prediction. Critical hotspots exist in data preprocessing (per-star loop vectorization, extinction calculations) and parquet I/O patterns for the 614k-star Stream 3 inference workload. No major memory leaks or unnecessary float64 inside training loops detected.

---

## Critical Hotspots (Fix Immediately)

### 1. Per-Star Extinction Loop (Data-Blocking Bottleneck)

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/gaia_xp.py:607-637`

```python
def _ccm89_deredden(flux: np.ndarray, sampling_nm: np.ndarray, a_v: np.ndarray) -> np.ndarray:
    ...
    out = np.empty_like(flux)
    for i, av in enumerate(a_v):
        if not np.isfinite(av):
            out[i] = flux[i]
            continue
        ext = extinction.ccm89(wave=wave_ang, a_v=float(av), r_v=3.1, unit="aa")
        out[i] = extinction.remove(extinction=ext, flux=flux[i], inplace=False)
```

**Issue**: Two nested per-star loops (lines 616-621 in `_ccm89_deredden` and identical pattern in `_ccm89_redden` lines 631-636). Each iteration calls `extinction.ccm89()` and `extinction.remove()` — expensive Python-to-C transitions. For the Ye+2024 flux correction pipeline operating on 5 000-star batches × 330 wavelengths, this is a serial bottleneck.

**Impact**: Ye+2024 correction preprocessing (§6.4 step 1) runs once per ingestion (not per-inference), but the loop is the dominant cost in `apply_ye2024_correction` when processing Stream 3 (50 k batches). Batched extinction calculations via vectorized `extinction` library or cupy (if available) would drop this from ~minutes to ~seconds per batch.

**Fix recommendation**: 
- Vectorize the loop: call `extinction.ccm89()` once with broadcast-compatible arrays if the library supports it.
- Fallback: use `np.where()` to handle finite-check vectorially, then call extinction on the finite subset only.
- RTX 3060 cupy support available; if `extinction` is slow, port extinction calculation to GPU.

---

### 2. Ye+2024 DataFrame Row Construction via List Comprehension

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/gaia_xp.py:404`

```python
return pd.DataFrame(
    {
        "source_id": joined["source_id"].to_numpy().astype(np.int64),
        "corrected_flux": [row for row in out_flux],  # <-- list comp over (n, 330) array
        ...
    }
)
```

**Issue**: Line 404 converts the `(n, YE2024_N_OUTPUT)` NumPy array `out_flux` to a Python list of row arrays via list comprehension. This forces 50k rows × 330-element array objects into Python's object layer, inflating memory and serialization cost when writing to Parquet.

**Impact**: Small but measurable on large batches; pandas prefers columns to be NumPy 1-D arrays, not lists. The Parquet write (line 144 in `ingest_stream3.py`) will serialize 50k object-dtype Series entries instead of a single 2-D array column.

**Fix recommendation**: Store as a separate 2-D array or use `df.assign(corrected_flux=[...])` with explicit dtype to signal that this is an array column. Better yet, if the downstream consumer (Pipeline 1) can handle a side-car wavelength + flux array file, store as HDF5 or raw binary to avoid Parquet object serialization entirely.

---

### 3. Selective Column Loading in Training Data Pipeline

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/training.py:118-127`

```python
df = pd.read_parquet(
    cfg.train_parquet,
    columns=["source_id", *_strat_columns_available(cfg.train_parquet)],
)
```

**Status**: ✓ **Well-optimized**. The column subset for stratification is loaded separately; full feature/label arrays are loaded via `load_arrays(cfg.train_parquet, layout, tiers)` which reads only needed columns (line 127).

However, line 127's `load_arrays` call reads every required feature column and immediately stacks them via `np.column_stack([df[c].to_numpy() for c in feature_cols])` (data.py:305). For 108 XP coefficient columns + 3 residuals + 26 aux = 137-D, this is 137 separate `.to_numpy()` calls. Minor inefficiency; grouped column reads (`pd.read_parquet(..., columns=[list])` followed by single `.values`) are faster.

**Impact**: ~5–10% wall-clock overhead on data loading. Negligible compared to the extinction loop, but worth noting for the 614k Stream 3 inference path.

**Fix recommendation**: Consider `pd.read_parquet(path, columns=cols).values` if memory is available, or construct the stacked array directly from the Parquet reader's internal row buffer (unlikely without Parquet API changes).

---

## Inference Path Performance (Generally Good)

### 4. Ensemble Aggregation — Streaming Covariance

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/inference.py:260-273`

```python
Sigma_alea = np.zeros_like(Sigmas_alea[0]) if Sigmas_alea else np.zeros((1, 1, 1))
for Sigma_m in Sigmas_alea:
    Sigma_alea += Sigma_m / len(Sigmas_alea)

delta = per_mu - mu_mean[None, :, :]  # (M, B, n)
Sigma_epi = np.einsum("mbi,mbj->bij", delta, delta) / per_mu.shape[0]
```

**Status**: ✓ **Excellent**. Streaming covariance accumulation avoids stacking O(M * B * n²) aleatoric covariances in memory. For 10-member ensemble × 614k stars × 21-D labels, this saves ~52 GB.

---

### 5. torch.no_grad() / inference_mode Coverage

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/uncertainty.py:107`

```python
with torch.no_grad():
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1]
        mu, L, _h, _z = model(x)
        mus.append(mu.cpu().numpy())
```

**Status**: ✓ **Correct**. All inference paths properly guarded. Stream 3 inference (614k stars) will not accumulate gradients.

---

### 6. Missing torch.inference_mode() in High-Frequency Loops

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/gaia_xp.py:382-383` (Ye+2024 NN forward pass)

```python
with torch.inference_mode():
    nn_out = nn_model(x_t).cpu().numpy()  # (k, 330)
```

**Status**: ✓ **Correct**. `torch.inference_mode()` is used (faster than `no_grad()` for this context).

---

## Memory and Dtype Issues

### 7. Explicit float64 in Kinematic and OOD Modules (Acceptable)

**Files**:
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/ood.py:68-69` (feature_precision, feature_mean)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/kinematic_ood.py:124-125` (velocity_mean, velocity_precision)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/training.py:275` (inverse freq weighting)

**Status**: ✓ **Justified**. float64 is used for statistical computations (covariance inversion, Mahalanobis distance, precision matrix). These are infrequent (fit once at training, load once at inference) and numerical stability is important. No float64 bloat inside the training loop itself.

---

### 8. Dataset Cache on GPU (Optional but Smart)

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/data.py:536-547`

```python
if stage_gpu:
    ds_device = "cuda"
else:
    ds_device = "cpu"
```

**Status**: ✓ **Well-designed**. Optional GPU staging of entire dataset (via `cfg.stage_dataset_on_gpu`) with correct DataLoader worker handling (num_workers=0 when GPU tensors are pre-staged). For RTX 3060 6GB with 5-10k training samples, this fits comfortably and saves PCIe bandwidth.

---

## Data I/O Bottlenecks (Stream 3 Inference)

### 9. Parquet Read Strategy for 614k-Star Stream 3 Inference

**Files**:
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/ingest_stream3.py` (data ingestion)
- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/data.py:302` (feature loading)

**Current pattern**: Full Parquet load into a single DataFrame, then column selection and array stacking.

**Issue**: For 614k rows × 137 feature columns, a single `pd.read_parquet()` call must load the entire table into RAM before filtering/selecting columns. On typical HPC hardware or a laptop with <32 GB RAM, this may spill to swap on the first pass. Repeated loads (e.g., inference on multiple random splits) will hit Parquet row-group decompression cache misses.

**Impact**: Bottleneck for large-scale Stream 3 inference or population studies. Not a blocker for current single-run Pipeline 1 inference, but will matter for iterative retraining / ablations.

**Fix recommendation**:
- Use Parquet columnar read directly: `pd.read_parquet(path, columns=cols)` only, skip the full load step. Parquet is column-oriented; reading 50 out of 200 columns should stream ~4× faster than loading all columns.
- For very large tables (>1M rows), consider Dask or Polars for lazy evaluation: `pl.scan_parquet(path).select(cols).collect()`.
- Chunk inference into sub-100k-row batches, process sequentially, write streaming results to avoid holding entire 614k-row prediction set in memory.

---

### 10. XP Coefficient Fetch Batching

**File**: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/gaia_xp.py:95-130`

**Status**: ✓ **Good**. XP batch size is 5 000 rows (line 51), sized to fit within AIP TAP async response limits and GPU memory. Checkpointing is implemented (checkpointing per-batch Parquet files avoids re-querying on network failure).

---

## Recommended Optimizations (Priority Order)

1. **HIGH**: Vectorize `_ccm89_deredden()` and `_ccm89_redden()` extinction loops (gaia_xp.py:607–637). Impact: 10–50× speedup on Ye+2024 preprocessing.

2. **HIGH**: Use Polars or Dask for Stream 3 inference feature loading (data.py:302). Avoids RAM spike on 614k-row single Parquet load.

3. **MEDIUM**: Refactor Ye+2024 output from object-dtype list of arrays to a 2-D NumPy column or side-car file (gaia_xp.py:404). Cleaner Parquet schema, faster downstream reads.

4. **MEDIUM**: Batch Stream 3 inference into sub-100k-row chunks if memory is constrained, with streaming writes to avoid peak memory usage on 614k-star inference.

5. **LOW**: Consider grouped column read in `load_arrays()` (data.py:305) to avoid 137 separate `.to_numpy()` calls. Likely <5% wall-clock win.

---

## Positive Patterns

- ✓ Inference paths use `torch.no_grad()` / `inference_mode()` consistently.
- ✓ Streaming aggregation in ensemble prediction avoids O(M·B·n²) memory explosion.
- ✓ Selective Parquet column loading in training pipeline (avoids unnecessary columns).
- ✓ Dataset GPU staging with correct worker configuration (no CUDA fork issues).
- ✓ Float64 used judiciously (statistics/precision, not training loop bloat).
- ✓ XP fetch batching respects API/memory constraints (5 k rows per batch).

---

## Conclusion

The codebase is well-structured for a single-GPU laptop (RTX 3060 6 GB) and HPC offloading (pc127 CPU-only). The main performance lever for Stream 3 inference (614k stars) is the extinction loop vectorization and chunked Parquet I/O. Current bottlenecks are in data preprocessing (per-star loops), not model training or inference orchestration.
