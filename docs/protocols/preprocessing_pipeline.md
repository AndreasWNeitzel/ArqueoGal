# Pipeline 1 Unified Preprocessing Protocol

**Version:** 1.0  
**Last Updated:** 2026-04-29  
**Implemented by:** `src/arqueogal/data/preprocessing.apply_pipeline1_preprocessing()`

## Overview

The unified preprocessing pipeline is the single source of truth for all three streams (Stream 1 / APOGEE×Gaia training, Stream 2 / Hon+2021 TESS asteroseismic giants, Stream 3 / Andrae+2023 application pool). All streams call `apply_pipeline1_preprocessing(df, mode)` with stream-specific input data and receive preprocessing-ready output suitable for training or inference.

The contract is deterministic: train and inference modes apply byte-identical transforms. The pipeline executes eight steps in fixed order, each conditional on the presence of required columns. The fingerprint check (step 5) enforces that the Hermite basis matches the frozen v1 statistics; this gate prevents silent schema drift.

## Steps 1-8 in Order

### Step 1: Lindegren+2021 Parallax Zero-Point

**Function:** `gaia_corrections.apply_parallax_zpt()`  
**Reference:** Lindegren et al., A&A 649, A4 (2021)

Applies the Gaia DR3 parallax zero-point correction from Lindegren et al. (2021). The correction is +18 micr as on average; per-magnitude and per-ecliptic-latitude color corrections are included. The corrected parallax `parallax_corr` replaces the raw Gaia column.

**Columns read:** `parallax`, `parallax_error`, `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`, `ecliptic_latitude`  
**Columns written:** `parallax_corr`, `parallax_error` (retains original)  
**Conditional skip logic:** If `parallax_corr` is already present, skip (caller has pre-applied).

### Step 2: Riello+2021 G-Band Correction

**Function:** `gaia_corrections.apply_g_mag_correction()`  
**Reference:** Riello et al., A&A 649, A3 (2021)

Applies the cubic G-magnitude correction from Riello et al. (2021) to account for small systematic biases in Gaia DR3 photometry at specific magnitude and color ranges. Replaces `phot_g_mean_mag` with the corrected value `phot_g_mean_mag_corr`.

**Columns read:** `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`  
**Columns written:** `phot_g_mean_mag_corr`  
**Conditional skip logic:** If `phot_g_mean_mag_corr` is present, skip.

### Step 3: XP Fetch and Ye+2024 NN Flux Correction

**Function:** `gaia_xp.fetch_xp_coefficients()` then `gaia_xp.apply_ye2024_correction()`  
**Reference:** Ye et al., arXiv:2411.19105 (2024)

Fetches Gaia XP spectral coefficients (55 BP + 55 RP Hermite basis functions) via AIP TAP. Applies Ye et al. (2024) neural-network flux-correction to re-normalize the spectrum and remove instrumental calibration biases. The corrected flux vector `corrected_flux` is the input to step 4.

**Columns read:** `source_id` (used for AIP TAP query)  
**Columns written:** `bp_coeffs`, `rp_coeffs`, `bp_coeff_errs`, `rp_coeff_errs`, `corrected_flux` (list of 110 floats)  
**Conditional skip logic:** If `skip_xp_fetch=True`, this step is entirely skipped; caller has pre-fetched and corrected. Callers invoking with `skip_xp_fetch=True` must provide `corrected_flux` already present.

**Batch processing:** XP fetch issues AIP TAP queries in batches of `xp_batch_size` (default 10,000 stars) to avoid timeout. Each batch is validated for non-null `corrected_flux` before proceeding to step 4.

### Step 4: Hermite Re-Projection

**Function:** `gaia_xp.reproject_ye_to_hermite()`

Re-projects the Ye+2024-corrected flux to Hermite basis coefficients and applies log-transformation and z-scoring using *frozen* statistics from the v1 training set. The frozen statistics (mean and std per coefficient) are loaded from the v1 checkpoint and are immutable across all subsequent training and inference runs.

**Columns read:** `corrected_flux`, `bp_coeff_errs`, `rp_coeff_errs`  
**Columns written:** `bp_coeffs_norm`, `rp_coeffs_norm`, `bp_coeff_errs_norm`, `rp_coeff_errs_norm`, `bp_c0_z`, `rp_c0_z`  
**Notes:** The normalized coefficients are dimensionless; z-scores are clipped to ±5 to avoid extreme outliers. The scalar `bp_c0_z` and `rp_c0_z` are separate columns for convenience in downstream analyses.

### Step 5: Frozen v1 Z-Score with Fingerprint Verification

**Function:** `frozen_stats.verify_basis_fingerprint()` then `frozen_stats.apply_frozen_zscore()`  
**Reference:** research_brief.md §3.3.2

Verifies that the Hermite basis fingerprint (SHA256 of the frozen mean/std arrays) matches the contract value `0d34b565...` (as of v1). This check enforces that no code path has drifted the basis. Aborts if the fingerprint does not match.

The z-scoring is then applied: each coefficient is standardized using the frozen v1 mean and std. These statistics are loaded from the frozen-stats JSON sidecar and are locked across all runs.

**Columns read:** `bp_coeffs_norm`, `rp_coeffs_norm` (from step 4)  
**Columns written:** (in-place modification of the above)  
**Conditional skip logic:** Never skipped; this is a mandatory gate.

**Fingerprint enforcement:** If the fingerprint does not match, the pipeline raises `ValueError` and halts. Callers must not suppress this check.

### Step 6: Yuan+2013 and CCM89 Broadband Dereddening

**Function:** `extinction.apply_extinction_corrections()`  
**Reference:** Yuan et al., MNRAS 430, 2188 (2013); Cardelli, Clayton, and Mathis, ApJ 345, 245 (1989)

Applies hybrid dereddening using Yuan+2013 extinction coefficients (J, H, K, W1, W2 bands) and CCM89 extinction law (R_V = 3.1) for Gaia G, BP, RP bands. The hybrid approach combines infrared and optical constraints, leveraging broadband photometry to refine the dust-map A_V estimate. Dereddened magnitudes (G_0, BP_0, RP_0) are emitted.

**Columns read:** `phot_g_mean_mag_corr`, `phot_bp_mean_mag`, `phot_rp_mean_mag`, `j_mag`, `h_mag`, `ks_mag`, `w1_mag`, `w2_mag`, `av_los`, `av_los_source`, `av_nbhd_median`  
**Columns written:** `g_mag_dereddened`, `bp_mag_dereddened`, `rp_mag_dereddened`  
**Conditional skip logic:** If `apply_extinction=False` (default is True) or if any required IR column is missing, step 6 is skipped. Callers can disable extinction correction for streams lacking dust-map data; the downstream model will use observed (reddened) photometry.

**Trust flags:** Distance-based selection of dust map (Edenhofer+2024 for d < 1.25 kpc; Lallement+2022 for 1.25–3 kpc; SFD for d > 3 kpc) is recorded in `av_los_source` (codes 0, 1, 2, -1=missing).

### Step 7: Distance Trust Flags and Av Source Flags

**Function:** (inline in `apply_pipeline1_preprocessing`)

Adds four boolean columns indicating data quality and source reliability.

**Columns written:**
- `dist_gaia_reliable` (bool): True if Gaia parallax fractional error < 0.2.
- `dist_photogeometric_fallback` (bool): True if Bailer-Jones distance was used (fallback when Gaia parallax SNR is low).
- `av_source_code` (int8): Categorical flag, {0=Edenhofer, 1=Lallement, 2=SFD, -1=missing}.

**Purpose:** Allows downstream consumers to filter by source reliability and to diagnose distance and extinction prior sensitivity.

### Step 8: Auxiliary-Feature Standardisation

**Function:** (inline in `apply_pipeline1_preprocessing`)

Ensures all auxiliary features (parallax, photometry, extinction, position) are present and have expected dtypes. If any auxiliary column is missing, it is filled with NaN and a warning is logged. The set of mandatory auxiliary columns is defined by `xp_abundances.main.DEFAULT_AUX_COLS`.

**Columns checked:** `parallax_corr`, `parallax_error`, `phot_g_mean_mag_corr`, `phot_bp_mean_mag`, `phot_rp_mean_mag`, `ra`, `dec`, `av_los`, `av_los_source`, `av_nbhd_median`, plus distance triple (r_med_photogeo, r_lo_photogeo, r_hi_photogeo).

**Conditional skip logic:** If a column is missing, it is created with NaN or a default sentinel value. Inference-time missing auxiliaries are tolerated (they feed into the model as NaN, triggering the `aux_missing_any` flag downstream); training-time missing auxiliaries trigger a warning and are imputed with the training-set median.

## Train vs. Inference Parity

Both modes apply the eight steps identically. The `mode` parameter is used only for logging context and sidecar provenance. A caller can invoke `apply_pipeline1_preprocessing(df, mode="train")` and `apply_pipeline1_preprocessing(df, mode="inference")` on the same input `df` and receive byte-identical output.

**Verification:** Unit tests in `tests/data/test_preprocessing.py` assert parity across all eight steps by comparing the output of train and inference modes on the same input fixture. The test suite exercises steps 1-7 end-to-end and confirms that the frozen-stats fingerprint check fires correctly.

## Verification Diagram

To convince yourself of train/inference parity and fingerprint correctness, check the following:

1. **Frozen-stats baseline fingerprint:** The value `0d34b565...` is stored in `src/arqueogal/data/frozen_stats.py:FROZEN_STATS_FINGERPRINT`. This is the commit-time SHA256 of the v1 Hermite basis mean/std arrays. Any attempt to change the basis or the z-scoring parameters triggers an abort.

2. **Unit test matrix:** Run `uv run pytest tests/data/test_preprocessing.py -v` to verify all steps. The test `test_apply_pipeline1_preprocessing_train_inference_parity` checks byte-equality for steps 1-8 on the same input.

3. **Extinction-law fingerprint:** The CCM89 + Yuan+2013 recipe has its own fingerprint in `src/arqueogal/data/extinction.py:DEFAULT_EXTINCTION_LAW.fingerprint()`. This ensures the extinction recipe is immutable across releases.

4. **MappingProxyType sentinel:** The `MESZAROS2025_ALLOWED_KEYS` read-only mapping in `src/arqueogal/xp_abundances/main/data.py` lists the exact 21 elements (Teff, log g, [M/H], + 18 abundances) that the v1.1 model is permitted to predict. Callers attempting to train a model on unlisted elements will fail at data-loading time.

## References

- **Cardelli, Clayton, and Mathis (1989):** CCM89 extinction law. ApJ 345, 245. [ADS](https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/)
- **Lindegren et al. (2021):** Gaia DR3 parallax zero-point. A&A 649, A4. [doi:10.1051/0004-6361/202039653](https://doi.org/10.1051/0004-6361/202039653)
- **Riello et al. (2021):** Gaia DR3 photometric calibration. A&A 649, A3. [doi:10.1051/0004-6361/202039587](https://doi.org/10.1051/0004-6361/202039587)
- **Yuan et al. (2013):** Extinction coefficients J, H, K, W1, W2. MNRAS 430, 2188. [doi:10.1093/mnras/sts684](https://doi.org/10.1093/mnras/sts684)
- **Ye et al. (2024):** XP neural-network flux correction. arXiv:2411.19105. [arXiv](https://arxiv.org/abs/2411.19105)
- **Edenhofer et al. (2024):** 3D dust map (d < 1.25 kpc). arXiv:2308.01295. [arXiv](https://arxiv.org/abs/2308.01295)
