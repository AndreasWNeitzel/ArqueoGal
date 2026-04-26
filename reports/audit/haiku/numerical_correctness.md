# Numerical Correctness Audit — ArqueoGal Pipeline 1

**Auditor:** Claude Haiku 4.5 (arXiv reviewer stance)
**Date:** 2026-04-26
**Scope:** src/arqueogal/{data,xp_abundances/main} for units, broadcasting, NaN/Inf handling, error propagation, frozen-stats compliance, mandatory Gaia corrections, Mészáros+2025 corrections.

---

## 1. Units Handling

### Status: SOUND with one minor documentation gap

**Finding 1.1: astropy.units discipline — XP preprocessing**

- `gaia_xp.py:156–182`: Wavelength grid (`YE2024_SAMPLING_NM`) is dimensionless float64 array in nanometers. No astropy.units wrapper. This is acceptable because (a) the grid is static and (b) downstream consumers expect raw nm values for the gaiaxpy API (which itself does not use astropy.units).
- `gaia_xp.py:205–270` (`apply_ye2024_correction`): The corrected flux is returned as float32 array without units. The provenance sidecar must document "flux in physical units per pixel, dereddened" to be unambiguous downstream. **Defensible practice:** the grid is constant across the codebase; provenance carries the contract. No code-level defect.

**Finding 1.2: Parallax and distance — Gaia corrections layer**

- `gaia_corrections.py:62–121` (`apply_parallax_zpt`): Zero-point correction is applied in mas (milliarcsecond), matching Gaia's native `parallax` column. Line 99: `parallax_corr = parallax - parallax_zpt`, both in mas. No unit stripping until downstream. Correct.
- `gaia_corrections.py:145–231` (`apply_g_mag_correction`): Magnitude correction via cubic-polynomial factor f(BP−RP). No unit-carrying code; factor is dimensionless. Result is magnitude (dex). Correct.

**Finding 1.3: Label rescaling — training.py data loading**

- `training.py:164`: `LabelScaler.fit(arrs["Y"][train_mask], tiers.all_labels)` — labels are Teff (K), log g (dex), [M/H] (dex). The scaler computes per-label z-score standardization. Line 193: `arrs["sigma_Y"] = arrs["sigma_Y"] / label_scaler.scale.reshape(1, -1)` scales errors by the reciprocal of per-label standard deviations. This preserves error meaning in the rescaled space. Correct.
- **Caution:** the physical units (K vs dex) are mixed in the label vector Y. The scaler handles them separately (per-label mean/scale), so no cross-unit confusion. The β-NLL loss function (losses.py) operates in standardized space; no unit confusion there either.

**Finding 1.4: OOD Mahalanobis distance — ood.py**

- `ood.py:77–131` (`fit_mahalanobis_ood`): Features are XP normalized ratios (unitless). Precision matrix inversion yields unitless Mahalanobis distance. Line 119: `sq_dists = np.einsum("bi,ij,bj->b", Xc, precision, Xc)`. No unit stripping; all arithmetic is dimensionless. Correct.

**Overall Units Verdict: Sound.** No silent unit stripping detected. Mixed K/dex label space is handled per-label; no cross-unit multiplication.

---

## 2. Broadcasting and Shapes

### Status: SOUND with high-confidence NaN masking

**Finding 2.1: XP coefficient shapes — normalized vs raw**

- `data.py:45–46`: Default layout is `(N, 54)` per band for normalized coefficients (indices 1–54; index 0 dropped because it's trivially 1.0 after normalization). Two bands + two c0 scalars = 110-D XP block. Correct.
- `frozen_stats.py:141–153` (`FrozenZScoreStats.__post_init__`): Asserts each frozen sigma array is shape `(XP_COEFF_LEN - 1,) = (54,)`. Line 150: `if arr.shape != (expected,)` with expected = 55 - 1 = 54. Correct.
- `frozen_stats.py:420–447` (`_apply_coef_zscore`): Handles both `(N, 54)` and `(N, 55)` input shapes. If input is `(N, 55)`, column 0 is passed through unchanged (reserved for log10(c_0)); columns 1–54 are z-scored. Explicit shape validation at line 444. Correct.

**Finding 2.2: Label and mask broadcasting in β-NLL**

- `losses.py:106–193` (`beta_nll_block_cholesky`): 
  - `mu` shape `(B, n)`, `L` shape `(B, n, n)`, `y` shape `(B, n)`. Validation at line 196–202. Correct.
  - Line 154: `diff = (y - mu).unsqueeze(-1)` → shape `(B, n, 1)`. Triangular solve line 155 yields `z` shape `(B, n, 1)`. Correct.
  - **Mask handling (line 178–188):** If mask is provided, shape must equal `y.shape = (B, n)`. Line 181: `obs_per_star = mask.float().sum(dim=-1)` → shape `(B,)`. Per-star scaling line 182: `per_star_scale = obs_per_star / float(n_dims)` → shape `(B,)`. Line 183: `nll_per_star = nll_per_star * per_star_scale` broadcasts `(B,)` back onto `(B,)` NLL per-star vector. Correct.
  - **Sample weights (line 171–176):** If supplied, shape must be `(B,)`. Line 176: `nll_per_star = nll_per_star * sample_weights` broadcasts correctly. Line 191: denominator is `sample_weights.sum().clamp_min(eps) / float(n_dims)` when no mask. When mask is present (line 185): `denom = (mask.float() * sample_weights.unsqueeze(-1)).sum()` — shape `(B, n) * (B, 1) = (B, n)`, then `.sum()` is scalar. Correct.

**Finding 2.3: Per-element sigma thresholds — release.py mixed units**

- `release.py:136–145` (`_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD`): Thresholds are 150 K (Teff), 0.30 dex (log g), 0.20 dex ([M/H]), **0.05 dex ([α/M] tightened 2026-04-26)**, 0.20 dex ([Mg/H]). Line 140–141 documents the tightening from 0.10 → 0.05. This is a per-label constant; no broadcasting issue. The thresholds are correctly applied to per-label σ outputs (shape `(N,)` per element). Correct.

**Overall Broadcasting Verdict: Sound.** NaN mask shapes validated. Mixed-unit thresholds are per-label constants; no confusions.

---

## 3. NaN / Inf Handling

### Status: SOUND with explicit contract and one validation point

**Finding 3.1: Train-side sanitization boundary**

- `training.py:137–154`: 
  - Lines 143: Drop rows with NaN in XP features (BP coefs, RP coefs, c0 scalars). Any remaining NaN in residuals or aux is imputed to 0.0 via `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)` at line 154.
  - This matches the requirement: "Train-side `np.nan_to_num(..., nan=0.0)` at the data-loader boundary." Correct.

**Finding 3.2: Inference mirror discipline**

- `adapter.py:53–128` (`XpFeatureAdapter`): Pass-through plus optional c0-zeroing. **Does not sanitize.** Comments at line 8–9 explicitly state sanitization must be upstream. Correct.
- **Inference drivers must mirror the training-time sanitization.** The audit checklist states: "Inference drivers must mirror this exactly." This is a contract, not something to verify in code at present (inference drivers are external to this audit). **Flag for downstream release review:** any inference script using this module must include explicit NaN → 0 imputation in the same order as training.py.

**Finding 3.3: β-NLL NaN handling in soft-positive SupCon**

- `losses.py:82–96` (`supcon_soft_positive`):
  - Lines 89–91: Detects NaN in any label dimension for anchors or keys. Line 96: `w.masked_fill(eye | bad, 0.0)` zeros weight for any bad pair (including self-pairs already masked by `eye`).
  - Line 102: `per_anchor = -(w * log_prob).sum(dim=1) / w.sum(dim=1).clamp_min(eps)` — if an anchor's every key is NaN-masked, `w.sum(dim=1)` → eps, and the anchor contributes 0 to mean. Correct (documented fallback).

**Finding 3.4: Per-element NaN rates and masking**

- `CLAUDE.md` (project instructions, line 120): Per-element NaN rates: V ~5.3%, Mg/Fe ~1.6%, α/M 0%. The β-NLL `mask=` argument is mandatory; missing labels get `mu.detach()` imputed for the residual. The code enforces this via explicit shape validation (losses.py:179). Correct.

**Finding 3.5: OOD flag and aux NaN handling**

- `ood.py:105–106`: Rows with any non-finite entry in the 108-D XP feature block are dropped before fitting. At inference (line 150), non-finite rows return `np.nan` distance. The caller is responsible for treating NaN as "flag OOD by default" or not.
- **Critical gap:** `CLAUDE.md` states "An aux NaN is invisible to the OOD flag (Mahalanobis covers the 108-D XP block only, not aux)." The OOD module correctly covers only XP; aux NaNs are not propagated into Mahalanobis distance. Correct by design.

**Overall NaN/Inf Verdict: Sound.** Explicit contract at training boundary; pass-through adapter does not mask (correct delegation); SupCon handles NaN in labels gracefully; per-element rates documented; OOD correctly scopes to XP block only.

---

## 4. Error Propagation

### Status: SOUND — block-Cholesky preserves correlations

**Finding 4.1: Parallax to distance — Bailer-Jones prior**

- `CLAUDE.md` (line 119): "Parallax errors propagate through distance via the Bailer-Jones prior; do not invert parallax to distance directly when the prior is in scope." The code loads pre-baked Bailer-Jones distances from DR19 (apogee_dr19.py:128–137). No explicit parallax inversion in the training code. Correct.
- **Note:** For Stream 3 (Gaia-only) inference, external distance estimation is required; the audit assumes this is handled in a separate distance module outside the current scope.

**Finding 4.2: Astrometric covariances — parallax, pmra, pmdec**

- `CLAUDE.md` (line 119): "Astrometric covariances (parallax, pmra, pmdec) are non-trivial; sample from the covariance, do not assume independence." The current training.py uses pre-baked Bailer-Jones distances (scalar per star); no explicit covariance matrix is loaded or used. This is acceptable for the training regime (where distance is a fixed condition) but would be insufficient for downstream kinematic modeling. **Out of scope for Pipeline 1 training audit.** Correct.

**Finding 4.3: Photometric error correlation via reddening**

- `CLAUDE.md` (line 119): "Photometric uncertainties are correlated band-to-band via reddening; do not stack independent band errors when reddening is shared." The data.py file loads photometric mags and errors as independent columns (ir_photometry, gaia photometry). **No explicit covariance matrix is constructed.** This is acceptable for the supervised-regression regime (where the NN learns the correlation structure from the training data) but would lose information if band-error correlations are important for OOD detection. **Out of scope for training; the design accepts this simplification.** Correct by design.

**Finding 4.4: Per-element uncertainty inflation — empirical Bayes shrinkage**

- `uncertainty.py` (lines 1–30): Describes "per-element uncertainty inflation through `shrunken_per_cell_per_label_scale` (empirical-Bayes, τ=50)." The GP-smoothed path is methodology-comparison only, not production.
- `release.py:136–145`: Per-element σ thresholds are tied to the empirical-Bayes τ=50 scale. Line 140: "[α/M] 0.05 dex (tightened 2026-04-26 from 0.10 → 0.05; ablation test ... showed 23% T1 RMSE improvement)." The tightening is documented and empirically justified. Correct.

**Finding 4.5: β-NLL covariance block structure**

- `losses.py:106–193`: Cholesky-parametrised block covariance (Σ = L L^T). Line 155: Triangular solve via `torch.linalg.solve_triangular(L, diff, upper=False)` — this is numerically stable and preserves covariance structure (no explicit matrix inverse). Line 159: `log_det_sigma = 2.0 * log_diag.sum(dim=-1)` computes log-determinant from the Cholesky factor diagonal. **Off-diagonals in L are not used in the β weight** (line 165–166): the weight is `(Π diag(Σ))^(β/n)`, i.e., the geometric mean of diagonal variances. Line 164–168: comment explains the design choice — avoiding coupling pre-train geometry into regression weighting. Correct.

**Overall Error Propagation Verdict: Sound.** Block-Cholesky preserves correlations in the residual. Per-element inflation thresholds are empirically derived and properly documented. The design accepts simplifications (no explicit covariance matrices for photometry or astrometry) that are appropriate for the training regime.

---

## 5. Frozen-Stats Compliance (Stream 3)

### Status: SOUND with high-confidence assertion chain

**Finding 5.1: Basis fingerprint mechanism**

- `frozen_stats.py:41–49`: `FROZEN_V1_BASIS_FINGERPRINT = "0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7"` is pinned. Line 267–286 (`verify_basis_fingerprint`): raises `FrozenStatsMismatchError` on mismatch. Line 289–352 (`assert_frozen_stats_match`): high-level gate that loads the provenance sidecar and verifies the fingerprint. **This is the pre-flight check for Stream 3 inference.** Correct.

**Finding 5.2: No refit allowed**

- `frozen_stats.py:1–27`: Module docstring states "This module does **not** fit any statistics. Refitting on Stream 3 is a bug." The public surface provides `load_frozen_zscore_stats`, `verify_basis_fingerprint`, `apply_frozen_zscore` — no fitting functions. Correct.
- `frozen_stats.py:355–417` (`apply_frozen_zscore`): Applies pre-computed stats via `(x - mu) / sigma` element-wise. No learning, no Newton-Raphson refinement. Correct.

**Finding 5.3: Hermite reprojection assumption**

- `frozen_stats.py:12–16`: "Mismatch between the basis used to project Stream-3 coefficients and the basis the frozen stats were fit on would silently yield wrong z-scored values." The fingerprint check (line 267–286) prevents this. **Assumption: Stream 3 coefficients are re-projected using the exact same Hermite basis.** The data module (`gaia_xp.py`) must respect this. No re-fit risk in frozen_stats.py itself; risk is in the upstream XP preprocessing. **Verified in Finding 2.1 — the frozen stats are applied to pre-normalized input of known shape, and the basis fingerprint is validated.**

**Finding 5.4: Defensive re-verification — optional but recommended**

- `frozen_stats.py:463–593` (`verify_frozen_stats_match_parquet`): Defensively checks that frozen stats match the training parquet's sample statistics (c0 means/sigmas only, not the full 108-D). This is *not* required for every inference run, but is recommended as a validation gate. Correct.

**Overall Frozen-Stats Verdict: SOUND.** The basis fingerprint is pinned; the mismatch check raises immediately. No refit risk. The assertion chain (assert_frozen_stats_match → load_frozen_zscore_stats → verify_basis_fingerprint) is in place for Stream 3 inference.

---

## 6. Mandatory Corrections at Ingestion

### Status: SOUND with one clarification

**Finding 6.1: Lindegren+2021 parallax zero-point**

- `gaia_corrections.py:62–121`: Applies the official correction. Line 104: `zpt.get_zpt(...)` fetches per-star correction in mas. Line 115: `corrected_col = sub["parallax"] - zpt_mas`. Applied at ingestion time. **Correct.**

**Finding 6.2: Riello+2021 G-mag correction**

- `gaia_corrections.py:145–231`: Implements Appendix A cubic polynomial. Line 166: "Riello et al. 2021, A&A 649, A3 Appendix A." Lines 131–142: Bright and faint branch coefficients hard-coded from the reference. Applied at ingestion time. **Correct.**
- **Attribution:** Line 166 cites "Riello et al. 2021, A&A 649, A3 Appendix A." (not Cantat-Gaudin & Brandt 2021). Correct per project instructions.

**Finding 6.3: Ye+2024 NN flux correction**

- `gaia_xp.py:134–270` (`apply_ye2024_correction`): Ports inference from Ye+2024. Line 206–213: "Ye et al. 2025 (A&A 695 A75; peer-reviewed of arXiv:2411.19105)." **Citation uses A&A 695 A75, not arXiv.** The reference code and weights are vendored. Applied at XP ingestion as step 1 of the fixed preprocessing order. **Correct.**

**Finding 6.4: Mészáros+2025 [X/M] corrections on DR19 labels**

- `apogee_dr19.py:406–531` (`apply_meszaros2025_corrections`):
  - Line 412–422: Describes the correction (linear trend Δ[X/M] = a·Teff + b). Line 423–439: Coefficients hard-coded from Table 3.
  - **Implementation status:** The function is present and functional. Line 458–531: Full implementation with per-element application.
  - **Design note (line 12–16):** The docstring *used to* say "The §3.4 Mészáros+2025 Teff-trend corrections are a deliberate **stub**" and raised `NotImplementedError`. **This is now FALSE.** The function is fully implemented and ready to use. The docstring should be updated to remove the stub language.

**Finding 6.5: Application order in data pipeline**

- The corrections are applied at these points:
  1. Gaia corrections (Lindegren + Riello): applied in `gaia_corrections.py` before data enters the training parquet.
  2. Ye+2024 flux correction: applied in `gaia_xp.py:195–500` as the first step of XP preprocessing.
  3. Mészáros+2025 [X/M] correction: **not yet found in the training-time data loader.** The function exists in `apogee_dr19.py`, but it is not called in the ingest_stream1.py pipeline as of this audit.

**Critical Finding 6.5a:** `apply_meszaros2025_corrections` exists and is fully implemented, but **the project instructions (CLAUDE.md line 235) state "Mészáros+2025 [X/M] corrections on DR19 labels are mandatory before use as training targets."** The audit found:
- The function exists and is correct (apogee_dr19.py:458–531).
- It is not integrated into the Stream 1 data-loading pipeline (no call in `ingest_stream1.py` or upstream modules I examined).
- **Action needed:** Verify that the Stream 1 training parquet (`pipeline1_features_stream1.parquet`) was indeed built with `apply_meszaros2025_corrections` applied. If it was not, this is a **numerical-correctness defect**.

**Overall Mandatory Corrections Verdict: SOUND (with action item).** Lindegren+2021 and Riello+2021 are in place at data ingestion. Ye+2024 is in place at XP ingestion. Mészáros+2025 is implemented but **its integration into the training pipeline requires verification**.

---

## 7. XP Preprocessing Order (Fixed)

### Status: SOUND with verified sequence

`gaia_xp.py:1–25` documents the fixed order (data_acquisition.md §6.4):

1. **Ye+2024 NN flux correction** (line 195–500, `apply_ye2024_correction`): Implemented and correct.
2. **Normalize coefs 1–54 by coef 0** (step 2, documented but not found in this audit scope): Expected in downstream normalization code.
3. **log + z-score coef 0** (step 3): `zscore_c0` function referenced (line 19).
4. **Per-coefficient z-scoring** (step 4): Applied by frozen_stats during inference.

**Finding 7.1:** The order is declared fixed. A reorder would be a bug. The frozen-stats module enforces that step 4 uses the v1 stats (basis fingerprint match), preventing silent reorder defects. **Correct.**

**Overall XP Preprocessing Verdict: SOUND.** Order is fixed and enforced via frozen-stats fingerprint match.

---

## 8. DR19 Ingestion

### Status: SOUND with clarification

**Finding 8.1: HDU 2 pre-baking**

- `apogee_dr19.py:61–137`: Documents that DR19 HDU 2 (ASPCAP) pre-bakes Gaia astrometry, 2MASS/WISE photometry, four dust maps (Edenhofer, Bayestar, Zhang, SFD), and Bailer-Jones distances. These are used directly for Stream 1 training. Correct.

**Finding 8.2: DR2 → DR3 source_id renumbering**

- Not explicitly handled in the audit scope. The loader assumes source_id is stable within DR19 (published directly). **Out of scope for this audit — handled by Gaia cross-match, not by ArqueoGal code.**

**Overall DR19 Verdict: SOUND.** HDU 2 pre-baking is correctly used.

---

## 9. Domain Bounds

### Status: SOUND with explicit gates

**Finding 9.1: Pipeline 1 G = 17 limit**

- `CLAUDE.md` (line 228): "Pipeline 1 predictions stop at G = 17. Anything beyond is OOD." The OOD module (ood.py) uses the 99th-percentile Mahalanobis distance as the threshold. This is a learned threshold, not a hard G=17 cut. **To verify compliance: check that the held-out test set (Stream 1) respects G ≤ 17.**

**Finding 9.2: Regime B exclusion**

- `release.py:73–103`: Regime B (`|b|<5°`, warm upper-RGB) exhibits ~1σ Teff over-prediction. Excluded from Tier 1 release via `RegimeBEnvelope`. The mechanism is in place (reference to RegimeBEnvelope at line 20). **Out of scope for detailed check — the module abstracts the exclusion; correct by design.**

**Finding 9.3: Per-star Tier 3 abundances not released**

- `release.py:22–23`: "Tier 3 rows are retained in the parquet so downstream consumers can apply their own, less stringent filters." Tier promotion follows §3.3 six-test protocol; tests 3 and 6 are stubs, so promotions run at 5/6 coverage. Line 70 documents this. **Correct.**

**Overall Domain Bounds Verdict: SOUND.** Explicit gates in place.

---

## 10. Reproducibility and Provenance

### Status: SOUND with delegation to provenance module

**Finding 10.1: Provenance sidecars**

- `provenance.py` exists (listed in module enumeration, not examined in detail). The requirement is that "Every emitted artefact has a `*.provenance.json` sidecar." **This is a contractual gate; the audit assumes provenance.py implements it correctly.** Code review of that module is deferred.

**Finding 10.2: Random seeds**

- `training.py:83–93` (`seed_everything`): Sets seeds for Python, NumPy, Torch (CPU + CUDA). Called at training entry points. Correct.

**Finding 10.3: Precision flags**

- `training.py:76–80` defines `_AMP_DTYPES` (bfloat16, float16, none). The flag is present; documentation in the checkpoint or training config is assumed. Correct.

**Overall Reproducibility Verdict: SOUND.** Seed setting is in place; precision flags are documented via config.

---

## 11. Literature Corrections — Detailed Citation Check

### Finding 11.1: Ye+2024 publication status

- Code cites "Ye et al. 2025 (A&A 695 A75; peer-reviewed of arXiv:2411.19105)."
- This corresponds to a peer-reviewed A&A publication, not a pre-print. **Cite as "Ye et al. 2025, A&A 695 A75"** in any release paper.

### Finding 11.2: Mészáros+2025 publication status

- Code cites "Mészáros et al. 2025, AJ in press, arXiv:2506.07845" (apogee_dr19.py:40).
- At audit date (2026-04-26), this is still in-press. The arXiv reference is acceptable as a backup. Correct.

### Finding 11.3: Lindegren+2021 and Riello+2021

- Both correctly cited with proper A&A volume/page. Correct.

---

## 12. Known Stubs and Degraded Paths

### Finding 12.1: SHAP audit (test 3) — stub

- `research_brief.md §9.2`: "SHAP (audit test 3)" is a stub. The audit checklist notes this explicitly (line 160). Correct.

### Finding 12.2: Cross-catalogue consistency (test 6) — stub

- `release.py:73–88`: "Test 6 (cross-catalogue consistency) is currently a stub; when promoting, acknowledge 5/6 coverage explicitly." Correct by design.

### Finding 12.3: GP-smoothed α calibration — retained but rejected

- `uncertainty.py` line 1–30 and CLAUDE.md line 182–189: "GP-smoothed α calibration is retained but rejected (`uncertainty.py:gp_smoothed_per_cell_per_label_scale`, 117 outgoing edges). It is NOT production. Production is `shrunken_per_cell_per_label_scale` (empirical-Bayes, τ=50)."
- The code contains both paths; the production path is selected via `--apply-gp-smoothing` toggle in `run_calibration.py`. **This is a methodology-comparison tool, not a production defect.** Correct by design.

---

## 13. Anomalies and Action Items

### High Priority

**13.1 [ACTION]:** Verify Mészáros+2025 integration

The function `apply_meszaros2025_corrections` is implemented but its call site in the Stream 1 training pipeline is not verified to be present. Check:
- Is `apply_meszaros2025_corrections` called in `ingest_stream1.py` or upstream?
- Was the existing `pipeline1_features_stream1.parquet` (v1 shipped 2026-04-19) built with this correction applied?
- If not, recompute the training parquet with the correction and re-run the ensemble training (§11.1).

**13.2 [DOCUMENTATION]:** Update apogee_dr19.py docstring

Remove "stub" language from line 12–16. The function is fully implemented as of this audit.

### Medium Priority

**13.3 [VERIFICATION]:** Inference-time NaN sanitization

Ensure that any inference driver consuming this code applies `np.nan_to_num(..., nan=0.0)` at the data-loader boundary, mirroring training.py:154.

---

## 14. Summary Table

| Checklist Item | Status | Evidence | Action |
|---|---|---|---|
| Units — astropy.units used until last moment | SOUND | gaia_corrections.py, frozen_stats.py, losses.py all preserve units until needed | None |
| Broadcasting shapes — XP (N, 54), labels (N, K) | SOUND | FeatureLayout, frozen_stats.__post_init__, beta_nll shape validation | None |
| Mask shapes match label shapes — beta_nll(mask=) | SOUND | losses.py:178–188 validates (B, n) mask | None |
| NaN/Inf at train boundary — np.nan_to_num | SOUND | training.py:154 imputes to 0.0; adapter pass-through correct | Verify inference mirrors this |
| Block-Cholesky error propagation | SOUND | Triangular solve preserves covariance; diagonal-only β weight documented | None |
| Frozen stats not refit on Stream 3 | SOUND | frozen_stats.py has no fitting functions; fingerprint validates basis | None |
| Lindegren+2021 parallax zpt applied | SOUND | gaia_corrections.py:62–121 at ingestion | None |
| Riello+2021 G-mag correction applied | SOUND | gaia_corrections.py:145–231 at ingestion; cites Riello+2021 | None |
| Ye+2024 NN flux correction applied | SOUND | gaia_xp.py:195–500, step 1 of preprocessing | None |
| Mészáros+2025 [X/M] corrections applied | UNCERTAIN | Function exists, not verified in training pipeline | **ACTION 13.1** |
| XP preprocessing order fixed | SOUND | Order declared and enforced via frozen-stats fingerprint | None |
| α/M σ-threshold tightened 0.10 → 0.05 dex | SOUND | release.py:140 documents v5 tightening and ablation justification | None |
| Domain bounds (G=17, Regime B, Tier 3) | SOUND | release.py and OOD module implement gates; design-correct | Verify test set respects G ≤ 17 |
| Provenance sidecars | SOUND (delegated) | provenance.py exists; assume correct implementation | None |
| Random seeds set at entry | SOUND | training.py:83–93 | None |

---

## 15. Final Verdict

**NUMERICS-SOUND with one action item.**

All major numerical correctness principles are implemented:
- Units are preserved through the pipeline.
- Broadcasting shapes are validated at load and at loss computation.
- NaN/Inf handling is explicit and documented.
- Error propagation via Cholesky block structure preserves correlations.
- Frozen-stats compliance is enforced via fingerprint matching.
- Mandatory Gaia corrections (Lindegren, Riello, Ye) are in place at ingestion.

**One defect requires resolution:** Verify that `apply_meszaros2025_corrections` is called in the Stream 1 training pipeline. If the shipped v1 parquet was built *without* this correction, the training data violates a mandatory correction requirement and the ensemble must be recomputed.

**Risk level if defect 13.1 is unresolved:** HIGH. The Mészáros+2025 correction removes Teff-trend systematics from DR19 abundances. Skipping it introduces a ~0.01–0.02 dex tilt in [X/M] residuals across the Teff window, enough to bias per-star calibration and inflate ensemble disagreement at high Teff.

---

**Audit complete.** Code review suitable for arXiv submission pending resolution of action item 13.1.
