# Phase B kinematic-OOD detector integration plan

**Status:** plan ready, not yet executed
**Author:** Opus 4.7 autonomous overnight session, 2026-04-25 04:40 UTC
**Scope:** populate the v3 catalog's `kin_ood_flag` column with the actual detector output, replacing the v1 placeholder (all False).

This document is the recipe to land Phase B on a single 3-4 hour pass. The morning summary previously noted that Phase B was blocked because Stream 1 / Stream 3 features parquets lacked `pmra`, `pmdec`, `radial_velocity`. That assessment was incomplete: the **interim parquets carry the full kinematic block** (verified 2026-04-25 04:40 UTC), so no Gaia re-fetch is required. The feasibility upgrade is what motivates this plan.

---

## What the autonomous session already verified

`data/interim/stream1_apogee_gaia.parquet` (163 cols, 232 MB) carries:
- `pmra`, `pmra_error`, `pmdec`, `pmdec_error`
- All astrometric correlations: `ra_pmra_corr`, `ra_pmdec_corr`, `dec_pmra_corr`, `dec_pmdec_corr`, `parallax_pmra_corr`, `parallax_pmdec_corr`, `pmra_pmdec_corr`
- `radial_velocity`, `radial_velocity_error`

`data/interim/stream3_delta_gaia_dr3_corrected.parquet` (63 cols, 101 MB) carries the same fields.

`src/arqueogal/data/gaia_enrich.py` lines 49-59 confirms the TAP query already fetches these as part of the `ENRICHMENT_ADQL` template; the columns are simply dropped by the downstream features-build step (likely `scripts/build_pipeline1_features_stream3.py`, not yet inspected).

The detector module `src/arqueogal/xp_abundances/main/kinematic_ood.py` (~245 lines) is in place:
- `KinematicOODBundle` dataclass for the fit (mean, covariance, threshold).
- `fit_kinematic_ood(velocities, ...)` for Stream 1 fit.
- `score_kinematic_ood(bundle, velocities)` for per-star Mahalanobis distance.
- `flag_kinematic_ood(bundle, velocities)` for the boolean flag.

---

## Step-by-step recipe

### Step 1, add a velocity-computation module (~45 min)

Create `src/arqueogal/data/kinematics_galactocentric.py`:

```python
"""Galactocentric (v_R, v_phi, v_z) from Gaia DR3 6D astrometry.

Uses astropy.coordinates.SkyCoord with GalactocentricFrame defaults
(see galactic_frame_defaults.md ADR if it exists, otherwise McMillan17
parameters). Vectorised; handles ~600k rows in < 10 seconds.
"""
def compute_galactocentric_velocities(
    df: pd.DataFrame.
    *.
    ra_col: str = "ra".
    dec_col: str = "dec".
    parallax_col: str = "parallax_corr".
    pmra_col: str = "pmra".
    pmdec_col: str = "pmdec".
    rv_col: str = "radial_velocity".
    distance_col: str = "r_med_photogeo".
) -> pd.DataFrame:
    """Returns df with added columns v_R_kms, v_phi_kms, v_z_kms.

    Stars with non-finite parallax or RV propagate as NaN, which the
    downstream OOD scorer flags as outliers (conservative).
    """
```

Use `astropy.coordinates.ICRS(...).transform_to(astropy.coordinates.Galactocentric())` and pull `.v_x.value`, `.v_y.value`, `.v_z.value`, then convert to cylindrical `(v_R, v_phi, v_z)`.

Add a unit test in `tests/data/test_kinematics_galactocentric.py` exercising a star with known velocities (e.g., the Sun: v_R ~ 0, v_phi ~ -240 km/s, v_z ~ 0 in the LSR frame; pin to McMillan17 sun parameters).

### Step 2, fit the detector on Stream 1 (~30 min)

Write `scripts/fit_kinematic_ood.py`:

```python
# Load Stream 1 interim parquet.
df = pd.read_parquet("data/interim/stream1_apogee_gaia.parquet".
                     columns=["source_id", "ra", "dec", "parallax_corr".
                              "pmra", "pmdec", "radial_velocity"])
# Need a distance: load Stream 1 features for r_med_photogeo.
feat = pd.read_parquet("data/processed/pipeline1_features_stream1.parquet".
                       columns=["source_id", "r_med_photogeo"])
df = df.merge(feat, on="source_id", validate="many_to_one")
df = compute_galactocentric_velocities(df)

# Disc-only training subset: |z| < 1 kpc, |v_z| < 50 km/s would be ideal.
# but for an OOD detector that learns "where the bulk of disc kinematics
# is", we want all training stars regardless of z. Use the full APOGEE
# training set; halo and accreted-debris contamination is small (<5%)
# and the detector's job is to flag stars outside the dominant disc
# distribution, including those small contaminants.
velocities = df[["v_R_kms", "v_phi_kms", "v_z_kms"]].dropna().values
bundle = fit_kinematic_ood(velocities, p_threshold=0.99)

# Persist the bundle next to the model checkpoint.
bundle.save("models/main/xp_abundances/.../kinematic_ood_bundle.npz")
```

Validate: compute the mean and covariance against published Solar-neighborhood values (mean v_phi ~ -240 km/s, sigma_v_R ~ 35 km/s, etc.; cite Gaia Collaboration+2023, Robin+2017).

### Step 3, score Stream 3 and write a sidecar parquet (~30 min)

Write `scripts/score_kinematic_ood_stream3.py`:

```python
df = pd.read_parquet("data/interim/stream3_delta_gaia_dr3_corrected.parquet".
                     columns=["source_id", "ra", "dec", "parallax_corr".
                              "pmra", "pmdec", "radial_velocity"])
feat = pd.read_parquet("data/processed/pipeline1_features_stream3.parquet".
                       columns=["source_id", "r_med_photogeo"])
df = df.merge(feat, on="source_id", validate="one_to_one")
df = compute_galactocentric_velocities(df)
bundle = KinematicOODBundle.load("...")
df["kin_ood_flag"] = flag_kinematic_ood(bundle.
    df[["v_R_kms", "v_phi_kms", "v_z_kms"]].values)
df["kin_ood_score"] = score_kinematic_ood(bundle.
    df[["v_R_kms", "v_phi_kms", "v_z_kms"]].values)
# Write to a small sidecar parquet next to predictions.
df[["source_id", "kin_ood_flag", "kin_ood_score"]].to_parquet(
    "data/processed/pipeline1_kin_ood_stream3.parquet")
# Provenance sidecar.
```

Expected: ~5-15% of Stream 3 stars flagged kinematic-OOD (halo, retrograde, hot-thick-disc).

### Step 4, modify the release pipeline to merge in the kinematic-OOD column (~20 min)

In `src/arqueogal/data/release_pipeline.py`:
- Add `kin_ood_path` parameter to `run_release_pipeline`.
- After the join, merge in the kin-OOD parquet on `source_id`, validate=one_to_one.
- The downstream `release.assign_per_element_release_tier` will now consume the live `kin_ood_flag` (currently it consumes the placeholder all-False from `release.assign_kin_ood_flag`).

In `src/arqueogal/xp_abundances/main/release.py`:
- `assign_kin_ood_flag` becomes a fallback when the column is not present (backwards-compatible).
- Pre-existing logic at line 317 (aux-assisted demotion via kin_ood) already consumes the flag, no further changes needed.

### Step 5, regenerate the Stream 3 release artefacts (~30 min)

Re-run `release_pipeline.run_release_pipeline()` end-to-end (the new join brings in the kin_ood column; annotate_parquet now produces the demoted [α/M] / [Mg/H] tiers). Expected outcomes:
- Tier 1 [α/M] count drops by 5-15% (halo and accreted stars demoted).
- Tier 1 [M/H], T_eff, log g unchanged (spectrum-dominant).
- Per-element panel (b) of `fig_alpha_m_vs_fe_h.pdf` gains visible structure (the kin-OOD-True hexbin will be sparser than the all-Tier-1 panel).
- Manifest JSON updates the `kin_ood_flag_pct` field.

### Step 6, regenerate the diagnostic figures (~10 min)

Re-run `.expert_review_2026-04-24/reports/codebase_fixes/generate_real_data_plots.py`. The α-bimodality panel (b) now shows the actual demotion pattern instead of the placeholder caveat overlay.

### Step 7, update CATALOG_SCHEMA.md and DESIGN.md (~15 min)

`docs/CATALOG_SCHEMA.md`: replace the `kin_ood_flag` row's "v1 placeholder, all False" language with the live behaviour. Cite the kin-OOD-bundle path.

`src/arqueogal/data/DESIGN.md`: add a `kinematics_galactocentric.py` entry to the module-layout tree.

`docs/plan/03_stream3_inference.md`: mark Phase B kin-OOD as done with date.

---

## Potential pitfalls

1. **Stream 1 RV missingness.** APOGEE provides spectroscopic RV (`vhelio` or `rv_apogee` column) which is more reliable than Gaia's RV for cool giants. Decide whether to use Gaia RV (simpler, works for both streams), APOGEE RV when available (requires a fallback), or coast on tangential velocity only (drop v_R into 2D Mahalanobis). For autonomous follow-up: start with Gaia RV; iterate later.

2. **Distance choice.** `r_med_photogeo` (Bailer-Jones+2021) is the Stream 3 default. Stream 1 also has it via DR19's pre-baked Bailer-Jones distance (column `bj_r_med_photogeo` or similar). Verify the column name when loading Stream 1; the audit-corrections memo did not check this.

3. **Frame conventions.** astropy's `Galactocentric` defaults are NOT McMillan17. Pin a custom `GalactocentricFrame` with McMillan17 sun parameters (galcen_distance=8.21 kpc, z_sun=20.8 pc, galcen_v_sun=(11.1, 248.27, 7.25) km/s) to match the project convention used elsewhere (`src/arqueogal/data/kinematics.py` for the older galpy-based code).

4. **NaN propagation.** Stars without RV or with non-finite parallax produce NaN velocities. The detector should flag those as kin-OOD True (conservative). Ensure the score function treats NaN inputs as infinite distance.

5. **Threshold calibration.** The 99th percentile is applied to the **fit-set** velocity Mahalanobis distances, not to chi-squared theory. With ~700k Stream 1 stars, the empirical 99th percentile is well-determined, but the chi-squared p=0.99 (df=3) value of D² ~ 11.34 is a useful sanity check.

6. **Backward compatibility.** Keep `assign_kin_ood_flag` as a fallback in `release.py` for anyone who runs the annotate without the kin-OOD merge. Document the recommended path in the docstring.

---

## Estimated total time, 3-4 hours

| Step | Estimate |
|---|---|
| 1. Velocity-computation module + test | 45 min |
| 2. Fit on Stream 1 | 30 min |
| 3. Score Stream 3, write sidecar parquet | 30 min |
| 4. Wire into release_pipeline | 20 min |
| 5. Regenerate Stream 3 release artefacts | 30 min |
| 6. Regenerate diagnostic figures | 10 min |
| 7. Doc updates | 15 min |
| Buffer for debugging coordinate transforms | 30 min |

Total: ~3-3.5 hours, well within a single afternoon.

---

## Why this was deferred autonomously

Even with the data accessible, the work spans 7 distinct steps including coordinate-transform code, multi-source merges, and overwriting the existing release artefacts. A single subtle bug in the McMillan17 frame definition could ship a quietly-wrong kin-OOD flag. User oversight at the velocity-validation step is the natural risk gate. The plan document is the autonomous-friendly deliverable; the execution belongs in a daytime working session.
