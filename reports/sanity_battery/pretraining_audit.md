# Pipeline-1 Pre-training Sanity Battery

**Result: all six gates pass.**

- Input: `data/processed/pipeline1_features_stream1.parquet` (324,054 rows × 316 cols)
- Checks: 6 (3 hard-fail, 3 soft-fail)
- Hard-fail count: **0**
- Soft-fail count: **0**

## Per-check results

| # | Check | Level | Result | Summary |
|---|---|---|---|---|
| 1 | `xp_feature_nan_invariant` | HARD | **PASS** | no unexpected NaN in 110 XP feature columns (flagged-out rows: 14,183/324,054) |
| 2 | `tier1_label_completeness` | HARD | **PASS** | all 3 Tier-1 atmospheric labels finite on 324,054 flag_bad==0 rows |
| 3 | `parameter_bounds` | HARD | **PASS** | all 5 parameters within physical bounds |
| 4 | `per_element_nan_rates` | SOFT | **PASS** | per-element NaN baseline on 324,054 flag_bad==0 rows; top-3: v_h_apogee 5.35%, s_h_apogee 4.87%, k_h_apogee 4.87% |
| 5 | `zscore_validity` | SOFT | **PASS** | z-score stats within tolerance on 309,871 reference rows: BP mu=-0.0000 σ=1.0000, RP mu=+0.0000 σ=1.0000 |
| 6 | `dedup_idempotency` | SOFT | **PASS** | dedup rows_out=292,948 matches expected 292,948 |

### 1. `xp_feature_nan_invariant` — HARD — PASS

*no unexpected NaN in 110 XP feature columns (flagged-out rows: 14,183/324,054)*

```json
{
  "n_flagged_out_rows": 14183,
  "n_xp_columns_checked": 110,
  "surprise_counts_per_column": {},
  "expected_mask": "ye2024_flag != 0 OR xp_fit_flag_residual_high != 0/NA OR bp_coef_0 <= 0 OR rp_coef_0 <= 0"
}
```

### 2. `tier1_label_completeness` — HARD — PASS

*all 3 Tier-1 atmospheric labels finite on 324,054 flag_bad==0 rows*

```json
{
  "n_flag_bad_zero_rows": 324054,
  "tier1_atmospheric_labels": [
    "teff_apogee",
    "logg_apogee",
    "mh_apogee"
  ],
  "nan_counts_per_label": {}
}
```

### 3. `parameter_bounds` — HARD — PASS

*all 5 parameters within physical bounds*

```json
{
  "violations": {},
  "bounds": {
    "teff_apogee": [
      3000.0,
      8000.0
    ],
    "logg_apogee": [
      -0.5,
      5.5
    ],
    "fe_h_apogee": [
      -4.0,
      1.1
    ],
    "mh_apogee": [
      -4.0,
      1.0
    ],
    "alpha_m_apogee": [
      -0.8,
      0.8
    ]
  }
}
```

### 4. `per_element_nan_rates` — SOFT — PASS

*per-element NaN baseline on 324,054 flag_bad==0 rows; top-3: v_h_apogee 5.35%, s_h_apogee 4.87%, k_h_apogee 4.87%*

```json
{
  "n_flag_bad_zero_rows": 324054,
  "rates": {
    "fe_h_apogee": {
      "n_nan": 5234,
      "rate": 0.016151629049479407
    },
    "alpha_m_apogee": {
      "n_nan": 0,
      "rate": 0.0
    },
    "mg_h_apogee": {
      "n_nan": 5222,
      "rate": 0.016114598184253242
    },
    "c_h_apogee": {
      "n_nan": 15689,
      "rate": 0.04841477037777654
    },
    "n_h_apogee": {
      "n_nan": 15766,
      "rate": 0.04865238509631111
    },
    "o_h_apogee": {
      "n_nan": 15766,
      "rate": 0.04865238509631111
    },
    "na_h_apogee": {
      "n_nan": 15766,
      "rate": 0.04865238509631111
    },
    "al_h_apogee": {
      "n_nan": 15689,
      "rate": 0.04841477037777654
    },
    "si_h_apogee": {
      "n_nan": 15766,
      "rate": 0.04865238509631111
    },
    "s_h_apogee": {
      "n_nan": 15771,
      "rate": 0.04866781462348868
    },
    "k_h_apogee": {
      "n_nan": 15771,
      "rate": 0.04866781462348868
    },
    "ca_h_apogee": {
      "n_nan": 15689,
      "rate": 0.04841477037777654
    },
    "ti_h_apogee": {
      "n_nan": 15766,
      "rate": 0.04865238509631111
    },
    "v_h_apogee": {
      "n_nan": 17347,
      "rate": 0.05353120158985848
    },
    "cr_h_apogee": {
      "n_nan": 15689,
      "rate": 0.04841477037777654
    },
    "mn_h_apogee": {
      "n_nan": 15771,
      "rate": 0.04866781462348868
    },
    "ni_h_apogee": {
      "n_nan": 15766,
      "rate": 0.04865238509631111
    },
    "ce_h_apogee": {
      "n_nan": 15689,
      "rate": 0.04841477037777654
    }
  }
}
```

### 5. `zscore_validity` — SOFT — PASS

*z-score stats within tolerance on 309,871 reference rows: BP mu=-0.0000 σ=1.0000, RP mu=+0.0000 σ=1.0000*

```json
{
  "n_reference_rows": 309871,
  "bp_mean": -1.575756591876143e-08,
  "bp_std": 1.0,
  "rp_mean": 3.151513050525523e-09,
  "rp_std": 0.9999998807907104,
  "mean_tol": 0.001,
  "std_tol": 0.005
}
```

### 6. `dedup_idempotency` — SOFT — PASS

*dedup rows_out=292,948 matches expected 292,948*

```json
{
  "rows_in": 324054,
  "rows_out": 292948,
  "expected_rows_out": 292948,
  "n_duplicate_stars": 25531,
  "max_duplicates_per_star": 15,
  "sort_column": "snr",
  "duplicate_histogram": {
    "1": 267417,
    "2": 21332,
    "3": 3426,
    "4": 553,
    "5": 123,
    "6": 37,
    "7": 11,
    "8": 3,
    "9": 3,
    "10": 7,
    "11": 9,
    "12": 14,
    "13": 6,
    "14": 2,
    "15": 5
  }
}
```

## Check 4 — Distribution plots (operator eyeball)

Compare the shapes below against the published Mészáros+2025 ASPCAP-giant distributions. Shape match matters more than statistical p-values; our cut imposes known differences vs the full ASPCAP pool.

### Kiel diagram

![Kiel diagram](kiel.png)

### [α/M] — [Fe/H]

![Tinsley-Wallerstein](tinsley_wallerstein.png)

## Continuity embedding — UMAP on 43-D XP subspace

Feature vector: `bp_coef_norm_1..19` + `rp_coef_norm_1..22` + `bp_c0_z` + `rp_c0_z` (43 dims). Normal-population subset. Expected: a clean smooth Teff gradient across the embedding, no isolated clusters, no random scatter. A broken gradient indicates a data-plumbing bug (column order, accidental shuffling, or miscomputed z-score) that the other checks missed.

![UMAP continuity](umap_continuity.png)

## Why these specific bounds and checks

The checks below are deliberately calibrated to APOGEE DR19 realism, not textbook-generic finiteness expectations. Keeping them DR19-aware is the gating criterion for whether the battery catches a real future drift versus false-alarming on genuine ASPCAP behaviour. If any of this rationale changes, update `sanity.py` and this rationale together.

- **Tier-1 completeness gates on `{teff_apogee, logg_apogee, mh_apogee}` only** — not `fe_h_apogee`. ASPCAP DR19 fits [M/H] globally over Fe-peak + α lines, then runs per-element [Fe/H] afterwards; the per-element fit legitimately fails on saturated Fe lines (metal-rich regime), insufficient SNR per individual line in the blue, or unresolved Fe blends in cool giants. A ~1.6% NaN rate on `fe_h_apogee` at flag_bad==0 is the DR19 baseline, not a pipeline bug. See `sanity.py::TIER1_ATMOSPHERIC` and research_brief §3.2. [Fe/H] is treated as a Tier-2 per-element label (NaN-masked in training).

- **Parameter bounds match ASPCAP DR19 dynamic range**: `fe_h_apogee ∈ [-4, 1.1]` widens the canonical +1.0 upper bound to cover DR19's metal-rich tail (per-element σ ≈ 0.03 dex); `alpha_m_apogee ∈ [-0.8, 0.8]` widens the disk-dominated [-0.5, 0.7] envelope to include genuine halo/CEMP α-poor tails and the upper α-rich fits. Tighter bounds would false-alarm on real DR19 stars; looser bounds would not flag a future ASPCAP drift to an even wider dynamic range (e.g. a DR20 recalibration pushing α/M to [-1, 1] would correctly trip this check).

- **RGB window [Teff 4000-5500 K, log g 1.0-3.5] is enforced as an explicit builder-time cut** (in `scripts/build_pipeline1_features_stream1.py`). At 2026-04-18 the emergent intersection of (ASPCAP flag_bad==0) ∩ (SNR>70) ∩ (Gaia XP available) already falls inside this box — the cut drops zero rows on current data. It is retained as a named cut so a future APOGEE DR20 rebuild with different SNR statistics or XP coverage surfaces any drift immediately as a nonzero drop count in provenance, not weeks into training audit.

- **Per-element NaN rates (check `per_element_nan_rates`) are report-only** — no threshold. The baseline rates captured above are the 2026-04-18 snapshot; future rebuilds should compare against these. A large delta on any element (e.g. Mg or Al rates doubling) is a signal to investigate upstream ASPCAP changes before training.

## Bottom line

All six gate checks pass. The data-layer contract is validated against the 2026-04-18 feature-matrix emit.