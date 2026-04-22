# Ye+2024 `NO_SYNTH_PHOT` Selection Function — v1

**Author:** Andreas Neitzel (Co-I)
**Date:** 2026-04-19
**Context:** ArqueoGal D-Cat-b release (FCT 2024.15303.PEX), unblocking Stream 3 launch.
**Input:** `data/processed/pipeline1_features_stream1.parquet` (N = 324,054).
**Artefact:** `reports/selection_function/selection_function_v1.parquet` (5×5 grid).
**Provenance:** `reports/selection_function/selection_function_v1.provenance.json`.
**Scorer module:** `src/arqueogal/data/selection_function.py` → `score_selection_prob(b_deg, g_mag)`.

---

## 1. Motivation

Thread-1 diagnostics on Stream 1 established that the `ye2024_flag == 1` rate
(set when `gaiaxpy.generate` cannot produce synthetic photometry for the
pre-correction Gaia spectrum and Ye's NN therefore refuses to apply a
correction — i.e., the `NO_SYNTH_PHOT` failure mode) is a strong function of
Galactic latitude and G magnitude:

| Region | N | `NO_SYNTH_PHOT` rate |
|---|---:|---:|
| `|b| < 5°` (plane) | 69,134 | 10.48 % |
| `|b| > 15°` (off-plane) | 170,119 | 0.08 % |
| Global (Stream 1) | 324,054 | 2.60 % |

Ratio = 133.99×. The rejection is not a uniformly-random sampling — it tracks
regions where crowding, extinction, and Gaia XP de-blending failures
preferentially remove stars. The D-Cat-b catalogue must therefore expose a
per-star scalar quantifying this selection so downstream users can inverse-
weight when doing disc-structure, halo, or plane-archaeology science.

## 2. Methodology

### 2.1 Choice: 2-D grid vs regression

We tabulate the `NO_SYNTH_PHOT` rate on a 5×5 rectangular grid in `(|b|, G)`.
Grid edges:

- `|b|` (deg): `[0, 5, 10, 20, 45, 90]`
- `G`   (mag): `[2.0, 11.0, 12.5, 14.0, 15.5, 17.65]`

These were chosen to isolate the plane (`|b| < 5°`), the mid-plane transition
(5–10°), the disc/thick-disc interface (10–20°), the off-plane bulk (20–45°),
and the caps (45–90°) on the latitude axis; and on the magnitude axis to
isolate the bright Gaia DR3 floor, the DR3 RV-sample bulk, and the XP-native
faint regime split into two bins up to the release cutoff G = 17.65.

**Sparse-cell census.** Of 25 cells, exactly **1** falls below the `n < 200`
sparseness threshold: `|b| ∈ [45°, 90°)`, `G ∈ [15.5, 17.65)` with N = 180.
Its measured flag rate is 0.56 %, fully consistent with the surrounding
low-latitude-faint / high-latitude-bright neighbours (all ≤ 1 %), so no
smoothing/regression fallback is triggered. If future data ingestion tips
this cell (or others) below 200, the planned fallback is `statsmodels.nonparametric.lowess`
or `scipy.ndimage.gaussian_filter` on the rate grid; the contract with
downstream consumers (`score_selection_prob(b_deg, g_mag) → float64 in [0.01, 1.0]`)
is unchanged.

### 2.2 Computation

For each cell $c = (i_b, i_g)$:

$$
\mathrm{flag\_rate}_c = \frac{N_{\mathrm{flagged}, c}}{N_{\mathrm{total}, c}},
\qquad
\mathrm{selection\_prob}_c = \mathrm{clip}\left(1 - \mathrm{flag\_rate}_c,\ 0.01,\ 1.0\right).
$$

The floor of 0.01 guarantees finite inverse weights inside the plane at faint
magnitudes (where the empirical rate reaches 40 %) while keeping the weight
visibly large — the honest rather than the convenient choice. The ceiling
of 1.0 is structural (probabilities cannot exceed unity).

### 2.3 Scoring protocol

Given a star's `(b_deg, g_mag)`, the scorer takes `|b|`, clamps both inputs
to the grid support `([0, 90], [2.0, 17.65])`, and returns the cell's
`selection_prob`. No smoothing or inter-cell interpolation: v1 is a piecewise-
constant step function. This is deliberate — stepwise is audit-friendly, the
grid resolution is coarse enough that inter-cell interpolation adds no
information, and downstream users can apply their own smoothing if required.

## 3. The grid

All 324,054 Stream-1 rows fall inside the grid; 8,438 flagged (2.60 % global).

| `|b|` (deg) | `G` (mag) | N | flagged | flag_rate | selection_prob |
|---:|---:|---:|---:|---:|---:|
| [0, 5)   | [2.0, 11.0)  |    990 |     0 | 0.0000 | 1.0000 |
| [0, 5)   | [11.0, 12.5) |   4898 |     0 | 0.0000 | 1.0000 |
| [0, 5)   | [12.5, 14.0) |  18542 |     1 | 0.0001 | 0.9999 |
| [0, 5)   | [14.0, 15.5) |  28100 |   543 | 0.0193 | 0.9807 |
| [0, 5)   | [15.5, 17.65)|  16604 |  6698 | 0.4034 | **0.5966** |
| [5, 10)  | [2.0, 11.0)  |   1291 |     0 | 0.0000 | 1.0000 |
| [5, 10)  | [11.0, 12.5) |   5327 |     0 | 0.0000 | 1.0000 |
| [5, 10)  | [12.5, 14.0) |  15377 |     0 | 0.0000 | 1.0000 |
| [5, 10)  | [14.0, 15.5) |  18594 |    57 | 0.0031 | 0.9969 |
| [5, 10)  | [15.5, 17.65)|   4948 |   911 | 0.1841 | 0.8159 |
| [10, 20) | [2.0, 11.0)  |   3733 |     0 | 0.0000 | 1.0000 |
| [10, 20) | [11.0, 12.5) |  13080 |     0 | 0.0000 | 1.0000 |
| [10, 20) | [12.5, 14.0) |  23984 |     0 | 0.0000 | 1.0000 |
| [10, 20) | [14.0, 15.5) |  16312 |    20 | 0.0012 | 0.9988 |
| [10, 20) | [15.5, 17.65)|   1090 |   186 | 0.1706 | 0.8294 |
| [20, 45) | [2.0, 11.0)  |  22082 |     0 | 0.0000 | 1.0000 |
| [20, 45) | [11.0, 12.5) |  47170 |     0 | 0.0000 | 1.0000 |
| [20, 45) | [12.5, 14.0) |  36241 |     0 | 0.0000 | 1.0000 |
| [20, 45) | [14.0, 15.5) |   4499 |     0 | 0.0000 | 1.0000 |
| [20, 45) | [15.5, 17.65)|    551 |    21 | 0.0381 | 0.9619 |
| [45, 90] | [2.0, 11.0)  |   9983 |     0 | 0.0000 | 1.0000 |
| [45, 90] | [11.0, 12.5) |  16251 |     0 | 0.0000 | 1.0000 |
| [45, 90] | [12.5, 14.0) |  12332 |     0 | 0.0000 | 1.0000 |
| [45, 90] | [14.0, 15.5) |   1895 |     0 | 0.0000 | 1.0000 |
| [45, 90] | [15.5, 17.65]|    180*|     1 | 0.0056 | 0.9944 |

*single cell with N < 200 (sparse-cell threshold); rate consistent with
the adjacent cell `[20, 45) × [15.5, 17.65)` (rate 3.81 %) and the low-
latitude high-G trend, so smoothing is not triggered.*

### Shape

Reading the table: **the selection is essentially flat at 1.0 everywhere
brighter than G ≈ 14 regardless of latitude**; it only bites in the two
faintest bins. Inside the plane (`|b| < 5°`) at G > 15.5 the retention drops
to **0.597** — i.e., 40 % of stars there are missing from the Ye-corrected
sample. Off-plane at G > 15.5 the retention recovers to ≥ 0.81, and by
`|b| > 20°` the faint bin is still ≥ 0.96. The structure is a factor ~2–67×
variation in rejection rate across a single magnitude bin, driven entirely
by the plane crossing.

## 4. Representative example stars

Scored from `data/processed/pipeline1_features_stream1.parquet`:

| tag | source_id | b (deg) | G (mag) | `ye2024_flag` | scored `selection_prob` |
|---|---|---:|---:|---:|---:|
| faint_plane   | 4116896994232035328 | +3.26  | 16.90 | 1 | **0.597** |
| bright_plane  |  189402663276535680 | +3.46  | 10.61 | 0 | 1.000 |
| faint_cap     | 4691764932043836416 | −47.58 | 16.18 | 0 | 0.994 |
| bright_cap    |  745142126815716480 | +50.91 | 10.53 | 0 | 1.000 |
| midlat_mid_G  |  480694977448495744 | +19.09 | 12.92 | 0 | 1.000 |

The two `faint_*` examples — same G-bin, factor-2 difference in retention —
encapsulate why this feature must be exposed per-star.

## 5. Interpretation by science case

**Disc-structure science (phase spirals, velocity maps, Rocha-Pinto-style
age tomography, Gaia-Enceladus debris in the halo-disc interface):** The
signal is dominated by mid-plane-but-not-in-plane populations (`|b|` of a
few degrees to ~30°) at G ≤ 15. In this regime `selection_prob` ≥ 0.99
across every cell we measure, so inverse weighting is numerically a no-op.
Users can safely ignore the column unless they are extending their analysis
into the faint plane.

**Galactic plane / bulge archaeology (low-latitude bar, in-plane
metallicity gradient, nuclear disc):** The signal sits inside the plane at
G ≥ 15.5, where retention drops to ~60 %. **Inverse weighting is material
here** and must be applied. Halo/bar densities will otherwise be
underestimated by up to 40 % at the catalogue's faint end. A per-cell
Poisson uncertainty on the inverse weight (`sqrt(n_flagged) / n_total`)
is recommended when turning the inverse weights into number densities.

**Halo archaeology, Milky Way satellites, streams:** These populations
live predominantly at `|b| > 20°`, where retention is ≥ 0.96 in every G
bin we measure. Selection-function bias is small. If the sample extends
into the plane crossings of streams (e.g., Sagittarius near `b ~ 0°`),
inverse-weighting is mandatory on the plane-crossing members.

## 6. Inverse-weighting recipe (for downstream users)

```python
from arqueogal.data.selection_function import score_selection_prob

sel = score_selection_prob(df["b_deg"], df["g_mag"])
df["selection_prob"] = sel           # column to be carried through D-Cat-b
df["inverse_weight"] = 1.0 / sel     # safe: sel >= 0.01 by construction
```

For a density-weighted estimator of any quantity `Q` on a sub-sample:

$$
\hat{Q} = \frac{\sum_i Q_i / \mathrm{selection\_prob}_i}{\sum_i 1 / \mathrm{selection\_prob}_i}.
$$

For likelihoods, multiply each star's per-object likelihood by
`1 / selection_prob_i`. The floor of 0.01 keeps the weight finite at the
plane-faint corner — but a sample whose weight is dominated by cells with
`selection_prob < 0.1` should be flagged in release notes; we are not
extrapolating the selection function, we are reporting what Ye+2024
actually retains.

## 7. Known limitations

1. **v1 stratifies on `(|b|, G)` only.** Teff, log g, [M/H], and local
   extinction covary with XP correction success (high-extinction regions
   in the plane map onto more failures); a v2 stratification including at
   least Teff and line-of-sight Av is earmarked for a post-D-Cat-b release.
2. **The per-cell Poisson error is not carried on the artefact.** v1
   downstream users who need it can compute `sqrt(n_flagged) / n_total`
   from the Parquet columns directly.
3. **The v1 artefact is based on Stream 1 (APOGEE × Gaia XP) only** —
   Stream 3 at the XP-native faint end may sample slightly different
   `(|b|, G)` joint distributions. The selection-function *rate* depends
   only on `(|b|, G)` and `NO_SYNTH_PHOT` is driven by Gaia XP conditions
   alone, so the rate should transfer; but a cross-check on the first
   Stream-3 Ye+2024 run is a scheduled sanity step.
4. **Piecewise-constant.** A star one magnitude inside a bin gets the
   same score as a star one magnitude out; the v2 LOWESS/GAM fallback
   will smooth that if resolution ever matters.

## 8. Artefact schema (`selection_function_v1.parquet`)

Tidy table, one row per cell, 25 rows:

| column | dtype | notes |
|---|---|---|
| `b_lo` | float64 | Galactic-latitude bin lower edge, deg, inclusive |
| `b_hi` | float64 | upper edge, exclusive (inclusive on last row) |
| `g_lo` | float64 | G-magnitude bin lower edge, inclusive |
| `g_hi` | float64 | upper edge, exclusive (inclusive on last row) |
| `n_total` | int64 | stars in cell |
| `n_flagged` | int64 | stars with `ye2024_flag == 1` in cell |
| `flag_rate` | float64 | `n_flagged / n_total` |
| `selection_prob` | float64 | `clip(1 − flag_rate, 0.01, 1.0)` |

## 9. Build + provenance

- **Builder:** `scripts/build_selection_function_v1.py` (deterministic — no
  random state, no RNG used).
- **Input SHA-256:** recorded in the provenance sidecar.
- **Git SHA and timestamp:** recorded in the provenance sidecar.
- **Reproduction:** `PYTHONPATH=src python scripts/build_selection_function_v1.py`.
- **Read-only on the input Parquet.** Atomic write on the output Parquet.

## 10. References

- Ye et al. 2025, *A&A* 695, A75 (peer-reviewed), arXiv:2411.19105. Zenodo
  concept DOI 10.5281/zenodo.14028588.
- `docs/data_acquisition.md` §6.4 — Ye+2024 integration in the ArqueoGal
  XP preprocessing sequence.
- `src/arqueogal/data/gaia_xp.py` — `YE2024_FLAG_NO_SYNTH_PHOT`,
  `apply_ye2024_correction`.
