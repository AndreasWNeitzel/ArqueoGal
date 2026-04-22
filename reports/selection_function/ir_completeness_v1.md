# IR-Completeness Selection-Function Component — v1

**Input:** `data/processed/pipeline1_features_stream1.parquet` (N = 324,054)
**Artefact:** `reports/selection_function/ir_completeness_v1.parquet` (4-D grid + |b|×G marginal stacked in two tables; this sidecar lists the 4-D).
**Scorer module:** `src/arqueogal/data/selection_function.py` → `score_ir_completeness(b_deg, g_mag, teff, logg)`, `score_compound_selection_prob(...)`.

---

## 1. Definition

A row is **IR-complete** iff all of `j_mag`, `h_mag`, `k_mag`, `w1_mag`, `w2_mag` are finite and non-zero. The zero-sentinel is excluded because downstream inference uses `nan_to_num(0.0)` on missing IR rows, so at inference time `mag == 0` is indistinguishable from "no counterpart"; both conditions are enforced for definitional transferability from training to inference domains.

**Stream 1 global P(IR-complete) = 99.6319 %** (N_complete = 322,861 / N_total = 324,054).

The IR-dependency diagnostic referenced a ~99.9 % training-domain completeness heuristic; the empirical value is 99.63 %, in the same ballpark but slightly lower, driven by faint-in-plane stars.

## 2. Binning

Four-dimensional grid, matching the v1 Ye-retention |b|×G axes for compositional ease:

- `|b|` (deg): [0.0, 5.0, 10.0, 20.0, 45.0, 90.0]  → 5 bins
- `G`   (mag): [2.0, 11.0, 12.5, 14.0, 15.5, 17.65]  → 5 bins
- `Teff` (K):  [3000.0, 4400.0, 4900.0, 6500.0]  → 3 bins (cool / mid / warm giants)
- `log g`:     [0.0, 2.5, 5.0]  → 2 bins (luminous giants / lower-RGB+RC)

Total possible cells: 150 (5×5×3×2). Populated cells: **145**. Dense cells (n ≥ 100): **112**. Sparse cells (below threshold, scorer backs off to |b|×G marginal at runtime): **33**.

Per-cell Laplace smoothing: `p_ir_complete = (n_complete + 1) / (n_total + 2)`, then clipped to `[0.01, 1]`. The Laplace correction prevents any cell from scoring exactly 0 or 1; the floor/ceil structural.

## 3. |b|×G marginal (N_complete / N_total per cell)

The 25-cell |b|×G marginal is the operational fallback when Teff/log g are unavailable at inference (e.g., before Pipeline 1 predicts them, or for sparse 4-D cells). It is also the compositional partner of the Ye v1 retention table.

| `|b|` (deg) | `G` (mag) | N_total | N_complete | rate | p_ir_complete (Laplace) |
|---:|---:|---:|---:|---:|---:|
| [   0,    5) | [ 2.00, 11.00) |    990 |    930 | 0.9394 | 0.9385 |
| [   0,    5) | [11.00, 12.50) |   4898 |   4779 | 0.9757 | 0.9755 |
| [   0,    5) | [12.50, 14.00) |  18542 |  18431 | 0.9940 | 0.9940 |
| [   0,    5) | [14.00, 15.50) |  28100 |  27923 | 0.9937 | 0.9937 |
| [   0,    5) | [15.50, 17.65) |  16604 |  16342 | 0.9842 | 0.9842 |
| [   5,   10) | [ 2.00, 11.00) |   1291 |   1278 | 0.9899 | 0.9892 |
| [   5,   10) | [11.00, 12.50) |   5327 |   5309 | 0.9966 | 0.9964 |
| [   5,   10) | [12.50, 14.00) |  15377 |  15350 | 0.9982 | 0.9982 |
| [   5,   10) | [14.00, 15.50) |  18594 |  18564 | 0.9984 | 0.9983 |
| [   5,   10) | [15.50, 17.65) |   4948 |   4938 | 0.9980 | 0.9978 |
| [  10,   20) | [ 2.00, 11.00) |   3733 |   3710 | 0.9938 | 0.9936 |
| [  10,   20) | [11.00, 12.50) |  13080 |  13047 | 0.9975 | 0.9974 |
| [  10,   20) | [12.50, 14.00) |  23984 |  23926 | 0.9976 | 0.9975 |
| [  10,   20) | [14.00, 15.50) |  16312 |  16254 | 0.9964 | 0.9964 |
| [  10,   20) | [15.50, 17.65) |   1090 |   1086 | 0.9963 | 0.9954 |
| [  20,   45) | [ 2.00, 11.00) |  22082 |  22038 | 0.9980 | 0.9980 |
| [  20,   45) | [11.00, 12.50) |  47170 |  47138 | 0.9993 | 0.9993 |
| [  20,   45) | [12.50, 14.00) |  36241 |  36205 | 0.9990 | 0.9990 |
| [  20,   45) | [14.00, 15.50) |   4499 |   4485 | 0.9969 | 0.9967 |
| [  20,   45) | [15.50, 17.65) |    551 |    549 | 0.9964 | 0.9946 |
| [  45,   90) | [ 2.00, 11.00) |   9983 |   9964 | 0.9981 | 0.9980 |
| [  45,   90) | [11.00, 12.50) |  16251 |  16235 | 0.9990 | 0.9990 |
| [  45,   90) | [12.50, 14.00) |  12332 |  12310 | 0.9982 | 0.9981 |
| [  45,   90) | [14.00, 15.50) |   1895 |   1890 | 0.9974 | 0.9968 |
| [  45,   90) | [15.50, 17.65) |    180 |    180 | 1.0000 | 0.9945 |

## 4. Structure beyond |b|×G?

Across the **4-D grid**, the spread in `p_ir_complete` at constant (|b|, G) — i.e., the residual variation unlocked by Teff, log g stratification — is summarised by the worst/best cells:

- **Worst** (lowest p_ir_complete): |b| ∈ [20, 45), G ∈ [14.00, 15.50), Teff ∈ [3000, 4400), log g ∈ [2.50, 5.00] → **p = 0.6667** (N = 1, complete = 1, sparse — backs off to marginal).
- **Best** (highest p_ir_complete): |b| ∈ [20, 45), G ∈ [11.00, 12.50), Teff ∈ [4400, 4900), log g ∈ [2.50, 5.00] → **p = 0.9997** (N = 15861).

Spread (max − min) across populated dense cells: **0.0741** in probability units. This is load-bearing for the plane-faint corner and a near-no-op elsewhere — broadly the same structure the Ye-retention component exhibits.

## 5. Compound v1.1 |b|×G table (`p_compound = p_ye · p_ir`)

Joined view of the v1 Ye retention and the v1 IR-completeness marginal. Used directly by `score_compound_selection_prob` when Teff/log g are unavailable, and carried in the v1.1 Parquet artefact for release.

| `|b|` (deg) | `G` (mag) | p_ye_retained | p_ir_complete | p_compound_bg |
|---:|---:|---:|---:|---:|
| [   0,    5) | [ 2.00, 11.00) | 1.0000 | 0.9385 | 0.9385 |
| [   0,    5) | [11.00, 12.50) | 1.0000 | 0.9755 | 0.9755 |
| [   0,    5) | [12.50, 14.00) | 0.9999 | 0.9940 | 0.9939 |
| [   0,    5) | [14.00, 15.50) | 0.9807 | 0.9937 | 0.9745 |
| [   0,    5) | [15.50, 17.65) | 0.5966 | 0.9842 | 0.5872 |
| [   5,   10) | [ 2.00, 11.00) | 1.0000 | 0.9892 | 0.9892 |
| [   5,   10) | [11.00, 12.50) | 1.0000 | 0.9964 | 0.9964 |
| [   5,   10) | [12.50, 14.00) | 1.0000 | 0.9982 | 0.9982 |
| [   5,   10) | [14.00, 15.50) | 0.9969 | 0.9983 | 0.9953 |
| [   5,   10) | [15.50, 17.65) | 0.8159 | 0.9978 | 0.8141 |
| [  10,   20) | [ 2.00, 11.00) | 1.0000 | 0.9936 | 0.9936 |
| [  10,   20) | [11.00, 12.50) | 1.0000 | 0.9974 | 0.9974 |
| [  10,   20) | [12.50, 14.00) | 1.0000 | 0.9975 | 0.9975 |
| [  10,   20) | [14.00, 15.50) | 0.9988 | 0.9964 | 0.9952 |
| [  10,   20) | [15.50, 17.65) | 0.8294 | 0.9954 | 0.8256 |
| [  20,   45) | [ 2.00, 11.00) | 1.0000 | 0.9980 | 0.9980 |
| [  20,   45) | [11.00, 12.50) | 1.0000 | 0.9993 | 0.9993 |
| [  20,   45) | [12.50, 14.00) | 1.0000 | 0.9990 | 0.9990 |
| [  20,   45) | [14.00, 15.50) | 1.0000 | 0.9967 | 0.9967 |
| [  20,   45) | [15.50, 17.65) | 0.9619 | 0.9946 | 0.9567 |
| [  45,   90) | [ 2.00, 11.00) | 1.0000 | 0.9980 | 0.9980 |
| [  45,   90) | [11.00, 12.50) | 1.0000 | 0.9990 | 0.9990 |
| [  45,   90) | [12.50, 14.00) | 1.0000 | 0.9981 | 0.9981 |
| [  45,   90) | [14.00, 15.50) | 1.0000 | 0.9968 | 0.9968 |
| [  45,   90) | [15.50, 17.65) | 0.9944 | 0.9945 | 0.9890 |

## 6. Scoring protocol

Given `(b_deg, g_mag, teff, logg)`, the scorer takes `|b|`, looks up the 4-D cell, and returns the smoothed `p_ir_complete`. Fallbacks:

- If the 4-D cell is sparse (n < 100), return the |b|×G marginal value.
- If `teff` or `logg` is NaN / missing at call time, return the |b|×G marginal value directly (no 4-D lookup attempted).
- Out-of-range `|b|`, `G`, `teff`, `logg` are clamped to the nearest edge.

## 7. Compound probability contract

```
p_compound = p_ye_retained · p_ir_complete · p_parallax · p_extinction
```

`p_parallax` and `p_extinction` are simple 0/1 gates in v1.1 (True → 1.0, False → 0.0). They are placeholders for later work; the user's requirement is the Ye × IR composition. All four are clamped to `[0.0001, 1]` — the joint floor is the product of the two component floors, so inverse weights at the plane-faint corner remain finite.

## 8. Build + provenance

- **Builder:** `scripts/build_selection_function_v11.py`. Deterministic.
- **Inputs:** `data/processed/pipeline1_features_stream1.parquet` (read-only; SHA-256 in sidecar); v1 Ye retention grid at `reports/selection_function/selection_function_v1.parquet` (read-only).
- **Atomic writes** on all three outputs (Parquet + MD + provenance JSON).
- **Reproduction:** `PYTHONPATH=src python scripts/build_selection_function_v11.py`.

## 9. Known limitations

1. **Parallax + extinction components are 0/1 gates.** v1.2 should replace them with smooth per-star availability or uncertainty-weighted probabilities.
2. **Stream 1 basis.** The IR-completeness table is computed on the Stream 1 (APOGEE × Gaia XP) joint selection. Stream 3 at the XP-native faint end may sample slightly different (|b|, G, Teff, log g) joint distributions; a cross-check at first Stream 3 ingestion is scheduled (same protocol as v1).
3. **Piecewise-constant inside each 4-D cell.** A star one bin-width inside a cell gets the same score as one right at the edge. Acceptable at this stratification depth — smoothing is v1.3 work if the spread ever grows.
4. **Teff bin granularity is coarse (3 bins).** If future data ingestion produces material 4-D structure inside any of the three Teff bins, refine.

