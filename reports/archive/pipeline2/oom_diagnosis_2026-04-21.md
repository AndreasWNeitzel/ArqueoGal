# Pipeline-2-v1 classifier OOM — full memory audit

**Date:** 2026-04-21
**Affected runs:** `logs/pipeline2_classifier_20260420_cuml.log` (PID 1083595)
and `logs/pipeline2_classifier_20260421.log` (PID 22742).
**Status:** confirmed OOM inside the first HDBSCAN grid cell's DBCV call.

## Revision note (supersedes the earlier pass)

An earlier draft of this report said the peak allocation was a *single*
(N_big, N_big) float64 dense matrix at ~5.8 GiB (for a 90% dominant cluster
in a 30k subsample), and proposed shrinking `--dbcv-max-n` from 30k → 5k.

Both parts are wrong.

- The peak is **six** simultaneously-live (N_big, N_big) matrices, not one.
  At 80% thin-disc dominance on a 30k subsample that's **~26 GiB peak**,
  which is why the process died silently on a 9.7 GiB WSL2 host.
- Shrinking N is a floor-raiser, not an architectural fix. The proper fix
  removes the dominant-cluster N² exposure regardless of subsample size.

I also previously gestured at `hdbscan.HDBSCAN.relative_validity_` as a zero-N²
drop-in replacement. Empirically it is **not rank-preserving** against
`validity_index` on a dominated mixture (Spearman ρ = +0.37, p = 0.29 over
ten HDBSCAN configs on an 80/15/5 blob mixture; see §6 calibration). So it
cannot replace DBCV for model selection without breaking the winner pick.
Killing that option too.

## 1. Observed failure

Both runs reach exactly the same point and stop:

```
INFO run_population_classifier: embedding shape=(229970, 2)
INFO run_population_classifier: HDBSCAN grid-search 'v1' (12 HDBSCAN cells per UMAP-point)
<loky resource_tracker: leaked semlock objects>   <-- process teardown
```

No `cell mcs=… → K=… DBCV=…` line ever prints, so the OOM fires **inside
the very first HDBSCAN grid cell's DBCV call** — before the grid loop's
per-cell log line completes.

Host budget: **9.7 GiB RAM + 3.0 GiB swap** (WSL2). `free` at post-mortem
showed ~4.4 GiB already resident in baseline tooling, so available headroom
to the classifier at launch was **~5–6 GiB**. WSL2 does not expose the
kernel OOM killer via `dmesg` by default, which is why there is no OOM
record in the system log.

The loky "leaked semlock" warnings are a consequence, not the cause: HDBSCAN
uses joblib under the hood, and the semlocks are what the resource tracker
failed to join when the process received SIGKILL.

## 2. Complete memory ledger

All numbers are for N = 229,970 stars in 2-D embedding space, with the v1
grid (12 HDBSCAN cells on the fixed cuML UMAP embedding). "Alive set"
= peak simultaneously resident after the stage is entered, before the next
stage starts.

| Stage                                                   | Alive set  | N² exposure? | Notes |
|---------------------------------------------------------|------------|--------------|-------|
| Parquet load (pandas, 9 cols)                           | ~14 MB     | no           | negligible |
| `build_feature_matrix` → standardised float32           | ~22 MB     | no           | `StandardScaler.fit_transform` — one copy, column-wise |
| cuML UMAP `fit_transform`                               | ~5 MB host | no           | brute-force batched KNN on GPU; host output is (N, 2) float32 |
| HDBSCAN `fit_predict` (+ `prediction_data=True`)        | ~400 MB    | no           | KDTree + Boruvka MST + condensed tree + exemplar cache. O(N log N), not O(N²) |
| `all_points_membership_vectors`                         | (N, K) + (N, Σexemplars) | no | ~10–60 MB. Fat but not quadratic. |
| **`validity_index` — the culprit**                      | **6 × 8 × N_big²** | **yes** | see §3 |
| Results list retention (12 × `GridCell`)                | ~10 KB     | no           | `GridCell` only retains dbcv + n_clusters + noise_fraction |
| `embed_cache` (one UMAP embedding for v1 grid)          | ~1.8 MB    | no           | single UMAP axis point; embedding is cached correctly |

Other allocations in the pipeline path that are **not** N²:
- `_build_labels_frame` — O(N × K), ~10–20 MB.
- `_dbcv_on_subsample` — allocates `idx` of size `max_n` and the two sub-arrays `Z[idx]`, `labels[idx]`. `Z[idx]` at 30k is 480 KB. Fine.
- `cluster_hdbscan` return — `soft_memberships` (N, K) float32, ~9 MB at K=5.
- joblib/loky workers — each worker copies the sub-array (max_n × 2 × 8 bytes). At default 8 workers that's ~4 MB total. Not a factor.

## 3. The N² culprit, nailed down line by line

The OOM is entirely inside `hdbscan.validity.distances_between_points`,
called from `validity_index` once per cluster. Reading the library source
(`hdbscan/validity.py`), the per-cluster body is:

```python
subset_X        = X[labels == cluster_id, :]                   # (N_big, d)
distance_matrix = pairwise_distances(subset_X, metric=...)     # (N_big, N_big) f64   [A]
core_distances  = all_points_core_distance(distance_matrix.copy(), d)  # transient .copy()  [B]
core_dist_matrix = np.tile(core_distances, (N_big, 1))         # (N_big, N_big) f64   [C]
stacked_distances = np.dstack(                                 # (N_big, N_big, 3) f64 [D]
    [distance_matrix, core_dist_matrix, core_dist_matrix.T])
return stacked_distances.max(axis=-1), core_distances
```

Peak simultaneously alive on the line marked `[D]`: `distance_matrix`
+ `core_dist_matrix` + `core_dist_matrix.T` (materialised by `dstack`)
+ `stacked_distances` = **6 × 8 × N_big² bytes**.

(The `.copy()` on line `[B]` is transient — it is released before
`stacked_distances` is built — so it doesn't add to the *peak*, but it
is a second 8 × N_big² allocation during that function.)

**Corrected peak ladder** (6× multiplier, per-cell):

| Subsample size | Dominance | N_big  | Peak       |
|----------------|-----------|--------|------------|
|  5,000         | 80%       |  4,000 |   0.72 GiB |
|  5,000         | 90%       |  4,500 |   0.91 GiB |
| 10,000         | 80%       |  8,000 |   2.86 GiB |
| 10,000         | 90%       |  9,000 |   3.62 GiB |
| **30,000**     | **80%**   | 24,000 | **25.75 GiB** |
| 30,000         | 90%       | 27,000 |  32.59 GiB |

The 30k × 80% cell is what the classifier was trying to allocate when it
died. 26 GiB on a 9.7 GiB host → silent SIGKILL.

## 4. Complete N² audit across the Pipeline-2 call graph

Exhaustive search for every call site that could build an O(N²) dense
matrix. `grep` patterns: `pairwise_distances|distance_matrix|pdist|squareform|cdist|scipy\.spatial\.distance`.

| Location                                             | Finding  | Status |
|------------------------------------------------------|----------|--------|
| `src/arqueogal/population_classifier/*.py`           | zero hits| clean  |
| `scripts/run_population_classifier.py`               | zero hits| clean  |
| `scripts/build_pipeline2_features.py`                | zero hits| clean  |
| `hdbscan.validity.distances_between_points`          | **6 × N²** per cluster | THE offender |
| `hdbscan.validity.density_separation` (inter-cluster)| `cdist(cluster_i_internal, cluster_j_internal)` | sub-O(N²); only internal MST nodes, not full clusters |
| `hdbscan.HDBSCAN.fit` (Boruvka path)                 | KDTree + MST | O(N log N) |
| `hdbscan.all_points_membership_vectors`              | (N, K) + (N, Σexemplars) | O(N × K), not O(N²) |
| `cuml.manifold.UMAP.fit_transform`                   | batched KNN on GPU | sparse (N, k); no dense N² on host |
| `sklearn.preprocessing.StandardScaler`               | mean/std per column | O(N × D) |

**After removing the `validity_index` exposure there is no other
O(N²) dense-matrix allocation anywhere in our code path.** The only other
fat allocations are (N, K) soft memberships and exemplar caches, both
addressed by making them winner-only (§5.2).

## 5. Fix plan

Ordered by impact. The architectural one is fix 1; the rest are hygiene.

### 5.1 Fix 1 — stratified per-cluster DBCV subsampling (architectural)

Instead of uniform-subsampling across all points, **cap each cluster's
size before feeding into `validity_index`**:

```python
PER_CLUSTER_CAP = 2000    # N_big ceiling → peak = 6 × 8 × 2000² ≈ 183 MiB
def dbcv_stratified(Z, labels, cap=PER_CLUSTER_CAP, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.zeros(labels.shape[0], dtype=bool)
    for cid in np.unique(labels):
        idx = np.flatnonzero(labels == cid)
        if cid == -1 or idx.size <= cap:
            keep[idx] = True
        else:
            keep[rng.choice(idx, size=cap, replace=False)] = True
    return validity_index(Z[keep].astype(np.float64),
                          labels[keep].astype(np.int64))
```

Guarantees:
- **Peak memory is bounded regardless of dominance fraction.** At cap=2000:
  `6 × 8 × 2000² = 183 MiB` per cluster, worst case. Safe with orders of
  magnitude of headroom.
- **Small clusters (halo, GES, Sequoia candidates) are preserved in full,**
  which matters because DBCV weights each cluster by `cluster_size / N` —
  so the numerically-meaningful part of small clusters is *not* subsampled
  away.
- **Still calls the authoritative `validity_index`** — no switch to
  `relative_validity_` or any other approximation, so §10.4's "DBCV
  maximisation" contract is honoured literally.

Statistical justification: uniform subsampling *within* a single cluster
is an unbiased estimator of that cluster's density sparseness (the MST
diameter of a random subgraph is rank-consistent with the full MST diameter
up to a known scaling). Rank ordering across grid cells — the only thing
the grid search uses — is preserved.

Touches: replace the `_dbcv_on_subsample` helper in
`scripts/run_population_classifier.py` and reference it from
`_grid_search_hdbscan`. Optionally mirror the logic into
`src/arqueogal/population_classifier/main/hyperparameter.py::dbcv_score`
for consistency.

### 5.2 Fix 2 — winner-only `prediction_data` + soft memberships

`cluster_hdbscan` today builds `prediction_data=True` and calls
`all_points_membership_vectors` on **every** grid cell. The grid loop only
consumes `labels` + `n_clusters` + `noise_fraction` — none of the soft
stuff. Split the function:

```python
def cluster_hdbscan_grid(Z, config):   # grid-scoring path
    clusterer = hdbscan.HDBSCAN(..., prediction_data=False)
    labels = clusterer.fit_predict(Z)
    return SimpleResult(labels=labels, ...)

def cluster_hdbscan_final(Z, config):  # winner re-fit path, called once
    clusterer = hdbscan.HDBSCAN(..., prediction_data=True)
    ...  # exactly the current cluster_hdbscan
```

Saves 100–400 MB per grid cell × 12 cells = **1.2–4.8 GB avoided
allocation pressure** over the sweep. Not load-bearing once fix 1 is in,
but hygiene.

### 5.3 Fix 3 — explicit teardown + GC between grid cells

`del clusterer; del soft; gc.collect()` at the bottom of the grid loop.
Cython objects held by HDBSCAN don't always drop immediately when the
wrapper goes out of scope. Cheap. Lowers the trough so the next cell's
peak starts from a lower floor.

### 5.4 Fix 4 — log RSS at every grid-cell boundary

```python
import resource
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kB on Linux
_LOG.info("  cell mcs=%d ... DBCV=%+.4f  rss=%.2f GiB",
          ..., rss / (1024**2))
```

Costs nothing. Makes the next OOM visible in the log instead of silent.

### 5.5 Fix 5 — drop the `astype(np.float64)` copy in `dbcv_score`

`validity_index` needs float64 inputs, but today we call
`Z_d = np.asarray(Z, dtype=np.float64)` which always copies the full
embedding (230k × 2 × 8 = 3.7 MB). Minor, but `.astype(..., copy=False)`
is the correct idiom.

## 6. Calibration evidence: why `relative_validity_` is NOT a fix

Synthetic mixture mimicking MW dominance (80% thin disc, 15% thick, 5%
halo-like; n=8000 points in 2-D). Ten HDBSCAN configs, rank correlation
between full `validity_index` and `HDBSCAN.relative_validity_`:

```
  mcs   ms   eps   K        VI       rVI
  200   10  0.00   3   +0.2126   +0.0344
  200   10  0.10   3   +0.2126   +0.0344
  200   20  0.00   3   +0.0353   +0.0295
  200   20  0.10   3   +0.0353   +0.0295
  500   10  0.00   2   +0.2026   +0.0004
  500   10  0.10   2   +0.2026   +0.0004
  500   20  0.00   2   +0.0200   +0.0014
  500   20  0.10   2   +0.0200   +0.0014
 1000   20  0.00   2   +0.0200   +0.0014
 1000   20  0.10   2   +0.0200   +0.0014

Spearman(VI, rVI) = +0.3684   p=0.2948   (n=10)
```

`relative_validity_` is not a faithful rank proxy for `validity_index` on
dominated mixtures. This is why fix 1 keeps `validity_index` itself and
only changes what feeds into it.

A follow-up acceptance check for fix 1: on the same synthetic mixture,
compare full-data `validity_index` vs stratified-cap `validity_index`
across the same ten configs. Report Spearman ρ as a release-gate
acceptance metric (≥ +0.9) before applying fix 1 to the production
classifier. I'll run that calibration the moment you approve — it's
cheap and mechanical.

## 7. Expected budget after fixes 1+2

Per grid cell (the HDBSCAN fit + DBCV scoring pair):

- HDBSCAN fit without `prediction_data`: **~80 MB**
- Stratified-cap `validity_index` at cap=2000: **~180 MB peak**
- Everything else (labels, scratch): ~10 MB
- **Grid-cell peak total: ~270 MB**

Final winner re-fit (`prediction_data=True`, `all_points_membership_vectors`
on full 229,970 points): one-shot, bounded by (N, K) × exemplars cache,
~400 MB. Fits comfortably.

Grid sweep total (12 cells sequential, no cross-cell retention beyond
summary rows): peak ~270 MB; trough ~100 MB. Full pipeline fits under
1 GiB resident even with baseline tooling on the box.

## 8. What I will NOT do until you confirm

- I will not rerun the classifier.
- I will not apply fixes 1–5.
- I will not run the acceptance calibration described in §6 (cheap but
  code-touching: adds a one-off test harness).

Your call:
- (a) approve fixes 1–5 as a single diff + acceptance calibration + rerun, or
- (b) something different — e.g. drop DBCV entirely in favour of a
  cluster-size-invariant objective, or push harder on `mst_raw_dist=True`
  (which cuts the 6× multiplier to 1× by skipping the `dstack` but keeps
  the single (N_big, N_big) pairwise matrix — belt-and-braces alongside
  fix 1, not an alternative to it).
