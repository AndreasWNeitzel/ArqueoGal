# Input Validation Audit: ArqueoGal Data Module

Date: 2026-04-26  
Scope: `src/arqueogal/data/` boundary validation (disk reads, HTTP downloads, CLI flags)

---

## Summary

The data module demonstrates solid structural validation (parquet schema contracts via `master_schema.py`, required-column checks at module boundaries) but exhibits gaps in numeric bounds checking on user-controllable parameters and missing source_id type guards in several ingestion paths. Percentile cutoff validation exists in kinematic OOD but lacks defensive range checks in per-element release-tier functions.

---

## Findings

### CRITICAL: Missing source_id dtype validation at ingest boundary

**Files affected:**
- `src/arqueogal/data/ingest_xp.py:38` — casts to `int64` after fetch, but no upstream type guard
- `src/arqueogal/data/enrich_geometry.py` — `.astype(np.int64)` after read, assumes data is already numeric
- `src/arqueogal/data/ir_photometry.py:126–140` — `np.asarray(expected_ids, dtype=np.int64)` without validating input iterable is non-empty or contains valid integers

**Risk:** Silent truncation if source_ids arrive as float64 or str; no pre-ingest validation that IDs are in the valid Gaia DR3 range (≥0, ≤2^63 - 1).

**Mitigation:** Add an early-stage `validate_source_ids` function that:
1. Checks dtype is int64 or coercible without loss
2. Rejects NaN, negative, or zero values
3. Flags duplicates if the contract requires uniqueness (used in `gaia_enrich.py` docstring)

---

### HIGH: Missing numeric bounds on user CLI flags

**Files affected:**
- `src/arqueogal/data/stream3_selection.py:97–98` — `per_cell` parameter validated as positive (`per_cell <= 0` raises ValueError at line 127) but no upper bound (e.g., `per_cell > 1_000_000` could exhaust memory)
- `src/arqueogal/data/stream3_selection.py:387–389` — `distance_cut_kpc` and `n_target` validated as positive but no upper bounds (e.g., `distance_cut_kpc = 1e10` or `n_target = 999_999_999`)
- `src/arqueogal/xp_abundances/main/kinematic_ood.py:137` — `p_threshold: float = 0.99` accepts any float; no guard that `0 < p_threshold < 1` (e.g., `p_threshold = 1.5` would compute a nonsensical percentile)

**Risk:** Memory exhaustion, nonsensical quantile computation, or OOM in downstream operations.

**Mitigation:** Wrap flags in `[min, max]` range checks. Example:
```python
if not (0 < p_threshold < 1):
    raise ValueError(f"p_threshold must be in (0, 1), got {p_threshold}")
if per_cell < 1 or per_cell > 100_000:
    raise ValueError(f"per_cell must be in [1, 100_000], got {per_cell}")
```

---

### HIGH: Missing parquet schema validation at read time (inconsistent application)

**Files affected:**
- `src/arqueogal/data/stream3_selection.py:123–126` — validates required columns but does **not** call `master_schema.validate()` on the input frame; implicitly assumes caller has already validated against schema
- `src/arqueogal/data/ingest_stream3.py:100–101` — calls `load_andrae2023()` without validating the result against an expected schema before passing to `stratified_subsample`
- `src/arqueogal/data/release_pipeline.py:80–100` — reads parquets with `pd.read_parquet()` without calling the schema validator; relies on downstream `.astype()` casts to catch dtype mismatches

**Risk:** Silent dtype mismatches (e.g., float32 instead of float64) propagate to analysis; NaN columns accepted without audit.

**Mitigation:**
1. Define lightweight ingestion schemas (subset of `master_schema`) for intermediate formats (Andrae+2023, AIP responses)
2. Call `schema.validate(df, check_array_lengths=True)` immediately after every parquet/FITS/TAP read
3. Centralize in a guard decorator `@validate_schema(PIPELINE1_INFERENCE_SCHEMA)`

---

### MEDIUM: No defensive bounds on Mahalanobis percentile in OOD functions

**Files affected:**
- `src/arqueogal/xp_abundances/main/kinematic_ood.py:134–180` — `p_threshold` parameter lacks validation; also `from_dict()` at line 122 does **not** verify that `p_threshold ∈ (0, 1)` when deserializing from JSON (silent corruption if a checkpoint has `p_threshold = 1.5`)

**Risk:** Nonsensical percentile thresholds silently corrupt the OOD bundle; downstream tier assignment uses invalid threshold.

**Mitigation:** Add a `__post_init__` or explicit validation:
```python
@dataclass
class KinematicOODBundle:
    ...
    def __post_init__(self):
        if not (0 < self.p_threshold < 1):
            raise ValueError(f"p_threshold must be in (0, 1), got {self.p_threshold}")
```

---

### MEDIUM: Missing batch-size bounds in enrichment operations

**Files affected:**
- `src/arqueogal/data/ingest_stream3.py:55` — `enrich_batch_size: int = 10_000` parameter accepted without bounds; no guard that `enrich_batch_size > 0` or ≤ AIP's 100 KB inline-IN limit (~5k IDs depending on precision)

**Risk:** If user passes `enrich_batch_size = 50_000`, the TAP query will exceed the 100 KB inline-IN limit and fail at runtime (AIP-specific footgun documented in CLAUDE.md).

**Mitigation:** Add explicit bounds with a warning if the size approaches the TAP limit:
```python
if enrich_batch_size < 1:
    raise ValueError(f"enrich_batch_size must be >= 1, got {enrich_batch_size}")
if enrich_batch_size > 5_000:
    logger.warning(
        "enrich_batch_size %d may exceed AIP TAP inline-IN 100 KB limit; "
        "consider reducing to ~5k or using tap.batched_upload_fetch_df()",
        enrich_batch_size,
    )
```

---

### MEDIUM: Parquet artifact path existence checks incomplete

**Files affected:**
- `src/arqueogal/data/selection_function.py:92–113` — `_load_grid()` reads parquet with `pd.read_parquet(artifact_path)` without checking path existence first; if the file is missing, pandas raises an opaque I/O error
- `src/arqueogal/data/frozen_stats.py:177–180` — `load_frozen_zscore_stats()` checks the JSON file exists implicitly (raises KeyError), but does **not** validate that `provenance_path` is a `.provenance.json` file (could be any text file, leading to confusing error messages)

**Risk:** Opaque error messages if required artifacts are missing or corrupted; no pre-flight check on application startup.

**Mitigation:** Add explicit existence and format checks:
```python
def _load_grid(artifact_path: str) -> tuple[...]:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"selection-function artifact not found: {path}")
    if path.suffix != ".parquet":
        raise ValueError(f"expected .parquet file, got {path.suffix}")
    ...
```

---

### LOW: Bin-edge validation missing in stratification

**Files affected:**
- `src/arqueogal/data/stream3_selection.py:86–241` — `bins_teff`, `bins_logg`, `bins_mh`, `bins_g` parameters are accepted without validation; no check that edges are monotone increasing or non-empty

**Risk:** If user passes unsorted or empty bins, `np.digitize()` returns meaningless indices; stratification silently fails or over-samples one cell.

**Mitigation:**
```python
def stratified_subsample(...):
    for name, bins in [("teff", bins_teff), ("logg", bins_logg), ("mh", bins_mh), ("g", bins_g)]:
        bins_arr = np.asarray(bins)
        if len(bins_arr) < 2:
            raise ValueError(f"{name} bins must have at least 2 edges, got {len(bins_arr)}")
        if not np.all(np.diff(bins_arr) > 0):
            raise ValueError(f"{name} bins must be strictly increasing, got {bins_arr}")
```

---

### LOW: Missing validation on argparse-sourced Path arguments

**Files affected:**
- `src/arqueogal/scripts/assign_release_tier.py:38–42` — `--target` argument is typed as `Path` but no existence check; the script logs a warning if the path doesn't exist but continues (line 52) — this is intentional but could mask typos
- `src/arqueogal/scripts/export_catalog_to_fits.py:54–60` — `--input` and `--output` arguments are bare strings; no validation that `--input` exists or is a parquet file

**Risk:** Silent skips or opaque errors if CLI user mistypes a path; FITS export may overwrite unintended files.

**Mitigation:** Use argparse custom type:
```python
def path_must_exist(s: str) -> Path:
    p = Path(s)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {p}")
    return p

parser.add_argument("--input", type=path_must_exist, ...)
```

---

## Scope Exclusions

The following are **not** covered by this audit (out of scope for input-validation focused review):

- Downstream inference-path NaN-handling in `xp_abundances/` (covered separately; see CLAUDE.md footgun on `nan_to_num` boundary)
- FITS/VOTable export column selection (covered in export script)
- Gaia DR3 astrometric correlation-matrix validation (assumed valid from TAP)
- Stellar-label NaN rates in APOGEE (documented as expected variation)

---

## Recommended Priority

1. **CRITICAL → Source_id type guard:** Essential for data integrity; 2h to implement, high impact.
2. **HIGH → Numeric bounds on user flags:** Prevents OOM; straightforward checks, 1h.
3. **HIGH → Schema validation at ingest:** Catch dtype mismatches early; 3h to refactor read paths.
4. **MEDIUM → Percentile bounds:** Prevent invalid OOD thresholds; 30m per affected function.
5. **LOW → Bin-edge validation:** Defensive; 1h.

---

## Test Coverage Notes

- `src/arqueogal/data/master_schema.py` has an `__post_init__` validator for `FrozenZScoreStats` (line 141–153) — good pattern to replicate elsewhere
- Stratification has a 100-row smoke test in `tests/data/` but does **not** test out-of-bounds parameters or unsorted bins
- Frozen-stats loading has explicit KeyError cascades (good) but lacks schema validation on deserialized NumPy arrays

