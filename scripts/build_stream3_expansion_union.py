"""Phase 3a — Stream 3 Option C expansion: build the source_id union.

Produces two new Stream-3 sub-samples on top of the Andrae+2023 vetted RGB
catalogue (10.48 M stars) with on-disk BJ21 photogeometric distances:

- **uniform** — ~400 k stars, stratified in ``(Teff, log g, [M/H], G)``.
  The existing 168,099 Stream-3 selection (``stream3_selected_source_ids``,
  ``per_cell=600``, ``rng_seed=0``) is reused as a prior subset; the delta is
  drawn from the remaining Andrae pool with a second stratified draw sized
  to add ~232 k so the union reaches 400 k.
- **volume_limited** — ~250 k stars drawn uniformly at random from the
  BJ21-qualified ``r_med_photogeo ≤ 2500 pc`` pool (4,020,951 rows) MINUS
  the uniform arm's source_ids, enforcing disjointness.

Outputs
-------
- ``data/interim/stream3_expansion_union.parquet``
    Columns ``source_id, sample, g_mag, b_deg, ra, dec, distance_pc,
    teff_andrae, logg_andrae, mh_andrae``.
    ``sample`` is ``"uniform"`` or ``"volume_limited"`` (mutually exclusive).
- ``data/interim/stream3_expansion_delta_source_ids.parquet``
    ``source_id`` only for the stars whose XP/Ye/IR must still be fetched
    (i.e. union MINUS the 168,099 already on disk). Consumed by
    ``scripts/fetch_gaia_xp_delta.py``, ``scripts/apply_ye2024_xp_delta.py``,
    and ``scripts/fetch_ir_photometry.py``.
- ``*.provenance.json`` sidecar for both.

Halt conditions
---------------
- delta > 600 k (selection went wrong, over-fill)
- delta < 300 k (over-reuse, something filtered too aggressively)

User ratification recap
-----------------------
- Natural density for downstream density-based clustering (volume-limited,
  no stratification oversample) per user instruction; if Starfold v1 shows a
  population under-represented, revisit in v2.
- Disjointness enforced: volume-limited is drawn from
  ``(BJ21 d≤2.5kpc pool) \\ uniform_source_ids``.
- Seed 0 throughout for reproducibility.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar
from arqueogal.data.stream3_selection import (
    DEFAULT_BINS_G,
    DEFAULT_BINS_LOGG,
    DEFAULT_BINS_MH,
    DEFAULT_BINS_TEFF,
    stratified_subsample,
    volume_limited_subsample,
)

REPO = Path(__file__).resolve().parents[1]

ANDRAE_PARQUET = REPO / "data" / "raw" / "andrae2023" / "andrae2023_rgb.parquet"
BJ21_PARQUET = REPO / "data" / "raw" / "bailer_jones_2021" / "andrae_pool_bj21.parquet"
EXISTING_STREAM3_IDS = REPO / "data" / "interim" / "stream3_selected_source_ids.parquet"

OUT_UNION = REPO / "data" / "interim" / "stream3_expansion_union.parquet"
OUT_DELTA_IDS = REPO / "data" / "interim" / "stream3_expansion_delta_source_ids.parquet"
LOG_PATH = REPO / "logs" / "stream3_expansion_union_20260419.log"

# Targets per task spec
N_UNIFORM_TARGET = 400_000
N_VOLUME_TARGET = 250_000
DISTANCE_CUT_KPC = 2.5
SEED = 0

# Halt thresholds
DELTA_MIN = 300_000
DELTA_MAX = 600_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stream3_union")


def _setup_file_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _galactic_b(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Compute Galactic latitude b [deg] from ICRS (ra, dec) in degrees.

    Uses astropy SkyCoord — vectorised, accurate to <1e-4 deg; good enough
    for the ``selection_prob`` stratification downstream (|b| bin edges at
    5 / 10 / 20 / 45 deg).
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return c.galactic.b.deg.astype(np.float32)


def main() -> None:  # noqa: PLR0912, PLR0915 — linear driver
    _setup_file_logging()
    t0 = time.time()
    logger.info("=== Phase 3a Stream 3 expansion — source_id union — start ===")

    # ------------------------------------------------------------------
    # 1. Inputs
    # ------------------------------------------------------------------
    for p in (ANDRAE_PARQUET, BJ21_PARQUET, EXISTING_STREAM3_IDS):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading Andrae+2023: %s", ANDRAE_PARQUET)
    andrae = pd.read_parquet(
        ANDRAE_PARQUET,
        columns=["source_id", "ra_deg", "dec_deg", "g_mag", "teff", "logg", "fe_h"],
    )
    andrae["source_id"] = andrae["source_id"].astype("int64")
    logger.info("  Andrae rows: %d", len(andrae))

    logger.info("loading BJ21: %s", BJ21_PARQUET)
    bj21 = pd.read_parquet(
        BJ21_PARQUET,
        columns=["source_id", "r_med_photogeo"],
    )
    bj21["source_id"] = bj21["source_id"].astype("int64")
    logger.info("  BJ21 rows: %d", len(bj21))

    logger.info("loading existing Stream 3 source_ids: %s", EXISTING_STREAM3_IDS)
    existing = pd.read_parquet(EXISTING_STREAM3_IDS, columns=["source_id"])
    existing["source_id"] = existing["source_id"].astype("int64")
    existing_ids = set(existing["source_id"].tolist())
    logger.info("  existing Stream 3 source_ids: %d", len(existing_ids))

    # Merge Andrae × BJ21 (left join — many Andrae stars have no BJ21 match,
    # which is fine for stratification but matters for the volume-limited arm).
    logger.info("left-joining Andrae × BJ21 on source_id")
    merged = andrae.merge(bj21, on="source_id", how="left")
    merged["distance_pc"] = merged["r_med_photogeo"].astype(np.float32)
    logger.info(
        "  n with BJ21 distance: %d / %d",
        merged["distance_pc"].notna().sum(), len(merged),
    )

    # ------------------------------------------------------------------
    # 2. Uniform arm: existing 168k + stratified fill to reach 400k
    # ------------------------------------------------------------------
    #
    # Strategy. The existing stratified selection used per_cell=600 on the
    # (teff, logg, fe_h, g_mag) grid. Its source_ids are kept as-is to
    # preserve disk reuse of XP+Ye+IR. We then run a second stratified draw
    # over the Andrae \\ existing pool with per_cell tuned so the total
    # uniform sample lands near 400 k. The two draws together are *not*
    # exactly the same as a single stratified draw at per_cell≈1400, but
    # they preserve the same stratification axes and cell-level balance,
    # and they reuse 100% of the existing Stream-3 on-disk Ye+IR.
    # ------------------------------------------------------------------

    n_fill = N_UNIFORM_TARGET - len(existing_ids)
    logger.info(
        "uniform arm: reuse %d existing + fill %d more → target %d",
        len(existing_ids), n_fill, N_UNIFORM_TARGET,
    )

    # Andrae minus existing — the pool for the fill stratification.
    fill_pool = merged.loc[~merged["source_id"].isin(existing_ids)].reset_index(drop=True)
    logger.info("  Andrae \\ existing pool size: %d", len(fill_pool))

    # First stratified call counted 9,337,992 available rows and placed
    # 168,099 per_cell=600 into 872 non-empty cells. To add ~232 k, the
    # fill_pool has roughly the same Andrae minus 168 k = 9.17 M available
    # rows. Per_cell ≈ 232 k / 872 ≈ 266. We search around that target.
    #
    # Since the available rows per cell after excluding existing are not
    # uniform, we calibrate by running stratified_subsample at per_cell=266
    # and then adjust once if the result overshoots/undershoots by > 10 %.

    def _fill_stratify(per_cell: int) -> pd.DataFrame:
        result = stratified_subsample(
            fill_pool,
            teff_col="teff",
            logg_col="logg",
            mh_col="fe_h",
            g_col="g_mag",
            bins_teff=DEFAULT_BINS_TEFF,
            bins_logg=DEFAULT_BINS_LOGG,
            bins_mh=DEFAULT_BINS_MH,
            bins_g=DEFAULT_BINS_G,
            per_cell=per_cell,
            rng_seed=SEED,
        )
        return result.sample

    per_cell_try = max(1, round(n_fill / 872))
    logger.info("  stratified fill per_cell estimate: %d", per_cell_try)
    fill_sample = _fill_stratify(per_cell_try)
    logger.info("  fill draw at per_cell=%d → %d rows", per_cell_try, len(fill_sample))

    # Bias-correct once if |miss| > 10 %.
    if abs(len(fill_sample) - n_fill) / max(n_fill, 1) > 0.10:
        scale = n_fill / max(len(fill_sample), 1)
        per_cell_try = max(1, round(per_cell_try * scale))
        fill_sample = _fill_stratify(per_cell_try)
        logger.info(
            "  rebalanced fill at per_cell=%d → %d rows",
            per_cell_try, len(fill_sample),
        )

    # Uniform arm = existing ∪ fill.
    existing_merged = merged.loc[merged["source_id"].isin(existing_ids)].reset_index(drop=True)
    uniform_sample_df = pd.concat([existing_merged, fill_sample], ignore_index=True)
    uniform_ids = uniform_sample_df["source_id"].astype("int64").to_numpy()
    uniform_ids_set = set(uniform_ids.tolist())
    logger.info(
        "uniform arm total: %d (existing reused: %d, fill: %d)",
        len(uniform_sample_df), len(existing_merged), len(fill_sample),
    )

    # Tag.
    uniform_sample_df["sample"] = "uniform"

    # ------------------------------------------------------------------
    # 3. Volume-limited arm: uniform draw from BJ21 d≤2.5kpc \\ uniform
    # ------------------------------------------------------------------
    vol_pool = merged.loc[
        (~merged["source_id"].isin(uniform_ids_set))
        & (merged["distance_pc"].notna())
    ].reset_index(drop=True)
    # Convert pc → kpc for the subsampler contract (kpc).
    vol_pool["distance_kpc"] = vol_pool["distance_pc"].to_numpy(dtype=np.float32) / 1000.0
    logger.info("volume-limited pool (Andrae \\ uniform, with BJ21): %d", len(vol_pool))

    vol_result = volume_limited_subsample(
        vol_pool,
        distance_col="distance_kpc",
        distance_cut_kpc=DISTANCE_CUT_KPC,
        n_target=N_VOLUME_TARGET,
        seed=SEED,
    )
    vol_sample_df = vol_result.sample.copy()
    vol_sample_df["sample"] = "volume_limited"
    logger.info(
        "volume-limited arm: %d below cut, %d selected (target %d)",
        vol_result.n_below_cut, vol_result.n_selected, N_VOLUME_TARGET,
    )

    # ------------------------------------------------------------------
    # 4. Build union + enforce schema
    # ------------------------------------------------------------------
    keep_cols = [
        "source_id", "sample",
        "g_mag", "ra_deg", "dec_deg",
        "distance_pc",
        "teff", "logg", "fe_h",
    ]

    uniform_out = uniform_sample_df[keep_cols].copy()
    vol_out = vol_sample_df[keep_cols].copy()
    union = pd.concat([uniform_out, vol_out], ignore_index=True)
    logger.info("union rows (uniform + volume_limited): %d", len(union))

    # Disjointness sanity check.
    dup = union["source_id"].duplicated().sum()
    if dup:
        raise SystemExit(
            f"BUG: {dup} source_ids appear in both uniform and volume_limited arms"
        )
    logger.info("disjointness confirmed: 0 source_ids in both arms")

    # Compute Galactic b for the selection-prob stratification downstream.
    logger.info("computing Galactic latitude b_deg")
    b_deg = _galactic_b(
        union["ra_deg"].to_numpy(dtype=float),
        union["dec_deg"].to_numpy(dtype=float),
    )
    union["b_deg"] = b_deg.astype(np.float32)

    # Rename to the schema specified in the task.
    union = union.rename(
        columns={
            "ra_deg": "ra",
            "dec_deg": "dec",
            "teff": "teff_andrae",
            "logg": "logg_andrae",
            "fe_h": "mh_andrae",
        }
    )
    union = union[[
        "source_id", "sample",
        "g_mag", "b_deg", "ra", "dec", "distance_pc",
        "teff_andrae", "logg_andrae", "mh_andrae",
    ]]
    # Type tightening.
    union["source_id"] = union["source_id"].astype("int64")
    for c in ("g_mag", "b_deg", "ra", "dec", "distance_pc",
              "teff_andrae", "logg_andrae", "mh_andrae"):
        union[c] = union[c].astype(np.float32)
    union["sample"] = union["sample"].astype("category")

    # ------------------------------------------------------------------
    # 5. Delta source_ids (XP/Ye/IR fetch input)
    # ------------------------------------------------------------------
    delta_mask = ~union["source_id"].isin(existing_ids)
    delta_ids = union.loc[delta_mask, "source_id"].astype("int64").to_numpy()
    n_delta = int(len(delta_ids))
    n_uniform = int((union["sample"] == "uniform").sum())
    n_uniform_reused = int(union["sample"].eq("uniform").sum() - (delta_mask & union["sample"].eq("uniform")).sum())
    n_uniform_new = int((delta_mask & union["sample"].eq("uniform")).sum())
    n_vol = int((union["sample"] == "volume_limited").sum())
    n_vol_new = int((delta_mask & union["sample"].eq("volume_limited")).sum())

    logger.info(
        "delta to fetch (XP+Ye+IR): %d "
        "(uniform_new=%d, volume_limited_new=%d)",
        n_delta, n_uniform_new, n_vol_new,
    )
    logger.info(
        "breakdown: uniform total=%d (reused=%d + new=%d), volume_limited total=%d (all new=%d)",
        n_uniform, n_uniform_reused, n_uniform_new, n_vol, n_vol_new,
    )

    # Halt conditions BEFORE writing outputs.
    if n_delta > DELTA_MAX:
        raise SystemExit(
            f"HALT: delta={n_delta} > {DELTA_MAX} — selection went wrong"
        )
    if n_delta < DELTA_MIN:
        raise SystemExit(
            f"HALT: delta={n_delta} < {DELTA_MIN} — over-reuse / under-fill"
        )
    logger.info("halt conditions: all clear (delta in [%d, %d])", DELTA_MIN, DELTA_MAX)

    # ------------------------------------------------------------------
    # 6. Atomic writes
    # ------------------------------------------------------------------
    _write_parquet_atomic(union, OUT_UNION)
    union_sha = _sha256_of(OUT_UNION)
    union_size_mb = OUT_UNION.stat().st_size / 1024**2
    logger.info(
        "wrote %s (%.2f MB, sha256=%s…)",
        OUT_UNION, union_size_mb, union_sha[:12],
    )

    delta_df = pd.DataFrame({"source_id": delta_ids}).sort_values("source_id").reset_index(drop=True)
    _write_parquet_atomic(delta_df, OUT_DELTA_IDS)
    delta_sha = _sha256_of(OUT_DELTA_IDS)
    delta_size_mb = OUT_DELTA_IDS.stat().st_size / 1024**2
    logger.info(
        "wrote %s (%.2f MB, sha256=%s…)",
        OUT_DELTA_IDS, delta_size_mb, delta_sha[:12],
    )

    # ------------------------------------------------------------------
    # 7. Provenance sidecars
    # ------------------------------------------------------------------
    def _now_iso() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    prov_union_extra: dict[str, Any] = {
        "phase": "Phase 3a — Stream 3 Option C expansion",
        "rng_seed": SEED,
        "distance_cut_kpc": DISTANCE_CUT_KPC,
        "uniform_target": N_UNIFORM_TARGET,
        "uniform_total": n_uniform,
        "uniform_existing_reused": n_uniform_reused,
        "uniform_fill_new": n_uniform_new,
        "uniform_fill_per_cell": per_cell_try,
        "uniform_stratification_axes": ["teff", "logg", "fe_h", "g_mag"],
        "uniform_bins_teff": DEFAULT_BINS_TEFF.tolist(),
        "uniform_bins_logg": DEFAULT_BINS_LOGG.tolist(),
        "uniform_bins_mh": DEFAULT_BINS_MH.tolist(),
        "uniform_bins_g": DEFAULT_BINS_G.tolist(),
        "volume_limited_target": N_VOLUME_TARGET,
        "volume_limited_total": n_vol,
        "volume_limited_pool_size": int(vol_result.n_below_cut),
        "volume_limited_total_andrae_le_2500pc": int(
            (merged["distance_pc"].notna() & (merged["distance_pc"] <= 2500.0)).sum()
        ),
        "union_total_rows": int(len(union)),
        "disjointness_duplicates": int(dup),
        "delta_total": n_delta,
        "delta_from_uniform_arm": n_uniform_new,
        "delta_from_volume_limited_arm": n_vol_new,
        "halt_conditions_tripped": {
            "delta_over_max": n_delta > DELTA_MAX,
            "delta_under_min": n_delta < DELTA_MIN,
        },
        "built_at_utc": _now_iso(),
        "wallclock_seconds": time.time() - t0,
    }

    prov_union = Provenance(
        output_file=str(OUT_UNION.relative_to(REPO)),
        script="scripts/build_stream3_expansion_union.py",
        sources=[
            LocalSource(
                name="Andrae+2023 vetted RGB catalogue (VizieR J/MNRAS/537/1984)",
                path=str(ANDRAE_PARQUET.relative_to(REPO)),
                sha256=_sha256_of(ANDRAE_PARQUET),
            ),
            LocalSource(
                name="BJ21 photogeometric distances for the Andrae pool",
                path=str(BJ21_PARQUET.relative_to(REPO)),
                sha256=_sha256_of(BJ21_PARQUET),
            ),
            LocalSource(
                name="Existing Stream 3 stratified source_ids (per_cell=600, seed=0)",
                path=str(EXISTING_STREAM3_IDS.relative_to(REPO)),
                sha256=_sha256_of(EXISTING_STREAM3_IDS),
            ),
        ],
        cuts_applied=[
            "uniform arm: existing 168k ∪ stratified_subsample(Andrae \\ existing, "
            f"per_cell≈{per_cell_try}, seed=0) on (teff, logg, fe_h, g_mag)",
            (
                "volume_limited arm: volume_limited_subsample("
                "Andrae \\ uniform, distance_col='distance_kpc', "
                f"distance_cut_kpc={DISTANCE_CUT_KPC}, n_target={N_VOLUME_TARGET}, seed=0)"
            ),
            "disjointness: volume_limited pool excludes all uniform source_ids",
        ],
        corrections=[
            "distance_pc = BJ21 r_med_photogeo (float32)",
            "distance_kpc = distance_pc / 1000 (used only for volume subsampler)",
            "b_deg = astropy SkyCoord(ra, dec).galactic.b.deg (float32)",
            "float32 cast on g_mag, b_deg, ra, dec, distance_pc, teff, logg, fe_h",
            "source_id int64, sample category",
        ],
        row_count_before=int(len(andrae)),
        row_count_after=int(len(union)),
        notes=(
            "Phase 3a source_id union for Stream 3 Option C expansion. "
            "Uniform arm targets 400 k stratified stars for Pipeline 1 "
            "inference; volume-limited arm targets 250 k natural-density "
            "stars with d ≤ 2.5 kpc for downstream density-based clustering "
            "in Starfold (separate repo). User-ratified "
            "under the 10 GB disk ceiling: no stratification oversampling "
            "on the volume-limited arm; seed 0; disjointness enforced."
        ),
        extra=prov_union_extra,
    )
    write_sidecar(prov_union)
    logger.info("wrote provenance: %s", OUT_UNION.with_suffix(".provenance.json"))

    prov_delta = Provenance(
        output_file=str(OUT_DELTA_IDS.relative_to(REPO)),
        script="scripts/build_stream3_expansion_union.py",
        sources=[
            LocalSource(
                name="Phase 3a union",
                path=str(OUT_UNION.relative_to(REPO)),
                sha256=union_sha,
            ),
            LocalSource(
                name="Existing Stream 3 stratified source_ids (to be reused, not re-fetched)",
                path=str(EXISTING_STREAM3_IDS.relative_to(REPO)),
                sha256=_sha256_of(EXISTING_STREAM3_IDS),
            ),
        ],
        cuts_applied=[
            "delta = union.source_id \\ existing_stream3_selected.source_id",
        ],
        corrections=[],
        row_count_before=int(len(union)),
        row_count_after=int(len(delta_df)),
        notes=(
            "XP + Ye + IR fetch input for Phase 3a. The existing 168,099-row "
            "Stream-3 XP+Ye+IR artefacts are reused intact; this delta file "
            "lists the net-new source_ids that scripts/fetch_gaia_xp_delta.py, "
            "scripts/apply_ye2024_xp_delta.py, and scripts/fetch_ir_photometry.py "
            "must cover to complete Phase 3a."
        ),
        extra={
            "n_delta": n_delta,
            "delta_from_uniform_arm": n_uniform_new,
            "delta_from_volume_limited_arm": n_vol_new,
            "existing_reused_source_ids": len(existing_ids),
            "output_sha256": delta_sha,
        },
    )
    write_sidecar(prov_delta)
    logger.info("wrote provenance: %s", OUT_DELTA_IDS.with_suffix(".provenance.json"))

    # ------------------------------------------------------------------
    # 8. Console summary
    # ------------------------------------------------------------------
    dt = time.time() - t0
    print("\n=== Phase 3a union summary ===")
    print(f"  Andrae pool:                      {len(andrae):>10,}")
    print(f"  BJ21 coverage (Andrae pool):      {bj21['source_id'].nunique():>10,}")
    print(f"  Existing Stream-3 reused:         {len(existing_ids):>10,}")
    print(f"  Uniform arm (total):              {n_uniform:>10,}")
    print(f"    ... existing reused:            {n_uniform_reused:>10,}")
    print(f"    ... new (stratified fill):      {n_uniform_new:>10,}")
    print(f"  Volume-limited arm (total):       {n_vol:>10,}")
    print(f"    ... all new:                    {n_vol_new:>10,}")
    print(f"  Union total:                      {len(union):>10,}")
    print(f"  Delta to fetch (XP+Ye+IR):        {n_delta:>10,}")
    print(f"    ... from uniform arm:           {n_uniform_new:>10,}")
    print(f"    ... from volume-limited arm:    {n_vol_new:>10,}")
    print(f"  Wall-clock:                       {dt:>10.1f} s")
    print(f"  Outputs:")
    print(f"    {OUT_UNION.relative_to(REPO)} ({union_size_mb:.2f} MB)")
    print(f"    {OUT_DELTA_IDS.relative_to(REPO)} ({delta_size_mb:.2f} MB)")
    print("=== done ===\n")
    logger.info("=== done in %.1f s ===", dt)


if __name__ == "__main__":
    main()
