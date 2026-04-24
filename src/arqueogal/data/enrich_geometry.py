"""Level-4 enrichment — distances + extinction.

Applies §7 (distances) and §8 (dust) to an already-enriched stream Parquet:

    input frame (source_id, ra, dec, ag_gspphot, …)
        → GAVO Bailer-Jones+2021 photogeometric distances
        → [optional] AIP StarHorse2 v2 (APOGEE-overlap cross-check only)
        → merge_distances (§7.3)  → dist_primary_pc / σ / dist_conflict
        → compose_av (§8.2)       → av_los / av_los_source
        → neighborhood_av_features (§8.3) → av_nbhd_median / std / n
        → write ``<basename>_geom.parquet``
        → write ``<basename>_geom.provenance.json``

The three 3D-dust queries are dependency-injected; in production the
caller passes :func:`get_default_queries` (needs ``dustmaps`` installed),
and tests pass plain callables. This decouples the orchestrator from the
map files on disk and makes the entire module offline-testable.

References
----------
data_acquisition.md §7, §8, §11 (Level 4), §14 (provenance).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.distances import (
    BAILERJONES_ADQL,
    DIST_CONFLICT_LOG_THRESHOLD,
    STARHORSE2_ADQL_TEMPLATE,
    STARHORSE2_SAMPLE_TABLES,
    fetch_bailerjones,
    fetch_starhorse2,
    merge_distances,
)
from arqueogal.data.dust_maps import (
    DEFAULT_NEIGHBORHOOD_RADIUS_PC,
    FAR_BOUNDARY_KPC,
    MIN_NEIGHBORS_FOR_MEDIAN,
    NEAR_BOUNDARY_KPC,
    SFD_TO_AV_COEFF,
    DustQuery,
    compose_av,
    neighborhood_av_features,
)
from arqueogal.data.provenance import Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, GAVO_TAP_URL

logger = logging.getLogger(__name__)


def enrich_geometry(  # noqa: PLR0913, PLR0915 — keyword-only tuning knobs with safe defaults
    df: pd.DataFrame,
    *,
    output_path: Path | str,
    gavo: TAPService,
    aip: TAPService | None = None,
    include_starhorse2: bool = False,
    sh2_sample: str = "apogee_dr17",
    bj_batch_size: int = 10_000,
    sh2_batch_size: int = 10_000,
    bj_checkpoint_dir: Path | str | None = None,
    sh2_checkpoint_dir: Path | str | None = None,
    dust_queries: tuple[DustQuery, DustQuery, DustQuery] | None = None,
    nbhd_radius_pc: float = DEFAULT_NEIGHBORHOOD_RADIUS_PC,
    nbhd_min_neighbors: int = MIN_NEIGHBORS_FOR_MEDIAN,
    ag_gspphot_col: str = "ag_gspphot",
) -> Path:
    """Add §7 distances and §8 extinction columns to ``df``; write Parquet.

    Parameters
    ----------
    df
        Frame carrying ``source_id``, ``ra``, ``dec`` at minimum. For §8.3
        neighborhood-median features, also needs ``ag_gspphot`` (the column
        name can be overridden via ``ag_gspphot_col``).
    output_path
        Destination Parquet path. Parent directory is created; a provenance
        sidecar ``<output_path>.provenance.json`` is written alongside.
    gavo
        GAVO TAP service (Bailer-Jones+2021).
    aip
        AIP TAP service, required only if ``include_starhorse2=True``.
    include_starhorse2
        If True, also fetch StarHorse2 v2 posteriors for the APOGEE-overlap
        cross-check (§7.2). Off by default because Stream 3 does not have
        SH2 coverage beyond the APOGEE subset.
    sh2_sample
        StarHorse2 parent-sample table key; must be in
        :data:`STARHORSE2_SAMPLE_TABLES`.
    bj_batch_size, sh2_batch_size
        ``IN (...)`` chunk sizes.
    bj_checkpoint_dir, sh2_checkpoint_dir
        Per-batch checkpoint directories for resumable fetches.
    dust_queries
        Three-tuple ``(near, mid, far)`` of :class:`DustQuery` callables.
        ``None`` resolves to the production wiring via
        :func:`arqueogal.data.dust_maps.get_default_queries` — which will
        raise ``ImportError`` if ``dustmaps`` is not installed. Tests pass
        synthetic callables directly.
    nbhd_radius_pc, nbhd_min_neighbors
        Forwarded to :func:`neighborhood_av_features`. Defaults match §8.3.
    ag_gspphot_col
        Column name of the Gaia GSP-Phot A_G in ``df``. Default
        ``"ag_gspphot"`` matches :data:`gaia_enrich.ENRICHMENT_ADQL`.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.
    """
    required = {"source_id", "ra", "dec"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"enrich_geometry input missing columns {sorted(missing)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_ids = df["source_id"].astype(np.int64)
    n_in = len(df)

    logger.info("Level-4: fetching Bailer-Jones+2021 for %d sources", n_in)
    bj = fetch_bailerjones(
        gavo,
        source_ids,
        batch_size=bj_batch_size,
        checkpoint_dir=bj_checkpoint_dir,
    )
    logger.info("Level-4: BJ21 returned %d rows", len(bj))

    sh2: pd.DataFrame | None = None
    if include_starhorse2:
        if aip is None:
            raise ValueError("include_starhorse2=True requires an AIP TAP service via `aip=...`")
        logger.info("Level-4: fetching StarHorse2 (%s) for %d sources", sh2_sample, n_in)
        sh2 = fetch_starhorse2(
            aip,
            source_ids,
            sample=sh2_sample,
            batch_size=sh2_batch_size,
            checkpoint_dir=sh2_checkpoint_dir,
        )
        logger.info("Level-4: SH2 returned %d rows", len(sh2))

    logger.info("Level-4: merge_distances (BJ21 + SH2)")
    distances = merge_distances(bj, sh2)

    enriched = df.merge(distances, on="source_id", how="left")

    if dust_queries is None:
        # Lazy import so offline tests don't require dustmaps.
        from arqueogal.data.dust_maps import get_default_queries

        dust_queries = get_default_queries()
    near_q, mid_q, far_q = dust_queries

    logger.info("Level-4: §8.2 A_V composition")
    ra = enriched["ra"].to_numpy(dtype=float)
    dec = enriched["dec"].to_numpy(dtype=float)
    dist_pc = enriched["dist_primary_pc"].to_numpy(dtype=float)
    composed = compose_av(
        ra,
        dec,
        dist_pc,
        near_query=near_q,
        mid_query=mid_q,
        far_query=far_q,
    )
    enriched["av_los"] = composed.av
    enriched["av_los_source"] = composed.source

    if ag_gspphot_col in enriched.columns:
        logger.info("Level-4: §8.3 neighborhood-median A_G (r=%.0f pc)", nbhd_radius_pc)
        nbhd = neighborhood_av_features(
            ra,
            dec,
            dist_pc,
            enriched[ag_gspphot_col].to_numpy(dtype=float),
            radius_pc=nbhd_radius_pc,
            min_neighbors=nbhd_min_neighbors,
        )
        enriched["av_nbhd_median"] = nbhd.av_nbhd_median
        enriched["av_nbhd_std"] = nbhd.av_nbhd_std
        enriched["av_nbhd_n_neighbors"] = nbhd.n_neighbors
    else:
        logger.warning(
            "Level-4: %s not in input frame — skipping §8.3 neighborhood features",
            ag_gspphot_col,
        )

    logger.info("Level-4: writing %s", output_path)
    _write_parquet_atomic(enriched, output_path)

    n_bj_batches = (n_in + bj_batch_size - 1) // bj_batch_size
    sources = [
        TapSource(
            name="GAVO Bailer-Jones+2021 (gedr3dist.main)",
            endpoint=GAVO_TAP_URL,
            query=BAILERJONES_ADQL,
            n_batches=n_bj_batches,
            batch_size=bj_batch_size,
        ),
    ]
    if include_starhorse2 and sh2 is not None:
        n_sh2_batches = (n_in + sh2_batch_size - 1) // sh2_batch_size
        sources.append(
            TapSource(
                name=f"AIP StarHorse2 v2 ({STARHORSE2_SAMPLE_TABLES[sh2_sample]})",
                endpoint=AIP_TAP_URL,
                query=STARHORSE2_ADQL_TEMPLATE.format(
                    table=STARHORSE2_SAMPLE_TABLES[sh2_sample],
                    placeholder="__batch__",
                ),
                n_batches=n_sh2_batches,
                batch_size=sh2_batch_size,
            )
        )

    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/enrich_geometry.py",
        sources=sources,
        cuts_applied=[],
        corrections=[
            "Bailer-Jones+2021 photogeometric distance",
            f"§8.2 A_V composition (near<{NEAR_BOUNDARY_KPC} kpc, "
            f"mid<{FAR_BOUNDARY_KPC} kpc, SFD×{SFD_TO_AV_COEFF} beyond)",
            f"§8.3 neighborhood-median A_G (r={nbhd_radius_pc} pc)",
        ],
        row_count_before=n_in,
        row_count_after=len(enriched),
        notes=(
            "Level-4 geometry: adds BJ21 distances, optional SH2 cross-check, "
            "§8.2 composed A_V, and §8.3 neighborhood-median A_G."
        ),
        extra={
            "bj_rows": len(bj),
            "sh2_rows": (int(len(sh2)) if sh2 is not None else 0),
            "sh2_sample": sh2_sample if include_starhorse2 else None,
            "dist_conflict_threshold_log10": DIST_CONFLICT_LOG_THRESHOLD,
            "nbhd_radius_pc": nbhd_radius_pc,
            "nbhd_min_neighbors": nbhd_min_neighbors,
        },
    )
    write_sidecar(prov)
    logger.info("Level-4: done (%d rows → %s)", len(enriched), output_path)
    return output_path


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


__all__ = ["enrich_geometry"]
