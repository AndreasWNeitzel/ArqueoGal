"""Stream 2 end-to-end ingestion — TESS Hon+2021 × TIC v8.2 × Gaia DR3.

Ties the single-purpose modules into the §2.2 / §4 Stream 2 pipeline:

    Hon+2021 VizieR fetch (Prob > 0.95)
        → TIC v8.2 VizieR lookup (TIC → DR2 ``GAIA``)
        → drop TICs without a DR2 counterpart
        → DR2 → DR3 via ``gaiadr3.dr2_neighbourhood`` (§4.3 cuts + tie-break)
        → AIP Gaia DR3 enrichment (async TAP, checkpointed)
        → AIP Gaia XP fetch + Ye+2024 correction + Hermite re-projection
        → Lindegren+2021 parallax zero-point
        → Riello+2021 G-band flux/magnitude correction
        → write ``stream2_tess_gaia.parquet``
        → write ``stream2_tess_gaia.provenance.json``

As with Stream 1, the two correction stages
(:func:`gaia_corrections.apply_parallax_zpt`,
:func:`gaia_corrections.apply_g_mag_correction`) are stubs that halt loudly
rather than silently producing uncorrected astrometry; see §3.7 for the
rationale.

XP coefficients are now fetched and corrected via Ye+2024 NN flux-correction
(mandatory per §6.4), then re-projected onto the Hermite basis using the
frozen v1 z-score statistics. This enables Stream 2 to feed the same
feature-matrix pipeline as Stream 1 / 3, ensuring chemical abundance
inference and Task 4 age work are directly comparable.

References
----------
data_acquisition.md §2.2 (Stream 2), §4 (Hon+2021 + TIC + DR2→DR3),
§6 (XP extraction + Ye+2024), §14 (layout + provenance).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyvo.dal.tap import TAPService

from arqueogal.data.crossmatch import (
    DEFAULT_ANGULAR_DISTANCE_MAS,
    DEFAULT_MAG_DIFF_LIMIT,
    DR2_NEIGHBOURHOOD_ADQL,
    crossmatch_dr2_to_dr3,
)
from arqueogal.data.gaia_enrich import ENRICHMENT_ADQL, enrich_source_ids
from arqueogal.data.gaia_xp import (
    XP_BATCH_SIZE,
    apply_ye2024_correction,
    fetch_xp_coefficients,
)
from arqueogal.data.preprocessing import apply_pipeline1_preprocessing
from arqueogal.data.provenance import Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, VIZIER_TAP_URL, aip_service, vizier_service
from arqueogal.data.tess_hon2021 import (
    HON2021_DEFAULT_PROB_THRESHOLD,
    build_hon2021_adql,
    fetch_hon2021,
)
from arqueogal.data.tic_v82 import TIC_V82_ADQL, fetch_tic_v82
from arqueogal.utils.io import save_parquet

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "stream2_tess_gaia.parquet"


def ingest_stream2(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    data_dir: Path | str,
    *,
    vizier: TAPService | None = None,
    aip: TAPService | None = None,
    prob_threshold: float = HON2021_DEFAULT_PROB_THRESHOLD,
    tic_batch_size: int = 5_000,
    xmatch_batch_size: int = 5_000,
    enrich_batch_size: int = 10_000,
    xp_batch_size: int = XP_BATCH_SIZE,
    max_angular_distance_mas: float = DEFAULT_ANGULAR_DISTANCE_MAS,
    max_mag_diff: float = DEFAULT_MAG_DIFF_LIMIT,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> Path:
    """Run the full Stream 2 pipeline, returning the output Parquet path.

    Parameters
    ----------
    data_dir
        Repository ``data/`` root. Creates:

        - ``{data_dir}/interim/xmatch_batches/stream2_dr2_nbh/`` — DR2→DR3 checkpoints.
        - ``{data_dir}/interim/xmatch_batches/stream2_tic/`` — TIC v8.2 checkpoints.
        - ``{data_dir}/interim/enrich_batches/stream2/`` — Gaia enrichment checkpoints.
        - ``{data_dir}/interim/stream2_tess_gaia.parquet`` + provenance sidecar.
    vizier
        VizieR TAP service (Hon+2021 + TIC v8.2). ``None`` calls
        :func:`vizier_service` lazily.
    aip
        Authenticated AIP TAP service (DR2→DR3 + Gaia enrichment). ``None``
        calls :func:`aip_service` lazily.
    prob_threshold
        Hon+2021 detection-probability cut. Default 0.95 per §4.1.
    tic_batch_size, xmatch_batch_size, enrich_batch_size, xp_batch_size
        ``IN (...)`` chunk sizes for the batched TAP queries. xp_batch_size
        defaults to :data:`gaia_xp.XP_BATCH_SIZE` (5000).
    max_angular_distance_mas, max_mag_diff
        §4.3 DR2→DR3 cuts. Defaults 300 mas / 0.1 mag.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.

    Raises
    ------
    NotImplementedError
        From the Lindegren+2021 or Riello+2021 correction stubs.
    """
    data_dir = Path(data_dir)
    interim_dir = data_dir / "interim"
    tic_ckpt = interim_dir / "xmatch_batches" / "stream2_tic"
    dr2_ckpt = interim_dir / "xmatch_batches" / "stream2_dr2_nbh"
    enrich_ckpt = interim_dir / "enrich_batches" / "stream2"
    xp_ckpt = interim_dir / "enrich_batches" / "stream2_xp"
    output_path = interim_dir / output_filename

    interim_dir.mkdir(parents=True, exist_ok=True)

    vz = vizier if vizier is not None else vizier_service()
    ap = aip if aip is not None else aip_service()

    logger.info("Stream 2: fetching Hon+2021 (Prob > %s) from VizieR", prob_threshold)
    hon = fetch_hon2021(vz, prob_threshold=prob_threshold)
    n_hon = len(hon)
    logger.info("Stream 2: Hon+2021 post-cut %d rows", n_hon)

    logger.info("Stream 2: TIC v8.2 lookup for %d TICs", n_hon)
    tic = fetch_tic_v82(
        vz,
        hon["TIC"].tolist(),
        batch_size=tic_batch_size,
        checkpoint_dir=tic_ckpt,
    )
    # Drop TICs without a DR2 counterpart before we spend a TAP batch on NaNs.
    tic_valid = tic.dropna(subset=["GAIA"]).copy()
    tic_valid["GAIA"] = tic_valid["GAIA"].astype("int64")
    n_tic_total = len(tic)
    n_tic_with_dr2 = len(tic_valid)
    logger.info(
        "Stream 2: TIC rows with DR2: %d / %d",
        n_tic_with_dr2,
        n_tic_total,
    )

    logger.info(
        "Stream 2: DR2→DR3 (angular<%s mas, |Δmag|<%s)",
        max_angular_distance_mas,
        max_mag_diff,
    )
    xmatch = crossmatch_dr2_to_dr3(
        ap,
        tic_valid["GAIA"].tolist(),
        batch_size=xmatch_batch_size,
        checkpoint_dir=dr2_ckpt,
        max_angular_distance_mas=max_angular_distance_mas,
        max_mag_diff=max_mag_diff,
    )
    n_xmatch = len(xmatch)
    logger.info("Stream 2: DR3-resolved %d / %d DR2 sources", n_xmatch, n_tic_with_dr2)

    logger.info("Stream 2: AIP Gaia DR3 enrichment (%d source_ids)", n_xmatch)
    enriched = enrich_source_ids(
        ap,
        xmatch["dr3_source_id"],
        batch_size=enrich_batch_size,
        checkpoint_dir=enrich_ckpt,
    )

    logger.info("Stream 2: XP fetch for %d source_ids", n_xmatch)
    raw_xp = fetch_xp_coefficients(
        ap,
        xmatch["dr3_source_id"],
        batch_size=xp_batch_size,
        checkpoint_dir=xp_ckpt,
    )
    n_xp_fetched = len(raw_xp)
    logger.info("Stream 2: XP TAP returned %d rows", n_xp_fetched)

    logger.info("Stream 2: Ye+2024 NN flux correction")
    coords_for_ye = enriched[["source_id", "ra", "dec"]].copy()
    corrected_xp = apply_ye2024_correction(raw_xp, coords_for_ye)
    n_ye_ok = (corrected_xp["ye2024_flag"] == 0).sum()
    logger.info(
        "Stream 2: Ye+2024 done: %d OK / %d total",
        n_ye_ok,
        len(corrected_xp),
    )

    # Note: Hermite reprojection, z-scoring, extinction, and aux-column
    # standardisation are now handled by the unified preprocessing pipeline below.
    # Store corrected_xp for later merge.

    logger.info("Stream 2: assembling joined frame")
    # Join order: Hon+2021 ── TIC (on TIC) ── DR2→DR3 (on GAIA=dr2_source_id)
    # ── Gaia enrichment (on dr3_source_id == source_id).
    #
    # Seismic columns (νmax, e_numax, asteroseismic Teff, R) returned by
    # ``fetch_hon2021`` propagate through the merges below. ArqueoGal's Pipeline 1
    # only consumes the cross-matched DR3 ids plus astrometry/photometry from the
    # AIP enrichment, but the seismic columns are deliberately *retained* here
    # (rather than dropped) so Task 4 (asteroseismic ages, led by Campante /
    # Miglio) can read them straight from ``stream2_tess_gaia.parquet`` without
    # re-fetching Hon+2021. See ``data_acquisition.md`` §4.4.
    joined = hon.merge(tic_valid, on="TIC", how="inner", suffixes=("", "_tic"))
    joined = joined.merge(
        xmatch,
        left_on="GAIA",
        right_on="dr2_source_id",
        how="inner",
    )
    joined = joined.merge(
        enriched,
        left_on="dr3_source_id",
        right_on="source_id",
        how="inner",
    )
    # Merge in the Ye+2024-corrected XP (skip full z-scoring+hermite steps here,
    # they're handled by the unified preprocessing pipeline below)
    joined = joined.merge(
        corrected_xp[["source_id", "corrected_flux", "a_v_sfd", "ye2024_flag"]],
        on="source_id",
        how="inner",
    )
    n_joined_pre_preprocessing = len(joined)
    logger.info("Stream 2: joined %d rows (pre-preprocessing)", n_joined_pre_preprocessing)

    logger.info("Stream 2: unified preprocessing pipeline (steps 1-8)")
    processed = apply_pipeline1_preprocessing(
        joined,
        mode="inference",
        aip=ap,
        skip_xp_fetch=True,
        apply_extinction=False,
    )
    n_joined = len(processed)

    logger.info("Stream 2: writing %s", output_path)
    save_parquet(processed, output_path)

    n_tic_batches = (n_hon + tic_batch_size - 1) // tic_batch_size
    n_xmatch_batches = (n_tic_with_dr2 + xmatch_batch_size - 1) // xmatch_batch_size
    n_enrich_batches = (n_xmatch + enrich_batch_size - 1) // enrich_batch_size
    n_xp_batches = (n_xmatch + xp_batch_size - 1) // xp_batch_size
    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/ingest_stream2.py",
        sources=[
            TapSource(
                name="VizieR Hon+2021 (J/ApJ/919/131/table1)",
                endpoint=VIZIER_TAP_URL,
                query=build_hon2021_adql(prob_threshold=prob_threshold),
                n_batches=1,
            ),
            TapSource(
                name="VizieR TIC v8.2 (IV/39/tic82)",
                endpoint=VIZIER_TAP_URL,
                query=TIC_V82_ADQL,
                n_batches=n_tic_batches,
                batch_size=tic_batch_size,
            ),
            TapSource(
                name="AIP Gaia DR2 neighbourhood (gaiadr3.dr2_neighbourhood)",
                endpoint=AIP_TAP_URL,
                query=DR2_NEIGHBOURHOOD_ADQL,
                n_batches=n_xmatch_batches,
                batch_size=xmatch_batch_size,
            ),
            TapSource(
                name="AIP Gaia DR3 enrichment (gaiadr3.gaia_source ⨝ astrophysical_parameters)",
                endpoint=AIP_TAP_URL,
                query=ENRICHMENT_ADQL,
                n_batches=n_enrich_batches,
                batch_size=enrich_batch_size,
            ),
            TapSource(
                name="AIP Gaia DR3 XP (gaiadr3.xp_continuous_mean_spectrum)",
                endpoint=AIP_TAP_URL,
                query=(
                    "SELECT x.source_id, ... FROM "
                    "gaiadr3.xp_continuous_mean_spectrum AS x WHERE x.source_id IN (...)"
                ),
                n_batches=n_xp_batches,
                batch_size=xp_batch_size,
            ),
        ],
        cuts_applied=[
            f"Hon+2021 Prob > {prob_threshold}",
            f"angular_distance < {max_angular_distance_mas} mas",
            f"|magnitude_difference| < {max_mag_diff}",
        ],
        corrections=[
            "Ye+2024 NN flux correction (A&A 695, A75; Zenodo 10.5281/zenodo.14712749)",
            "Hermite re-projection (frozen v1 basis)",
            "Frozen v1 z-score projection (Stream-1 training statistics)",
            "Lindegren+2021 parallax zero-point",
            "Riello+2021 G-band flux/magnitude correction (A&A 649, A3 Appendix A)",
        ],
        row_count_before=n_hon,
        row_count_after=n_joined,
        notes=(
            "Stream 2 for Task 4 asteroseismic ages and chemical abundance inference: "
            "Hon+2021 × TIC v8.2 × Gaia DR3 × XP. XP coefficients processed via "
            "Ye+2024 NN correction and re-projected onto Hermite basis with frozen stats."
        ),
        extra={
            "hon2021_rows": n_hon,
            "tic_rows_total": n_tic_total,
            "tic_rows_with_dr2": n_tic_with_dr2,
            "dr3_resolved": n_xmatch,
            "gaia_enriched": len(enriched),
            "xp_fetched": n_xp_fetched,
            "ye2024_ok": int(n_ye_ok),
            "joined_pre_preprocessing": n_joined_pre_preprocessing,
            "preprocessed": n_joined,
            "tic_checkpoint_dir": str(tic_ckpt),
            "dr2_xmatch_checkpoint_dir": str(dr2_ckpt),
            "enrich_checkpoint_dir": str(enrich_ckpt),
            "xp_checkpoint_dir": str(xp_ckpt),
        },
    )
    write_sidecar(prov)
    logger.info("Stream 2: done (%d rows → %s)", n_joined, output_path)
    return output_path


__all__ = ["DEFAULT_OUTPUT_FILENAME", "ingest_stream2"]
