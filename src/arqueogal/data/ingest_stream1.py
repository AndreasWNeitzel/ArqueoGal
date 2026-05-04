"""Stream 1 end-to-end ingestion — APOGEE DR19 × Gaia DR3.

Ties the single-purpose data-layer modules into the §2.1 Stream 1 pipeline:

    download DR19 summary
        → load FITS → quality cuts → derive [C/N]
        → Mészáros+2025 [X/M] corrections
        → AIP Gaia DR3 enrichment (async TAP, checkpointed)
        → inner join on ``source_id``
        → Lindegren+2021 parallax zero-point
        → Riello+2021 G-band flux/magnitude correction
        → CCM89 R_V=3.1 + Yuan+2013 IR dereddening (when broadband cols exist)
        → write ``stream1_apogee_gaia.parquet``
        → write ``stream1_apogee_gaia.provenance.json``

All four corrections are live: Mészáros+2025 (Table 3 linear Δ[X/M] fit to
open clusters), Lindegren+2021 parallax zero-point (via ``gaiadr3-zeropoint``),
Riello+2021 (A&A 649 A3 Appendix A) G-band flux/magnitude correction (see
``gaia_corrections.apply_g_mag_correction``), and the v1 frozen extinction
recipe — CCM89 R_V=3.1 with Yuan+2013 broadband ratios applied to whichever
of (J, H, Ks, W1, W2) the merged frame carries; see ``data.extinction`` and
``docs/protocols/extinction_correction.md``. The dereddening step is a
no-op when the broadband columns are absent (Stream 1 carries them only
after the IR cross-match step).
AIP Gaia enrichment is the only remaining user-gated step.

References
----------
data_acquisition.md §2.1 (Stream 1), §3 (DR19), §3.6 (enrichment),
§3.7 (corrections), §14 (layout + provenance);
docs/protocols/extinction_correction.md (v1 dereddening contract).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.apogee_dr19 import (
    DEFAULT_SUMMARY_FILENAME,
    DR19_SUMMARY_URL,
    QualityCuts,
    apply_meszaros2025_corrections,
    apply_quality_cuts,
    derive_c_n,
    load_dr19,
)
from arqueogal.data.downloads import download, sha256_file
from arqueogal.data.extinction import DEFAULT_EXTINCTION_LAW
from arqueogal.data.gaia_enrich import ENRICHMENT_ADQL, enrich_source_ids
from arqueogal.data.preprocessing import apply_pipeline1_preprocessing
from arqueogal.data.provenance import (
    HttpSource,
    LocalSource,
    Provenance,
    TapSource,
    write_sidecar,
)
from arqueogal.data.tap import AIP_TAP_URL, aip_service
from arqueogal.utils.io import save_parquet

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "stream1_apogee_gaia.parquet"


def ingest_stream1(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    data_dir: Path | str,
    *,
    service: TAPService | None = None,
    cuts: QualityCuts | None = None,
    enrich_batch_size: int = 10_000,
    summary_url: str = DR19_SUMMARY_URL,
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    download_progress: bool = True,
) -> Path:
    """Run the full Stream 1 pipeline, returning the output Parquet path.

    Parameters
    ----------
    data_dir
        Repository ``data/`` root. Subdirectories are created as needed:

        - ``{data_dir}/raw/apogee_dr19/`` — DR19 summary download.
        - ``{data_dir}/interim/enrich_batches/stream1/`` — TAP batch checkpoints.
        - ``{data_dir}/interim/`` — final ``stream1_apogee_gaia.parquet`` plus
          ``stream1_apogee_gaia.provenance.json``.
    service
        Authenticated AIP TAP service. ``None`` calls :func:`aip_service`
        lazily (honouring YAML credentials and the ``GAIA_AIP_TOKEN`` env
        fallback).
    cuts
        Override the §3.3 defaults in :class:`QualityCuts`.
    enrich_batch_size
        ``IN (...)`` chunk size for the Gaia enrichment TAP query.

    Returns
    -------
    Path
        Absolute path to the written Parquet file. The provenance sidecar
        lives next to it.

    Raises
    ------
    ImportError
        From :func:`apply_parallax_zpt` when ``gaiadr3-zeropoint`` is not
        installed. Install with ``pip install gaiadr3-zeropoint --no-deps``.
    """
    cuts = cuts or QualityCuts()
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw" / "apogee_dr19"
    interim_dir = data_dir / "interim"
    checkpoint_dir = interim_dir / "enrich_batches" / "stream1"
    summary_path = raw_dir / summary_filename
    output_path = interim_dir / output_filename

    interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stream 1: downloading DR19 summary to %s", summary_path)
    dl = download(summary_url, summary_path, progress=download_progress)

    logger.info("Stream 1: loading DR19 summary")
    raw = load_dr19(summary_path)

    logger.info("Stream 1: quality cuts")
    cut_df, cut_stats = apply_quality_cuts(raw, cuts)

    logger.info("Stream 1: derive [C/N]")
    cut_df = derive_c_n(cut_df)

    logger.info("Stream 1: Mészáros+2025 [X/M] corrections")
    corrected = apply_meszaros2025_corrections(cut_df)

    logger.info("Stream 1: AIP Gaia DR3 enrichment (%d source_ids)", len(corrected))
    tap = service if service is not None else aip_service()
    enriched = enrich_source_ids(
        tap,
        corrected["source_id"],
        batch_size=enrich_batch_size,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("Stream 1: inner merge APOGEE × Gaia on source_id")
    # APOGEE allStar carries multiple ASPCAP solutions per Gaia source_id (different
    # visits, programs, reduction runs). The Gaia DR3 enrichment is one row per
    # source_id by construction. Hence many_to_one is the load-bearing contract:
    # we expect repeated source_ids on the left and exactly one on the right.
    # validate="many_to_one" enforces this; it raises if a future Gaia enrichment
    # query regression produces duplicates on the right side. The downstream training
    # driver dedups (training.py:122-135) before splitting so the multi-spectrum
    # rows do not leak across train/val/test.
    merged = corrected.merge(enriched, on="source_id", how="inner", validate="many_to_one")
    n_merged_pre_preprocessing = len(merged)
    logger.info(
        "Stream 1: merged %d rows (APOGEE post-cut %d × Gaia %d)",
        n_merged_pre_preprocessing,
        len(corrected),
        len(enriched),
    )

    logger.info("Stream 1: unified preprocessing pipeline (steps 1-8)")
    processed = apply_pipeline1_preprocessing(
        merged,
        mode="train",
        aip=tap,
        apply_extinction=True,
    )
    extinction_applied = any(c in processed.columns for c in ("j_mag_dered",))
    n_merged = len(processed)

    logger.info("Stream 1: writing %s", output_path)
    save_parquet(processed, output_path)

    n_enrich_batches = (len(corrected) + enrich_batch_size - 1) // enrich_batch_size
    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/ingest_stream1.py",
        sources=[
            HttpSource(
                name="APOGEE DR19 ASPCAP summary (Mészáros+2025)",
                url=dl.url,
                size_bytes=dl.size_bytes,
                sha256=dl.sha256,
            ),
            LocalSource(
                name="APOGEE DR19 ASPCAP summary (local)",
                path=str(summary_path),
                sha256=dl.sha256,
            ),
            TapSource(
                name="AIP Gaia DR3 enrichment (gaiadr3.gaia_source ⨝ astrophysical_parameters)",
                endpoint=AIP_TAP_URL,
                query=ENRICHMENT_ADQL,
                n_batches=n_enrich_batches,
                batch_size=enrich_batch_size,
            ),
        ],
        cuts_applied=cuts.as_predicates(),
        corrections=(
            [
                "Mészáros+2025 [X/M]/Teff polynomial corrections (DR19)",
                "Lindegren+2021 parallax zero-point",
                "Riello+2021 G-band flux/magnitude correction (A&A 649, A3 Appendix A)",
            ]
            + (
                [
                    f"Extinction: {DEFAULT_EXTINCTION_LAW.name} "
                    "(Yuan+2013 broadband ratios, dust-map fusion via "
                    "data.extinction.apply_extinction_corrections)"
                ]
                if extinction_applied
                else []
            )
        ),
        row_count_before=cut_stats["before"],
        row_count_after=n_merged,
        notes=(
            "Stream 1 training set: APOGEE DR19 × Gaia DR3. "
            "Enrichment cross-matches via Gaia source_id published in DR19."
        ),
        extra={
            "quality_cut_stage_counts": cut_stats,
            "apogee_post_cut": len(corrected),
            "gaia_enriched": len(enriched),
            "merged": n_merged,
            "enrich_checkpoint_dir": str(checkpoint_dir),
            # Persist the Mészáros+2025 correction summary so the audit trail
            # survives the parquet write (pandas drops .attrs by default).
            # Keys preserved verbatim from apply_meszaros2025_corrections().
            "meszaros_correction_summary": (
                corrected.attrs.get("meszaros_correction_summary").to_dict("records")
                if isinstance(corrected.attrs.get("meszaros_correction_summary"), pd.DataFrame)
                else corrected.attrs.get("meszaros_correction_summary")
            ),
            # Extinction-correction provenance: full law fingerprint + flag
            # firing rates so the v1 contract is reproducible from the
            # sidecar alone (no need to read source code).
            "extinction_correction": (
                {
                    "applied": True,
                    "law": DEFAULT_EXTINCTION_LAW.fingerprint(),
                    "flag_counts": {
                        "av_is_neighborhood_fallback": int(
                            processed["av_is_neighborhood_fallback"].sum()
                        )
                        if "av_is_neighborhood_fallback" in processed.columns
                        else 0,
                        "av_distance_prior_dominated": int(
                            processed["av_distance_prior_dominated"].sum()
                        )
                        if "av_distance_prior_dominated" in processed.columns
                        else 0,
                        "av_neighbourhood_high_dispersion": int(
                            processed["av_neighbourhood_high_dispersion"].sum()
                        )
                        if "av_neighbourhood_high_dispersion" in processed.columns
                        else 0,
                    },
                }
                if extinction_applied
                else {"applied": False, "reason": "broadband or av_layer columns absent"}
            ),
        },
    )
    write_sidecar(prov)
    logger.info("Stream 1: done (%d rows → %s)", n_merged, output_path)
    return output_path


__all__ = [
    "DEFAULT_OUTPUT_FILENAME",
    "ingest_stream1",
    "sha256_file",
]
