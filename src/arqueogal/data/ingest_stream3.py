"""Stream 3 end-to-end ingestion — Andrae+2023 vetted-RGB × Gaia DR3.

Ties the single-purpose modules into the §2.3 / §5 Stream 3 pipeline:

    Andrae+2023 FITS load (local, from Zenodo 7945154)
        → §5.3 stratified sub-sample in (Teff, logg, [M/H], G)
        → AIP Gaia DR3 enrichment (async TAP, checkpointed)
        → inner join on ``source_id``
        → Lindegren+2021 parallax zero-point
        → Riello+2021 G-band flux/magnitude correction
        → write ``stream3_gaia_rgbrc.parquet``
        → write ``stream3_gaia_rgbrc.provenance.json``

XP coefficients are **not** fetched here — they're a separate pass
through :mod:`arqueogal.data.gaia_xp`. The on-disk accounting in §12
stores XP in its own Parquet and joins at use time to avoid duplicating
the 1.5 M-row coefficient arrays into every intermediate frame.

References
----------
data_acquisition.md §2.3 (Stream 3), §5 (selection + stratification),
§14 (layout + provenance).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.andrae2023 import ANDRAE2023_ZENODO_RECORD, load_andrae2023
from arqueogal.data.downloads import sha256_file
from arqueogal.data.gaia_corrections import apply_g_mag_correction, apply_parallax_zpt
from arqueogal.data.gaia_enrich import ENRICHMENT_ADQL, enrich_source_ids
from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.stream3_selection import DEFAULT_PER_CELL, stratified_subsample
from arqueogal.data.tap import AIP_TAP_URL, aip_service

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "stream3_gaia_rgbrc.parquet"


def ingest_stream3(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    data_dir: Path | str,
    *,
    andrae_fits: Path | str,
    service: TAPService | None = None,
    per_cell: int = DEFAULT_PER_CELL,
    rng_seed: int = 0,
    enrich_batch_size: int = 10_000,
    bins_teff: np.ndarray | None = None,
    bins_logg: np.ndarray | None = None,
    bins_mh: np.ndarray | None = None,
    bins_g: np.ndarray | None = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> Path:
    """Run the full Stream 3 pipeline, returning the output Parquet path.

    Parameters
    ----------
    data_dir
        Repository ``data/`` root. Creates
        ``{data_dir}/interim/enrich_batches/stream3/`` for TAP checkpoints
        and writes ``stream3_gaia_rgbrc.parquet`` + provenance to
        ``{data_dir}/interim/``.
    andrae_fits
        Local path to the Zenodo-downloaded Andrae+2023 FITS.
    service
        Authenticated AIP TAP service. ``None`` calls :func:`aip_service`
        lazily.
    per_cell, rng_seed, bins_*
        Forwarded to :func:`stratified_subsample`. Defaults reproduce §5.3.
    enrich_batch_size
        ``IN (...)`` chunk size for the Gaia enrichment query.

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
    checkpoint_dir = interim_dir / "enrich_batches" / "stream3"
    output_path = interim_dir / output_filename

    interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stream 3: loading Andrae+2023 FITS from %s", andrae_fits)
    andrae = load_andrae2023(andrae_fits)
    n_raw = len(andrae)
    logger.info("Stream 3: %d vetted-RGB rows loaded", n_raw)

    logger.info("Stream 3: §5.3 stratified sub-sample (per_cell=%d, seed=%d)",
                per_cell, rng_seed)
    strat_kwargs = {"per_cell": per_cell, "rng_seed": rng_seed}
    if bins_teff is not None:
        strat_kwargs["bins_teff"] = bins_teff
    if bins_logg is not None:
        strat_kwargs["bins_logg"] = bins_logg
    if bins_mh is not None:
        strat_kwargs["bins_mh"] = bins_mh
    if bins_g is not None:
        strat_kwargs["bins_g"] = bins_g
    strat_result = stratified_subsample(andrae, **strat_kwargs)
    sampled = strat_result.sample
    n_sampled = len(sampled)
    logger.info("Stream 3: %d stars after stratification", n_sampled)

    tap = service if service is not None else aip_service()
    logger.info("Stream 3: AIP Gaia DR3 enrichment (%d source_ids)", n_sampled)
    enriched = enrich_source_ids(
        tap,
        sampled["source_id"],
        batch_size=enrich_batch_size,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("Stream 3: inner merge Andrae × Gaia on source_id")
    merged = sampled.merge(enriched, on="source_id", how="inner", suffixes=("", "_gaia"))
    n_merged = len(merged)
    logger.info(
        "Stream 3: merged %d rows (sampled %d × Gaia %d)",
        n_merged, n_sampled, len(enriched),
    )

    logger.info("Stream 3: Lindegren+2021 parallax zero-point")
    merged = apply_parallax_zpt(merged)

    logger.info("Stream 3: Riello+2021 G-mag correction")
    merged = apply_g_mag_correction(merged)

    logger.info("Stream 3: writing %s", output_path)
    _write_parquet_atomic(merged, output_path)

    n_enrich_batches = (n_sampled + enrich_batch_size - 1) // enrich_batch_size
    strat_prov = strat_result.to_provenance()
    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/ingest_stream3.py",
        sources=[
            LocalSource(
                name=f"Andrae+2023 vetted-RGB FITS (Zenodo {ANDRAE2023_ZENODO_RECORD})",
                path=str(Path(andrae_fits)),
                sha256=sha256_file(andrae_fits),
            ),
            TapSource(
                name="AIP Gaia DR3 enrichment (gaiadr3.gaia_source ⨝ astrophysical_parameters)",
                endpoint=AIP_TAP_URL,
                query=ENRICHMENT_ADQL,
                n_batches=n_enrich_batches,
                batch_size=enrich_batch_size,
            ),
        ],
        cuts_applied=[
            f"stratified_subsample per_cell={per_cell}",
            f"stratified_subsample rng_seed={rng_seed}",
        ],
        corrections=[
            "Lindegren+2021 parallax zero-point",
            "Riello+2021 G-band flux/magnitude correction (A&A 649, A3 Appendix A)",
        ],
        row_count_before=n_raw,
        row_count_after=n_merged,
        notes=(
            "Stream 3 inference / Pipeline-2 feature basis: "
            "Andrae+2023 vetted-RGB, stratified in (Teff, logg, [M/H], G)."
        ),
        extra={
            "andrae_rows_loaded": n_raw,
            "stratified_selected": n_sampled,
            "gaia_enriched": len(enriched),
            "merged": n_merged,
            "enrich_checkpoint_dir": str(checkpoint_dir),
            "stratification": strat_prov,
        },
    )
    write_sidecar(prov)
    logger.info("Stream 3: done (%d rows → %s)", n_merged, output_path)
    return output_path


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file via temp + rename so crashes never leave a partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


__all__ = ["DEFAULT_OUTPUT_FILENAME", "ingest_stream3"]
