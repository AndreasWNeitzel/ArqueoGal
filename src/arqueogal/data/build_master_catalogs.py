"""Level-6 master-catalog builders — §10 / §11 Level 6.

Two pure-join functions over already-produced intermediates:

- :func:`build_pipeline1_training` joins a Level-4 stream 1 parquet
  (``stream1_apogee_gaia_geom.parquet``) with ``xp_coeffs.parquet`` on
  ``source_id`` and validates the result against
  :data:`PIPELINE1_TRAINING_SCHEMA`.
- :func:`build_pipeline1_inference` does the same for stream 3
  (``stream3_gaia_rgbrc_geom.parquet`` × ``xp_coeffs.parquet``), validating
  against :data:`PIPELINE1_INFERENCE_SCHEMA`.

Why a pure join and not another enrichment stage: by Level 6 every
upstream stage has already attached its contribution (APOGEE labels,
Gaia enrichment + corrections, BJ21 distances, §8.2/§8.3 extinction, XP
preprocessing). The master-catalog step is just the *contract boundary*
between data ingestion and modelling — schema-validate, write, record
row-count losses from the inner join.

The downstream chrono-chemo-kinematic feature matrix (formerly §10.3
``pipeline2_features.parquet``) is no longer built in this repository.
Population classification moved to the separate **Starfold** repo on
2026-04-22; Starfold is responsible for its own feature-matrix assembly
on top of this repo's Pipeline 1 prediction parquets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from arqueogal.data.downloads import sha256_file
from arqueogal.data.master_schema import (
    PIPELINE1_INFERENCE_SCHEMA,
    PIPELINE1_TRAINING_SCHEMA,
    MasterSchema,
)
from arqueogal.data.provenance import (
    LocalSource,
    Provenance,
    sidecar_path,
    write_sidecar,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_FILENAME = "pipeline1_training.parquet"
DEFAULT_INFERENCE_FILENAME = "pipeline1_inference.parquet"


def build_pipeline1_training(
    stream1_geom_path: Path | str,
    xp_coeffs_path: Path | str,
    *,
    output_path: Path | str,
    check_array_lengths: bool = False,
) -> Path:
    """Join stream 1 (APOGEE×Gaia, geometry-enriched) with XP, validate, write.

    Parameters
    ----------
    stream1_geom_path
        Path to the Level-4 output from
        :func:`arqueogal.data.enrich_geometry.enrich_geometry` applied to
        the Stream 1 interim parquet — carries APOGEE labels, corrected
        Gaia astrometry/photometry, BJ21 distances, §8.2/§8.3 extinction.
    xp_coeffs_path
        Path to :func:`arqueogal.data.ingest_xp.ingest_xp` output — the
        per-star normalised / z-scored XP coefficient arrays.
    output_path
        Destination Parquet path (typically
        ``data/processed/pipeline1_training.parquet``).
    check_array_lengths
        Opt-in O(N) check that every XP array cell has length 55.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.

    Raises
    ------
    SchemaError
        If the joined frame does not satisfy
        :data:`PIPELINE1_TRAINING_SCHEMA`.
    """
    return _build_master_catalog(
        stream_path=stream1_geom_path,
        xp_path=xp_coeffs_path,
        output_path=output_path,
        schema=PIPELINE1_TRAINING_SCHEMA,
        check_array_lengths=check_array_lengths,
        stream_label="stream1_geom (APOGEE × Gaia × geometry)",
        notes=(
            "Pipeline 1 training set — APOGEE DR19 labels + Gaia astrometry/"
            "photometry + BJ21 distances + §8 extinction + XP coefficients. "
            "Inner join on source_id; row-count losses logged to provenance."
        ),
    )


def build_pipeline1_inference(
    stream3_geom_path: Path | str,
    xp_coeffs_path: Path | str,
    *,
    output_path: Path | str,
    check_array_lengths: bool = False,
) -> Path:
    """Join stream 3 (Gaia RGB+RC, geometry-enriched) with XP, validate, write.

    Same shape as :func:`build_pipeline1_training` but with the §10.2
    inference schema (Andrae+2023 diagnostics in place of APOGEE labels).
    """
    return _build_master_catalog(
        stream_path=stream3_geom_path,
        xp_path=xp_coeffs_path,
        output_path=output_path,
        schema=PIPELINE1_INFERENCE_SCHEMA,
        check_array_lengths=check_array_lengths,
        stream_label="stream3_geom (Gaia RGB+RC × geometry)",
        notes=(
            "Pipeline 1 inference set — Stream 3 Gaia RGB+RC (Andrae+2023 "
            "diagnostics) + BJ21 distances + §8 extinction + XP coefficients. "
            "No APOGEE labels. Inner join on source_id; row-count losses "
            "logged to provenance."
        ),
    )


def _build_master_catalog(  # noqa: PLR0913 — keyword-only args are all distinct
    *,
    stream_path: Path | str,
    xp_path: Path | str,
    output_path: Path | str,
    schema: MasterSchema,
    check_array_lengths: bool,
    stream_label: str,
    notes: str,
) -> Path:
    stream_path = Path(stream_path)
    xp_path = Path(xp_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Level-6 [%s]: reading %s", schema.name, stream_path)
    stream = pd.read_parquet(stream_path)
    logger.info("Level-6 [%s]: stream has %d rows", schema.name, len(stream))

    logger.info("Level-6 [%s]: reading %s", schema.name, xp_path)
    xp = pd.read_parquet(xp_path)
    logger.info("Level-6 [%s]: xp has %d rows", schema.name, len(xp))

    if "source_id" not in stream.columns:
        raise KeyError(f"{stream_path}: missing source_id")
    if "source_id" not in xp.columns:
        raise KeyError(f"{xp_path}: missing source_id")

    n_stream = len(stream)
    n_xp = len(xp)
    logger.info("Level-6 [%s]: inner-join on source_id", schema.name)
    merged = stream.merge(xp, on="source_id", how="inner", suffixes=("", "_xp"))
    n_merged = len(merged)
    logger.info(
        "Level-6 [%s]: merged %d rows (stream %d × xp %d)",
        schema.name, n_merged, n_stream, n_xp,
    )

    logger.info("Level-6 [%s]: schema validation", schema.name)
    schema.validate(merged, check_array_lengths=check_array_lengths)

    logger.info("Level-6 [%s]: writing %s", schema.name, output_path)
    _write_parquet_atomic(merged, output_path)

    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/build_master_catalogs.py",
        sources=[
            LocalSource(
                name=stream_label,
                path=str(stream_path),
                sha256=sha256_file(stream_path),
            ),
            LocalSource(
                name="xp_coeffs (Level-3 XP extraction)",
                path=str(xp_path),
                sha256=sha256_file(xp_path),
            ),
        ],
        cuts_applied=[
            "inner join on source_id (drops rows missing from either side)",
        ],
        corrections=[],
        row_count_before=n_stream,
        row_count_after=n_merged,
        notes=notes,
        extra={
            "schema_name": schema.name,
            "stream_rows": n_stream,
            "xp_rows": n_xp,
            "merged_rows": n_merged,
            "rows_dropped_by_join": n_stream - n_merged,
            "array_length_check": check_array_lengths,
        },
    )
    write_sidecar(prov, path=sidecar_path(output_path))
    logger.info("Level-6 [%s]: done (%d rows → %s)", schema.name, n_merged, output_path)
    return output_path


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


__all__ = [
    "DEFAULT_INFERENCE_FILENAME",
    "DEFAULT_TRAINING_FILENAME",
    "build_pipeline1_inference",
    "build_pipeline1_training",
]
