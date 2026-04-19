"""Level-5 enrichment — galpy actions on a geometry-enriched stream.

Applies §9 (kinematics) to a frame already carrying BJ21 distances,
Gaia 6D astrometry (ra, dec, pmra, pmdec, radial_velocity):

    input frame (source_id, ra, dec, pmra, pmdec, radial_velocity,
                 r_med_photogeo, …)
        → compute_actions (McMillan17, Staeckel delta=0.45)
        → merge action columns back on source_id (left-join, NaN where
          galpy couldn't solve)
        → write ``<basename>_kin.parquet``
        → write ``<basename>_kin.provenance.json``

Pure local computation — no TAP, no HTTP. The provenance sidecar records
the full :class:`KinematicsConfig` so a downstream run can be replayed
without guessing which potential was used.

References
----------
data_acquisition.md §9 (kinematics), §11 (Level 5), §14 (provenance).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from arqueogal.data.kinematics import (
    OUTPUT_COLS,
    REQUIRED_INPUT_COLS,
    KinematicsConfig,
    compute_actions,
)
from arqueogal.data.provenance import Provenance, write_sidecar

logger = logging.getLogger(__name__)


def enrich_kinematics_stream(
    df: pd.DataFrame,
    *,
    output_path: Path | str,
    config: KinematicsConfig | None = None,
) -> Path:
    """Add §9 galpy-action columns to ``df`` and write Parquet.

    Parameters
    ----------
    df
        Input frame. Must carry :data:`REQUIRED_INPUT_COLS`
        (``source_id``, ``ra``, ``dec``, ``r_med_photogeo``, ``pmra``,
        ``pmdec``, ``radial_velocity``). Rows with any NaN in the required
        columns are dropped inside :func:`compute_actions`; the returned
        frame still has one row per *input* star (left-joined), with NaN
        action columns where the drop happened.
    output_path
        Destination Parquet path. Parent directory is created; a provenance
        sidecar ``<output_path>.provenance.json`` is written alongside.
    config
        Override defaults (potential, ``ro``/``vo``, Staeckel delta, solar
        motion). ``None`` → :class:`KinematicsConfig` defaults from §9.2.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.

    Raises
    ------
    KeyError
        If ``df`` is missing any of :data:`REQUIRED_INPUT_COLS`.
    """
    missing = set(REQUIRED_INPUT_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"enrich_kinematics_stream input missing columns {sorted(missing)}")

    cfg = config or KinematicsConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = len(df)
    logger.info("Level-5: compute_actions on %d rows (potential=%s)", n_in, cfg.potential)
    actions = compute_actions(df, config=cfg)
    n_solved = len(actions)
    logger.info("Level-5: %d/%d actions computed", n_solved, n_in)

    # Left-join so the output row-count matches the input, with NaN where
    # galpy dropped a row for non-finite phase-space inputs.
    action_cols_only = [c for c in OUTPUT_COLS if c != "source_id"]
    enriched = df.merge(actions, on="source_id", how="left", suffixes=("", "_kin"))

    logger.info("Level-5: writing %s", output_path)
    _write_parquet_atomic(enriched, output_path)

    cfg_dict = asdict(cfg)
    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/enrich_kinematics.py",
        sources=[],
        cuts_applied=[],
        corrections=[
            f"galpy actions ({cfg.potential}, Staeckel delta={cfg.staeckel_delta})",
        ],
        row_count_before=n_in,
        row_count_after=len(enriched),
        notes=(
            "Level-5 kinematics: adds galpy actions (J_R, J_z, L_z), orbital "
            "shape (ecc, r_peri, r_apo, z_max), energy E, and Galactocentric "
            "cylindrical velocities under the configured potential."
        ),
        extra={
            "kinematics_config": cfg_dict,
            "rows_solved": n_solved,
            "rows_unsolved": n_in - n_solved,
            "action_columns": action_cols_only,
        },
    )
    write_sidecar(prov)
    logger.info("Level-5: done (%d rows → %s)", len(enriched), output_path)
    return output_path


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


__all__ = ["enrich_kinematics_stream"]
