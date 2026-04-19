"""Andrae+2023 vetted-RGB catalogue loader — §5.2.

Small single-purpose wrapper around :func:`astropy.io.fits.open` that
reads the trimmed column set we use for Stream 3 stratified sub-sampling.

Reference: Andrae, Rix & Chandra 2023, ApJS 267:8 (arXiv:2302.02611);
Zenodo record 7945154. The full vetted catalogue is ~800 MB and ships
XGBoost-derived ``teff_xgboost`` / ``logg_xgboost`` / ``mh_xgboost`` labels
used here purely as a *selection* input (never as Pipeline 1 training
labels — that would be circular, §5.1 weakness note).

The download itself is a separate concern (a straightforward HTTPS fetch
from Zenodo handled by :mod:`arqueogal.data.downloads`); this module only
reads a local FITS file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

ANDRAE2023_ZENODO_RECORD: Final[str] = "7945154"
"""Zenodo record id for the Andrae+2023 vetted-RGB catalogue (§5.2)."""

ANDRAE2023_DEFAULT_HDU: Final[int] = 1
"""Primary catalogue HDU for the Zenodo-published FITS table."""

# Minimal column set for §5.3 stratified sub-sampling.
KEPT_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "teff_xgboost",
    "logg_xgboost",
    "mh_xgboost",
    "phot_g_mean_mag",
)

# VizieR-route canonical columns — produced by scripts/fetch_andrae2023_vizier.py
# using table ``J/MNRAS/537/1984/a23`` (Ardern-Arentsen+2024 reissue of
# Andrae+2023). Substitutes the 3.59 GB Zenodo FITS, per
# reports/extraction_budget_20260418.md. Stratification columns here are
# ``teff``, ``logg``, ``fe_h``, ``g_mag`` — pass those via ``teff_col``/etc.
# kwargs to :func:`arqueogal.data.stream3_selection.stratified_subsample`.
VIZIER_KEPT_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "ra_deg", "dec_deg",
    "g_mag", "parallax_mas", "parallax_err_mas",
    "ebv", "bp_rp_0", "g_mag_0",
    "teff", "e_teff", "s_teff",
    "logg", "e_logg", "s_logg",
    "fe_h", "e_fe_h", "s_fe_h",
    "c_fe", "e_c_fe", "s_c_fe",
    "c_cor", "energy", "l_z",
    "pvar",
)


def load_andrae2023(
    path: Path | str,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    hdu: int = ANDRAE2023_DEFAULT_HDU,
) -> pd.DataFrame:
    """Read the Andrae+2023 FITS into a pandas DataFrame.

    Parameters
    ----------
    path
        Local path to the Zenodo-downloaded FITS (optionally gzipped).
    columns
        Subset of columns to materialise. Defaults to :data:`KEPT_COLUMNS`,
        which is the minimum required by §5.3 stratified sub-sampling.
    hdu
        HDU index (default 1, the primary catalogue table).

    Returns
    -------
    pd.DataFrame
        One row per vetted-RGB star, columns per ``columns``.

    Raises
    ------
    KeyError
        If the FITS HDU is missing any requested column. Column naming in
        Zenodo re-uploads has drifted before — consult the record page and
        pass an updated ``columns`` tuple explicitly when that happens.
    """
    path = Path(path)
    cols = tuple(columns) if columns is not None else KEPT_COLUMNS

    with fits.open(path, memmap=True) as hdul:
        table = hdul[hdu].data
        present = {name.lower() for name in table.columns.names}
        missing = [c for c in cols if c.lower() not in present]
        if missing:
            raise KeyError(
                f"{path}: Andrae+2023 HDU {hdu} missing expected columns {missing}. "
                "Column naming may have changed — check the Zenodo record page."
            )
        case_map = {name.lower(): name for name in table.columns.names}
        frame: dict[str, np.ndarray] = {}
        for col in cols:
            arr = np.asarray(table[case_map[col.lower()]])
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
            frame[col] = arr

    df = pd.DataFrame(frame)
    logger.info("loaded %s: %d rows × %d cols", path.name, len(df), len(df.columns))
    return df


def load_andrae2023_parquet(
    path: Path | str,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Read the VizieR-route Andrae+2023 Parquet into a pandas DataFrame.

    Parameters
    ----------
    path
        Local path to ``andrae2023_rgb.parquet`` written by
        ``scripts/fetch_andrae2023_vizier.py``.
    columns
        Subset of columns to materialise. Defaults to :data:`VIZIER_KEPT_COLUMNS`.

    Raises
    ------
    KeyError
        If the Parquet is missing any requested column.
    """
    path = Path(path)
    cols = tuple(columns) if columns is not None else VIZIER_KEPT_COLUMNS
    df = pd.read_parquet(path, columns=list(cols))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: Andrae+2023 Parquet missing columns {missing}")
    logger.info("loaded %s: %d rows × %d cols", path.name, len(df), len(df.columns))
    return df


__all__ = [
    "ANDRAE2023_DEFAULT_HDU",
    "ANDRAE2023_ZENODO_RECORD",
    "KEPT_COLUMNS",
    "VIZIER_KEPT_COLUMNS",
    "load_andrae2023",
    "load_andrae2023_parquet",
]
