"""Mandatory Gaia DR3 astrometry/photometry corrections (data_acquisition.md §3.7).

Two independent fixes must be applied at ingestion, before any downstream
use of Gaia parallax or G-band photometry:

1. **Lindegren+2021 parallax zero-point**. The published ``parallax`` column
   contains a colour/magnitude/ecliptic-latitude-dependent bias. The official
   correction code (Lindegren et al. 2021, https://www.cosmos.esa.int/web/gaia/edr3-code)
   is distributed as the ``gaiadr3-zeropoint`` PyPI package. This module is a
   thin wrapper that loads the tables on first use and applies the correction
   column-wise to a DataFrame.

2. **Riello+2021 G-band flux/magnitude correction** (A&A 649, A3 Appendix A).
   Applies to sources with 2- or 6-parameter astrometric solutions at G ≥ 13,
   where the published ``phot_g_mean_flux`` suffers a colour-dependent bias.
   Implemented per the agabrown/gaiaedr3-6p-gband-correction reference code
   (cubic-polynomial factor f(BP−RP), bright and faint branches).

Both corrections must be recorded in the provenance sidecar (see
``arqueogal.data.provenance``).

Install note
------------
``gaiadr3-zeropoint`` is NOT currently in the rapidsenv venv. Install with::

    pip install gaiadr3-zeropoint --no-deps

The ``--no-deps`` flag keeps RAPIDS pins intact (the package is pure-Python
with trivial deps).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Gaia DR3 ``astrometric_params_solved`` values. Bitmask semantics; these
# are the three observed values in DR3.
SOL_TWO_PARAM: Final[int] = 3
SOL_FIVE_PARAM: Final[int] = 31
SOL_SIX_PARAM: Final[int] = 95

# ``zpt.get_zpt`` returns the correction already in mas (docstring is explicit:
# "correction in mas (milliarcsecond, not micro)"), matching the native unit of
# Gaia's ``parallax`` column. No unit conversion needed.

_ZPT_REQUIRED_COLS: Final[tuple[str, ...]] = (
    "parallax",
    "phot_g_mean_mag",
    "nu_eff_used_in_astrometry",
    "pseudocolour",
    "ecl_lat",
    "astrometric_params_solved",
)


def apply_parallax_zpt(
    df: pd.DataFrame,
    *,
    zpt_col: str = "parallax_zpt",
    corrected_col: str = "parallax_corr",
) -> pd.DataFrame:
    """Apply Lindegren+2021 parallax zero-point. Returns a new DataFrame.

    Adds two columns:

    - ``parallax_zpt`` (mas) — the per-star zero-point. ``NaN`` for 2-param
      solutions, for which no correction is defined.
    - ``parallax_corr`` (mas) — ``parallax - parallax_zpt`` where the zpt is
      defined; equal to ``parallax`` where it isn't (2-param).

    Downstream code must use ``parallax_corr``, never raw ``parallax``.

    Raises
    ------
    ImportError
        If ``gaiadr3-zeropoint`` is not installed. The exception message
        names the pip install command.
    KeyError
        If any of the required Gaia columns is missing.
    """
    zpt = _load_zpt_module()

    missing = [c for c in _ZPT_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"apply_parallax_zpt requires columns {missing}")

    # ``load_tables`` is idempotent; safe to call every time.
    zpt.load_tables()

    five_or_six = df["astrometric_params_solved"].isin((SOL_FIVE_PARAM, SOL_SIX_PARAM))
    out = df.copy()
    out[zpt_col] = np.nan
    out[corrected_col] = out["parallax"].astype("float64")

    if five_or_six.any():
        sub = df.loc[five_or_six]
        zpt_mas = np.asarray(
            zpt.get_zpt(
                sub["phot_g_mean_mag"].to_numpy(),
                sub["nu_eff_used_in_astrometry"].to_numpy(),
                sub["pseudocolour"].to_numpy(),
                sub["ecl_lat"].to_numpy(),
                sub["astrometric_params_solved"].to_numpy(),
                _warnings=False,
            ),
            dtype="float64",
        )
        out.loc[five_or_six, zpt_col] = zpt_mas
        out.loc[five_or_six, corrected_col] = (
            sub["parallax"].to_numpy().astype("float64") - zpt_mas
        )

    n_two = int((~five_or_six).sum())
    if n_two:
        logger.info(
            "Lindegren zpt: %d 2-param (or unknown) solutions left uncorrected", n_two
        )
    logger.info("Lindegren zpt applied to %d/%d rows", int(five_or_six.sum()), len(df))
    return out


# Riello+2021 (A&A 649 A3) Appendix A cubic-polynomial factor applied to
# sources with 2-parameter (astrometric_params_solved == 3) or 6-parameter
# (== 95) astrometric solutions. Reference Python code is hosted at
# agabrown/gaiaedr3-6p-gband-correction.

_G_CORR_BP_RP_CLIP_LO: Final[float] = 0.25
_G_CORR_BP_RP_CLIP_HI: Final[float] = 3.0
_G_CORR_BRIGHT_COEFFS: Final[tuple[float, float, float, float]] = (
    1.00876, -0.02540, 0.01747, -0.00277,
)
_G_CORR_FAINT_COEFFS: Final[tuple[float, float, float, float]] = (
    1.00525, -0.02323, 0.01740, -0.00253,
)


def apply_g_mag_correction(
    df: pd.DataFrame,
    *,
    g_mag_col: str = "phot_g_mean_mag",
    bp_rp_col: str = "bp_rp",
    sol_col: str = "astrometric_params_solved",
    flux_col: str = "phot_g_mean_flux",
    corrected_mag_col: str = "phot_g_mean_mag_corr",
    corrected_flux_col: str = "phot_g_mean_flux_corr",
) -> pd.DataFrame:
    """Riello+2021 G-band flux/mag correction for Gaia EDR3/DR3 sources.

    Applies the Appendix A cubic-polynomial factor f(BP−RP) to sources with
    2-parameter (``astrometric_params_solved == 3``) or 6-parameter (``== 95``)
    astrometric solutions at G ≥ 13. Bright (13 ≤ G ≤ 16) and faint (G > 16)
    branches use distinct coefficients; BP−RP is clipped to [0.25, 3.0] before
    evaluation. 5-parameter solutions (``== 31``), sources with G < 13, and
    sources lacking BP−RP are passed through unchanged.

    Reference implementation:
    https://github.com/agabrown/gaiaedr3-6p-gband-correction
    (Riello et al. 2021, A&A 649, A3 Appendix A.)

    Parameters
    ----------
    df
        Gaia DR3-columns DataFrame. Must contain ``g_mag_col``, ``bp_rp_col``,
        ``sol_col``. ``flux_col`` is optional — only needed if the caller
        wants a corrected flux column.
    g_mag_col, bp_rp_col, sol_col, flux_col
        Input column names. Override if the caller renamed Gaia columns
        (e.g. DR19's ``g_mag`` instead of ``phot_g_mean_mag``).
    corrected_mag_col, corrected_flux_col
        Output column names.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``corrected_mag_col`` (always) and
        ``corrected_flux_col`` (iff ``flux_col`` present) added.

    Raises
    ------
    KeyError
        If any required input column is missing.
    """
    required = [g_mag_col, bp_rp_col, sol_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"apply_g_mag_correction requires columns {missing}")

    out = df.copy()
    bp_rp = out[bp_rp_col].to_numpy(dtype=np.float64, copy=False)
    g_mag = out[g_mag_col].to_numpy(dtype=np.float64, copy=False)
    sol = out[sol_col].to_numpy(dtype=np.int64, copy=False)

    do_not_correct = np.isnan(bp_rp) | (g_mag < 13.0) | (sol == SOL_FIVE_PARAM)
    bright = ~do_not_correct & (g_mag >= 13.0) & (g_mag <= 16.0)
    faint = ~do_not_correct & (g_mag > 16.0)
    bp_rp_c = np.clip(bp_rp, _G_CORR_BP_RP_CLIP_LO, _G_CORR_BP_RP_CLIP_HI)

    factor = np.ones_like(g_mag)
    cb = _G_CORR_BRIGHT_COEFFS
    cf = _G_CORR_FAINT_COEFFS
    factor[bright] = (
        cb[0] + cb[1] * bp_rp_c[bright] + cb[2] * bp_rp_c[bright] ** 2
        + cb[3] * bp_rp_c[bright] ** 3
    )
    factor[faint] = (
        cf[0] + cf[1] * bp_rp_c[faint] + cf[2] * bp_rp_c[faint] ** 2
        + cf[3] * bp_rp_c[faint] ** 3
    )

    out[corrected_mag_col] = g_mag - 2.5 * np.log10(factor)
    if flux_col in df.columns:
        out[corrected_flux_col] = (
            out[flux_col].to_numpy(dtype=np.float64, copy=False) * factor
        )

    n_corr = int(bright.sum() + faint.sum())
    logger.info(
        "Riello+2021 G-mag correction applied to %d/%d rows (bright=%d, faint=%d)",
        n_corr, len(df), int(bright.sum()), int(faint.sum()),
    )
    return out


def _load_zpt_module():
    """Lazy import of the Lindegren zero-point package with a helpful error."""
    try:
        from zero_point import zpt  # noqa: PLC0415 — deliberate lazy import
    except ImportError as exc:
        raise ImportError(
            "gaiadr3-zeropoint is not installed. Install with:\n"
            "    pip install gaiadr3-zeropoint --no-deps\n"
            "Reference: Lindegren+2021, https://www.cosmos.esa.int/web/gaia/edr3-code"
        ) from exc
    return zpt
