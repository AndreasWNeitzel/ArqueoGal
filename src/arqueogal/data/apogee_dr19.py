"""APOGEE DR19 (Mészáros+2025) summary file ingestion. Stream 1, step 1.

Reads the Astra/ASPCAP summary FITS (HDU 2), keeps only the columns we need
for training Pipeline 1, applies the §3.3 quality cuts, derives ``c_n`` if
absent, and hands off to provenance.

**Does not** download automatically at import time; pass a local ``Path`` to
``load_dr19``. For a one-shot "fetch + load" pipeline use
:func:`ingest_dr19_summary`, which composes this module with
``arqueogal.data.downloads``.

The §3.4 Mészáros+2025 Teff-trend corrections are implemented in
:func:`apply_meszaros2025_corrections`, which rewrites the 14 per-element
abundances listed in Mészáros Table 3 (α, O, Na, Mg, Al, Si, S, K, Ca, Ti,
Cr, Mn, Ni, Ce). [M/H], [Fe/H], C, N, V, Cu are intentionally left
uncorrected per Mészáros §4.1-4.2; do not extend the correction to those
labels. The correction is mandatory before use as training targets (see
``docs/data_acquisition.md`` §3.4).

DR19 ASPCAP summary naming conventions
--------------------------------------
The v0.6.0 summary file ``astraAllStarASPCAP-0.6.0.fits.gz`` uses:

- ``sdss4_apogee_id`` (not ``apogee_id``),
- ``gaia_dr3_source_id`` (not ``source_id``),
- ``v_sini`` / ``v_micro`` (not ``vsini`` / ``vmicro``); no ``vmacro``,
- ``{el}_h`` / ``e_{el}_h`` for calibrated [X/H] (not ``{el}_h_atm``),
- ``v_rad`` / ``e_v_rad`` for radial velocity (not ``vhelio_avg`` / ``_err``),
- ``task_pk`` (not ``task_id``),
- no per-parameter flags like ``teff_flag``; bitmasks live in
  ``flag_bad``, ``flag_warn``, ``result_flags``, ``initial_flags``,
  ``calibrated_flags``.
- no ``c_fe`` / ``n_fe``; we synthesise them from
  ``c_h``/``n_h``/``fe_h`` at load time (needed for [C/N]).

:data:`COLUMN_ALIASES` maps our canonical names → DR19 FITS names. The
loader also accepts either name directly, so test fixtures that pre-date
the DR19 schema still work.

References
----------
Mészáros et al. 2025, AJ in press, arXiv:2506.07845
data_acquisition.md §3 (columns, cuts, corrections, cross-match)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

DR19_SUMMARY_URL = (
    "https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraAllStarASPCAP-0.6.0.fits.gz"
)
DEFAULT_SUMMARY_FILENAME = "astraAllStarASPCAP-0.6.0.fits.gz"

# HDU 2 is the aggregated ASPCAP catalogue per data_acquisition.md §3.2.
ASPCAP_HDU = 2

# --- Columns we keep (data_acquisition.md §3.2) -------------------------------

IDENTIFIER_COLS = ("sdss_id", "apogee_id", "source_id")
"""Cross-match keys. ``source_id`` is Gaia DR3 (published directly in DR19)."""

POSITION_COLS = ("ra_deg", "dec_deg")
"""Equatorial coordinates from DR19 (ASPCAP cross-match positions)."""

ASTROMETRY_COLS = (
    "parallax_mas",
    "parallax_err_mas",
    "pmra_mas_yr",
    "pmra_err_mas_yr",
    "pmdec_mas_yr",
    "pmdec_err_mas_yr",
)
"""Gaia DR3 astrometry propagated into DR19 (``plx``, ``pmra``, ``pmde`` + errs).
Zero-point correction and G-mag correction still happen in :mod:`gaia_corrections`."""

PHOTOMETRY_COLS = (
    "g_mag",
    "bp_mag",
    "rp_mag",
    "j_mag",
    "e_j_mag",
    "h_mag",
    "e_h_mag",
    "k_mag",
    "e_k_mag",
    "w1_mag",
    "e_w1_mag",
    "w2_mag",
    "e_w2_mag",
)
"""Gaia DR3 + 2MASS + WISE photometry as ingested by Astra (DR19)."""

DUST_FROM_DR19_COLS = (
    "ebv",
    "e_ebv",
    "ebv_edenhofer_2023",
    "e_ebv_edenhofer_2023",
    "ebv_bayestar_2019",
    "e_ebv_bayestar_2019",
    "ebv_zhang_2023",
    "e_ebv_zhang_2023",
    "ebv_sfd",
    "e_ebv_sfd",
)
"""Per-star 3D + 2D dust values pre-baked into DR19.

Coverage on the full DR19 catalogue (pre-quality-cuts):

    ebv               100.0%   (composite default column)
    ebv_edenhofer_2023  51.8%  (3D, d < 1.25 kpc)
    ebv_bayestar_2019   86.8%  (3D, all-sky N of δ = -30°)
    ebv_zhang_2023      88.0%  (XP-based, alternative 3D)
    ebv_sfd            100.0%  (2D upper limit, full sky)

This supersedes the planned separate Edenhofer+Lallement+SFD composition
for Stream 1 (see ``docs/data_acquisition.md`` §6) — the DR19 catalogue
already carries all four maps per star. Stream 3 (Gaia-only ~2 M RGB
subsample) still needs an external 3D dust map since those stars were not
observed by APOGEE."""

BJ_DISTANCE_COLS = (
    "r_med_photogeo",
    "r_lo_photogeo",
    "r_hi_photogeo",
    "r_med_geo",
    "r_lo_geo",
    "r_hi_geo",
)
"""Bailer-Jones+2021 distances pre-joined into DR19. Same values as the
GAVO ``gedr3dist.main`` fetch — handy as an independent cross-check."""

ATMOS_COLS = (
    "teff",
    "e_teff",
    "logg",
    "e_logg",
    "m_h_atm",
    "e_m_h_atm",
    "alpha_m_atm",
    "e_alpha_m_atm",
    "vsini",
    "vmicro",
)
"""Atmospheric parameters. DR19 does not publish macroturbulence (``vmacro``)."""

ABUNDANCE_ELEMENTS = (
    "c",
    "n",
    "o",
    "na",
    "mg",
    "al",
    "si",
    "s",
    "k",
    "ca",
    "ti",
    "v",
    "cr",
    "mn",
    "fe",
    "ni",
    "ce",
)
"""[X/H] abundances kept from DR19 (calibrated). DR19 names them ``{el}_h``;
we expose them downstream as ``{el}_h_atm`` to match data_acquisition.md §3.2."""

FLAG_COLS = (
    "flag_bad",
    "flag_warn",
    "snr",
    "result_flags",
    "initial_flags",
    "calibrated_flags",
    "vhelio_avg",
    "vhelio_err",
)
"""DR19 quality flags and radial velocity. ``flag_bad``, ``flag_warn``,
``result_flags``, ``initial_flags``, ``calibrated_flags`` are bitmasks."""

META_COLS = ("v_astra", "task_id", "spectrum_pk")


# --- DR19 column aliases: canonical → FITS actual ----------------------------

COLUMN_ALIASES: dict[str, str] = {
    "apogee_id": "sdss4_apogee_id",
    "source_id": "gaia_dr3_source_id",
    "ra_deg": "ra",
    "dec_deg": "dec",
    "parallax_mas": "plx",
    "parallax_err_mas": "e_plx",
    "pmra_mas_yr": "pmra",
    "pmra_err_mas_yr": "e_pmra",
    "pmdec_mas_yr": "pmde",
    "pmdec_err_mas_yr": "e_pmde",
    "vsini": "v_sini",
    "vmicro": "v_micro",
    "vhelio_avg": "v_rad",
    "vhelio_err": "e_v_rad",
    "task_id": "task_pk",
    **{f"{el}_h_atm": f"{el}_h" for el in ABUNDANCE_ELEMENTS},
    **{f"e_{el}_h_atm": f"e_{el}_h" for el in ABUNDANCE_ELEMENTS},
}


# Columns this loader can synthesise from others (so kept_columns() may include
# them even when the DR19 file does not publish them directly).
_SYNTHESISABLE: frozenset[str] = frozenset({"c_fe", "e_c_fe", "n_fe", "e_n_fe"})


def kept_columns() -> list[str]:
    """Full list of DR19 columns this module expects / reads."""
    cols: list[str] = []
    cols.extend(IDENTIFIER_COLS)
    cols.extend(POSITION_COLS)
    cols.extend(ASTROMETRY_COLS)
    cols.extend(PHOTOMETRY_COLS)
    cols.extend(DUST_FROM_DR19_COLS)
    cols.extend(BJ_DISTANCE_COLS)
    cols.extend(ATMOS_COLS)
    for el in ABUNDANCE_ELEMENTS:
        cols.extend([f"{el}_h_atm", f"e_{el}_h_atm"])
    # [X/Fe] derivatives for C and N — synthesised from [X/H] and [Fe/H] if
    # the FITS does not publish them directly (DR19 does not).
    cols.extend(["c_fe", "e_c_fe", "n_fe", "e_n_fe"])
    cols.extend(FLAG_COLS)
    cols.extend(META_COLS)
    return cols


# --- Quality cuts (data_acquisition.md §3.3) ---------------------------------


@dataclass(frozen=True, slots=True)
class QualityCuts:
    """Training-set quality cuts.

    Kiel-diagram cuts (Teff, log g) were dropped on 2026-04-29 — the OOD-flag
    machinery downstream is responsible for catching evolutionary-stage
    out-of-distribution stars at inference time. Only spectroscopic-quality
    and metallicity-range cuts remain.
    """

    min_snr: float = 70.0
    m_h_min: float = -2.0
    m_h_max: float = 0.5

    def as_predicates(self) -> list[str]:
        return [
            "flag_bad == 0",
            f"snr > {self.min_snr}",
            f"m_h_atm in [{self.m_h_min}, {self.m_h_max}]",
        ]


# --- I/O -----------------------------------------------------------------------


def load_dr19(
    path: Path | str,
    *,
    columns: list[str] | None = None,
    hdu: int = ASPCAP_HDU,
) -> pd.DataFrame:
    """Load the DR19 summary FITS into a pandas DataFrame.

    Parameters
    ----------
    columns
        Subset of columns to keep. Defaults to :func:`kept_columns`. Names are
        the canonical ones listed in the module-level constants (e.g.
        ``source_id``, ``vsini``, ``c_h_atm``). DR19's actual FITS column
        names are resolved via :data:`COLUMN_ALIASES`.
    hdu
        HDU index. Defaults to 2 (ASPCAP aggregated catalogue).

    Notes
    -----
    - Reads gzipped FITS natively (astropy.io.fits).
    - For each requested canonical column, attempts in order:
      (1) the aliased DR19 FITS name, (2) the canonical name itself
      (so test fixtures and any pre-DR19 files still work), and
      (3) synthesis from primary DR19 columns (for ``c_fe``, ``n_fe`` and
      their errors, built from ``{el}_h_atm`` - ``fe_h_atm``).
    - If a requested column can be neither found nor synthesised, raises
      ``KeyError`` naming the missing columns explicitly.
    - Byte-order-swaps native-endian arrays (FITS is big-endian; pandas wants
      native) before wrapping in a DataFrame.
    """
    path = Path(path)
    columns = list(columns) if columns is not None else kept_columns()

    with fits.open(path, memmap=True) as hdul:
        table = hdul[hdu].data
        present: dict[str, str] = {name.lower(): name for name in table.columns.names}

        frame: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for col in columns:
            candidates = [COLUMN_ALIASES.get(col, col), col]
            fits_name: str | None = None
            for cand in candidates:
                hit = present.get(cand.lower())
                if hit is not None:
                    fits_name = hit
                    break
            if fits_name is not None:
                arr = np.asarray(table[fits_name])
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
                frame[col] = arr
            elif col in _SYNTHESISABLE:
                # Defer to the synthesis pass below.
                continue
            else:
                missing.append(col)
        if missing:
            raise KeyError(
                f"{path}: DR19 HDU {hdu} missing expected columns {missing}. "
                f"Column naming may have changed — check the Astra schema docs."
            )

    df = pd.DataFrame(frame)

    # Synthesise [X/Fe] = [X/H] - [Fe/H] and quadrature-sum errors where
    # requested but not published directly (the DR19 case).
    for col in columns:
        if col in df.columns:
            continue
        if col == "c_fe":
            df["c_fe"] = df["c_h_atm"].astype(np.float64) - df["fe_h_atm"].astype(np.float64)
        elif col == "n_fe":
            df["n_fe"] = df["n_h_atm"].astype(np.float64) - df["fe_h_atm"].astype(np.float64)
        elif col == "e_c_fe":
            df["e_c_fe"] = np.sqrt(
                df["e_c_h_atm"].astype(np.float64) ** 2 + df["e_fe_h_atm"].astype(np.float64) ** 2
            )
        elif col == "e_n_fe":
            df["e_n_fe"] = np.sqrt(
                df["e_n_h_atm"].astype(np.float64) ** 2 + df["e_fe_h_atm"].astype(np.float64) ** 2
            )

    # Reorder to requested column order.
    df = df[columns]
    logger.info("loaded %s: %d rows × %d cols", path.name, len(df), len(df.columns))
    return df


# --- Quality cuts & derivations -----------------------------------------------


def apply_quality_cuts(
    df: pd.DataFrame, cuts: QualityCuts | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply §3.3 cuts, return (post-cut DataFrame, stage-by-stage counts)."""
    cuts = cuts or QualityCuts()
    stats: dict[str, int] = {"before": len(df)}

    mask = df["flag_bad"] == 0
    stats["after_flag_bad"] = int(mask.sum())
    mask &= df["snr"] > cuts.min_snr
    stats["after_snr"] = int(mask.sum())
    # Teff / log g Kiel cuts removed 2026-04-29 (OOD flags handle evolutionary
    # out-of-distribution stars downstream).
    mask &= df["m_h_atm"].between(cuts.m_h_min, cuts.m_h_max)
    stats["after_m_h"] = int(mask.sum())

    out = df.loc[mask].reset_index(drop=True)
    stats["after"] = len(out)
    logger.info("quality cuts: %s", stats)
    return out, stats


def derive_c_n(df: pd.DataFrame) -> pd.DataFrame:
    """Derive [C/N] = [C/Fe] - [N/Fe] in-place (if absent) and propagate errors.

    The [C/Fe] and [N/Fe] inputs are either read directly from the FITS
    (legacy naming) or synthesised by :func:`load_dr19` from the DR19
    ``{el}_h`` calibrated abundances.
    """
    if "c_n" in df.columns:
        return df

    if not {"c_fe", "n_fe"}.issubset(df.columns):
        raise KeyError("derive_c_n requires both 'c_fe' and 'n_fe' columns")

    df["c_n"] = df["c_fe"].astype(np.float64) - df["n_fe"].astype(np.float64)
    if {"e_c_fe", "e_n_fe"}.issubset(df.columns):
        df["e_c_n"] = np.sqrt(
            df["e_c_fe"].astype(np.float64) ** 2 + df["e_n_fe"].astype(np.float64) ** 2
        )
    return df


# --- Mészáros+2025 Teff-trend corrections ------------------------------------

MESZAROS2025_TEFF_MIN = 3500.0
MESZAROS2025_TEFF_MAX = 6000.0
MESZAROS2025_LOGG_MAX = 3.8

# Mészáros+2025 (arXiv:2506.07845) Table 3. Linear trend Δ[X/M] = a·Teff + b
# fit to open-cluster members in 3500 < Teff < 6000 K and log g < 3.8.
# Outside the Teff window the constant boundary offsets are used (evaluation
# of a·Teff+b pinned to the nearest endpoint).
# Corrected [X/M] = raw [X/M] − Δ[X/M]. Since [X/H] = [X/M] + [M/H], the
# correction applies identically to [X/H] (we overwrite the _h_atm column).
# C and N are deliberately omitted: first-dredge-up / thermohaline mixing
# means cluster giants are not born with solar [C/N], so the cluster-based
# detrending is not astrophysically valid for those elements. Fe is the
# reference (Δ[Fe/M] ≡ 0 by construction). V and Cu are not published in
# Mészáros+2025 Table 3 so they receive no correction here.
# Allowlist: the exact 14 columns Mészáros+2025 Table 3 publishes coefficients
# for. Any new key in MESZAROS2025_COEFFS must be peer-reviewed *and* added to
# this allowlist deliberately. The module-import-time assertion catches drift
# (typo, accidental copy/paste, double-correction of an unpublished element)
# before any pipeline runs; the ``MappingProxyType`` wrapper below freezes the
# dict at *runtime* too, so a contributor cannot widen the contract by
# mutating the dict after import. See AGENTS.md invariant 13.
MESZAROS2025_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "alpha_m_atm",
        "o_h_atm",
        "na_h_atm",
        "mg_h_atm",
        "al_h_atm",
        "si_h_atm",
        "s_h_atm",
        "k_h_atm",
        "ca_h_atm",
        "ti_h_atm",
        "cr_h_atm",
        "mn_h_atm",
        "ni_h_atm",
        "ce_h_atm",
    }
)

_MESZAROS2025_COEFFS_RAW: dict[str, tuple[float, float, float, float]] = {
    # canonical column → (a, b, offset_hot_Teff>6000, offset_cold_Teff<3500)
    "alpha_m_atm": (-2.2918e-5, 0.0861, -0.0514, 0.0059),
    "o_h_atm": (-4.0909e-5, 0.1651, -0.0804, 0.0219),
    "na_h_atm": (-8.2173e-5, 0.4586, -0.0344, 0.1710),
    "mg_h_atm": (-4.3932e-5, 0.1733, -0.0903, 0.0195),
    "al_h_atm": (3.0850e-5, -0.1734, 0.0117, -0.0654),
    "si_h_atm": (1.1688e-5, -0.0594, 0.0107, -0.0185),
    "s_h_atm": (-1.0886e-6, 0.0214, 0.0149, 0.0176),
    "k_h_atm": (-7.0765e-5, 0.3220, -0.1026, 0.0743),
    "ca_h_atm": (5.4495e-5, -0.2832, 0.0438, -0.0925),
    "ti_h_atm": (-1.1466e-4, 0.5137, -0.1743, 0.1124),
    "cr_h_atm": (-6.4991e-6, 0.0099, -0.0291, -0.0128),
    "mn_h_atm": (-1.0168e-4, 0.4999, -0.1102, 0.1440),
    "ni_h_atm": (-2.3203e-5, 0.0806, -0.0586, -0.0006),
    "ce_h_atm": (1.3833e-4, -0.5431, 0.2869, -0.0589),
}
if frozenset(_MESZAROS2025_COEFFS_RAW) != MESZAROS2025_ALLOWED_KEYS:
    raise RuntimeError(
        "MESZAROS2025_COEFFS drifted from the published Table 3 allowlist. "
        f"Coefficient keys: {sorted(_MESZAROS2025_COEFFS_RAW)}. "
        f"Allowed keys: {sorted(MESZAROS2025_ALLOWED_KEYS)}. "
        "Updating either set requires the corresponding peer-reviewed source "
        "and a documented rationale."
    )

# Public read-only view: ``MappingProxyType`` rejects __setitem__/__delitem__ at
# runtime so the 14-element contract cannot be widened post-import without
# touching this file (which then re-runs the allowlist check). Every consumer
# below reads it as a normal mapping; only mutation paths trip.
MESZAROS2025_COEFFS: Mapping[str, tuple[float, float, float, float]] = MappingProxyType(
    _MESZAROS2025_COEFFS_RAW
)


def _meszaros_delta(
    teff: np.ndarray, logg: np.ndarray, a: float, b: float, hot: float, cold: float
) -> np.ndarray:
    """Per-star Δ[X/M] for a single element. NaN wherever logg ≥ 3.8."""
    teff64 = np.asarray(teff, dtype=np.float64)
    logg64 = np.asarray(logg, dtype=np.float64)
    delta = np.where(
        teff64 > MESZAROS2025_TEFF_MAX,
        hot,
        np.where(teff64 < MESZAROS2025_TEFF_MIN, cold, a * teff64 + b),
    )
    delta[~np.isfinite(teff64) | ~np.isfinite(logg64)] = np.nan
    delta[logg64 >= MESZAROS2025_LOGG_MAX] = np.nan
    return delta


def apply_meszaros2025_corrections(
    df: pd.DataFrame, *, elements: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Apply Mészáros+2025 Teff-trend corrections to DR19 [X/M] abundances.

    Rewrites ``alpha_m_atm`` and ``{el}_h_atm`` in place as
    ``raw − (a·Teff + b)`` on a copy of ``df`` (original unchanged).
    Outside the calibrated window (Teff ∉ [3500, 6000] or log g ≥ 3.8) the
    constant boundary offsets from Mészáros+2025 Table 3 are used; stars
    with log g ≥ 3.8 or non-finite Teff/logg are left uncorrected (NaN
    delta → no shift). The full set of elements actually corrected is
    returned in the ``meszaros_correction_summary`` DataFrame attribute.

    Parameters
    ----------
    df
        Post-quality-cut DR19 DataFrame. Must carry ``teff``, ``logg``, and
        at least one of the supported ``{el}_h_atm`` / ``alpha_m_atm``
        columns.
    elements
        Optional restriction to a subset of keys from
        :data:`MESZAROS2025_COEFFS`. Defaults to every supported element
        that is present in ``df``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with corrected abundances. A
        ``meszaros_correction_summary`` attribute (pandas.DataFrame) is
        attached via ``df.attrs`` holding per-element (n, mean_shift, rms_shift).
    """
    required = {"teff", "logg"}
    if not required.issubset(df.columns):
        raise KeyError(f"apply_meszaros2025_corrections requires columns {sorted(required)}")

    out = df.copy()
    teff = out["teff"].to_numpy(dtype=np.float64, copy=False)
    logg = out["logg"].to_numpy(dtype=np.float64, copy=False)

    keys = tuple(elements) if elements is not None else tuple(MESZAROS2025_COEFFS)
    unknown = [k for k in keys if k not in MESZAROS2025_COEFFS]
    if unknown:
        raise KeyError(f"no Mészáros+2025 coefficients for: {unknown}")

    summary_rows: list[dict[str, float]] = []
    for col in keys:
        if col not in out.columns:
            logger.info("skipping %s: column absent from frame", col)
            continue
        a, b, hot, cold = MESZAROS2025_COEFFS[col]
        delta = _meszaros_delta(teff, logg, a, b, hot, cold)
        raw = out[col].to_numpy(dtype=np.float64, copy=True)
        corrected = np.where(np.isfinite(delta), raw - delta, raw)
        out[col] = corrected.astype(out[col].dtype, copy=False)
        applied = np.isfinite(delta) & np.isfinite(raw)
        shift = (corrected - raw)[applied]
        summary_rows.append(
            {
                "element": col,
                "n_applied": int(applied.sum()),
                "mean_shift": float(np.mean(shift)) if shift.size else float("nan"),
                "rms_shift": float(np.sqrt(np.mean(shift**2))) if shift.size else float("nan"),
            }
        )
        logger.info(
            "Mészáros+2025 correction applied to %s: n=%d, <Δ>=%.4f dex, rms=%.4f dex",
            col,
            summary_rows[-1]["n_applied"],
            summary_rows[-1]["mean_shift"],
            summary_rows[-1]["rms_shift"],
        )

    out.attrs["meszaros_correction_summary"] = pd.DataFrame(summary_rows)
    return out
