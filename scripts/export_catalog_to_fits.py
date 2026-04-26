"""Export a Pipeline 1 release Parquet to FITS binary table format.

Why FITS
--------

Astronomy consumers default to TOPCAT, astropy.io.fits, glue, and DS9 — all
FITS-native. Without a FITS export the D-Cat-b release is invisible to the
community's standard discovery and visualisation workflow (META_META §14.4
P0; data_preparation_output.md CRITICAL VizieR/FITS blocker).

This script reads a Parquet annotated by ``release.annotate_parquet`` and
writes a FITS binary table with proper unit headers, UCDs (Unified Content
Descriptors per IVOA), and per-column descriptions sourced from the
``CATALOG_SCHEMA.md`` reference.

Usage
-----

    python scripts/export_catalog_to_fits.py \\
        --input data/processed/pipeline1_inference_v1.parquet \\
        --output release/D-Cat-b/D-Cat-b_v1.0.fits \\
        --release-tag pipeline1-v1-2026-04-19

The output FITS file carries:

- HDU 0: PRIMARY with provenance keywords (release tag, git SHA, schema
  version, frozen-stats fingerprint, timestamp).
- HDU 1: BINTABLE with the catalog rows. Per-column TUNIT, TUCD (where
  applicable), TCOMM (description) headers.

Format choices
--------------

- ``str`` columns are written as ``TFORM=NA`` with appropriate length.
- ``int8`` tiers and ``bool`` flags are preserved as ``TFORM=B`` (1-byte
  unsigned integer; FITS does not have a native boolean dtype).
- ``float32`` predictions are preserved (no upcast); float64 columns
  remain float64.

Astropy is used for IO. The script is conservative about memory: large
catalogs (>1 GB Parquet) are read with pyarrow, then handed to astropy
table.from_pandas().

Status
------

This script is the FIRST FITS-export path for ArqueoGal. It is exercised
by tests/scripts/test_export_catalog_to_fits.py (when present) and by
manual verification on a 100-row holdout.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Lazy imports so the module is importable in environments without astropy.
# Actual export requires astropy.


# -----------------------------------------------------------------------------
# UCD (Unified Content Descriptor) catalogue.
# References: IVOA UCD1+ specification, Gaia DR3 column UCDs (ESA / DPAC).
# Where a column has no obvious UCD, we leave it blank (FITS allows omission).
# -----------------------------------------------------------------------------

_COLUMN_UCDS: dict[str, str] = {
    "source_id": "meta.id;meta.main",
    "ra": "pos.eq.ra;meta.main",
    "dec": "pos.eq.dec;meta.main",
    "parallax": "pos.parallax",
    "parallax_error": "stat.error;pos.parallax",
    "pmra": "pos.pm;pos.eq.ra",
    "pmdec": "pos.pm;pos.eq.dec",
    "phot_g_mean_mag": "phot.mag;em.opt",
    "phot_bp_mean_mag": "phot.mag;em.opt.B",
    "phot_rp_mean_mag": "phot.mag;em.opt.R",
    "teff_pred": "phys.temperature.effective",
    "teff_sigma": "stat.error;phys.temperature.effective",
    "logg_pred": "phys.gravity",
    "logg_sigma": "stat.error;phys.gravity",
    "mh_pred": "phys.abund.Z",
    "mh_sigma": "stat.error;phys.abund.Z",
    "alpha_m_pred": "phys.abund",
    "alpha_m_sigma": "stat.error;phys.abund",
    "mg_h_pred": "phys.abund.Mg",
    "mg_h_sigma": "stat.error;phys.abund.Mg",
    "release_tier": "meta.code.qual",
    "release_tier__teff": "meta.code.qual",
    "release_tier__logg": "meta.code.qual",
    "release_tier__mh": "meta.code.qual",
    "release_tier__alpha_m": "meta.code.qual",
    "release_tier__mg_h": "meta.code.qual",
    "ood_joint_flag": "meta.code.qual",
    "ood_aux_mahalanobis_flag": "meta.code.qual",
    "kin_ood_flag": "meta.code.qual",
    "dist_prior_dominated": "meta.code.qual",
    "regime_b_flag": "meta.code.qual",
    "g_mag_bin": "meta.code.class",
    "xp_abundance_type__teff": "meta.code.class",
    "xp_abundance_type__logg": "meta.code.class",
    "xp_abundance_type__mh": "meta.code.class",
    "xp_abundance_type__alpha_m": "meta.code.class",
    "xp_abundance_type__mg_h": "meta.code.class",
}


_COLUMN_UNITS: dict[str, str] = {
    "ra": "deg",
    "dec": "deg",
    "parallax": "mas",
    "parallax_error": "mas",
    "pmra": "mas/yr",
    "pmdec": "mas/yr",
    "phot_g_mean_mag": "mag",
    "phot_bp_mean_mag": "mag",
    "phot_rp_mean_mag": "mag",
    "teff_pred": "K",
    "teff_sigma": "K",
    # logg, mh, alpha_m, mg_h, predictions and sigmas are dimensionless dex.
    "logg_pred": "dex",
    "logg_sigma": "dex",
    "mh_pred": "dex",
    "mh_sigma": "dex",
    "alpha_m_pred": "dex",
    "alpha_m_sigma": "dex",
    "mg_h_pred": "dex",
    "mg_h_sigma": "dex",
}


_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "release_tier": "Composite per-row tier; max across release_tier__<element>.",
    "release_tier__teff": "Per-element tier for Teff; 1=Tier 1, 2=Tier 2, 3=Tier 3.",
    "release_tier__logg": "Per-element tier for logg.",
    "release_tier__mh": "Per-element tier for [M/H].",
    "release_tier__alpha_m": "Per-element tier for [alpha/M] (aux-assisted; demoted on kin_ood_flag).",
    "release_tier__mg_h": "Per-element tier for [Mg/H] (aux-assisted; demoted on kin_ood_flag).",
    "kin_ood_flag": "Kinematic OOD flag; True = star kinematically anomalous vs disc training.",
    "dist_prior_dominated": "Distance is prior-dominated (parallax SNR < 5 per Bailer-Jones+2021).",
    "ood_aux_mahalanobis_flag": "Aux-feature Mahalanobis OOD flag.",
    "xp_abundance_type__alpha_m": "spectrum_dominant or aux_assisted (CMI < 0.02 nats given aux).",
    "xp_abundance_type__mg_h": "spectrum_dominant or aux_assisted (CMI < 0.02 nats given aux).",
    "xp_abundance_type__teff": "spectrum_dominant or aux_assisted.",
    "xp_abundance_type__logg": "spectrum_dominant or aux_assisted.",
    "xp_abundance_type__mh": "spectrum_dominant or aux_assisted.",
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_to_fits(
    parquet_path: Path,
    fits_path: Path,
    *,
    release_tag: str,
    schema_version: int = 3,
    frozen_stats_fingerprint: str | None = None,
    overwrite: bool = False,
) -> dict[str, str | int]:
    """Export a release Parquet to a FITS binary table.

    Parameters
    ----------
    parquet_path : Path
        Input Parquet (must be release-annotated by ``release.annotate_parquet``).
    fits_path : Path
        Output FITS file path. Parent dir created if missing.
    release_tag : str
        Git tag (e.g., ``pipeline1-v1-2026-04-19``) recorded in PRIMARY HDU.
    schema_version : int
        Catalog schema version; recorded in PRIMARY HDU.
    frozen_stats_fingerprint : str, optional
        Hermite basis fingerprint; if provided, recorded in PRIMARY HDU.
    overwrite : bool
        Pass-through to astropy fits.writeto.

    Returns
    -------
    dict
        Summary: ``{"n_rows": ..., "fits_sha256": ..., "git_sha": ...}``.

    Notes
    -----
    Astropy import is lazy. If astropy is not available, raises ImportError
    with installation hint.
    """
    try:
        import pyarrow.parquet as pq  # noqa: F401
        from astropy.io import fits
        from astropy.table import Table
    except ImportError as e:
        raise ImportError(
            "FITS export requires astropy and pyarrow. Install via "
            "`uv add astropy pyarrow` or `pip install astropy pyarrow`.",
        ) from e

    fits_path.parent.mkdir(parents=True, exist_ok=True)

    # Use astropy.table directly from the Parquet via from_pandas — keeps
    # numerical dtypes intact and supports unit / description metadata.
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    table = Table.from_pandas(df)

    # Apply per-column units, UCDs, descriptions where defined.
    for col_name in table.colnames:
        if col_name in _COLUMN_UNITS:
            # Non-numeric columns may reject units; that's fine.
            with contextlib.suppress(Exception):
                table[col_name].unit = _COLUMN_UNITS[col_name]
        if col_name in _COLUMN_UCDS:
            table[col_name].meta["UCD"] = _COLUMN_UCDS[col_name]
        if col_name in _COLUMN_DESCRIPTIONS:
            table[col_name].description = _COLUMN_DESCRIPTIONS[col_name]

    # Build PRIMARY header with provenance.
    primary_header = fits.Header()
    primary_header["DATE"] = (
        datetime.now(tz=UTC).isoformat(),
        "Date of this FITS file creation (UTC)",
    )
    primary_header["RELTAG"] = (release_tag, "ArqueoGal release tag")
    primary_header["SCHEMVER"] = (schema_version, "Catalog schema version")
    primary_header["GIT_SHA"] = (_git_sha()[:12], "Git SHA at export time")
    if frozen_stats_fingerprint is not None:
        primary_header["FROZSTAT"] = (
            frozen_stats_fingerprint[:16],
            "Hermite basis fingerprint",
        )
    primary_header["INPSHA"] = (_file_sha256(parquet_path)[:16], "Input parquet SHA-256 (head)")
    primary_header["NROW"] = (int(len(df)), "Number of rows in BINTABLE")
    primary_header["AUTHOR"] = ("Andreas W. Neitzel et al.", "ArqueoGal collaboration")

    primary_hdu = fits.PrimaryHDU(header=primary_header)
    bintable_hdu = fits.BinTableHDU(data=table.as_array(), name="DCATB")

    # Re-apply column-level metadata to the bintable HDU since astropy.io.fits
    # constructs from the masked recarray and may not propagate all unit/UCD/desc.
    for i, col_name in enumerate(bintable_hdu.columns.names):
        if col_name in _COLUMN_UNITS:
            with contextlib.suppress(Exception):
                bintable_hdu.columns[i].unit = _COLUMN_UNITS[col_name]
        if col_name in _COLUMN_UCDS:
            bintable_hdu.header[f"TUCD{i + 1}"] = _COLUMN_UCDS[col_name]
        if col_name in _COLUMN_DESCRIPTIONS:
            bintable_hdu.header[f"TCOMM{i + 1}"] = _COLUMN_DESCRIPTIONS[col_name][:68]

    hdul = fits.HDUList([primary_hdu, bintable_hdu])
    # checksum=True writes CHECKSUM/DATASUM cards into each HDU header per the
    # FITS standard. Required for archival catalogs (CDS / VizieR strongly
    # prefer it for downstream integrity verification).
    hdul.writeto(fits_path, overwrite=overwrite, checksum=True)

    return {
        "n_rows": int(len(df)),
        "fits_sha256": _file_sha256(fits_path),
        "git_sha": _git_sha(),
        "release_tag": release_tag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet path.")
    parser.add_argument("--output", type=Path, required=True, help="Output FITS path.")
    parser.add_argument("--release-tag", type=str, required=True, help="Git tag identifier.")
    parser.add_argument(
        "--schema-version",
        type=int,
        default=3,
        help="Catalog schema version (default 3 — Phase A2-followup).",
    )
    parser.add_argument(
        "--frozen-stats-fingerprint",
        type=str,
        default=None,
        help="Hermite basis fingerprint (optional).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FITS file.",
    )
    args = parser.parse_args()

    summary = export_to_fits(
        parquet_path=args.input,
        fits_path=args.output,
        release_tag=args.release_tag,
        schema_version=args.schema_version,
        frozen_stats_fingerprint=args.frozen_stats_fingerprint,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
