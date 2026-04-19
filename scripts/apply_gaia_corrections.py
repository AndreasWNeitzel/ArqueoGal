"""Apply the mandatory Gaia DR3 corrections to a raw enrichment Parquet.

Per ``docs/data_acquisition.md`` §3.7, every downstream use of Gaia DR3
astrometry/photometry must go through:

1. Lindegren+2021 parallax zero-point (via ``gaiadr3-zeropoint``); writes
   ``parallax_zpt`` and ``parallax_corr`` columns.
2. Riello+2021 (Appendix A, A&A 649 A3) G-band flux/mag correction for
   2-/6-parameter astrometric solutions; writes ``phot_g_mean_mag_corr`` and
   ``phot_g_mean_flux_corr`` (the latter only if the flux column is present —
   it isn't in the AIP enrichment bundle, so typically only mag is written).

Produces ``<stem>_corrected.parquet`` + provenance sidecar next to the input.

Usage
-----
    python scripts/apply_gaia_corrections.py data/interim/stream1_gaia_dr3_raw.parquet
    python scripts/apply_gaia_corrections.py data/interim/stream3_gaia_dr3_raw.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.gaia_corrections import (
    apply_g_mag_correction,
    apply_parallax_zpt,
)
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apply_gaia_corrections")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path, help="Path to stream{1,3}_gaia_dr3_raw.parquet")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path (default: <stem>_corrected.parquet).",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    raw: Path = args.raw.resolve()
    if not raw.exists():
        raise SystemExit(f"missing input: {raw}")

    if args.out is not None:
        out = args.out.resolve()
    else:
        stem = raw.stem
        suffix = "_corrected.parquet"
        if stem.endswith("_raw"):
            stem = stem[:-4]
        out = raw.with_name(stem + suffix)

    logger.info("loading %s", raw)
    df = pd.read_parquet(raw)
    n_in = len(df)
    logger.info("loaded %d rows × %d cols", n_in, len(df.columns))

    df_zpt = apply_parallax_zpt(df)
    df_corr = apply_g_mag_correction(df_zpt)

    # Cast correction columns to float32 — 1 µmas precision is overkill.
    for col in ("parallax_zpt", "parallax_corr", "phot_g_mean_mag_corr"):
        if col in df_corr.columns and df_corr[col].dtype == np.float64:
            df_corr[col] = df_corr[col].astype(np.float32)

    _write_parquet_atomic(df_corr, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", out, size_mb, len(df_corr.columns))

    n_zpt_applied = int(df_corr["parallax_zpt"].notna().sum())
    raw_sha = _sha256_of(raw)
    prov = Provenance(
        output_file=str(out.relative_to(repo)) if out.is_relative_to(repo) else str(out),
        script="scripts/apply_gaia_corrections.py",
        sources=[
            LocalSource(
                name="Gaia DR3 raw enrichment",
                path=str(raw.relative_to(repo)) if raw.is_relative_to(repo) else str(raw),
                sha256=raw_sha,
            ),
        ],
        cuts_applied=[],
        corrections=[
            "Lindegren+2021 parallax zero-point (gaiadr3-zeropoint) — "
            f"{n_zpt_applied}/{n_in} rows (5-/6-param solutions)",
            "Riello+2021 G-band flux/mag correction (gaiaedr3-6p-gband-correction) — "
            "2-/6-param solutions at G ≥ 13",
        ],
        row_count_before=n_in,
        row_count_after=len(df_corr),
        notes=(
            "Adds parallax_zpt, parallax_corr, phot_g_mean_mag_corr columns. "
            "Downstream code must use parallax_corr and phot_g_mean_mag_corr — "
            "never the raw columns (docs/data_acquisition.md §3.7)."
        ),
        extra={
            "zpt_rows_applied": n_zpt_applied,
            "zpt_rows_skipped_two_param": n_in - n_zpt_applied,
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
