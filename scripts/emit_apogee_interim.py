"""Re-emit ``data/interim/apogee_dr19_precorrected.parquet`` with the richer
DR19 column set (Gaia astrometry + 2MASS/WISE photometry + per-star dust +
Bailer-Jones distances now exposed by :func:`arqueogal.data.apogee_dr19.load_dr19`).

Pre-corrections Parquet; Mészáros+2025 [X/M]/Teff corrections are applied
downstream by ``emit_apogee_corrected.py``. Gaia-side Lindegren+2021 zpt and
Riello+2021 G-mag corrections are applied in the Gaia enrichment scripts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from arqueogal.data.apogee_dr19 import (
    QualityCuts,
    apply_quality_cuts,
    derive_c_n,
    kept_columns,
    load_dr19,
)
from arqueogal.data.downloads import sha256_file
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("emit_apogee_interim")


def _cast_floats_float32(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = repo / "data" / "raw" / "apogee_dr19" / "astraAllStarASPCAP-0.6.0.fits.gz"
    out = repo / "data" / "interim" / "apogee_dr19_precorrected.parquet"

    logger.info("loading %s", raw)
    df = load_dr19(raw)
    logger.info("loaded %d rows × %d cols", len(df), len(df.columns))

    cut_df, stats = apply_quality_cuts(df, QualityCuts())
    logger.info("post-cut: %d rows", len(cut_df))

    cut_df = derive_c_n(cut_df)
    cut_df = _cast_floats_float32(cut_df)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    cut_df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", out, size_mb, len(cut_df.columns))

    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/emit_apogee_interim.py",
        sources=[
            LocalSource(
                name="APOGEE DR19 ASPCAP summary v0.6.0",
                path=str(raw.relative_to(repo)),
                sha256=sha256_file(raw),
            ),
        ],
        cuts_applied=QualityCuts().as_predicates(),
        corrections=[
            "cast float64 → float32 on all float cols",
            "c_fe = c_h_atm - fe_h_atm; n_fe = n_h_atm - fe_h_atm "
            "(+ quadrature errors) synthesised from DR19 [X/H] columns",
            "c_n = c_fe - n_fe (+ quadrature error)",
        ],
        row_count_before=stats["before"],
        row_count_after=stats["after"],
        notes=(
            "DR19 post-quality-cuts Parquet, PRE Mészáros+2025 [X/M] trend correction. "
            "Gaia-side Lindegren+2021 zpt and Riello+2021 G-mag corrections are applied "
            "to the Gaia enrichment stream separately (see stream1_gaia_dr3_corrected). "
            "Now includes Gaia DR3 astrometry (ra/dec/plx/pm), Gaia+2MASS+WISE "
            "photometry, Edenhofer+2023 / SFD ebv, and Bailer-Jones+2021 distances "
            "pre-joined by Astra — saves a separate 3 GB Edenhofer bulk fetch for "
            "Stream 1 training."
        ),
        extra={
            "quality_cut_stage_counts": stats,
            "kept_columns": kept_columns(),
            "n_unique_source_ids": int(cut_df["source_id"].nunique()),
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
