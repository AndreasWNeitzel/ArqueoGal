"""Apply Mészáros+2025 Teff-trend corrections to the post-cuts APOGEE DR19
interim Parquet and emit ``data/interim/apogee_dr19_corrected.parquet``.

Reads ``apogee_dr19_precorrected.parquet`` (produced by emit_apogee_interim.py),
applies :func:`arqueogal.data.apogee_dr19.apply_meszaros2025_corrections` to
``alpha_m_atm`` and the supported ``{el}_h_atm`` columns, and writes a new
Parquet + provenance sidecar. The pre-corrected file is left untouched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from arqueogal.data.apogee_dr19 import (
    MESZAROS2025_COEFFS,
    apply_meszaros2025_corrections,
)
from arqueogal.data.downloads import sha256_file
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("emit_apogee_corrected")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data" / "interim" / "apogee_dr19_precorrected.parquet"
    out = repo / "data" / "interim" / "apogee_dr19_corrected.parquet"

    if not src.exists():
        raise SystemExit(f"missing {src}; run scripts/emit_apogee_interim.py first")

    logger.info("loading %s", src)
    df = pd.read_parquet(src)
    logger.info("loaded %d rows × %d cols", len(df), len(df.columns))

    corrected = apply_meszaros2025_corrections(df)
    summary: pd.DataFrame = corrected.attrs["meszaros_correction_summary"]
    for _, row in summary.iterrows():
        logger.info(
            "  %-12s n=%7d  <Δ>=%+7.4f dex  rms=%7.4f dex",
            row["element"], int(row["n_applied"]), row["mean_shift"], row["rms_shift"],
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    # Strip DataFrame-valued attrs: parquet metadata must be JSON-serialisable.
    to_write = corrected.copy()
    to_write.attrs = {}
    to_write.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", out, size_mb, len(corrected.columns))

    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/emit_apogee_corrected.py",
        sources=[
            LocalSource(
                name="APOGEE DR19 post-cut interim (pre-correction)",
                path=str(src.relative_to(repo)),
                sha256=sha256_file(src),
            ),
        ],
        cuts_applied=[],
        corrections=[
            "Mészáros+2025 Table 3 Δ[X/M] = a·Teff + b applied to "
            + ", ".join(sorted(MESZAROS2025_COEFFS)),
            "Mészáros+2025 domain: 3500 < Teff < 6000 K and log g < 3.8; "
            "boundary offsets outside Teff window; log g ≥ 3.8 left uncorrected",
            "C, N, Fe, V, Cu deliberately uncorrected (see Mészáros+2025 §4.1-4.2)",
        ],
        row_count_before=len(df),
        row_count_after=len(corrected),
        notes=(
            "Mészáros+2025 (arXiv:2506.07845) Teff-trend correction applied per "
            "docs/data_acquisition.md §3.4. Gaia-side Lindegren+2021 parallax "
            "zpt and Riello+2021 (A&A 649, A3 Appendix A) G-mag corrections are "
            "applied separately in the Gaia enrichment stream. This file is the "
            "Stream-1 training-pool APOGEE side; Gaia is joined in build_stream1_apogee_gaia."
        ),
        extra={
            "meszaros_correction_summary": summary.to_dict(orient="records"),
            "meszaros_n_elements_corrected": int((summary["n_applied"] > 0).sum()),
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
