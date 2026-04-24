"""Stream 3 stratified selection from the VizieR-route Andrae+2023 parquet.

Runs the §5.3 (Teff, logg, [Fe/H], G)-stratified sub-sample on the output
of ``fetch_andrae2023_vizier.py`` and writes the selected sample plus a
source_id-only companion Parquet ready for the (AIP-credential-gated) Gaia
DR3 enrichment + XP coefficient pulls that come next.

Selection alone is unblocked — AIP-dependent enrichment is handled by
``ingest_stream3.py`` once credentials are present.

Outputs
-------
``data/interim/stream3_selected.parquet``
    Full VIZIER_KEPT_COLUMNS for the selected ≈1.5 M subset.
``data/interim/stream3_selected_source_ids.parquet``
    Single column ``source_id`` — tiny, for handoff to enrichment scripts.
Both files get a ``*.provenance.json`` sidecar.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from arqueogal.data.andrae2023 import (
    ANDRAE2023_ZENODO_RECORD,
    load_andrae2023_parquet,
)
from arqueogal.data.downloads import sha256_file
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar
from arqueogal.data.stream3_selection import (
    DEFAULT_PER_CELL,
    stratified_subsample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("select_stream3_stars")


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    andrae_pq = repo / "data" / "raw" / "andrae2023" / "andrae2023_rgb.parquet"
    out_full = repo / "data" / "interim" / "stream3_selected.parquet"
    out_ids = repo / "data" / "interim" / "stream3_selected_source_ids.parquet"

    if not andrae_pq.exists():
        raise SystemExit(f"missing {andrae_pq}; run scripts/fetch_andrae2023_vizier.py first")

    logger.info("loading %s", andrae_pq)
    df = load_andrae2023_parquet(andrae_pq)
    n_loaded = len(df)
    logger.info("Andrae+2023 rows loaded: %d", n_loaded)

    # VizieR renames: teff/logg/fe_h/g_mag. Andrae+2023 is pure RGB at
    # [Fe/H] ≳ -2.5 in low-α regime → [Fe/H] ≈ [M/H] for stratification.
    result = stratified_subsample(
        df,
        teff_col="teff",
        logg_col="logg",
        mh_col="fe_h",
        g_col="g_mag",
        per_cell=DEFAULT_PER_CELL,
        rng_seed=0,
    )
    sample = result.sample
    n_selected = len(sample)
    logger.info("stratified subsample: %d rows", n_selected)

    _write_parquet_atomic(sample, out_full)
    logger.info("wrote %s (%.1f MB)", out_full, out_full.stat().st_size / 1024**2)

    ids_df = sample[["source_id"]].drop_duplicates().reset_index(drop=True)
    _write_parquet_atomic(ids_df, out_ids)
    logger.info(
        "wrote %s (%.1f MB, %d unique ids)", out_ids, out_ids.stat().st_size / 1024**2, len(ids_df)
    )

    src = LocalSource(
        name=f"Andrae+2023 VizieR parquet (reissue of Zenodo {ANDRAE2023_ZENODO_RECORD})",
        path=str(andrae_pq.relative_to(repo)),
        sha256=sha256_file(andrae_pq),
    )
    strat_prov = result.to_provenance()

    full_prov = Provenance(
        output_file=str(out_full.relative_to(repo)),
        script="scripts/select_stream3_stars.py",
        sources=[src],
        cuts_applied=[
            f"stratified_subsample per_cell={DEFAULT_PER_CELL}",
            "stratified_subsample rng_seed=0",
            "stratification_columns=(teff, logg, fe_h, g_mag)",
        ],
        corrections=[],
        row_count_before=n_loaded,
        row_count_after=n_selected,
        notes=(
            "§5.3 stratified sub-sample of Andrae+2023 vetted RGB. "
            "AIP Gaia DR3 enrichment is separate (user-blocked on credentials); "
            "this file carries the full VizieR column set for the selected stars."
        ),
        extra={"stratification": strat_prov},
    )
    write_sidecar(full_prov)

    ids_prov = Provenance(
        output_file=str(out_ids.relative_to(repo)),
        script="scripts/select_stream3_stars.py",
        sources=[src],
        cuts_applied=full_prov.cuts_applied,
        corrections=[],
        row_count_before=n_loaded,
        row_count_after=len(ids_df),
        notes="Source_id-only handoff Parquet for downstream AIP enrichment + XP pulls.",
        extra={"stratification": strat_prov},
    )
    write_sidecar(ids_prov)


if __name__ == "__main__":
    main()
