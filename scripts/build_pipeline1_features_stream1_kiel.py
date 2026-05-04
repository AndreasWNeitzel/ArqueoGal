"""Emit a Kiel-bounded Stream-1 training parquet (logg ∈ [1.0, 3.5], Teff ∈ [4000, 5500]).

Background
----------
Stream 1 (APOGEE × Gaia DR3) carries stars across many evolutionary phases.
For the production xp_abundances regressor we want to constrain training to
the bright-RGB / red-clump bounding box only. Re-querying APOGEE is not
required: the canonical features parquet already contains the Tier-1 truth
columns (teff_apogee, logg_apogee). This script applies the bbox mask at the
parquet boundary and writes a sibling artefact suitable as a drop-in training
input for the contrastive pretrain, the supervised ensemble fine-tune, the
kNN-rescue training pool, the OOD Mahalanobis training pool, and the latent-
support gate's reference set.

The frozen Hermite z-score basis (provenance JSON) is **shared** between the
canonical and the Kiel-masked parquet: the mask filters rows in label space,
not feature space, so the encoder sees the same inputs as before — just on a
narrower training population.

Outputs
-------
- ``data/processed/pipeline1_features_stream1_kiel.parquet``
- ``data/processed/pipeline1_features_stream1_kiel.provenance.json`` —
  copies the canonical frozen-stats JSON verbatim (basis fingerprint
  unchanged), then appends a ``kiel_mask`` block recording the bbox limits
  and the row counts before/after.

Usage
-----
::

    PYTHONPATH=src python scripts/build_pipeline1_features_stream1_kiel.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("build_kiel")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data/processed/pipeline1_features_stream1_kiel.parquet"
DEFAULT_INPUT_PROV = REPO_ROOT / "data/processed/pipeline1_features_stream1.provenance.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-provenance", type=Path, default=DEFAULT_INPUT_PROV)
    parser.add_argument("--teff-min", type=float, default=4000.0)
    parser.add_argument("--teff-max", type=float, default=5500.0)
    parser.add_argument("--logg-min", type=float, default=1.0)
    parser.add_argument("--logg-max", type=float, default=3.5)
    args = parser.parse_args()

    _LOG.info("reading %s", args.input)
    table = pq.read_table(args.input)
    n_total = table.num_rows
    teff = table.column("teff_apogee").to_numpy()
    logg = table.column("logg_apogee").to_numpy()
    mask = (
        (teff >= args.teff_min)
        & (teff <= args.teff_max)
        & (logg >= args.logg_min)
        & (logg <= args.logg_max)
    )
    n_keep = int(mask.sum())
    _LOG.info(
        "kiel bbox: Teff [%.0f, %.0f] K, log g [%.2f, %.2f] dex; "
        "keep %d / %d rows (%.1f%%)",
        args.teff_min, args.teff_max, args.logg_min, args.logg_max,
        n_keep, n_total, 100.0 * n_keep / max(n_total, 1),
    )
    if n_keep == 0:
        raise RuntimeError("Kiel mask kept zero rows — check teff/logg columns")

    table_kiel = table.filter(pa.array(mask))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table_kiel, args.output, compression="zstd")
    _LOG.info("wrote %s (%d rows, %.1f MB)",
              args.output, table_kiel.num_rows,
              args.output.stat().st_size / 1e6)

    # Provenance: copy frozen-stats payload, append kiel_mask block.
    prov_path = args.output.with_suffix("").with_suffix(".provenance.json")
    if prov_path.suffix != ".json":
        prov_path = args.output.with_name(args.output.stem + ".provenance.json")

    if args.input_provenance.exists():
        prov = json.loads(args.input_provenance.read_text())
    else:
        prov = {}
    prov["kiel_mask"] = {
        "teff_min_K": args.teff_min,
        "teff_max_K": args.teff_max,
        "logg_min_dex": args.logg_min,
        "logg_max_dex": args.logg_max,
        "n_rows_before": n_total,
        "n_rows_after": n_keep,
        "fraction_kept": float(n_keep / max(n_total, 1)),
        "source_parquet": str(args.input.relative_to(REPO_ROOT)),
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    _LOG.info("wrote %s", prov_path)


if __name__ == "__main__":
    main()
