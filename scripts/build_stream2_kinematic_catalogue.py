"""End-to-end Stream-2 kinematic-catalogue driver — D-Cat-b MVP.

Workflow:

    Stream 2 base parquet (Hon+2021 × TIC × Gaia DR3 enrichment, from
    `arqueogal.data.ingest_stream2`)
        → BJ21 photogeometric distances (GAVO TAP, optional)
        → dust-map fusion av_los (Edenhofer+2024 / Lallement+2022 / SFD)
        → broadband dereddening (CCM89 R_V=3.1 + Yuan+2013) — no-op if
          IR cross-match is absent
        → distance trust flags (BJ21 percentile spread, parallax sign)
        → galpy actions under McMillan+2017
        → write {output_path} + {output_path}.provenance.json

Run:

    PYTHONPATH=src python scripts/build_stream2_kinematic_catalogue.py \\
        --stream2-parquet data/interim/stream2_tess_gaia.parquet \\
        --bj21-parquet    data/interim/stream2_bj21.parquet \\
        --av-parquet      data/interim/stream2_av_layer.parquet \\
        --out             release/D-Cat-b/stream2_kinematic_catalogue.parquet

The BJ21 and dust-map parquets are optional; if absent, the corresponding
steps are skipped and the trust flags reflect the missing inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from arqueogal.data.build_stream2_kinematic_catalogue import (
    build_stream2_kinematic_catalogue,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream2-parquet",
        type=Path,
        required=True,
        help="Stream-2 base parquet from ingest_stream2.",
    )
    parser.add_argument(
        "--bj21-parquet",
        type=Path,
        default=None,
        help="Optional Bailer-Jones+2021 parquet (source_id, r_med_photogeo, ...).",
    )
    parser.add_argument(
        "--av-parquet",
        type=Path,
        default=None,
        help="Optional dust-map fusion parquet (source_id, av_edenhofer, ...).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination parquet for the kinematic catalogue.",
    )
    parser.add_argument(
        "--no-extinction",
        action="store_true",
        help="Skip the broadband dereddening step (debug/methodology only).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Loading Stream 2 base from %s", args.stream2_parquet)
    stream2 = pd.read_parquet(args.stream2_parquet)

    bj21 = None
    if args.bj21_parquet is not None:
        logger.info("Loading BJ21 frame from %s", args.bj21_parquet)
        bj21 = pd.read_parquet(args.bj21_parquet)

    av_layer = None
    if args.av_parquet is not None:
        logger.info("Loading dust-map fusion frame from %s", args.av_parquet)
        av_layer = pd.read_parquet(args.av_parquet)

    out_path = build_stream2_kinematic_catalogue(
        stream2,
        output_path=args.out,
        bj21_df=bj21,
        av_layer_df=av_layer,
        apply_extinction=not args.no_extinction,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
