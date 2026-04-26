"""Build the hybrid (regressor + kNN) release catalog end to end.

Convenience wrapper around
:func:`arqueogal.data.release_pipeline.run_hybrid_release_pipeline`. Use this
when the kNN-rescue parquet is already on disk (from
``scripts/run_knn_rescue.py``) and you want the full annotated catalog with
``<elem>_hybrid_pred`` / ``_sigma`` / ``_source`` / ``_tier`` columns.

The output directory contains:

- ``predictions_with_features.parquet`` (joined + annotated + hybrid columns)
- ``predictions_with_features.release_tier.json`` (release-tier sidecar)
- ``derivatives/`` (HRD-ready, kinematic-ready, Tier-1, per-cell, per-magnitude)
- ``release_pipeline_manifest.json`` (full manifest, including ``hybrid`` block)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from arqueogal.data.release_pipeline import run_hybrid_release_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("build_hybrid_release")

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PRED = REPO_ROOT / "data/processed/pipeline1_predictions_stream3.parquet"
_DEFAULT_FEAT = REPO_ROOT / "data/processed/pipeline1_features_stream3.parquet"
_DEFAULT_KNN = REPO_ROOT / "data/processed/pipeline1_knn_rescue.parquet"
_DEFAULT_OUT = REPO_ROOT / "release/D-Cat-b/hybrid_pipeline_run"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=_DEFAULT_PRED)
    parser.add_argument("--features", type=Path, default=_DEFAULT_FEAT)
    parser.add_argument(
        "--knn-rescue",
        type=Path,
        default=_DEFAULT_KNN,
        help="kNN-rescue parquet (run scripts/run_knn_rescue.py to produce one).",
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--no-derivatives", action="store_true")
    parser.add_argument("--no-partition", action="store_true")
    args = parser.parse_args()

    _LOG.info(
        "hybrid release pipeline → %s\n  predictions = %s\n  features    = %s\n  knn_rescue  = %s",
        args.output_dir,
        args.predictions,
        args.features,
        args.knn_rescue,
    )
    knn_path = args.knn_rescue if args.knn_rescue.exists() else None
    if knn_path is None:
        _LOG.warning(
            "kNN-rescue parquet not found; running degraded hybrid (regressor + caveat only)"
        )

    manifest = run_hybrid_release_pipeline(
        args.predictions,
        args.features,
        knn_path,
        args.output_dir,
        build_derivatives=not args.no_derivatives,
        build_partition=not args.no_partition,
    )
    print(json.dumps(manifest.get("hybrid", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
