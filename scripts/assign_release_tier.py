"""Assign ``release_tier`` to Pipeline 1 Stream-3 prediction parquets in place.

Reads each of the default targets:

- ``data/processed/pipeline1_predictions_stream3_joint.parquet`` (union)
- ``data/processed/pipeline1_predictions_stream3_joint_volume.parquet``
- ``data/processed/pipeline1_predictions_stream3_joint_uniform.parquet``

Applies :func:`arqueogal.xp_abundances.main.release.assign_release_tier`
and writes each back in place with the new ``release_tier`` column plus a
``*.release_tier.json`` sidecar capturing tier counts and flag-column
provenance. Idempotent.

Run: ``PYTHONPATH=src python scripts/assign_release_tier.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from arqueogal.xp_abundances.main.release import annotate_parquet

_REPO = Path(__file__).resolve().parents[1]
_DEF_TARGETS: tuple[Path, ...] = (
    _REPO / "data/processed/pipeline1_predictions_stream3_joint.parquet",
    _REPO / "data/processed/pipeline1_predictions_stream3_joint_volume.parquet",
    _REPO / "data/processed/pipeline1_predictions_stream3_joint_uniform.parquet",
)

log = logging.getLogger("assign_release_tier")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, action="append", default=None,
        help="Override the default target list (repeatable).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    targets: tuple[Path, ...] = tuple(args.target) if args.target else _DEF_TARGETS

    for path in targets:
        if not path.exists():
            log.warning("Skipping missing parquet %s", path)
            continue
        summary = annotate_parquet(path)
        counts = summary["counts"]
        n = summary["n_rows"]
        log.info(
            "%s: n=%d  tier1=%d (%.1f%%)  tier2=%d (%.1f%%)  tier3=%d (%.1f%%)",
            path.name, n,
            counts[1], 100 * counts[1] / n,
            counts[2], 100 * counts[2] / n,
            counts[3], 100 * counts[3] / n,
        )


if __name__ == "__main__":
    _main()
