"""CLI wrapper for the five derivative release artefacts + g_mag_bin partitioning.

Builds, in one invocation:

1. HRD-ready subset (Tier ≤ 1).
2. Kinematic-ready subset (Tier ≤ 2).
3. Tier-1-only full subset (release_tier == 1, all elements spectrum_dominant).
4. Per-cell summary (default group: g_mag_bin).
5. Per-magnitude reliability table (g_mag_bin × element).
6. Partitioned dataset by g_mag_bin (zstd level 10, 25k row groups).

Each output goes under ``--output-dir``; partitioning produces a sub-directory.

Source: data_preparation_output.md (5 derivative artefacts + partitioning recommendation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arqueogal.data.release_artefacts import (
    build_hrd_ready_subset,
    build_kinematic_ready_subset,
    build_per_cell_summary,
    build_per_magnitude_reliability,
    build_tier1_only_subset,
    partition_by_g_mag_bin,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input release-annotated Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for all output artefacts.",
    )
    parser.add_argument(
        "--skip-partition",
        action="store_true",
        help="Skip the g_mag_bin partitioning step (large for D-Cat-b ~1.5M rows).",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=25_000,
        help="Pyarrow row-group size for partitioned dataset (default 25_000).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    summaries.append(
        build_hrd_ready_subset(
            args.input,
            args.output_dir / "hrd_ready.parquet",
        ),
    )
    summaries.append(
        build_kinematic_ready_subset(
            args.input,
            args.output_dir / "kinematic_ready.parquet",
        ),
    )
    summaries.append(
        build_tier1_only_subset(
            args.input,
            args.output_dir / "tier1_only_full.parquet",
        ),
    )
    summaries.append(
        build_per_cell_summary(
            args.input,
            args.output_dir / "per_cell_summary.parquet",
        ),
    )
    summaries.append(
        build_per_magnitude_reliability(
            args.input,
            args.output_dir / "per_magnitude_reliability.parquet",
        ),
    )

    if not args.skip_partition:
        summaries.append(
            partition_by_g_mag_bin(
                args.input,
                args.output_dir / "partitioned_by_g_mag_bin",
                row_group_size=args.row_group_size,
            ),
        )

    manifest_path = args.output_dir / "build_release_artefacts_manifest.json"
    manifest_path.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
