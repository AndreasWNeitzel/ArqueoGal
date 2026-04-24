"""Merge latent_knn_dist + latent_support_flag into Stream-3 prediction parquets.

Reads the gate output (``pipeline1_latent_support_stream3.parquet``) and
left-joins ``latent_knn_dist`` + ``latent_support_flag`` on ``source_id``
into each of:

- ``pipeline1_predictions_stream3_joint.parquet`` (union)
- ``pipeline1_predictions_stream3_joint_volume.parquet``
- ``pipeline1_predictions_stream3_joint_uniform.parquet``

Writes back in place. Idempotent: if the columns already exist they are
dropped before the merge. The original ``ood_joint_flag`` column is never
touched — the latent-support flag is additive, not a replacement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_DEF_GATE = _REPO / "data/processed/pipeline1_latent_support_stream3.parquet"
_DEF_TARGETS: tuple[Path, ...] = (
    _REPO / "data/processed/pipeline1_predictions_stream3_joint.parquet",
    _REPO / "data/processed/pipeline1_predictions_stream3_joint_volume.parquet",
    _REPO / "data/processed/pipeline1_predictions_stream3_joint_uniform.parquet",
)

_NEW_COLS = ("latent_knn_dist", "latent_support_flag")


def _merge_one(target: Path, gate: pd.DataFrame) -> None:
    df = pd.read_parquet(target)
    before = len(df)
    drop = [c for c in _NEW_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    merged = df.merge(
        gate[["source_id", *_NEW_COLS]],
        on="source_id",
        how="left",
    )
    if len(merged) != before:
        raise RuntimeError(
            f"{target.name}: merge changed row count {before} → {len(merged)}",
        )
    missing = merged["latent_knn_dist"].isna().sum()
    merged.to_parquet(target, index=False)
    print(
        f"{target.name}: merged ({before:,} rows, "
        f"{missing:,} missing gate values), flag_rate="
        f"{merged['latent_support_flag'].astype(bool).mean():.4%}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", type=Path, default=_DEF_GATE)
    ap.add_argument("--targets", type=Path, nargs="+", default=list(_DEF_TARGETS))
    args = ap.parse_args()

    gate = pd.read_parquet(args.gate)
    print(
        f"gate: {len(gate):,} rows, flag_rate={gate['latent_support_flag'].astype(bool).mean():.4%}"
    )
    for t in args.targets:
        _merge_one(t, gate)


if __name__ == "__main__":
    main()
