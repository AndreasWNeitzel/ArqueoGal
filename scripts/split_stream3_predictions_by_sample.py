"""Split Pipeline-1 Stream-3 predictions into uniform and volume-limited arms.

Stream-3 is a union of two logically-independent selection arms (Option C):
- ``uniform`` — the (Teff, logg, [M/H], G) stratified uniform sample for
  downstream tests that require a clean selection function.
- ``volume_limited`` — the volume-limited complete sample for downstream
  density-based clustering and kinematic analysis in Starfold (separate
  repo) and Task 5.

The inference driver runs once on the union and emits predictions keyed by
``source_id`` only. This splitter joins predictions with the feature-matrix
``sample`` column and writes two outputs:

``data/processed/pipeline1_predictions_stream3_uniform.parquet``
``data/processed/pipeline1_predictions_stream3_volume.parquet``

Each split carries a provenance sidecar pointing back to the union output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("split_stream3_predictions")


def _atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_pred = repo / "data" / "processed" / "pipeline1_predictions_stream3.parquet"
    default_feat = repo / "data" / "processed" / "pipeline1_features_stream3.parquet"
    default_out_u = repo / "data" / "processed" / "pipeline1_predictions_stream3_uniform.parquet"
    default_out_v = repo / "data" / "processed" / "pipeline1_predictions_stream3_volume.parquet"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, default=default_pred)
    ap.add_argument("--feat", type=Path, default=default_feat)
    ap.add_argument("--out-uniform", type=Path, default=default_out_u)
    ap.add_argument("--out-volume", type=Path, default=default_out_v)
    args = ap.parse_args()

    pred = args.pred.resolve()
    feat = args.feat.resolve()
    out_u = args.out_uniform.resolve()
    out_v = args.out_volume.resolve()

    logger.info("loading predictions %s", pred)
    df_pred = pd.read_parquet(pred)
    logger.info("  %d rows x %d cols", *df_pred.shape)

    logger.info("loading feature matrix sample column from %s", feat)
    df_sample = pd.read_parquet(feat, columns=["source_id", "sample"])
    logger.info("  %d rows x %d cols", *df_sample.shape)

    logger.info("joining on source_id")
    df = df_pred.merge(df_sample, on="source_id", how="inner", validate="one_to_one")
    if len(df) != len(df_pred):
        logger.warning(
            "join shrank predictions: %d → %d (loss=%d); one or more source_ids "
            "missing the sample column",
            len(df_pred),
            len(df),
            len(df_pred) - len(df),
        )

    counts = df["sample"].value_counts().to_dict()
    logger.info("sample composition post-join: %s", counts)

    for name, path in (("uniform", out_u), ("volume_limited", out_v)):
        arm = df[df["sample"] == name].drop(columns=["sample"]).reset_index(drop=True)
        logger.info("writing %s arm: %d rows → %s", name, len(arm), path)
        _atomic(arm, path)
        size_mb = path.stat().st_size / 1024**2
        logger.info("  %.1f MB on disk", size_mb)

        # Per-arm halt diagnostics — propagate the Phase 3b halt-cell criteria.
        ood = float(arm["ood_joint_flag"].mean())
        rgb = float(arm["regime_b_flag"].mean())
        aux = float(arm["aux_missing_any"].mean())
        # Prediction NaN rate uses teff_pred as a representative column; the
        # 5-label block NaNs as one unit so any column is equivalent.
        pred_nan = float(arm["teff_pred"].isna().mean())
        logger.info(
            "  halt diagnostics: OOD=%.4f  RegimeB=%.4f  aux_missing=%.4f  pred_NaN=%.4f",
            ood,
            rgb,
            aux,
            pred_nan,
        )

        sidecar = {
            "output_file": str(path.relative_to(repo)),
            "script": "scripts/split_stream3_predictions_by_sample.py",
            "sources": [
                {
                    "name": "Stream-3 Pipeline-1 predictions (union)",
                    "path": str(pred.relative_to(repo)),
                },
                {
                    "name": "Stream-3 feature matrix (sample column source)",
                    "path": str(feat.relative_to(repo)),
                },
            ],
            "row_count": int(len(arm)),
            "sample_arm": name,
            "halt_diagnostics": {
                "ood_joint_rate": ood,
                "regime_b_rate": rgb,
                "aux_missing_any_rate": aux,
                "prediction_nan_rate": pred_nan,
            },
        }
        sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
        tmp_p = sidecar_path.with_suffix(sidecar_path.suffix + ".part")
        with tmp_p.open("w") as f:
            json.dump(sidecar, f, indent=2, default=str)
        os.replace(tmp_p, sidecar_path)
        logger.info("  wrote provenance %s", sidecar_path)


if __name__ == "__main__":
    main()
