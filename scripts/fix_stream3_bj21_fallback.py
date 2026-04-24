"""Post-fix Stream 3 stage A parquet: fill r_med_photogeo NaN from distance_pc.

The initial builder run (PID 944532) resolved only 435,225/613,939 BJ21
lookups because 178k Stream-3 source_ids are absent from the Andrae per-chunk
BJ21 files. The Phase 3a union carries the same quantity (BJ21
photogeometric r_med_photogeo) under the name `distance_pc`, so we fill
the missing r_med_photogeo entries from that column. r_lo_photogeo and
r_hi_photogeo remain NaN (inference treats them as aux-missing — the
ensemble never required BJ21 lower/upper bounds as features).

Writes atomically (temp file → rename). Updates the provenance JSON with
the fill counts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fix_stream3_bj21_fallback")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    feat = repo / "data" / "processed" / "pipeline1_features_stream3.parquet"
    prov = feat.with_suffix(".provenance.json")

    logger.info("loading %s", feat)
    df = pd.read_parquet(feat)
    logger.info("  %d rows x %d cols", *df.shape)

    r_med = df["r_med_photogeo"].to_numpy(dtype=np.float64)
    dist_pc = df["distance_pc"].to_numpy(dtype=np.float64)

    n_before = int(np.isfinite(r_med).sum())
    use_fallback = ~np.isfinite(r_med) & np.isfinite(dist_pc)
    n_fill = int(use_fallback.sum())
    logger.info("  r_med_photogeo finite before: %d", n_before)
    logger.info("  fallback candidates (r_med NaN & distance_pc finite): %d", n_fill)

    if n_fill:
        df.loc[use_fallback, "r_med_photogeo"] = dist_pc[use_fallback].astype(np.float32)
        r_med_after = df["r_med_photogeo"].to_numpy(dtype=np.float64)
        n_after = int(np.isfinite(r_med_after).sum())
        logger.info("  r_med_photogeo finite after: %d (+%d)", n_after, n_after - n_before)
        n_still_nan = len(df) - n_after
        logger.info("  r_med_photogeo still NaN: %d", n_still_nan)

    tmp = feat.with_suffix(feat.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, feat)
    size_mb = feat.stat().st_size / 1024**2
    logger.info("rewrote %s (%.1f MB)", feat, size_mb)

    if prov.exists():
        with prov.open() as f:
            meta = json.load(f)
        extra = meta.setdefault("extra", {})
        extra["bj21_fallback_fills_from_union_distance_pc"] = n_fill
        extra["r_med_photogeo_finite_after_fallback"] = int(np.isfinite(df["r_med_photogeo"]).sum())
        corr = meta.setdefault("corrections", [])
        corr.append(
            "r_med_photogeo fallback: filled from union distance_pc where "
            f"per-chunk BJ21 lookup missed ({n_fill} rows)"
        )
        tmp_p = prov.with_suffix(prov.suffix + ".part")
        with tmp_p.open("w") as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp_p, prov)
        logger.info("updated provenance %s", prov)
    else:
        logger.warning("provenance sidecar %s not found; skipping update", prov)


if __name__ == "__main__":
    main()
