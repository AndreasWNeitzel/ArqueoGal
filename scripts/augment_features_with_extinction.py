"""One-shot script: augment pipeline1_features_stream{1,2,3}.parquet with
the extinction-derived columns required by the current 140-D FeatureLayout.

Adds: av_los, av_los_source, j_mag_dered, h_mag_dered, k_mag_dered,
w1_mag_dered, w2_mag_dered.

Reads the parquet, calls arqueogal.data.extinction.apply_extinction_corrections,
writes back atomically. Skips streams whose parquet already carries av_los.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.data.extinction import apply_extinction_corrections  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def augment(parquet_path: Path) -> None:
    log.info("loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  %d rows x %d cols", len(df), df.shape[1])

    if "av_los" in df.columns and "j_mag_dered" in df.columns:
        log.info("  already augmented; skipping")
        return

    needed_av = [c for c in ("av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median") if c in df.columns]
    needed_ir = [c for c in ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag") if c in df.columns]
    log.info("  available av layers: %s", needed_av)
    log.info("  available IR bands:  %s", needed_ir)
    if not needed_av or not needed_ir:
        log.warning("  cannot augment: missing extinction layers or IR photometry; skipping")
        return

    out = apply_extinction_corrections(df, inplace=False)
    log.info("  -> %d rows x %d cols", len(out), out.shape[1])
    for c in ("av_los", "av_los_source", "j_mag_dered", "h_mag_dered", "k_mag_dered", "w1_mag_dered", "w2_mag_dered"):
        if c not in out.columns:
            log.error("  expected column '%s' missing after augmentation; aborting", c)
            return

    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(parquet_path)
    log.info("  wrote %s (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)


def main() -> int:
    for stream in (1, 2, 3):
        p = REPO / "data" / "processed" / f"pipeline1_features_stream{stream}.parquet"
        if not p.exists():
            log.warning("missing %s — skipping", p)
            continue
        augment(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
