"""Precompute per-star A_V for Stream 3 delta 449k Ye-OK (Phase 3b).

Delta variant of ``precompute_stream3_av.py``. Uses BJ21 photogeometric
distance (distance_pc from stream3_expansion_union.parquet) rather than
1/parallax — BJ21 is the preferred distance estimator for d > ~1 kpc,
and the Phase 3a union already carries it on every star. The existing
168k A_V used Andrae 1/parallax; this ~10% distance-provenance drift
between existing and delta arms is noted in the Phase 3b report and is
not load-bearing since A_V_los enters Pipeline 1 as a rough prior (the
neighborhood-median is the more-trusted feature).

Outputs
-------
``data/interim/stream3_delta_av.parquet``
    source_id, av_edenhofer, av_sfd_path, av_los, av_los_source per star.

``*.provenance.json`` sidecar.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_stream3_av_delta")

NEAR_BOUNDARY_KPC = 1.25
SFD_TO_AV = 2.742


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    from dustmaps.config import config as dm_config

    repo = Path(__file__).resolve().parents[1]
    dm_config["data_dir"] = str(repo / "data" / "external" / "dustmaps")

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from dustmaps.edenhofer2023 import Edenhofer2023Query
    from dustmaps.sfd import SFDQuery

    from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

    union_path = repo / "data" / "interim" / "stream3_expansion_union.parquet"
    ye_ok_path = repo / "data" / "interim" / "stream3_delta_ye_ok_source_ids.parquet"
    out = repo / "data" / "interim" / "stream3_delta_av.parquet"

    logger.info("loading %s", union_path)
    union = pd.read_parquet(union_path, columns=["source_id", "ra", "dec", "distance_pc"])
    logger.info("union: %d rows", len(union))

    logger.info("loading %s", ye_ok_path)
    ye_ok = pd.read_parquet(ye_ok_path)["source_id"].astype("int64")
    logger.info("Ye-OK delta: %d source_ids", len(ye_ok))

    df = union[union["source_id"].isin(ye_ok)].reset_index(drop=True)
    n = len(df)
    logger.info("delta A_V subset: %d rows (union ∩ ye_ok)", n)
    if n != len(ye_ok):
        logger.warning(
            "mismatch: %d Ye-OK vs %d in union — %d missing", len(ye_ok), n, len(ye_ok) - n
        )

    d_pc = df["distance_pc"].to_numpy(dtype=np.float64)
    valid_d = np.isfinite(d_pc) & (d_pc > 0)
    d_kpc = np.where(valid_d, d_pc / 1000.0, np.nan)
    near_mask = valid_d & (d_kpc < NEAR_BOUNDARY_KPC)
    far_mask = valid_d & (d_kpc >= NEAR_BOUNDARY_KPC)
    logger.info(
        "distance bins: %d near (<%.2f kpc), %d far (≥%.2f kpc), %d no-distance",
        int(near_mask.sum()),
        NEAR_BOUNDARY_KPC,
        int(far_mask.sum()),
        NEAR_BOUNDARY_KPC,
        int((~valid_d).sum()),
    )

    av_eden = np.full(n, np.nan, dtype=np.float32)
    av_sfd = np.full(n, np.nan, dtype=np.float32)
    av_los = np.full(n, np.nan, dtype=np.float32)
    source = np.full(n, -1, dtype=np.int8)

    logger.info("loading Edenhofer+2024 healpix cube")
    eden = Edenhofer2023Query()
    logger.info("loading SFD map")
    sfd = SFDQuery()

    ra = df["ra"].to_numpy(dtype=np.float64)
    dec = df["dec"].to_numpy(dtype=np.float64)

    logger.info("querying SFD for all %d stars", n)
    coords_all = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ebv_sfd_all = np.asarray(sfd(coords_all), dtype=np.float64)
    av_sfd[:] = (ebv_sfd_all * SFD_TO_AV).astype(np.float32)

    if near_mask.any():
        logger.info("querying Edenhofer at %d near-bin stars", int(near_mask.sum()))
        coords_near = SkyCoord(
            ra=ra[near_mask] * u.deg,
            dec=dec[near_mask] * u.deg,
            distance=(d_kpc[near_mask] * 1000.0) * u.pc,
            frame="icrs",
        )
        av_near = np.asarray(eden(coords_near), dtype=np.float32)
        av_eden[near_mask] = av_near
        av_los[near_mask] = av_near
        source[near_mask] = 0
        nan_near = ~np.isfinite(av_near)
        if nan_near.any():
            idx = np.flatnonzero(near_mask)[nan_near]
            source[idx] = -1
            av_los[idx] = np.nan

    if far_mask.any():
        av_los[far_mask] = av_sfd[far_mask]
        source[far_mask] = 2

    out_df = pd.DataFrame(
        {
            "source_id": df["source_id"].to_numpy(),
            "av_edenhofer": av_eden,
            "av_sfd_path": av_sfd,
            "av_los": av_los,
            "av_los_source": source,
        }
    )
    _write_parquet_atomic(out_df, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB)", out, size_mb)

    n_eden = int((source == 0).sum())
    n_sfd = int((source == 2).sum())
    n_nan = int((source == -1).sum())
    logger.info("composition: %d Edenhofer, %d SFD-fallback, %d no-coverage", n_eden, n_sfd, n_nan)

    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/precompute_stream3_av_delta.py",
        sources=[
            LocalSource(
                name="Stream 3 expansion union (Phase 3a)",
                path=str(union_path.relative_to(repo)),
                sha256=None,
            ),
            LocalSource(
                name="Stream 3 delta Ye-OK source_ids (Phase 3a)",
                path=str(ye_ok_path.relative_to(repo)),
                sha256=None,
            ),
            LocalSource(
                name="Edenhofer+2024 mean+std healpix (dustmaps cache)",
                path="data/external/dustmaps/edenhofer_2023/mean_and_std_healpix.fits",
                sha256=None,
            ),
            LocalSource(
                name="SFD dust (dustmaps cache)",
                path="data/external/dustmaps/sfd/",
                sha256=None,
            ),
        ],
        cuts_applied=["inner join on stream3_delta_ye_ok_source_ids"],
        corrections=[
            "distance = distance_pc / 1000 kpc (BJ21 photogeometric)",
            f"SFD E(B-V) → A_V via {SFD_TO_AV} (Schlafly & Finkbeiner 2011)",
        ],
        row_count_before=n,
        row_count_after=n,
        notes=(
            "Per-star A_V lookup for Stream 3 delta (Phase 3b). Uses BJ21 "
            "photogeometric distance from the Phase 3a union, rather than "
            "Andrae 1/parallax used for the existing 168k. Edenhofer+2024 "
            "for d<1.25 kpc, SFD (×2.742) for d≥1.25 kpc. Lallement+2022 "
            "upgrade still pending."
        ),
        extra={
            "n_near_edenhofer": n_eden,
            "n_far_sfd_fallback": n_sfd,
            "n_no_coverage": n_nan,
            "near_boundary_kpc": NEAR_BOUNDARY_KPC,
            "sfd_to_av_coeff": SFD_TO_AV,
            "distance_source": "BJ21 photogeometric (r_med_photogeo)",
            "phase": "Phase 3b delta A_V",
        },
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
