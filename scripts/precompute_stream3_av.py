"""Precompute per-star A_V for Stream 3 against Edenhofer+2024 + SFD.

Queries the Edenhofer+2024 3D dust map at each Stream 3 star's heliocentric
distance (from Andrae parallax) for d < 1.25 kpc, and falls back to SFD (2D,
× 2.742 to convert E(B-V) → A_V) for d ≥ 1.25 kpc. Lallement+2022 is not in
the installed ``dustmaps`` build, so the 1.25–3 kpc gap is currently served
by SFD too; add a Lallement wrapper later to upgrade this zone.

Outputs
-------
``data/interim/stream3_av.parquet``
    ``source_id``, ``av_edenhofer``, ``av_sfd_path``, ``av_los``,
    ``av_los_source`` per star. ~1.3 MB for 168 k stars.

``*.provenance.json`` sidecar alongside it.

Notes
-----
- Distance = 1 / parallax_mas (kpc) with NaN for parallax ≤ 0 or NaN.
  Dust-map resolution (14′, 516 distance bins over 1.25 kpc) is far coarser
  than per-star parallax uncertainty, so Bailer-Jones photogeometric
  distance is not needed for this lookup.
- Once ``av_los`` is written, the 3.2 GB ``mean_and_std_healpix.fits`` can
  be safely deleted — it is not needed downstream. Pipeline 1 only sees
  ``av_los`` and neighborhood-median A_G.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_stream3_av")

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

    from dustmaps.edenhofer2023 import Edenhofer2023Query
    from dustmaps.sfd import SFDQuery
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

    src = repo / "data" / "interim" / "stream3_selected.parquet"
    out = repo / "data" / "interim" / "stream3_av.parquet"
    logger.info("loading %s", src)
    df = pd.read_parquet(src, columns=["source_id", "ra_deg", "dec_deg", "parallax_mas"])
    n = len(df)
    logger.info("stream 3 rows: %d", n)

    plx = df["parallax_mas"].to_numpy(dtype=np.float64)
    valid_plx = np.isfinite(plx) & (plx > 0)
    d_kpc = np.where(valid_plx, 1.0 / plx, np.nan)
    near_mask = valid_plx & (d_kpc < NEAR_BOUNDARY_KPC)
    far_mask = valid_plx & (d_kpc >= NEAR_BOUNDARY_KPC)
    logger.info(
        "distance bins: %d near (<%.2f kpc), %d far (≥%.2f kpc), %d no-parallax",
        int(near_mask.sum()), NEAR_BOUNDARY_KPC,
        int(far_mask.sum()), NEAR_BOUNDARY_KPC,
        int((~valid_plx).sum()),
    )

    av_eden = np.full(n, np.nan, dtype=np.float32)
    av_sfd = np.full(n, np.nan, dtype=np.float32)
    av_los = np.full(n, np.nan, dtype=np.float32)
    source = np.full(n, -1, dtype=np.int8)

    logger.info("loading Edenhofer+2024 healpix cube")
    eden = Edenhofer2023Query()
    logger.info("loading SFD map")
    sfd = SFDQuery()

    ra = df["ra_deg"].to_numpy(dtype=np.float64)
    dec = df["dec_deg"].to_numpy(dtype=np.float64)

    # SFD over the whole sample (cheap, 2D lookup) — used both as the
    # ≥1.25 kpc line-of-sight integrator and as a diagnostic column for all.
    logger.info("querying SFD for all %d stars", n)
    coords_all = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ebv_sfd_all = np.asarray(sfd(coords_all), dtype=np.float64)
    av_sfd[:] = (ebv_sfd_all * SFD_TO_AV).astype(np.float32)

    # Edenhofer for the near bin only (needs distance-attached SkyCoord).
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
        source[near_mask] = 0  # 0 = Edenhofer
        # Guard against NaN returns from the map (outside footprint etc).
        nan_near = ~np.isfinite(av_near)
        if nan_near.any():
            idx = np.flatnonzero(near_mask)[nan_near]
            source[idx] = -1
            av_los[idx] = np.nan

    # Far bin (≥1.25 kpc): fall back to SFD × 2.742.
    if far_mask.any():
        av_los[far_mask] = av_sfd[far_mask]
        source[far_mask] = 2  # 2 = SFD (no Lallement installed)

    out_df = pd.DataFrame({
        "source_id": df["source_id"].to_numpy(),
        "av_edenhofer": av_eden,
        "av_sfd_path": av_sfd,
        "av_los": av_los,
        "av_los_source": source,
    })
    _write_parquet_atomic(out_df, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB)", out, size_mb)

    n_eden = int((source == 0).sum())
    n_sfd = int((source == 2).sum())
    n_nan = int((source == -1).sum())
    logger.info("composition: %d Edenhofer, %d SFD-fallback, %d no-coverage",
                n_eden, n_sfd, n_nan)

    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/precompute_stream3_av.py",
        sources=[
            LocalSource(
                name="Stream 3 selected (Andrae+2023 RGB stratified)",
                path=str(src.relative_to(repo)),
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
        cuts_applied=[],
        corrections=[
            "distance = 1/parallax_mas kpc (rough, parallax≤0 → NaN)",
            f"SFD E(B-V) → A_V via {SFD_TO_AV} (Schlafly & Finkbeiner 2011)",
        ],
        row_count_before=n,
        row_count_after=n,
        notes=(
            "Per-star A_V lookup for Stream 3. Edenhofer+2024 for d<1.25 kpc, "
            "SFD (×2.742) for d≥1.25 kpc. Lallement+2022 1.25–3 kpc upgrade "
            "pending — install a dustmaps build that ships lallement2022 or "
            "add a custom query. The 3.2 GB Edenhofer FITS can be deleted "
            "after this script runs; only the per-star av_los is kept downstream."
        ),
        extra={
            "n_near_edenhofer": n_eden,
            "n_far_sfd_fallback": n_sfd,
            "n_no_coverage": n_nan,
            "near_boundary_kpc": NEAR_BOUNDARY_KPC,
            "sfd_to_av_coeff": SFD_TO_AV,
        },
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
