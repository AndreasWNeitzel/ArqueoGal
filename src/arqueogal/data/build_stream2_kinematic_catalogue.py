"""Stream 2 kinematic catalogue — the D-Cat-b MVP astrometric-kinematic surface.

The Hon+2021 TESS asteroseismic-giant catalogue is the priority target for
D-Cat-b (Aug 2026). The kinematic component of that deliverable is a
parquet of TESS giants carrying:

- 3-D position: (ra, dec, distance) with Bailer-Jones+2021 photogeometric
  distance + percentile-spread trust flag.
- 3-D velocity: (pmra, pmdec, radial_velocity) — already attached by
  :mod:`arqueogal.data.ingest_stream2` from the AIP Gaia DR3 enrichment.
- Extinction: dust-map fusion (Edenhofer+2024 / Lallement+2022 / SFD +
  neighbourhood-median) → ``av_los`` + ``av_los_source`` + the three
  trust flags from :mod:`arqueogal.data.extinction`.
- Galpy actions: (J_R, L_z, J_z, ecc, r_peri, r_apo, z_max, E) under
  McMillan+2017 with the pinned :data:`utils.coordinates.GALACTOCENTRIC_FRAME`.

This module is the **call-graph wiring** the previous ingestion path was
missing (the brief: Section 7 of the 2026-04-29 audit). Each downstream
component already exists; this orchestrator runs them in order and emits
the kinematic-augmented parquet + a single combined provenance sidecar.

Chemistry inference on Stream 2 (the ML-predicted abundances) is
**downstream of this module**: once this kinematic catalogue is on disk,
``scripts/run_pipeline1_inference.py`` adds the predicted labels and
covariance.

References
----------
- ``docs/plan/00_overview.md`` (Phase B: Stream-2-as-priority).
- ``docs/protocols/extinction_correction.md`` (v1 dereddening contract).
- ``utils/coordinates.GALACTOCENTRIC_FRAME`` (frame conventions).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.extinction import (
    DEFAULT_EXTINCTION_LAW,
    apply_extinction_corrections,
)
from arqueogal.data.kinematics import KinematicsConfig, compute_actions
from arqueogal.data.provenance import Provenance, write_sidecar
from arqueogal.utils.io import save_parquet

logger = logging.getLogger(__name__)


# Trust-flag thresholds for the BJ21 distance percentile spread. Calibrated
# at v1 from the Stream-1 Stream-3 holdout; documented in
# docs/protocols/extinction_correction.md §3.
DEFAULT_DIST_RELATIVE_SPREAD_FLAG: float = 0.30
"""Threshold above which (r_hi - r_lo) / r_med flags a star as having a
prior-dominated distance. 0.30 corresponds to the ~3σ Bailer-Jones tail
where the Galactic prior dominates over the parallax constraint."""


def assign_distance_trust_flags(
    df: pd.DataFrame,
    *,
    relative_spread_threshold: float = DEFAULT_DIST_RELATIVE_SPREAD_FLAG,
) -> dict[str, np.ndarray]:
    """Per-star distance-trust booleans derived from the BJ21 percentile triple.

    Returns four flags keyed for direct attachment to the output frame:

    - ``dist_has_bj21``: True iff ``r_med_photogeo`` is finite.
    - ``dist_relative_spread_high``: True iff ``(r_hi − r_lo) / r_med``
      exceeds ``relative_spread_threshold``. The 16-84 % spread of the
      photogeometric posterior measured against the median; high spread
      means the parallax SNR is poor and the Galactic prior dominates.
    - ``dist_negative_parallax``: True iff the star has a negative
      parallax (Bailer-Jones still produces a posterior but the tail
      becomes prior-dominated).
    - ``dist_trustworthy``: True iff the star has a finite BJ21 distance,
      a non-high relative spread, and a positive parallax. This is the
      composite "use this star for science" gate.
    """
    n = len(df)
    has_bj = (
        df["r_med_photogeo"].notna().to_numpy()
        if "r_med_photogeo" in df.columns
        else np.zeros(n, dtype=bool)
    )
    if (
        "r_med_photogeo" in df.columns
        and "r_lo_photogeo" in df.columns
        and "r_hi_photogeo" in df.columns
    ):
        r_med = df["r_med_photogeo"].to_numpy(dtype=np.float64, copy=False)
        r_lo = df["r_lo_photogeo"].to_numpy(dtype=np.float64, copy=False)
        r_hi = df["r_hi_photogeo"].to_numpy(dtype=np.float64, copy=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.where(r_med > 0, (r_hi - r_lo) / r_med, np.inf)
        spread_high = np.where(np.isfinite(spread), spread > relative_spread_threshold, True)
    else:
        spread_high = np.ones(n, dtype=bool)

    if "parallax" in df.columns:
        plx = df["parallax"].to_numpy(dtype=np.float64, copy=False)
        neg_plx = np.where(np.isfinite(plx), plx < 0.0, False)
    else:
        neg_plx = np.zeros(n, dtype=bool)

    trustworthy = has_bj & ~spread_high & ~neg_plx

    return {
        "dist_has_bj21": has_bj.astype(bool),
        "dist_relative_spread_high": spread_high.astype(bool),
        "dist_negative_parallax": neg_plx.astype(bool),
        "dist_trustworthy": trustworthy.astype(bool),
    }


def build_stream2_kinematic_catalogue(  # noqa: PLR0913 — orthogonal scientific knobs
    stream2_df: pd.DataFrame,
    *,
    output_path: Path | str,
    bj21_df: pd.DataFrame | None = None,
    av_layer_df: pd.DataFrame | None = None,
    kinematics_config: KinematicsConfig | None = None,
    apply_extinction: bool = True,
    relative_spread_threshold: float = DEFAULT_DIST_RELATIVE_SPREAD_FLAG,
) -> Path:
    """Compose the Stream-2 kinematic catalogue end-to-end.

    The function is **TAP-free**: the caller supplies the three input
    frames (Stream-2 base, BJ21 distances, dust-map fusion). For an
    end-to-end run that hits the AIP / GAVO / VizieR services, see
    ``scripts/build_stream2_kinematic_catalogue.py``; this function is
    the testable core.

    Parameters
    ----------
    stream2_df
        Output of :func:`arqueogal.data.ingest_stream2.ingest_stream2` —
        carries Hon+2021 × TIC × DR2→DR3 × Gaia DR3 enrichment columns
        (source_id, ra, dec, parallax, pmra, pmdec, radial_velocity, ...).
    output_path
        Destination parquet. Provenance sidecar lives alongside.
    bj21_df
        Optional pre-fetched Bailer-Jones+2021 frame. Must carry
        ``source_id``, ``r_med_photogeo``, ``r_lo_photogeo``,
        ``r_hi_photogeo``. If ``None``, the function expects ``stream2_df``
        to already carry these columns (i.e. ``enrich_geometry`` was run
        upstream); otherwise the kinematic step skips.
    av_layer_df
        Optional pre-fetched dust-map fusion frame carrying ``source_id``,
        ``av_edenhofer``, ``av_lallement``, ``av_sfd``, ``av_nbhd_median``,
        ``av_nbhd_std``. If ``None``, the function expects ``stream2_df``
        to already carry them.
    kinematics_config
        Override the default :class:`KinematicsConfig` (McMillan+2017,
        Staeckel δ=0.45, GRAVITY-pinned R_0/V_0/z_sun).
    apply_extinction
        Run :func:`apply_extinction_corrections` after the dust-map join.
        Default True. Skipped if the broadband columns are absent.
    relative_spread_threshold
        Forwarded to :func:`assign_distance_trust_flags`.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = stream2_df.copy()
    n_in = len(out)
    logger.info("Stream 2 kinematic: %d input rows", n_in)

    # ── Distances ────────────────────────────────────────────────────────────
    if bj21_df is not None:
        if "source_id" not in bj21_df.columns:
            raise KeyError("bj21_df must carry 'source_id'")
        logger.info("Stream 2 kinematic: joining BJ21 (%d rows)", len(bj21_df))
        keep_cols = [
            c
            for c in ("source_id", "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo")
            if c in bj21_df.columns
        ]
        out = out.merge(bj21_df[keep_cols], on="source_id", how="left")
    has_bj_columns = "r_med_photogeo" in out.columns
    if not has_bj_columns:
        logger.warning(
            "Stream 2 kinematic: no BJ21 columns; trust flags will all be False "
            "and galpy actions will be skipped"
        )
        # Inject NaN columns so downstream code has them.
        for col in ("r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"):
            out[col] = np.nan

    trust_flags = assign_distance_trust_flags(
        out, relative_spread_threshold=relative_spread_threshold
    )
    for name, arr in trust_flags.items():
        out[name] = arr
    n_trust = int(trust_flags["dist_trustworthy"].sum())
    logger.info(
        "Stream 2 kinematic: %d / %d stars carry a trustworthy BJ21 distance",
        n_trust,
        n_in,
    )

    # ── Dust-map fusion + Yuan+2013 dereddening (broadband only) ─────────────
    extinction_applied = False
    if av_layer_df is not None:
        if "source_id" not in av_layer_df.columns:
            raise KeyError("av_layer_df must carry 'source_id'")
        logger.info("Stream 2 kinematic: joining dust-map fusion (%d rows)", len(av_layer_df))
        keep = [
            c
            for c in (
                "source_id",
                "av_edenhofer",
                "av_lallement",
                "av_sfd",
                "av_nbhd_median",
                "av_nbhd_std",
            )
            if c in av_layer_df.columns
        ]
        out = out.merge(av_layer_df[keep], on="source_id", how="left")

    has_any_av_layer = any(
        c in out.columns for c in ("av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median")
    )
    has_any_broadband = any(
        c in out.columns for c in ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag")
    )
    if apply_extinction and has_any_av_layer:
        logger.info(
            "Stream 2 kinematic: applying %s extinction recipe (broadband=%s)",
            DEFAULT_EXTINCTION_LAW.name,
            has_any_broadband,
        )
        out = apply_extinction_corrections(out, inplace=True)
        extinction_applied = True
    else:
        logger.info(
            "Stream 2 kinematic: skipping dereddening (av_layer=%s, broadband=%s); "
            "model inference must filter on av_los_source != missing",
            has_any_av_layer,
            has_any_broadband,
        )

    # ── Galpy actions ────────────────────────────────────────────────────────
    cfg = kinematics_config or KinematicsConfig()
    n_solved = 0
    if has_bj_columns and {"pmra", "pmdec", "radial_velocity"}.issubset(out.columns):
        logger.info(
            "Stream 2 kinematic: galpy actions on %d rows (potential=%s)", n_in, cfg.potential
        )
        try:
            actions = compute_actions(out, config=cfg)
            n_solved = len(actions)
            logger.info("Stream 2 kinematic: %d / %d actions solved", n_solved, n_in)
            out = out.merge(actions, on="source_id", how="left", suffixes=("", "_kin"))
        except ImportError as exc:
            logger.warning(
                "Stream 2 kinematic: skipping galpy actions (%s); rerun in the "
                "RAPIDS env once galpy is installed",
                exc,
            )
    else:
        logger.warning(
            "Stream 2 kinematic: skipping galpy actions (BJ21=%s, RV/PM=%s)",
            has_bj_columns,
            {"pmra", "pmdec", "radial_velocity"}.issubset(out.columns),
        )

    # ── Write parquet + sidecar ──────────────────────────────────────────────
    logger.info("Stream 2 kinematic: writing %s", output_path)
    save_parquet(out, output_path)

    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/build_stream2_kinematic_catalogue.py",
        sources=[],
        cuts_applied=[],
        corrections=[
            f"BJ21 percentile-spread distance-trust flag (threshold={relative_spread_threshold})",
            f"galpy actions ({cfg.potential}, Staeckel delta={cfg.staeckel_delta})"
            if n_solved > 0
            else "galpy actions (skipped — missing inputs)",
        ]
        + (
            [
                f"Extinction: {DEFAULT_EXTINCTION_LAW.name} "
                "(Yuan+2013 broadband ratios, dust-map fusion via "
                "data.extinction.apply_extinction_corrections)"
            ]
            if extinction_applied
            else []
        ),
        row_count_before=n_in,
        row_count_after=len(out),
        notes=(
            "Stream 2 D-Cat-b kinematic catalogue: TESS asteroseismic giants "
            "(Hon+2021) cross-matched to Gaia DR3, with BJ21 distances, "
            "dust-map fusion A_V, optional broadband dereddening, and McMillan17 "
            "galpy orbital parameters. Chemistry inference is downstream "
            "(scripts/run_pipeline1_inference.py)."
        ),
        extra={
            "kinematics_config": asdict(cfg),
            "rows_solved": n_solved,
            "rows_unsolved": n_in - n_solved,
            "distance_trust_counts": {k: int(v.sum()) for k, v in trust_flags.items()},
            "extinction_correction": (
                {
                    "applied": True,
                    "law": DEFAULT_EXTINCTION_LAW.fingerprint(),
                    "flag_counts": {
                        col: int(out[col].sum())
                        for col in (
                            "av_is_neighborhood_fallback",
                            "av_distance_prior_dominated",
                            "av_neighbourhood_high_dispersion",
                        )
                        if col in out.columns
                    },
                }
                if extinction_applied
                else {"applied": False}
            ),
        },
    )
    write_sidecar(prov)
    logger.info("Stream 2 kinematic: done (%d rows → %s)", len(out), output_path)
    return output_path


__all__ = [
    "DEFAULT_DIST_RELATIVE_SPREAD_FLAG",
    "assign_distance_trust_flags",
    "build_stream2_kinematic_catalogue",
]
