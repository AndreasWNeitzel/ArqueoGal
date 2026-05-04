"""Unified Pipeline 1 input preprocessing — single source of truth.

ALL three streams (Stream 1 = APOGEE×Gaia training, Stream 2 = Hon+2021
TESS giants, Stream 3 = Andrae+2023 application pool) call this module
identically. The preprocessing steps are applied in a fixed order:

1. Lindegren+2021 parallax zero-point (apply_parallax_zpt)
2. Riello+2021 G-band correction (apply_g_mag_correction)
3. Fetch + Ye+2024 correct XP coefficients (gaia_xp.fetch_xp_coefficients,
   gaia_xp.apply_ye2024_correction)
4. Hermite re-projection (gaia_xp.reproject_ye_to_hermite)
5. Frozen v1 z-score on the Hermite basis (frozen_stats.apply_frozen_zscore)
   with basis-fingerprint verification (verify_basis_fingerprint)
6. Yuan+2013 + CCM89 R_V=3.1 broadband dereddening on JHKW1W2 IF the IR
   columns are present (extinction.apply_extinction_corrections)
7. Add Bailer-Jones+2021 distance trust flags + Av source flags as
   per-star booleans
8. Verify output has all DEFAULT_AUX_COLS columns (auxiliary features)

The contract is identical for train and inference — mode is for logging
and sidecar provenance only; both apply byte-identical transforms.

References
----------
docs/plan/03_stream3_inference.md — preprocessing contract + justification
docs/protocols/extinction_correction.md — hybrid dereddening architecture
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.extinction import apply_extinction_corrections
from arqueogal.data.frozen_stats import (
    apply_frozen_zscore,
    load_frozen_zscore_stats,
    verify_basis_fingerprint,
)
from arqueogal.data.gaia_corrections import apply_g_mag_correction, apply_parallax_zpt
from arqueogal.data.gaia_xp import (
    XP_BATCH_SIZE,
    apply_ye2024_correction,
    fetch_xp_coefficients,
    reproject_ye_to_hermite,
)
from arqueogal.xp_abundances.main import DEFAULT_AUX_COLS

logger = logging.getLogger(__name__)

# Sentinel to indicate that XP fetch should be skipped in this pipeline stage.
# Used when a caller has already fetched XP coefficients and corrected them,
# and only wants steps 5-9 applied (e.g., re-projection + z-scoring + aux).
_SKIP_XP_FETCH: Final[str] = "__skip_xp_fetch__"


def apply_pipeline1_preprocessing(
    df: pd.DataFrame,
    *,
    mode: Literal["train", "inference"] = "inference",
    aip: TAPService | None = None,
    xp_batch_size: int = XP_BATCH_SIZE,
    apply_extinction: bool = True,
    frozen_stats_path: Path | str | None = None,
    skip_xp_fetch: bool = False,
    xp_coords: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the unified Pipeline 1 preprocessing to a feature frame.

    This is the single source of truth for input preprocessing. All streams
    call this with stream-specific input data and receive preprocessing-ready
    output suitable for training or inference.

    Parameters
    ----------
    df
        Input DataFrame with at minimum ``source_id`` column. Contents depend
        on the skip_xp_fetch flag and which preprocessing steps have already
        been applied upstream:

        - If ``skip_xp_fetch=False`` (default): df must carry ``source_id``.
          Steps 1-9 are applied. XP coefficients are fetched and corrected.
        - If ``skip_xp_fetch=True``: df must already have XP columns
          (``corrected_flux`` from Ye+2024) plus astrometry/photometry for
          steps 1-2 and 5-9. Useful when XP has been pre-processed elsewhere.

    mode
        ``"train"`` or ``"inference"``. Both apply identical transforms; the
        flag is for logging and provenance only.

    aip
        Authenticated AIP TAP service for XP fetch. Required if
        ``skip_xp_fetch=False``. If ``None``, will call
        ``arqueogal.data.tap.aip_service()`` to lazily initialize (honouring
        ``GAIA_AIP_TOKEN`` and YAML credential fallback).

    xp_batch_size
        ``IN (...)`` chunk size for XP coefficient fetching (default 5000).
        Ignored if ``skip_xp_fetch=True``.

    apply_extinction
        If ``True`` (default), apply Yuan+2013 + CCM89 broadband dereddening
        (step 6). If ``False``, skip extinction. Useful for diagnostics or
        when the frame is known to lack dust-map columns.

    frozen_stats_path
        Path to the frozen-stats provenance sidecar JSON
        (``data/processed/pipeline1_features_stream1.provenance.json``).
        If ``None``, will be inferred from the frame's surrounding context
        at preprocessing time. Required for step 5 (z-scoring).

    skip_xp_fetch
        If ``True``, assume df already has XP columns (``corrected_flux``,
        ``ye2024_flag``, ``a_v_sfd`` from Ye+2024 correction upstream) and
        skip steps 3-4 (fetch + Ye correction). Step 5 (Hermite reprojection)
        still runs on the pre-provided ``corrected_flux``.

    xp_coords
        Required if ``skip_xp_fetch=False``. Minimum columns: ``source_id``,
        ``ra``, ``dec`` (ICRS degrees at Ep=2016.0). Joined to XP during
        correction step. Ignored if ``skip_xp_fetch=True``.

    Returns
    -------
    pd.DataFrame
        Preprocessing-ready frame with:

        - Astrometry corrected (parallax ZPT, G-mag correction)
        - XP Hermite coefficients (``bp_coef_norm_{1..54}``,
          ``rp_coef_norm_{1..54}``, ``bp_c0_z``, ``rp_c0_z``)
        - Reprojection residuals (``reprojection_residual_rms*``)
        - Broadband photometry dereddened (``j_mag_dered``, etc.) if present
        - Extinction priors and flags retained
        - All auxiliary columns in :data:`DEFAULT_AUX_COLS` present (filled
          with NaN if not available in input)

    Raises
    ------
    KeyError
        If required columns are missing.
    FrozenStatsMismatchError
        If Hermite basis fingerprint does not match frozen stats.
    ValueError
        If XP reprojection fails or invalid mode/flags.
    """
    from arqueogal.data.tap import aip_service

    out = df.copy()
    logger.info(
        "Pipeline 1 preprocessing: mode=%s, skip_xp_fetch=%s, apply_extinction=%s, n_rows=%d",
        mode,
        skip_xp_fetch,
        apply_extinction,
        len(out),
    )

    # Step 1: Parallax zero-point
    logger.info("Step 1/9: Lindegren+2021 parallax ZPT")
    out = apply_parallax_zpt(out)

    # Step 2: G-mag correction
    logger.info("Step 2/9: Riello+2021 G-band correction")
    out = apply_g_mag_correction(out)

    # Steps 3-4: XP fetch + correction + Hermite reprojection
    if not skip_xp_fetch:
        if xp_coords is None:
            raise ValueError(
                "xp_coords required when skip_xp_fetch=False. "
                "Must contain (source_id, ra, dec) columns."
            )
        if "source_id" not in out.columns:
            raise KeyError("df must contain 'source_id' column for XP fetch")

        logger.info("Step 3/9: Fetch XP coefficients for %d source_ids", len(out))
        tap = aip if aip is not None else aip_service()
        raw_xp = fetch_xp_coefficients(
            tap,
            out["source_id"].tolist(),
            batch_size=xp_batch_size,
        )
        logger.info("Step 3/9: XP fetch returned %d rows", len(raw_xp))

        logger.info("Step 4/9: Ye+2024 NN flux correction")
        corrected_xp = apply_ye2024_correction(raw_xp, xp_coords)
        n_ye_ok = (corrected_xp["ye2024_flag"] == 0).sum()
        logger.info(
            "Step 4/9: Ye+2024 done: %d OK / %d total",
            n_ye_ok,
            len(corrected_xp),
        )

        logger.info("Step 5/9: Hermite re-projection of corrected flux")
        hermite_dict = reproject_ye_to_hermite(corrected_xp["corrected_flux"].to_numpy())
        logger.info(
            "Step 5/9: Hermite done (basis fingerprint %s)",
            hermite_dict["basis_fingerprint_sha256"],
        )

        # Now proceed to z-scoring with the hermite outputs + corrected_xp.
        # Merge back into out by source_id so we get both the XP and any
        # astrometry/photometry columns that were in out.
        cols_to_merge = ["source_id", "corrected_flux", "a_v_sfd", "ye2024_flag"]
        _xp_to_merge = corrected_xp[cols_to_merge].copy()
        _xp_to_merge["bp_coeffs"] = [row.astype(np.float32) for row in hermite_dict["bp_coeffs"]]
        _xp_to_merge["rp_coeffs"] = [row.astype(np.float32) for row in hermite_dict["rp_coeffs"]]
        _xp_to_merge["reprojection_residual_rms"] = hermite_dict["reprojection_residual_rms"]
        _xp_to_merge["reprojection_residual_rms_bp"] = hermite_dict["reprojection_residual_rms_bp"]
        _xp_to_merge["reprojection_residual_rms_rp"] = hermite_dict["reprojection_residual_rms_rp"]

        out = out.merge(_xp_to_merge, on="source_id", how="inner")
    else:
        # Caller provided XP already. Expect corrected_flux + Hermite basis
        # fingerprint to be determinable from the frame or context.
        if "corrected_flux" not in out.columns:
            raise KeyError("corrected_flux required when skip_xp_fetch=True")
        logger.info("Step 3-4/9: Skipped (XP provided by caller)")

        logger.info("Step 5/9: Hermite re-projection of pre-provided corrected flux")
        hermite_dict = reproject_ye_to_hermite(out["corrected_flux"].to_numpy())
        logger.info(
            "Step 5/9: Hermite done (basis fingerprint %s)",
            hermite_dict["basis_fingerprint_sha256"],
        )

        # Unpack hermite_dict into the dataframe
        out["bp_coeffs"] = [row.astype(np.float32) for row in hermite_dict["bp_coeffs"]]
        out["rp_coeffs"] = [row.astype(np.float32) for row in hermite_dict["rp_coeffs"]]
        out["reprojection_residual_rms"] = hermite_dict["reprojection_residual_rms"]
        out["reprojection_residual_rms_bp"] = hermite_dict["reprojection_residual_rms_bp"]
        out["reprojection_residual_rms_rp"] = hermite_dict["reprojection_residual_rms_rp"]

    # Step 5b: Frozen z-score with basis-fingerprint verification
    logger.info("Step 5b/9: Frozen z-score (basis fingerprint check + application)")
    _frozen_stats_path = frozen_stats_path or Path(__file__).parent.parent.parent / (
        "data/processed/pipeline1_features_stream1.provenance.json"
    )
    frozen_stats = load_frozen_zscore_stats(_frozen_stats_path)

    verify_basis_fingerprint(
        hermite_dict["basis_fingerprint_sha256"],
        frozen_stats,
    )

    # Compute normalized coefficients for z-scoring
    bp_coeffs_norm = out["bp_coeffs"] / (out["bp_coeffs"].apply(lambda x: x[0] + 1e-30))
    rp_coeffs_norm = out["rp_coeffs"] / (out["rp_coeffs"].apply(lambda x: x[0] + 1e-30))
    bp_c0_log = np.log10(np.maximum(out["bp_coeffs"].apply(lambda x: x[0]).to_numpy(), 1e-30))
    rp_c0_log = np.log10(np.maximum(out["rp_coeffs"].apply(lambda x: x[0]).to_numpy(), 1e-30))

    # Stack the normalized coefficients so apply_frozen_zscore gets
    # 2D arrays with shape (N, 54) and (N, 1) respectively.
    _bp_norm_stacked = np.array([x[1:] for x in bp_coeffs_norm.values])
    _rp_norm_stacked = np.array([x[1:] for x in rp_coeffs_norm.values])

    bp_norm_z, rp_norm_z, bp_c0_z, rp_c0_z = apply_frozen_zscore(
        _bp_norm_stacked,
        _rp_norm_stacked,
        bp_c0_log,
        rp_c0_log,
        frozen_stats,
    )

    # Store z-scored coefficients as flat columns: bp_coef_norm_1..54, etc.
    # Build new columns dict to avoid DataFrame fragmentation warnings
    new_cols = {}
    for i, coef_idx in enumerate(range(1, 55)):
        new_cols[f"bp_coef_norm_{coef_idx}"] = bp_norm_z[:, i].astype(np.float32)
        new_cols[f"rp_coef_norm_{coef_idx}"] = rp_norm_z[:, i].astype(np.float32)
    new_cols["bp_c0_z"] = bp_c0_z.astype(np.float32)
    new_cols["rp_c0_z"] = rp_c0_z.astype(np.float32)
    out = out.assign(**new_cols)

    logger.info("Step 5b/9: Z-scoring complete")

    # Step 6: Broadband dereddening (if requested and IR columns present)
    if apply_extinction:
        has_any_broadband = any(
            c in out.columns for c in ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag")
        )
        has_any_av_layer = any(
            c in out.columns for c in ("av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median")
        )
        if has_any_broadband and has_any_av_layer:
            logger.info("Step 6/9: CCM89 R_V=3.1 + Yuan+2013 broadband dereddening")
            out = apply_extinction_corrections(out, inplace=True)
        else:
            logger.info(
                "Step 6/9: Skipped dereddening (broadband=%s, av_layer=%s)",
                has_any_broadband,
                has_any_av_layer,
            )
    else:
        logger.info("Step 6/9: Skipped (apply_extinction=False)")

    # Step 7: BJ21 distance flags + Av source flags (if present)
    logger.info("Step 7/9: Distance + extinction trust flags (no-op if not present)")
    out = _ensure_trust_flags(out)

    # Step 8: Verify all DEFAULT_AUX_COLS present
    logger.info("Step 8/9: Verify output columns")
    _verify_output_columns(out)

    logger.info(
        "Pipeline 1 preprocessing complete: output shape %s, %d rows, mode=%s",
        out.shape,
        len(out),
        mode,
    )
    return out


def _ensure_trust_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add Bailer-Jones+2021 distance + Av source flags if missing.

    Returns a potentially-rebuilt frame. Already present columns are left
    as-is. If columns don't exist, they're created as NaN. This is a
    no-op if the frame was built by build_stream2_kinematic_catalogue,
    which already includes these flags.
    """
    new_cols = {}
    for col in (
        "r_med_photogeo_is_finite",
        "av_is_neighborhood_fallback",
        "av_distance_prior_dominated",
        "av_neighbourhood_high_dispersion",
    ):
        if col not in df.columns:
            new_cols[col] = np.nan
    if new_cols:
        df = df.assign(**new_cols)
    return df


def _verify_output_columns(df: pd.DataFrame) -> None:
    """Verify that output has all DEFAULT_AUX_COLS + required XP columns.

    Logs a warning for missing columns but does not raise — the frame is
    usable for diagnostics even with missing auxiliary columns.
    """
    xp_cols = (
        ["bp_c0_z", "rp_c0_z"]
        + [f"bp_coef_norm_{i}" for i in range(1, 55)]
        + [f"rp_coef_norm_{i}" for i in range(1, 55)]
        + list(DEFAULT_AUX_COLS)
    )
    missing = [c for c in xp_cols if c not in df.columns]
    if missing:
        logger.warning(
            "Output missing %d expected columns: %s",
            len(missing),
            missing[:5],  # Log first 5 to avoid spam
        )
    else:
        logger.info("Output has all %d expected feature columns", len(xp_cols))


__all__ = [
    "apply_pipeline1_preprocessing",
]
