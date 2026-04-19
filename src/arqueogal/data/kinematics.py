"""Orbital parameters via galpy — §9 of data_acquisition.md.

Computes per-star actions (J_R, J_z, L_z), orbital shape (ecc, r_peri,
r_apo, z_max), energy E, and Galactocentric cylindrical velocities
(v_R, v_T, v_z) in the McMillan+2017 Milky Way potential using the
Staeckel fudge method (Binney 2012) with ``delta=0.45`` (Mackereth+2019).

This module is the entry point for Pipeline 2's kinematic features and for
the D-Cat-d soft-membership outputs. MC uncertainty propagation is split
into two tiers per §9.5:

- :func:`compute_actions` — central-value only (Stream 3 bulk, 1.5 M stars).
- :func:`compute_actions_mc` — N-MC draws from the Gaia astrometric
  covariance for the D-Cat-d boundary-cluster subsample.

The McMillan+2017 potential ships with ``ro=8.21 kpc, vo=233.1 km/s``.
data_acquisition.md §9.2 prefers ``R_0=8.122 kpc`` (GRAVITY 2018) — the
two values differ by ~1%. We follow the potential's native ``ro``/``vo``
to avoid rescaling the potential away from its fitted values; solar
peculiar motion (Schönrich+Binney+Dehnen 2010) and ``z_0=20.8 pc``
(Bennett & Bovy 2019) from §9.2 are applied via ``Orbit(..., solarmotion,
zo)`` and are fully user-settable.

galpy is imported lazily inside the computation function — the module can
be imported and the non-galpy helpers tested without triggering galpy's
sizeable import chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# §9.2 — solar / galactic constants. These are the *defaults*; callers can
# override per run if needed (e.g., sensitivity tests).
R_0_KPC: float = 8.21
"""Sun-galactocentric distance. Matches McMillan+2017 ``ro``. §9.2 cites
GRAVITY+2018 R_0 = 8.122 kpc but keeping McMillan's native value avoids a
1% rescaling of the fitted potential."""

V_0_KMS: float = 233.1
"""Local circular speed — matches McMillan+2017 ``vo`` (Reid & Brunthaler
2020 value in §9.2)."""

Z_0_PC: float = 20.8
"""Sun's height above the Galactic plane (Bennett & Bovy 2019)."""

SOLAR_MOTION_KMS: tuple[float, float, float] = (-11.1, 12.24, 7.25)
"""Sun's peculiar motion (U, V, W) in km/s. Sign convention matches
``galpy.orbit.Orbit(solarmotion=...)``: ``U`` is positive toward the
Galactic centre, negated in the tuple per galpy convention, so Schönrich+
Binney+Dehnen+2010's ``U_sun = +11.1`` becomes ``-11.1`` here."""

STAECKEL_DELTA: float = 0.45
"""Staeckel-fudge focal-length parameter — standard choice for the MW
(Binney 2012; Mackereth+2019)."""

REQUIRED_INPUT_COLS: tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "r_med_photogeo",
    "pmra",
    "pmdec",
    "radial_velocity",
)

OUTPUT_COLS: tuple[str, ...] = (
    "source_id",
    "R_galcen_kpc",
    "z_galcen_kpc",
    "phi_galcen_rad",
    "v_R_kms",
    "v_T_kms",
    "v_z_kms",
    "J_R_kpc_kms",
    "L_z_kpc_kms",
    "J_z_kpc_kms",
    "ecc",
    "r_peri_kpc",
    "r_apo_kpc",
    "z_max_kpc",
    "E_kms2",
)


@dataclass(frozen=True, slots=True)
class KinematicsConfig:
    """Tunable knobs for :func:`compute_actions`. Defaults match §9.2."""

    ro_kpc: float = R_0_KPC
    vo_kms: float = V_0_KMS
    zo_pc: float = Z_0_PC
    solarmotion_kms: tuple[float, float, float] = SOLAR_MOTION_KMS
    staeckel_delta: float = STAECKEL_DELTA
    potential: Literal["mcmillan17", "mwpotential2014"] = "mcmillan17"


def compute_actions(
    df: pd.DataFrame,
    *,
    config: KinematicsConfig | None = None,
) -> pd.DataFrame:
    """Central-value actions for every row in ``df`` (§9, §9.5 bulk path).

    Parameters
    ----------
    df
        Must carry :data:`REQUIRED_INPUT_COLS`. Distances are in **pc**
        (``r_med_photogeo``), proper motions in **mas/yr**, RV in **km/s**.
        Rows with any NaN in the required columns are dropped with a log
        warning — galpy cannot handle partial phase-space coordinates.
    config
        Override defaults (e.g. swap potentials for a sensitivity check).

    Returns
    -------
    pd.DataFrame
        One row per surviving input star, columns :data:`OUTPUT_COLS`.
        All physical quantities carry SI-ish units as indicated by the
        column suffix (``_kpc``, ``_kms``, ``_kms2`` for energy).
    """
    cfg = config or KinematicsConfig()
    _validate_required_cols(df)
    clean = _drop_nan_rows(df)
    if clean.empty:
        logger.warning("compute_actions: no rows survived NaN filtering; returning empty frame")
        return pd.DataFrame(columns=list(OUTPUT_COLS))

    coords = _build_skycoord(clean)
    raw = _run_galpy(coords, cfg)
    out = pd.DataFrame({"source_id": clean["source_id"].to_numpy(), **raw})
    return out[list(OUTPUT_COLS)]


def compute_actions_mc(
    df: pd.DataFrame,
    *,
    n_samples: int = 100,
    rng_seed: int = 0,
    config: KinematicsConfig | None = None,
) -> pd.DataFrame:
    """N-MC draws from the Gaia 5×5 astrometric covariance (§9.5 subsample).

    Requires the covariance-correlation coefficients (``ra_dec_corr`` etc.)
    alongside the per-parameter errors. For each input star, draws
    ``n_samples`` realisations of (ra, dec, parallax, pmra, pmdec) from
    the multivariate normal, converts parallax → distance via ``1000/ϖ``
    (simple — not the Bailer-Jones prior; §9.5 is only for boundary
    cases where the prior dependence is second-order), and computes
    actions per draw.

    Returns one *long-format* row per (star, draw) so downstream code
    can compute per-star means, stds, and percentiles with a groupby.

    This is expensive — N_MC × N_stars × ~1 ms. §9.5 restricts it to
    ~10⁴ boundary-cluster stars × 100 draws ≈ 15 min.
    """
    cfg = config or KinematicsConfig()
    required = set(REQUIRED_INPUT_COLS) | {
        "parallax", "parallax_error",
        "pmra_error", "pmdec_error",
        "radial_velocity_error",
        "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
        "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
        "parallax_pmra_corr", "parallax_pmdec_corr",
        "pmra_pmdec_corr",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_actions_mc requires columns: {sorted(missing)}")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    clean = _drop_nan_rows(df[list(required)])
    if clean.empty:
        return pd.DataFrame(columns=["source_id", "draw", *OUTPUT_COLS[1:]])

    rng = np.random.default_rng(rng_seed)
    long_rows: list[pd.DataFrame] = []
    for draw_idx in range(n_samples):
        draw_df = _draw_astrometric_sample(clean, rng)
        acted = compute_actions(draw_df, config=cfg)
        acted.insert(1, "draw", draw_idx)
        long_rows.append(acted)
    return pd.concat(long_rows, ignore_index=True)


# -----------------------------------------------------------------------------
# internals
# -----------------------------------------------------------------------------


def _validate_required_cols(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_INPUT_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"compute_actions requires columns: {sorted(missing)}")


def _drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    finite = np.ones(len(df), dtype=bool)
    for col in REQUIRED_INPUT_COLS:
        if col == "source_id":
            continue
        finite &= np.isfinite(df[col].to_numpy(dtype=float))
    n_dropped = (~finite).sum()
    if n_dropped:
        logger.info(
            "kinematics: dropped %d/%d rows with non-finite phase-space inputs",
            int(n_dropped), len(df),
        )
    return df.loc[finite].reset_index(drop=True)


def _build_skycoord(df: pd.DataFrame):  # noqa: ANN202 — astropy SkyCoord
    """Build an astropy SkyCoord from the cleaned DataFrame."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        distance=df["r_med_photogeo"].to_numpy() * u.pc,
        pm_ra_cosdec=df["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=df["pmdec"].to_numpy() * u.mas / u.yr,
        radial_velocity=df["radial_velocity"].to_numpy() * u.km / u.s,
    )


def _resolve_potential(name: str):  # noqa: ANN202 — galpy potential list
    if name == "mcmillan17":
        from galpy.potential import McMillan17

        return McMillan17.McMillan17
    if name == "mwpotential2014":
        from galpy.potential import MWPotential2014

        return MWPotential2014
    raise ValueError(f"unknown potential {name!r}; expected 'mcmillan17' or 'mwpotential2014'")


def _run_galpy(coords, cfg: KinematicsConfig) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Invoke galpy. Returns a dict of per-star arrays keyed by output column."""
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit

    pot = _resolve_potential(cfg.potential)
    orbit = Orbit(
        coords,
        ro=cfg.ro_kpc,
        vo=cfg.vo_kms,
        zo=cfg.zo_pc / 1000.0,  # galpy takes zo in kpc
        solarmotion=list(cfg.solarmotion_kms),
    )
    aA = actionAngleStaeckel(pot=pot, delta=cfg.staeckel_delta)
    jR_nat, Lz_nat, Jz_nat = aA(orbit)

    # Actions in natural units → physical kpc·km/s.
    action_scale = cfg.ro_kpc * cfg.vo_kms
    jR = np.atleast_1d(jR_nat) * action_scale
    lz = np.atleast_1d(Lz_nat) * action_scale
    jz = np.atleast_1d(Jz_nat) * action_scale

    ecc = np.atleast_1d(orbit.e(analytic=True, pot=pot, delta=cfg.staeckel_delta))
    rperi = np.atleast_1d(orbit.rperi(analytic=True, pot=pot, delta=cfg.staeckel_delta))
    rap = np.atleast_1d(orbit.rap(analytic=True, pot=pot, delta=cfg.staeckel_delta))
    zmax = np.atleast_1d(orbit.zmax(analytic=True, pot=pot, delta=cfg.staeckel_delta))

    # Physical-unit outputs — galpy's ``use_physical=True`` returns values in
    # its ``ro``/``vo`` unit system.
    R = np.atleast_1d(orbit.R(use_physical=True))
    z = np.atleast_1d(orbit.z(use_physical=True))
    phi = np.atleast_1d(orbit.phi(use_physical=True))
    vR = np.atleast_1d(orbit.vR(use_physical=True))
    vT = np.atleast_1d(orbit.vT(use_physical=True))
    vz = np.atleast_1d(orbit.vz(use_physical=True))
    E = np.atleast_1d(orbit.E(pot=pot, use_physical=True))

    return {
        "R_galcen_kpc": R.astype(np.float64),
        "z_galcen_kpc": z.astype(np.float64),
        "phi_galcen_rad": phi.astype(np.float64),
        "v_R_kms": vR.astype(np.float64),
        "v_T_kms": vT.astype(np.float64),
        "v_z_kms": vz.astype(np.float64),
        "J_R_kpc_kms": jR.astype(np.float64),
        "L_z_kpc_kms": lz.astype(np.float64),
        "J_z_kpc_kms": jz.astype(np.float64),
        "ecc": ecc.astype(np.float64),
        "r_peri_kpc": rperi.astype(np.float64),
        "r_apo_kpc": rap.astype(np.float64),
        "z_max_kpc": zmax.astype(np.float64),
        "E_kms2": E.astype(np.float64),
    }


def _draw_astrometric_sample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One MC realisation of astrometry → DataFrame in the shape compute_actions expects.

    Samples (ra, dec, parallax, pmra, pmdec) from the 5D Gaia covariance
    and (radial_velocity) independently. Converts parallax → distance via
    ``1000 / parallax`` (§9.5: prior-dependence is second-order for the
    boundary-cluster subsample).
    """
    n = len(df)
    ra_col = df["ra"].to_numpy(dtype=float)
    dec_col = df["dec"].to_numpy(dtype=float)
    plx_col = df["parallax"].to_numpy(dtype=float)
    pmra_col = df["pmra"].to_numpy(dtype=float)
    pmdec_col = df["pmdec"].to_numpy(dtype=float)

    sig = np.zeros((n, 5))
    # Positional errors from the Gaia catalogue are in mas for plx/pm and
    # would require per-star ra_error/dec_error columns for full rigor —
    # the Gaia DR3 catalogue does not publish these for the main source
    # table in the enrichment query, so we treat (ra, dec) as fixed per
    # Recio-Blanco+2023 §3.2. Populate only (plx, pmra, pmdec) variances.
    sig[:, 2] = df["parallax_error"].to_numpy(dtype=float)
    sig[:, 3] = df["pmra_error"].to_numpy(dtype=float)
    sig[:, 4] = df["pmdec_error"].to_numpy(dtype=float)

    # 5×5 correlation matrix per star; (ra, dec) rows/cols are identity.
    # Build a batched sample via uncorrelated draws for (ra, dec) = 0 and
    # correlated draws for the (plx, pmra, pmdec) 3×3 block.
    rho_plx_pmra = df["parallax_pmra_corr"].to_numpy(dtype=float)
    rho_plx_pmdec = df["parallax_pmdec_corr"].to_numpy(dtype=float)
    rho_pmra_pmdec = df["pmra_pmdec_corr"].to_numpy(dtype=float)

    draws = np.empty((n, 3))
    for i in range(n):
        cov3 = np.array(
            [
                [sig[i, 2] ** 2,
                 rho_plx_pmra[i] * sig[i, 2] * sig[i, 3],
                 rho_plx_pmdec[i] * sig[i, 2] * sig[i, 4]],
                [rho_plx_pmra[i] * sig[i, 2] * sig[i, 3],
                 sig[i, 3] ** 2,
                 rho_pmra_pmdec[i] * sig[i, 3] * sig[i, 4]],
                [rho_plx_pmdec[i] * sig[i, 2] * sig[i, 4],
                 rho_pmra_pmdec[i] * sig[i, 3] * sig[i, 4],
                 sig[i, 4] ** 2],
            ]
        )
        draws[i] = rng.multivariate_normal([0.0, 0.0, 0.0], cov3)

    plx_s = plx_col + draws[:, 0]
    # Guard against negative parallaxes in the draw (undefined 1/ϖ).
    plx_s = np.where(plx_s > 0, plx_s, np.nan)
    dist_pc_s = 1000.0 / plx_s

    rv_col = df["radial_velocity"].to_numpy(dtype=float)
    rv_err_col = df["radial_velocity_error"].to_numpy(dtype=float)
    rv_s = rv_col + rng.normal(0.0, rv_err_col)

    out = pd.DataFrame(
        {
            "source_id": df["source_id"].to_numpy(),
            "ra": ra_col,
            "dec": dec_col,
            "r_med_photogeo": dist_pc_s,
            "pmra": pmra_col + draws[:, 1],
            "pmdec": pmdec_col + draws[:, 2],
            "radial_velocity": rv_s,
        }
    )
    return out


__all__ = [
    "OUTPUT_COLS",
    "REQUIRED_INPUT_COLS",
    "R_0_KPC",
    "SOLAR_MOTION_KMS",
    "STAECKEL_DELTA",
    "V_0_KMS",
    "Z_0_PC",
    "KinematicsConfig",
    "compute_actions",
    "compute_actions_mc",
]
