"""Galactic coordinate + velocity transformations — utils/DESIGN.md.

Thin wrappers around astropy's :class:`~astropy.coordinates.SkyCoord`
that vectorise across 1-D arrays of Gaia measurements. The heavier
orbital-action computation lives in :mod:`arqueogal.data.kinematics`
(galpy + McMillan+2017 potential) — this module is for pure coordinate
transforms.

All inputs are plain ``float`` ndarrays in the units named below; we
attach ``astropy.units`` internally, propagate through the frame
transforms, and strip units only at the output boundary.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "equatorial_to_galactic",
    "galactic_velocities_to_cylindrical",
]


def equatorial_to_galactic(  # noqa: PLR0913 — Gaia measurement signature
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    parallax_mas: np.ndarray,
    pmra_masyr: np.ndarray,
    pmdec_masyr: np.ndarray,
    rv_kms: np.ndarray,
    *,
    distance_kpc: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """ICRS → Galactic, returning (l, b, d, U, V, W) as ndarrays.

    Parameters
    ----------
    ra_deg, dec_deg
        ICRS right ascension / declination in degrees.
    parallax_mas
        Parallax in mas. Only used if ``distance_kpc`` is None; in that
        case distance is ``1/parallax`` (kpc) — **not** the Bayesian
        Bailer-Jones estimate. Pass ``distance_kpc`` explicitly if you
        already have Bailer-Jones distances (required for science use).
    pmra_masyr, pmdec_masyr
        Proper motions in mas/yr. ``pmra`` is ``pmra*cos(dec)``.
    rv_kms
        Radial velocity in km/s.
    distance_kpc
        Optional pre-computed distance. Overrides ``1/parallax``.

    Returns
    -------
    dict
        Keys: ``l_deg``, ``b_deg``, ``d_kpc``, ``U_kms``, ``V_kms``,
        ``W_kms``. UVW here is the **heliocentric** Galactic-Cartesian
        velocity with U toward the Galactic centre, V in the direction of
        rotation, W toward the NGP. No solar-motion correction applied —
        callers who want LSR velocities add it downstream.
    """
    from astropy import units as u
    from astropy.coordinates import ICRS, Galactic

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    parallax_mas = np.asarray(parallax_mas, dtype=np.float64)
    pmra = np.asarray(pmra_masyr, dtype=np.float64)
    pmdec = np.asarray(pmdec_masyr, dtype=np.float64)
    rv = np.asarray(rv_kms, dtype=np.float64)

    if distance_kpc is None:
        if np.any(parallax_mas <= 0):
            raise ValueError(
                "equatorial_to_galactic: non-positive parallax without "
                "explicit distance_kpc — pass Bailer-Jones distances instead",
            )
        distance_kpc = 1.0 / parallax_mas
    else:
        distance_kpc = np.asarray(distance_kpc, dtype=np.float64)

    icrs = ICRS(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        distance=distance_kpc * u.kpc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=rv * u.km / u.s,
    )
    gal = icrs.transform_to(Galactic())

    # Heliocentric Cartesian velocities (U, V, W) in km/s.
    v_cart = gal.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
    # d_xyz shape (3, N). Rows are (x, y, z) = (toward GC, rot, NGP) in
    # Galactic Cartesian — matches UVW convention.
    return {
        "l_deg": np.asarray(gal.l.to(u.deg).value, dtype=np.float64),
        "b_deg": np.asarray(gal.b.to(u.deg).value, dtype=np.float64),
        "d_kpc": distance_kpc,
        "U_kms": np.asarray(v_cart[0], dtype=np.float64),
        "V_kms": np.asarray(v_cart[1], dtype=np.float64),
        "W_kms": np.asarray(v_cart[2], dtype=np.float64),
    }


def galactic_velocities_to_cylindrical(  # noqa: PLR0913 — (U,V,W,R,phi) vector signature
    U_kms: np.ndarray,
    V_kms: np.ndarray,
    W_kms: np.ndarray,
    R_kpc: np.ndarray,
    phi_rad: np.ndarray,
) -> dict[str, np.ndarray]:
    """Galactic Cartesian (U,V,W) → Galactocentric cylindrical (v_R, v_phi, v_z).

    Standard rotation at Galactocentric azimuth ``phi`` (measured from
    the Sun → Galactic centre line, increasing in the direction of
    rotation):

        v_R   =  U cos(phi) + V sin(phi)
        v_phi = -U sin(phi) + V cos(phi)
        v_z   =  W

    This assumes the UVW have already been expressed in the
    *Galactocentric* frame (i.e. the Sun's peculiar motion + LSR
    circular motion have been added). If only heliocentric UVW is
    available, callers should add the solar motion first.
    """
    U_kms = np.asarray(U_kms, dtype=np.float64)
    V_kms = np.asarray(V_kms, dtype=np.float64)
    W_kms = np.asarray(W_kms, dtype=np.float64)
    R_kpc = np.asarray(R_kpc, dtype=np.float64)
    phi_rad = np.asarray(phi_rad, dtype=np.float64)

    if not (U_kms.shape == V_kms.shape == W_kms.shape == phi_rad.shape):
        raise ValueError(
            f"U/V/W/phi shape mismatch: {U_kms.shape}/{V_kms.shape}/{W_kms.shape}/{phi_rad.shape}",
        )

    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    v_R = U_kms * cos_phi + V_kms * sin_phi
    v_phi = -U_kms * sin_phi + V_kms * cos_phi
    v_z = W_kms
    return {
        "v_R_kms": v_R,
        "v_phi_kms": v_phi,
        "v_z_kms": v_z,
        "R_kpc": R_kpc,
    }
