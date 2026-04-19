"""Tests for utils.coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.utils.coordinates import (
    equatorial_to_galactic,
    galactic_velocities_to_cylindrical,
)


def test_equatorial_to_galactic_shapes() -> None:
    n = 4
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, n)
    dec = rng.uniform(-90, 90, n)
    parallax = np.full(n, 1.0)  # 1 kpc
    pmra = rng.standard_normal(n)
    pmdec = rng.standard_normal(n)
    rv = rng.standard_normal(n) * 10
    out = equatorial_to_galactic(ra, dec, parallax, pmra, pmdec, rv)
    for key in ("l_deg", "b_deg", "d_kpc", "U_kms", "V_kms", "W_kms"):
        assert out[key].shape == (n,)
    assert np.allclose(out["d_kpc"], 1.0)


def test_equatorial_to_galactic_uses_explicit_distance() -> None:
    ra = np.array([0.0])
    dec = np.array([0.0])
    parallax = np.array([5.0])  # 0.2 kpc implied, but overridden below.
    d = np.array([2.5])
    out = equatorial_to_galactic(
        ra, dec, parallax, np.zeros(1), np.zeros(1), np.zeros(1),
        distance_kpc=d,
    )
    assert out["d_kpc"] == pytest.approx(2.5)


def test_equatorial_to_galactic_rejects_nonpositive_parallax() -> None:
    with pytest.raises(ValueError, match="parallax"):
        equatorial_to_galactic(
            np.array([0.0]), np.array([0.0]),
            np.array([-1.0]),  # invalid
            np.zeros(1), np.zeros(1), np.zeros(1),
        )


def test_galactic_velocities_to_cylindrical_zero_phi_gives_UVW() -> None:  # noqa: N802
    # At phi=0 (Sun direction), v_R = U, v_phi = V, v_z = W.
    U, V, W = np.array([10.0]), np.array([20.0]), np.array([5.0])
    R = np.array([8.0])
    phi = np.array([0.0])
    out = galactic_velocities_to_cylindrical(U, V, W, R, phi)
    assert out["v_R_kms"] == pytest.approx(10.0)
    assert out["v_phi_kms"] == pytest.approx(20.0)
    assert out["v_z_kms"] == pytest.approx(5.0)


def test_galactic_velocities_to_cylindrical_pi_over_2() -> None:
    # phi = π/2 rotates: v_R = V, v_phi = -U.
    U, V, W = np.array([10.0]), np.array([20.0]), np.array([5.0])
    R = np.array([8.0])
    phi = np.array([np.pi / 2])
    out = galactic_velocities_to_cylindrical(U, V, W, R, phi)
    assert out["v_R_kms"] == pytest.approx(20.0)
    assert out["v_phi_kms"] == pytest.approx(-10.0)


def test_galactic_velocities_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        galactic_velocities_to_cylindrical(
            np.zeros(3), np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(4),
        )
