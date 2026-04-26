"""Kinematic out-of-distribution detector for Pipeline 1 release.

The Phase A2 catalog schema introduced ``kin_ood_flag`` as a placeholder
column (always False). This module is the first real implementation: it
fits a Gaussian model on the disc-kinematics distribution of the Stream 1
training set in 3D Galactocentric velocity space ``(v_R, v_phi, v_z)`` and
flags inference-time stars whose Mahalanobis distance to the disc mean
exceeds a calibrated threshold.

Why this exists
---------------

The fleet's strongest convergent finding is that ``[α/M]`` and ``[Mg/H]``
predictions carry zero conditional MI with the XP block once parallax,
photometry, extinction, and position are conditioned on (META_META §3 and
§14.3, ``physical_causality.md``, ``xp_information_content.md``,
``statistics_methodology.md``, ``outlier_flagging.md``). The model learns
those labels from the disc-kinematics population prior implicit in the
APOGEE-Gaia training set, not from spectroscopy. For halo, accreted-debris,
or counter-rotating-disc stars where the disc prior breaks down, those
predictions are unreliable but currently carry the same calibrated
uncertainty as in-distribution disc stars.

A kinematic-OOD flag is the operational lever that translates the
information-content audit into a per-star decision rule. With this flag
populated, ``release.assign_per_element_release_tier`` demotes
aux-assisted elements (``alpha_m``, ``mg_h``) to Tier 2 for any star that
is kinematically OOD, while leaving spectrum-dominant elements (Teff,
logg, [M/H]) at Tier 1.

Design choices
--------------

- **Single-Gaussian, not GMM.** A Gaussian-mixture disc/halo model would be
  more flexible but the threshold semantics are clearer with a single
  Gaussian fit on the disc-only training subset. The boundary between
  thin disc, thick disc, and accreted halo is fuzzy; for OOD purposes the
  disc-only mean and covariance plus a 99th-percentile threshold give a
  conservative cut. Future work can extend to GMM with per-component
  flags.
- **3D Galactocentric velocity, not action-angle.** Action-angle (J_R,
  L_z, J_z) is the textbook coordinate for galactic dynamics but requires
  a Galactic potential model (galpy / AGAMA) and is computationally
  heavier. The Mahalanobis-on-velocity choice is operationally cheap and
  captures the disc/halo distinction well enough for tier demotion. A
  follow-up module can swap to action-angle if the methods paper requires
  it.
- **Threshold from training-set quantile, not theoretical χ²₃.** The
  empirical 99th percentile is robust to non-Gaussian disc-velocity
  distributions (notably the Hercules stream, the Local Standard of Rest
  asymmetric drift) without requiring a perfectly Gaussian model.
- **Galactocentric frame transform** is left to a sibling utility
  (typically in ``arqueogal.utils.coordinates`` or ``data/enrich_kinematics.py``).
  This module assumes the caller has already produced ``(v_R, v_phi, v_z)``
  in the chosen Galactocentric system. The caller passes the velocity
  matrix; this module fits and flags.

The bundle ``KinematicOODBundle`` is serialised alongside the model
checkpoint and reloaded at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_KIN_REGULARIZATION_DEFAULT: float = 1.0  # km^2/s^2; small relative to disc σ ~ 30 km/s.


@dataclass
class KinematicOODBundle:
    """Frozen disc-kinematics distribution for OOD scoring.

    Attributes
    ----------
    velocity_mean : (3,) float64 array
        Mean of the disc-only training subset in (v_R, v_phi, v_z).
        Typical disc values: v_R ≈ 0, v_phi ≈ +220 km/s (Galactic
        rotation curve at solar circle), v_z ≈ 0.
    velocity_precision : (3, 3) float64 array
        Inverse of the disc-velocity covariance plus regularisation.
    threshold : float
        99th-percentile Mahalanobis distance of the disc training subset.
    p_threshold : float
        Quantile used at fit time (default 0.99).
    n_training : int
        Number of disc-only training stars used to fit the bundle.
    regularization : float
        Scalar added to the diagonal of the covariance before inversion.
    coordinate_system : str
        Free-form label identifying the Galactocentric frame convention
        (e.g. ``"galpy_default_v_LSR_220"`` or ``"AGAMA_StaeckelFudge"``).
        Stored for provenance; not used by the flagger.
    """

    velocity_mean: np.ndarray
    velocity_precision: np.ndarray
    threshold: float
    p_threshold: float
    n_training: int
    regularization: float
    coordinate_system: str

    @property
    def feature_dim(self) -> int:
        return int(self.velocity_mean.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_mean": self.velocity_mean.astype(np.float32),
            "velocity_precision": self.velocity_precision.astype(np.float32),
            "threshold": float(self.threshold),
            "p_threshold": float(self.p_threshold),
            "n_training": int(self.n_training),
            "regularization": float(self.regularization),
            "coordinate_system": str(self.coordinate_system),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> KinematicOODBundle:
        return cls(
            velocity_mean=np.asarray(blob["velocity_mean"], dtype=np.float64),
            velocity_precision=np.asarray(blob["velocity_precision"], dtype=np.float64),
            threshold=float(blob["threshold"]),
            p_threshold=float(blob["p_threshold"]),
            n_training=int(blob["n_training"]),
            regularization=float(blob["regularization"]),
            coordinate_system=str(blob.get("coordinate_system", "unspecified")),
        )


def fit_kinematic_ood(
    velocities: np.ndarray,
    *,
    p_threshold: float = 0.99,
    regularization: float = _KIN_REGULARIZATION_DEFAULT,
    coordinate_system: str = "galpy_default_v_LSR_220",
) -> KinematicOODBundle:
    """Fit the disc-kinematics OOD bundle.

    Parameters
    ----------
    velocities : (N, 3) array
        Galactocentric (v_R, v_phi, v_z) for the disc-only subset of the
        Stream 1 training set. The caller is expected to pre-filter to
        disc kinematics (e.g. by APOGEE survey footprint, or by a coarse
        ``|v_z| < 80`` km/s and ``v_phi > 100`` km/s cut). Rows with any
        non-finite entry are dropped before fitting.
    p_threshold : float
        Quantile of the disc Mahalanobis distance used as the OOD
        threshold. Default 0.99 keeps ~1 % of the disc set as "boundary"
        and gates anything tighter as in-distribution.
    regularization : float
        Diagonal jitter (km²/s²) added to the velocity covariance before
        inversion. The disc-velocity dispersion is ~30 km/s in v_R and
        v_z, ~20 km/s in v_phi after subtracting rotation; 1 km²/s² is
        small relative to these but stabilises ill-conditioned cases
        (e.g., bulge-only subsets where v_phi has zero variance).

    Returns
    -------
    KinematicOODBundle
        Frozen bundle ready for serialisation alongside the model
        checkpoint.

    Notes
    -----
    The fit uses an unbiased covariance estimator (``ddof=1``). For
    typical Stream 1 sizes (~150–300 k stars), the empirical covariance
    is well-conditioned without the regulariser; the regulariser exists
    for robustness on small training-set splits or pathological cases.
    """
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError(
            f"velocities must be 2D (N, 3) Galactocentric; got {velocities.shape}",
        )
    finite_mask = np.isfinite(velocities).all(axis=1)
    V = velocities[finite_mask].astype(np.float64)
    if V.shape[0] < 100:
        raise ValueError(
            f"need ≥100 finite training rows to fit kinematic OOD; got {V.shape[0]}",
        )

    mu = V.mean(axis=0)
    Vc = V - mu
    cov = (Vc.T @ Vc) / (V.shape[0] - 1)
    cov.flat[:: cov.shape[0] + 1] += regularization
    precision = np.linalg.inv(cov)

    sq_dists = np.einsum("bi,ij,bj->b", Vc, precision, Vc)
    sq_dists = np.clip(sq_dists, 0.0, None)
    dists = np.sqrt(sq_dists)
    threshold = float(np.quantile(dists, p_threshold))

    return KinematicOODBundle(
        velocity_mean=mu,
        velocity_precision=precision,
        threshold=threshold,
        p_threshold=float(p_threshold),
        n_training=int(V.shape[0]),
        regularization=float(regularization),
        coordinate_system=str(coordinate_system),
    )


def score_kinematic_ood(
    velocities: np.ndarray,
    bundle: KinematicOODBundle,
) -> np.ndarray:
    """Per-star Mahalanobis distance to the disc-kinematics mean.

    Non-finite velocity rows return ``np.nan`` — the caller can decide
    whether unknown-velocity stars (no parallax / no proper motion / no
    Galactocentric transform) should be flagged OOD. ``flag_kinematic_ood``
    treats NaN-distance rows as flagged.
    """
    if velocities.ndim != 2 or velocities.shape[1] != bundle.feature_dim:
        raise ValueError(
            f"velocities must be 2D (B, {bundle.feature_dim}); got {velocities.shape}",
        )
    row_ok = np.isfinite(velocities).all(axis=1)
    out = np.full(velocities.shape[0], np.nan, dtype=np.float64)
    if row_ok.any():
        Vc = velocities[row_ok].astype(np.float64) - bundle.velocity_mean
        sq_dists = np.einsum("bi,ij,bj->b", Vc, bundle.velocity_precision, Vc)
        sq_dists = np.clip(sq_dists, 0.0, None)
        out[row_ok] = np.sqrt(sq_dists)
    return out


def flag_kinematic_ood(
    velocities: np.ndarray,
    bundle: KinematicOODBundle,
) -> np.ndarray:
    """Per-star boolean: True = kinematically OOD relative to the disc training subset.

    Non-finite velocity rows return True (cannot be scored, conservative).
    """
    dists = score_kinematic_ood(velocities, bundle)
    return ~(dists <= bundle.threshold)


def assemble_galactocentric_velocity_safe(
    parallax_mas: np.ndarray,
    pmra_mas_yr: np.ndarray,
    pmdec_mas_yr: np.ndarray,
    radial_velocity_km_s: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> np.ndarray:
    """Compute Galactocentric (v_R, v_phi, v_z) using astropy + galpy defaults.

    A thin convenience that delegates to ``astropy`` and ``galpy.coords``
    using the canonical solar-motion + LSR conventions (``v_LSR = 220
    km/s``, ``z_sun = 25 pc`` from Galactic plane). Stars with non-finite
    inputs return rows of NaN; the OOD scorer treats those as flagged.

    Notes
    -----
    The actual project pipeline computes Galactocentric velocities in
    ``arqueogal.data.enrich_kinematics``; this function is offered here
    so a stand-alone caller can score kinematic OOD without depending on
    the full ingestion stack. If you already have ``(v_R, v_phi, v_z)``
    from the project's enrichment step, prefer that and skip this helper.

    Implementation deferred — the concrete astropy+galpy invocation
    requires importing those packages and is left to the caller's data-
    preparation step. Raise NotImplementedError to surface clearly when
    a caller relies on the convenience helper without supplying upstream
    velocities.
    """
    raise NotImplementedError(
        "assemble_galactocentric_velocity_safe is a placeholder; the project's "
        "data pipeline computes (v_R, v_phi, v_z) in arqueogal.data.enrich_kinematics. "
        "Pass those columns directly to fit_kinematic_ood / flag_kinematic_ood.",
    )


__all__ = [
    "KinematicOODBundle",
    "assemble_galactocentric_velocity_safe",
    "fit_kinematic_ood",
    "flag_kinematic_ood",
    "score_kinematic_ood",
]
