"""Out-of-distribution rejection for the Pipeline-1 inference release.

Two per-star flags, per DESIGN §OOD rejection at inference:

- ``ood_flag_mahalanobis``: Mahalanobis distance from the training XP
  feature distribution (the 108-D ``(bp_coef_norm_1..54, rp_coef_norm_1..54)``
  block). Fitted at training time; stars at inference above the training-set
  99th-percentile distance are flagged.
- ``ood_flag_ensemble``: ensemble-disagreement epistemic proxy. Per star,
  compute ``epistemic_σ / total_σ`` averaged across labels; flag above a
  configurable threshold (default 0.5).

Either flag alone is a yellow light; both firing is red (do not trust per-
star). The flags are metadata — inference runs unaltered — but D-Cat-b
release documentation notes that flagged-star predictions are valid for
population-level statistics only.

Motivated by research_brief §3.1 Ye+2024 blue-flux instability concentrating
on metal-poor / hot / high-Av stars under-represented in APOGEE, where the
network could produce confidently-wrong predictions without explicit
rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS: float = 1e-12


@dataclass
class MahalanobisOODBundle:
    """Training-set XP feature distribution — frozen at fit time.

    ``feature_precision`` is the inverse of the empirical covariance
    (regularised by adding ``regularization`` to the diagonal before
    inversion). ``threshold`` is the training-set quantile of the
    Mahalanobis distance at ``p_threshold`` (default 99 %).
    """

    feature_mean: np.ndarray
    feature_precision: np.ndarray
    threshold: float
    p_threshold: float
    n_training: int
    regularization: float

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.astype(np.float32),
            "feature_precision": self.feature_precision.astype(np.float32),
            "threshold": float(self.threshold),
            "p_threshold": float(self.p_threshold),
            "n_training": int(self.n_training),
            "regularization": float(self.regularization),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "MahalanobisOODBundle":
        return cls(
            feature_mean=np.asarray(blob["feature_mean"], dtype=np.float64),
            feature_precision=np.asarray(blob["feature_precision"], dtype=np.float64),
            threshold=float(blob["threshold"]),
            p_threshold=float(blob["p_threshold"]),
            n_training=int(blob["n_training"]),
            regularization=float(blob["regularization"]),
        )


def fit_mahalanobis_ood(
    features: np.ndarray,
    *,
    p_threshold: float = 0.99,
    regularization: float = 1e-6,
) -> MahalanobisOODBundle:
    """Fit mean, precision, and p-quantile threshold on a training-set feature matrix.

    Parameters
    ----------
    features : (N, F) array
        Training-set XP feature block (typically 108-D
        ``(bp_coef_norm_1..54, rp_coef_norm_1..54)``). Rows with any
        non-finite entry are dropped before fitting.
    p_threshold : float
        Quantile of the training-set Mahalanobis distance used as the
        flag threshold. Stars above this quantile at inference are
        OOD.
    regularization : float
        Added to the diagonal of the empirical covariance before
        inversion — stabilises the precision matrix against near-
        singular directions in high dimensions. The XP coefficient
        space is ~exponentially decaying in variance; without this,
        the last ~15 RP modes routinely produce catastrophic
        precision-matrix blow-up on the 108-D block.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (N, F), got {features.shape}")
    finite_mask = np.isfinite(features).all(axis=1)
    X = features[finite_mask].astype(np.float64)
    if X.shape[0] < X.shape[1] + 2:
        raise ValueError(
            f"need at least F+2 = {X.shape[1] + 2} finite rows to fit "
            f"covariance, got {X.shape[0]}",
        )

    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    cov.flat[:: cov.shape[0] + 1] += regularization
    precision = np.linalg.inv(cov)

    # Training-set Mahalanobis distances → quantile threshold.
    sq_dists = np.einsum("bi,ij,bj->b", Xc, precision, Xc)
    sq_dists = np.clip(sq_dists, 0.0, None)
    dists = np.sqrt(sq_dists)
    threshold = float(np.quantile(dists, p_threshold))

    return MahalanobisOODBundle(
        feature_mean=mu,
        feature_precision=precision,
        threshold=threshold,
        p_threshold=float(p_threshold),
        n_training=int(X.shape[0]),
        regularization=float(regularization),
    )


def score_mahalanobis_ood(
    features: np.ndarray,
    bundle: MahalanobisOODBundle,
) -> np.ndarray:
    """Per-star Mahalanobis distance to the training feature mean.

    Rows with any non-finite entry return ``np.nan`` — the caller can
    decide whether a non-finite-feature row should be flagged as OOD
    by default (``flag_mahalanobis_ood`` treats NaN as flagged).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (B, F), got {features.shape}")
    if features.shape[1] != bundle.feature_dim:
        raise ValueError(
            f"feature dim {features.shape[1]} != training dim {bundle.feature_dim}",
        )
    row_ok = np.isfinite(features).all(axis=1)
    out = np.full(features.shape[0], np.nan, dtype=np.float64)
    if row_ok.any():
        Xc = features[row_ok].astype(np.float64) - bundle.feature_mean
        sq_dists = np.einsum("bi,ij,bj->b", Xc, bundle.feature_precision, Xc)
        sq_dists = np.clip(sq_dists, 0.0, None)
        out[row_ok] = np.sqrt(sq_dists)
    return out


def flag_mahalanobis_ood(
    features: np.ndarray,
    bundle: MahalanobisOODBundle,
) -> np.ndarray:
    """Per-star boolean: ``True`` = OOD (above training ``p_threshold``).

    Non-finite-feature rows flag as OOD — they cannot be scored and are
    not safe to treat as in-distribution.
    """
    dists = score_mahalanobis_ood(features, bundle)
    return ~(dists <= bundle.threshold)  # NaN → True via negation of NaN≤x = False


def ensemble_disagreement_ratio(
    mu_per_member: np.ndarray,
    sigma_diag_per_member: np.ndarray,
) -> np.ndarray:
    """Per-star epistemic/total σ ratio, averaged across labels.

    ``mu_per_member`` and ``sigma_diag_per_member`` both have shape
    ``(M, B, n)``: ``M`` ensemble members, ``B`` stars, ``n`` labels.

    For each (star, label):

    - ``epistemic² = Var_across_members(μ)``
    - ``aleatoric² = Mean_across_members(σ²)``
    - ``total² = epistemic² + aleatoric²``

    The returned ratio is ``mean_across_labels( epistemic / total )``.
    A star with ratio near 1 is dominated by ensemble disagreement
    (model-uncertainty-limited); near 0 is aleatoric-limited (good).

    Returns
    -------
    (B,) float64 array, in [0, 1].
    """
    if mu_per_member.ndim != 3:
        raise ValueError(f"mu_per_member must be (M, B, n), got {mu_per_member.shape}")
    if mu_per_member.shape != sigma_diag_per_member.shape:
        raise ValueError(
            f"shape mismatch: mu {mu_per_member.shape} vs sigma "
            f"{sigma_diag_per_member.shape}",
        )
    if mu_per_member.shape[0] < 2:
        raise ValueError(
            f"ensemble disagreement requires ≥2 members, got {mu_per_member.shape[0]}",
        )

    epistemic_var = mu_per_member.var(axis=0, ddof=0)  # (B, n)
    aleatoric_var = (sigma_diag_per_member ** 2).mean(axis=0)  # (B, n)
    total_var = epistemic_var + aleatoric_var
    total_sigma = np.sqrt(np.clip(total_var, _EPS, None))
    epistemic_sigma = np.sqrt(np.clip(epistemic_var, 0.0, None))
    ratio_per_label = epistemic_sigma / total_sigma  # (B, n)
    return ratio_per_label.mean(axis=1).astype(np.float64)


def flag_ensemble_ood(
    mu_per_member: np.ndarray,
    sigma_diag_per_member: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Per-star boolean: ``True`` = epistemic/total ratio above ``threshold``."""
    ratio = ensemble_disagreement_ratio(mu_per_member, sigma_diag_per_member)
    return ratio > threshold


def combined_ood_status(
    flag_mahalanobis: np.ndarray,
    flag_ensemble: np.ndarray,
) -> np.ndarray:
    """Map (mahal, ensemble) pair to a 3-level status code.

    ``0`` = neither firing (green — trust per-star).
    ``1`` = exactly one firing (yellow — caution, population-level ok).
    ``2`` = both firing (red — population-level only).
    """
    mahal = np.asarray(flag_mahalanobis, dtype=bool)
    ens = np.asarray(flag_ensemble, dtype=bool)
    return (mahal.astype(np.int8) + ens.astype(np.int8)).astype(np.int8)


__all__ = [
    "MahalanobisOODBundle",
    "combined_ood_status",
    "ensemble_disagreement_ratio",
    "fit_mahalanobis_ood",
    "flag_ensemble_ood",
    "flag_mahalanobis_ood",
    "score_mahalanobis_ood",
]
