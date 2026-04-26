"""Ensemble inter-member correlation diagnostics.

The hostile-referee committee question 12 (`hostile_referee_committee.md`) and
the overfitting-mitigation review (`overfitting_mitigation.md` CRITICAL #3)
both demand an empirical check that ensemble members are diverse enough to
support meaningful epistemic-uncertainty estimates. If pairwise Pearson
correlation between member predictions is too high (median ρ > 0.95), the
ensemble adds little beyond a single overfit model, and the released
``epistemic_sigma`` underestimates true model uncertainty.

This module provides:

- :func:`pairwise_member_correlation` — per-label pairwise Pearson on
  validation predictions.
- :func:`ensemble_diversity_report` — high-level summary with median ρ,
  recommended inflation factor, and a per-label table.
- :func:`apply_epistemic_inflation` — applies sqrt(1 + ρ_median) inflation
  to a per-element ``epistemic_sigma`` Series.

The metrics-diagnostics review (`metrics_diagnostics.md`) treats this as
one of the eight CRITICAL P0 missing diagnostics. With this module in
place, the methods paper can include a "Figure 7" showing the inter-
member correlation heatmap and quote ``ρ_median = X`` per label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ABUNDANCE_ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
_DEFAULT_THRESHOLD = 0.95


@dataclass
class EnsembleDiversityReport:
    """Summary of ensemble inter-member diversity per element.

    Attributes
    ----------
    per_element : dict[str, float]
        Median pairwise Pearson ρ per element across ensemble members.
    inflation_factor : dict[str, float]
        Suggested multiplicative inflation for ``sigma_epistemic`` per
        element: ``sqrt(1 + ρ_median)`` when ``ρ_median > threshold``,
        else 1.0. The rationale: high inter-member correlation means the
        ensemble's empirical disagreement understates true model
        uncertainty by approximately ``1 / (1 - ρ²)``; the inflation is
        a conservative correction.
    threshold : float
        The ρ_median above which inflation is applied.
    fail_elements : list[str]
        Elements whose ρ_median exceeds the threshold; the methods paper
        must explicitly document these.
    """

    per_element: dict[str, float]
    inflation_factor: dict[str, float]
    threshold: float
    fail_elements: list[str]

    def to_dict(self) -> dict[str, dict[str, float] | float | list[str]]:
        return {
            "per_element_rho_median": self.per_element,
            "inflation_factor": self.inflation_factor,
            "threshold": self.threshold,
            "fail_elements": self.fail_elements,
        }


def pairwise_member_correlation(
    predictions_per_member: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Pearson ρ between ensemble members on a 1-D label.

    Parameters
    ----------
    predictions_per_member : (M, B) array
        ``M`` ensemble members, ``B`` validation rows. Single-label.

    Returns
    -------
    (M, M) symmetric correlation matrix with 1.0 on the diagonal.

    Notes
    -----
    Uses ``np.corrcoef`` after dropping rows where any member returned
    NaN. If fewer than 2 members or fewer than 2 finite rows remain,
    raises ValueError.
    """
    if predictions_per_member.ndim != 2:
        raise ValueError(
            f"predictions_per_member must be 2D (M, B); got {predictions_per_member.shape}",
        )
    M = predictions_per_member.shape[0]
    if M < 2:
        raise ValueError(f"need ≥2 ensemble members; got {M}")

    finite_mask = np.isfinite(predictions_per_member).all(axis=0)
    if finite_mask.sum() < 2:
        raise ValueError(
            f"need ≥2 finite validation rows for correlation; got {int(finite_mask.sum())}",
        )

    P = predictions_per_member[:, finite_mask]
    return np.corrcoef(P)


def _median_offdiag(corr: np.ndarray) -> float:
    """Median of the strict upper-triangle of a square correlation matrix."""
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.median(corr[iu]))


def ensemble_diversity_report(
    member_predictions: dict[str, np.ndarray],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> EnsembleDiversityReport:
    """Compute the diversity report across all elements.

    Parameters
    ----------
    member_predictions : dict[str, (M, B) array]
        Maps element name to per-member predictions on the validation set.
        All arrays must share the same M and B.
    threshold : float
        ρ_median above which to apply inflation. Default 0.95
        (overfitting_mitigation review's recommended boundary).

    Returns
    -------
    EnsembleDiversityReport
        Per-element ρ_median, inflation factor, fail list.
    """
    per_elem: dict[str, float] = {}
    inflation: dict[str, float] = {}
    fail: list[str] = []

    for elem in _ABUNDANCE_ELEMENTS:
        preds = member_predictions.get(elem)
        if preds is None:
            continue
        try:
            corr = pairwise_member_correlation(preds)
        except ValueError as e:
            logger.warning("skipping element %s: %s", elem, e)
            continue
        rho = _median_offdiag(corr)
        per_elem[elem] = rho
        if rho > threshold:
            fail.append(elem)
            # Inflation: epistemic σ is empirical std across members; under
            # high correlation it underestimates true model uncertainty by
            # ~ 1 / sqrt(1 - ρ²). We use the more conservative sqrt(1+ρ).
            inflation[elem] = float(np.sqrt(1.0 + rho))
        else:
            inflation[elem] = 1.0

    return EnsembleDiversityReport(
        per_element=per_elem,
        inflation_factor=inflation,
        threshold=float(threshold),
        fail_elements=fail,
    )


def apply_epistemic_inflation(
    epistemic_sigma: pd.Series,
    inflation: float,
) -> pd.Series:
    """Apply a multiplicative inflation factor to a per-row epistemic sigma.

    Used after :func:`ensemble_diversity_report` if a particular element's
    ρ_median exceeds the threshold. The methods paper should document the
    inflation factor and the threshold explicitly.
    """
    if inflation <= 0:
        raise ValueError(f"inflation must be positive; got {inflation}")
    return epistemic_sigma * inflation


__all__ = [
    "EnsembleDiversityReport",
    "apply_epistemic_inflation",
    "ensemble_diversity_report",
    "pairwise_member_correlation",
]
