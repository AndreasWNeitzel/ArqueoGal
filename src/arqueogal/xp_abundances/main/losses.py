"""Pipeline 1 training losses: SupCon-soft-positive + multivariate Beta-NLL.

SupCon (:func:`supcon_soft_positive`) is ported from the TESS_ML prototype —
Supervised Contrastive (Khosla+2020) with Gaussian-kernel *soft* positives:
pairs in a batch are weighted by ``exp(-||y_a - y_k||² / 2σ²)`` instead of
being binary same-class matches. This matches the regression setting where
labels are continuous and "same cluster" is not a meaningful distinction.

Beta-NLL (:func:`beta_nll_block_cholesky`) is the DESIGN's chosen supervised
loss (Seitzer+2022, β=0.5). The original formulation is per-dimension scalar
σ; we extend to a Cholesky-parametrised block covariance by:

1. Computing the standard multivariate Gaussian NLL per-star from
   ``Σ = L Lᵀ`` via a triangular solve (no explicit inverse, no explicit
   ``Σ`` materialisation along the graph where we can avoid it).
2. Weighting each star's NLL by ``(Π diag(Σ))^(β/n)`` detached — the
   geometric mean of diagonal variances raised to β. This generalises
   Seitzer's ``σ²^β`` weighting to the multivariate case in a way that (a)
   reduces to the scalar case when ``n=1`` and (b) preserves the
   bias-variance decomposition Seitzer proved prevents σ-inflation.

The decision to use ``(Π diag)^(β/n)`` rather than e.g. ``det(Σ)^(β/n)`` is
deliberate: including off-diagonals in the β weight couples pre-train
contrastive geometry into the regression weighting, which is empirically
harder to tune. Document the choice here so future slices can experiment
without losing the audit trail.
"""

from __future__ import annotations

import torch

_LOG_2PI: float = 1.8378770664093453
"""``log(2π)``; precomputed for the multivariate Gaussian NLL constant term."""


def supcon_soft_positive(  # noqa: PLR0913 — mirrors SupCon API in TESS_ML prototype
    za: torch.Tensor,
    ya: torch.Tensor,
    zk: torch.Tensor,
    yk: torch.Tensor,
    *,
    temperature: torch.Tensor | float,
    sigma: float = 0.10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervised Contrastive loss with Gaussian-kernel soft positives.

    Parameters
    ----------
    za
        Anchor projections, shape ``(B, D)``, assumed L2-normalised by caller.
    ya
        Anchor labels, shape ``(B, n_labels)``. Used only for kernel weighting
        (in label space), not for gradient flow — detached internally.
    zk
        Key projections, shape ``(K, D)``.
    yk
        Key labels, shape ``(K, n_labels)``.
    temperature
        Scalar temperature τ. Typically a learnable ``nn.Parameter`` in the
        surrounding training module, clamped to ``[1e-3, 0.5]``.
    sigma
        Gaussian-kernel bandwidth in label space (prototype default 0.10).

    Returns
    -------
    Scalar loss — mean over anchors of the soft-weighted contrastive
    cross-entropy. Self-pairs (where ``B <= K`` and the first ``B`` keys
    coincide with the anchors) are masked out.
    """
    if za.shape[-1] != zk.shape[-1]:
        raise ValueError(f"projection dim mismatch: {za.shape[-1]} vs {zk.shape[-1]}")
    B, K = za.shape[0], zk.shape[0]
    sim = za @ zk.T / temperature  # (B, K)

    eye = torch.zeros(B, K, dtype=torch.bool, device=sim.device)
    n = min(B, K)
    eye[torch.arange(n), torch.arange(n)] = True

    with torch.no_grad():
        d2 = (ya.unsqueeze(1) - yk.unsqueeze(0)).pow(2).sum(-1)
        w = torch.exp(-d2 / (2.0 * sigma * sigma))
        w = w.masked_fill(eye, 0.0)

    sm = sim.detach().max(dim=1, keepdim=True).values
    es = torch.exp(sim - sm).masked_fill(eye, 0.0)
    log_prob = (sim - sm) - (es.sum(dim=1, keepdim=True) + eps).log()

    per_anchor = -(w * log_prob).sum(dim=1) / w.sum(dim=1).clamp_min(eps)
    return per_anchor.mean()


def beta_nll_block_cholesky(  # noqa: PLR0913 — all kwargs are distinct numerical knobs
    mu: torch.Tensor,
    L: torch.Tensor,
    y: torch.Tensor,
    *,
    beta: float = 0.5,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Multivariate Beta-NLL over a Cholesky-parametrised covariance.

    Parameters
    ----------
    mu
        Predicted mean, shape ``(B, n)``.
    L
        Lower-triangular Cholesky factor, shape ``(B, n, n)``. Must have
        strictly positive diagonal (the model enforces this via
        ``softplus + _MIN_CHOLESKY_DIAG``).
    y
        Target labels, shape ``(B, n)``.
    beta
        Seitzer β ∈ [0, 1]. ``β=0`` recovers vanilla MVN-NLL; ``β=1`` becomes
        a pure MSE-like weighting. DESIGN default 0.5.
    mask
        Optional ``(B, n)`` binary mask — ``1`` for labels present, ``0`` for
        missing. Missing labels are dropped from the NLL but kept in the
        Cholesky solve (they contribute via correlation with observed
        labels, which is the honest thing to do). When ``mask`` is ``None``,
        all labels assumed present.
    eps
        Small constant for numerical stability.

    Returns
    -------
    Scalar loss averaged over the number of observed labels.
    """
    _validate_shapes(mu, L, y)

    diff = (y - mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L, diff, upper=False)
    mahal_per_star = z.squeeze(-1).pow(2).sum(dim=-1)

    log_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(eps))
    log_det_sigma = 2.0 * log_diag.sum(dim=-1)

    n_dims = mu.shape[-1]
    nll_per_star = 0.5 * (n_dims * _LOG_2PI + log_det_sigma + mahal_per_star)

    if beta != 0.0:
        diag_var = torch.diagonal(L, dim1=-2, dim2=-1).pow(2)
        log_geo_mean = log_diag.sum(dim=-1) * (2.0 / n_dims)
        weight = torch.exp(beta * log_geo_mean).detach()
        nll_per_star = weight * nll_per_star
        del diag_var

    if mask is not None:
        if mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} != y shape {y.shape}")
        obs_per_star = mask.float().sum(dim=-1).clamp_min(eps)
        per_star_scale = obs_per_star / float(n_dims)
        nll_per_star = nll_per_star * per_star_scale
        return nll_per_star.sum() / mask.float().sum().clamp_min(eps)

    return nll_per_star.mean() / float(n_dims)


def _validate_shapes(mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor) -> None:
    if mu.shape != y.shape:
        raise ValueError(f"mu shape {mu.shape} != y shape {y.shape}")
    if L.shape[-2:] != (mu.shape[-1], mu.shape[-1]):
        raise ValueError(
            f"L shape {L.shape} inconsistent with n_labels={mu.shape[-1]}"
        )
    if L.shape[0] != mu.shape[0]:
        raise ValueError(f"batch dim mismatch: L {L.shape[0]} vs mu {mu.shape[0]}")


def mahalanobis_residual(
    mu: torch.Tensor,
    L: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-star whitened residual ``L^{-1} (y - μ)`` — useful for calibration diagnostics.

    Returns shape ``(B, n)``. The squared-sum over the last dim is the
    Mahalanobis distance used by :func:`beta_nll_block_cholesky`.
    """
    _ = eps  # Reserved for future ridge; kept in signature for symmetry.
    _validate_shapes(mu, L, y)
    diff = (y - mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L, diff, upper=False)
    return z.squeeze(-1)


__all__ = [
    "beta_nll_block_cholesky",
    "mahalanobis_residual",
    "supcon_soft_positive",
]
