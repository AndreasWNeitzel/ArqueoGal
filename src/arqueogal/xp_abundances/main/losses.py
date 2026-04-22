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
import torch.nn.functional as F

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
        # NaN-safe kernel: pairs where either label row has NaN in any dim get
        # weight 0. Per-element abundances (V, Mg/Fe, …) carry 1-5% NaN after
        # Mészáros+25 correction; propagating NaN into d2 would NaN the whole
        # loss from epoch 0. An anchor whose every key pair is NaN-masked
        # contributes 0 to the mean (w.sum → eps via clamp_min), effectively
        # dropping it from the batch — a sane fallback for rare full-NaN rows.
        nan_a = torch.isnan(ya).any(dim=-1)
        nan_k = torch.isnan(yk).any(dim=-1)
        bad = nan_a.unsqueeze(1) | nan_k.unsqueeze(0)
        ya_f = torch.nan_to_num(ya, nan=0.0)
        yk_f = torch.nan_to_num(yk, nan=0.0)
        d2 = (ya_f.unsqueeze(1) - yk_f.unsqueeze(0)).pow(2).sum(-1)
        w = torch.exp(-d2 / (2.0 * sigma * sigma))
        w = w.masked_fill(eye | bad, 0.0)

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
    sample_weights: torch.Tensor | None = None,
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
    sample_weights
        Optional ``(B,)`` non-negative per-star weight. When supplied, the
        batch NLL is a *weighted* average over stars (numerator weighted by
        ``w_i``; denominator scaled to match). Used by inverse-frequency
        training (#198, v1.1) to up-weight rare-[M/H] stars that would
        otherwise regress to the disc mean. Pass ``None`` or tensor of ones
        for unweighted training (v1 default).
    eps
        Small constant for numerical stability.

    Returns
    -------
    Scalar loss averaged (or weighted-averaged if ``sample_weights``
    supplied) over the number of observed labels.
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

    if sample_weights is not None:
        if sample_weights.shape != (mu.shape[0],):
            raise ValueError(
                f"sample_weights shape {tuple(sample_weights.shape)} "
                f"!= (B,) = ({mu.shape[0]},)",
            )
        nll_per_star = nll_per_star * sample_weights

    if mask is not None:
        if mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} != y shape {y.shape}")
        obs_per_star = mask.float().sum(dim=-1).clamp_min(eps)
        per_star_scale = obs_per_star / float(n_dims)
        nll_per_star = nll_per_star * per_star_scale
        if sample_weights is not None:
            denom = (mask.float() * sample_weights.unsqueeze(-1)).sum().clamp_min(eps)
        else:
            denom = mask.float().sum().clamp_min(eps)
        return nll_per_star.sum() / denom

    if sample_weights is not None:
        return nll_per_star.sum() / sample_weights.sum().clamp_min(eps) / float(n_dims)

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


def barlow_twins_loss(h: torch.Tensor, lam: float = 0.005, eps: float = 1e-8) -> torch.Tensor:
    """Barlow-Twins redundancy-reduction on the trunk hidden state ``h``.

    Ported from the TESS_ML prototype. Z-scores ``h`` along the batch dim,
    builds the feature-feature cross-correlation ``C``, and penalises:

    - diagonal deviation from 1 (per-feature decorrelation target), and
    - off-diagonal magnitude (redundancy-reduction across features), with
      weight ``lam``.

    This is the third term in TESS_ML's joint loss ``l_c + l_r + l_b`` and is
    what prevents the latent from collapsing when the regression term pulls
    the encoder toward label-space geometry. Without it, joint training with
    SupCon + Gaussian-NLL on multi-modal chemistry collapses to the
    conditional mean (the +0.11 α/M attractor diagnosed in ADR-0014).

    Parameters
    ----------
    h
        Trunk hidden state, shape ``(B, D)``. Not the L2-normalised
        projection ``z`` — Barlow wants raw magnitudes.
    lam
        Off-diagonal weight. TESS_ML default ``0.005``.
    eps
        Variance floor for z-scoring.
    """
    zn = (h - h.mean(dim=0)) / (h.std(dim=0) + eps)
    C = zn.T @ zn / h.shape[0]
    d = C.diagonal()
    return (d - 1.0).pow(2).sum() + lam * (C - d.diag()).pow(2).sum()


class ContrastiveQueue:
    """Momentum queue of projections + labels for SupCon dense negatives.

    Faithful port of ``TESS_ML/src/contrastive/training.py:Queue``. SupCon in
    small batches is undertrained — with ``B=512`` each anchor sees only ~511
    negatives per step. A queue of 8192 slots raises the effective key count
    to ``B + 8192`` per step (the queue concatenated with the current batch),
    which is what TESS_ML's prototype achieved low/high-α disc separation with.

    The queue is detached from the graph: keys enter via :meth:`enqueue`
    after each optimisation step, so backward flows only through the
    in-batch anchors — exactly the MoCo / SupCon-with-memory-queue pattern.

    ``warm_start=True`` (default) initialises the queue full with unit-norm
    random vectors and zero labels; the SupCon kernel gives these ~zero
    weight (labels at 0 vs real labels at e.g. 4500 K are Gaussian-kernel
    suppressed), so they function as low-signal negatives until real keys
    overwrite them. The benefit is that ``K`` is static from step 1, which
    keeps CUDA graph capture stable and removes a cold-start edge case.

    Parameters
    ----------
    latent_dim
        Projection dimension ``D``. Must match the encoder's projection head.
    n_labels
        Label dimension. SupCon kernel uses label distances; store the label
        that accompanies each key.
    size
        Queue length. TESS_ML default ``8192``.
    device
        Torch device; defaults to CUDA if available, else CPU.
    warm_start
        Initialise the queue full of random unit-norm vectors. Default True.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        n_labels: int,
        size: int = 8192,
        device: torch.device | None = None,
        warm_start: bool = True,
    ) -> None:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.ptr = 0
        self.full = warm_start
        self.z = F.normalize(torch.randn(size, latent_dim, device=dev), dim=1)
        self.y = torch.zeros(size, n_labels, device=dev)

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """Write ``(z, y)`` into the ring buffer. Detached internally."""
        bs = z.shape[0]
        if bs >= self.size:
            self.z[:] = z[-self.size:].detach()
            self.y[:] = y[-self.size:].detach()
            self.ptr = 0
            self.full = True
            return
        end = (self.ptr + bs) % self.size
        if self.ptr + bs <= self.size:
            self.z[self.ptr:self.ptr + bs] = z.detach()
            self.y[self.ptr:self.ptr + bs] = y.detach()
        else:
            ov = self.ptr + bs - self.size
            self.z[self.ptr:] = z[:bs - ov].detach()
            self.y[self.ptr:] = y[:bs - ov].detach()
            self.z[:ov] = z[bs - ov:].detach()
            self.y[:ov] = y[bs - ov:].detach()
            self.full = True
        self.ptr = end
        if end == 0:
            self.full = True

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently-stored ``(z, y)``. Empty prefix if not warm."""
        if self.full:
            return self.z, self.y
        return self.z[:self.ptr], self.y[:self.ptr]


__all__ = [
    "ContrastiveQueue",
    "barlow_twins_loss",
    "beta_nll_block_cholesky",
    "mahalanobis_residual",
    "supcon_soft_positive",
]
