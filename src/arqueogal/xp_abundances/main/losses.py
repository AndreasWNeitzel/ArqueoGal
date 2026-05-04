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
    sigma: float | torch.Tensor = 0.10,
    label_scale: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervised Contrastive loss with Gaussian-kernel soft positives.

    The label-space kernel runs in **raw label units** when ``sigma`` is a
    1-D tensor of per-label bandwidths and ``label_scale`` is supplied — the
    standardised labels arriving in ``ya`` / ``yk`` are first un-scaled by
    multiplying by ``label_scale``, then squared distances are normalised by
    the per-label ``sigma`` vector, so the kernel is anisotropic and
    physically interpretable: ``sigma[i]`` has the same units as label ``i``.

    Falls back to the scalar isotropic kernel in standardised space when
    ``label_scale`` is omitted (legacy behaviour, used by unit tests with no
    scaler attached).

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
        Either a scalar bandwidth applied isotropically across all label
        dimensions (legacy standardised-space behaviour), or a 1-D tensor of
        per-label bandwidths in raw label units (production path). When a
        per-label tensor is supplied, ``label_scale`` is also required.
    label_scale
        Per-label std vector ``s``, shape ``(n_labels,)``, that the
        ``LabelScaler`` divided by during standardisation. Multiplied into
        ``ya`` and ``yk`` to recover raw-unit labels before the kernel.

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

        sigma_is_tensor = isinstance(sigma, torch.Tensor) and sigma.ndim >= 1
        if sigma_is_tensor:
            if label_scale is None:
                raise ValueError(
                    "supcon_soft_positive: per-label sigma supplied but "
                    "label_scale is None; both are required to evaluate "
                    "the kernel in raw label units.",
                )
            scale = label_scale.to(ya_f.dtype).to(ya_f.device)
            ya_raw = ya_f * scale
            yk_raw = yk_f * scale
            sig = sigma.to(ya_raw.dtype).to(ya_raw.device)
            diff = ya_raw.unsqueeze(1) - yk_raw.unsqueeze(0)  # (B, K, n)
            d2_per_dim = diff.pow(2) / sig.pow(2).clamp_min(eps)
            d2 = d2_per_dim.sum(-1)
            w = torch.exp(-0.5 * d2)
        else:
            sig_scalar = float(sigma)
            diff = ya_f.unsqueeze(1) - yk_f.unsqueeze(0)
            d2 = diff.pow(2).sum(-1)
            w = torch.exp(-d2 / (2.0 * sig_scalar * sig_scalar))
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
                f"sample_weights shape {tuple(sample_weights.shape)} != (B,) = ({mu.shape[0]},)",
            )
        nll_per_star = nll_per_star * sample_weights

    if mask is not None:
        if mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} != y shape {y.shape}")
        # ``obs_per_star`` is the number of observed labels for each star and may
        # be exactly zero for stars whose entire label vector was masked out by
        # upstream NaN handling. Earlier code clamped this to ``eps`` so the
        # per-star scale became ~1e-9 — that is *not* zero, so a fully-masked
        # star still contributed underflowed noise (and a tiny but non-zero
        # gradient) computed from whatever placeholder ``y`` value the NaN
        # sanitiser had written. The principled behaviour is for fully-masked
        # stars to contribute exactly zero to both the numerator and the
        # denominator of the batch-mean NLL. We achieve that by *not* clamping
        # ``obs_per_star`` here (so per_star_scale is exactly 0 for masked-out
        # stars) and only clamping the batch-level ``denom`` to avoid 0/0 in
        # the degenerate case where every label in the batch is masked.
        obs_per_star = mask.float().sum(dim=-1)
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
        raise ValueError(f"L shape {L.shape} inconsistent with n_labels={mu.shape[-1]}")
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


def soft_ari_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Soft Adjusted Rand Index loss for 2-D chemistry plane ([α/M], [M/H]).

    Computes a differentiable ARI via soft confusion matrix between predicted and
    true GMM-component assignments. The loss is ``(1 - soft_ARI)`` so minimising it
    pulls the predictions toward correct cluster membership.

    Parameters
    ----------
    y_pred
        Predicted soft-assignment probabilities, shape ``(B, K)`` where K=2
        (low-α, high-α components). Rows should sum to 1 (output of softmax).
    y_true
        Ground-truth soft-assignment probabilities, shape ``(B, K)``, also summing
        to 1 along axis 1.
    eps
        Numerical stability constant.

    Returns
    -------
    Scalar loss in [0, 1]. Value 0 means perfect agreement; value 1 means
    worst agreement. This is the negative of the ARI metric itself.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
    if y_pred.shape[1] != 2:
        raise ValueError(f"soft_ari_loss expects K=2 components, got shape {y_pred.shape}")

    B = y_pred.shape[0]
    n_total = torch.tensor(float(B), dtype=y_pred.dtype, device=y_pred.device)

    # Hubert-Arabie ARI extended to soft (continuous) assignments. Confusion
    # matrix is left in raw soft-COUNT space (n_ij = sum_k pred[k,i] * true[k,j]),
    # NOT normalised by B — normalisation collapses both the numerator and
    # denominator of the ARI ratio because rows of soft pred / true already
    # sum to 1, so the marginals saturate and the formula returns ±∞ → 0.
    n_ij = y_pred.T @ y_true            # (2, 2) soft counts; sums to B
    a = n_ij.sum(dim=1)                 # row sums (per pred component)
    b = n_ij.sum(dim=0)                 # col sums (per truth component)

    def comb2(x: torch.Tensor) -> torch.Tensor:
        # Continuous extension of n*(n-1)/2 valid for soft (real-valued) counts.
        return 0.5 * x * (x - 1.0)

    sum_comb_nij = comb2(n_ij).sum()
    sum_comb_a = comb2(a).sum()
    sum_comb_b = comb2(b).sum()
    comb_n = comb2(n_total)

    expected = (sum_comb_a * sum_comb_b) / (comb_n + eps)
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    soft_ari = (sum_comb_nij - expected) / (max_idx - expected + eps)
    soft_ari = torch.clamp(soft_ari, -1.0, 1.0)

    return 1.0 - soft_ari


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
            self.z[:] = z[-self.size :].detach()
            self.y[:] = y[-self.size :].detach()
            self.ptr = 0
            self.full = True
            return
        end = (self.ptr + bs) % self.size
        if self.ptr + bs <= self.size:
            self.z[self.ptr : self.ptr + bs] = z.detach()
            self.y[self.ptr : self.ptr + bs] = y.detach()
        else:
            ov = self.ptr + bs - self.size
            self.z[self.ptr :] = z[: bs - ov].detach()
            self.y[self.ptr :] = y[: bs - ov].detach()
            self.z[:ov] = z[bs - ov :].detach()
            self.y[:ov] = y[bs - ov :].detach()
            self.full = True
        self.ptr = end
        if end == 0:
            self.full = True

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently-stored ``(z, y)``. Empty prefix if not warm."""
        if self.full:
            return self.z, self.y
        return self.z[: self.ptr], self.y[: self.ptr]


def add_feature_noise(
    x: torch.Tensor,
    sigma_features: torch.Tensor,
    feature_cols: list[int] | None = None,
    *,
    training: bool = True,
) -> torch.Tensor:
    """Add Gaussian noise to input features at training time for robustness.

    Parameters
    ----------
    x
        Input feature tensor, shape ``(B, D)``.
    sigma_features
        Per-feature uncertainty (standard deviation), shape ``(D,)`` or broadcastable.
        Features with sigma_features[i] = 0 are not perturbed.
    feature_cols
        Optional list of feature indices to perturb. If None, all features are
        eligible (subject to sigma_features > 0). Useful to exclude auxiliary
        features that should not be noised.
    training
        If False, return x unchanged. This allows toggling noise injection in
        eval mode.

    Returns
    -------
    Noised features of the same shape as x. When training=False, returns x unchanged.
    """
    if not training:
        return x

    sigma_features = sigma_features.to(x.device)
    if sigma_features.shape != x.shape[1:]:
        msg = (
            f"sigma_features shape {sigma_features.shape} incompatible with "
            f"x shape[1:] {x.shape[1:]}"
        )
        raise ValueError(msg)

    if feature_cols is not None:
        sigma_apply = torch.zeros_like(sigma_features)
        sigma_apply[feature_cols] = sigma_features[feature_cols]
    else:
        sigma_apply = sigma_features

    noise = torch.randn_like(x) * sigma_apply.unsqueeze(0)
    return x + noise


def propagate_feature_noise_uncertainty(
    mu: torch.Tensor,
    L: torch.Tensor,
    x: torch.Tensor,
    sigma_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically marginalise feature-measurement noise into predicted uncertainty.

    After training, this method inflates the predicted covariance ``Σ = L Lᵀ`` to
    account for feature-space measurement uncertainties via:

    ``σ_total² = σ_pred² + Σ_i (∂μ/∂x_i)² · σ_i²``

    where the gradients are computed via autograd on the trained model.

    Parameters
    ----------
    mu
        Predicted label means, shape ``(B, n_labels)``. Should have
        ``requires_grad=True`` for gradient computation.
    L
        Predicted Cholesky factors, shape ``(B, n_labels, n_labels)``.
    x
        Input features, shape ``(B, D)``. Should have ``requires_grad=True``.
    sigma_features
        Per-feature uncertainty, shape ``(D,)``.

    Returns
    -------
    ``(mu, L_total)`` where ``L_total`` is a new lower-triangular Cholesky factor
    that accounts for propagated feature noise. The diagonal of ``L_total`` is
    inflated; off-diagonals remain unchanged.
    """
    B, n_labels = mu.shape
    device = mu.device

    sigma_features = sigma_features.to(device)
    if sigma_features.shape[0] != x.shape[1]:
        raise ValueError(
            f"sigma_features length {sigma_features.shape[0]} != x.shape[1] {x.shape[1]}"
        )

    sigma_pred_sq = (L @ L.transpose(-2, -1)).diagonal(dim1=-2, dim2=-1)

    grad_norms_sq = torch.zeros(B, n_labels, device=device)
    for i_label in range(n_labels):
        grad_mu_i = torch.autograd.grad(
            outputs=mu[:, i_label].sum(),
            inputs=x,
            create_graph=True,
            allow_unused=True,
            retain_graph=(i_label < n_labels - 1),
        )[0]
        if grad_mu_i is not None:
            grad_norms_sq[:, i_label] = grad_mu_i.pow(2) @ sigma_features.pow(2)
        else:
            grad_norms_sq[:, i_label] = 0.0

    sigma_total_sq = sigma_pred_sq + grad_norms_sq

    L_total = L.clone()
    L_diag = torch.sqrt(sigma_total_sq.clamp_min(1e-8))
    L_indices = torch.arange(n_labels, device=device)
    L_total[:, L_indices, L_indices] = L_diag

    return mu, L_total


__all__ = [
    "ContrastiveQueue",
    "add_feature_noise",
    "barlow_twins_loss",
    "beta_nll_block_cholesky",
    "mahalanobis_residual",
    "propagate_feature_noise_uncertainty",
    "soft_ari_loss",
    "supcon_soft_positive",
]
