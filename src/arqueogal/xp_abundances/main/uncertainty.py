"""Post-hoc uncertainty calibration + coverage + conformal prediction.

DESIGN §Release gates marks calibration as a *hard* gate: every released label
must have a reliability diagram (per Teff × log g × [Fe/H] cell) where
predicted σ tracks observed residual σ within 10%, and 68/95/99% credible
intervals must cover truth within 5 percentage points. Uncalibrated cells
are corrected with stratified temperature scaling. Conformal prediction
intervals are released as an alternative (distribution-free) product.

This module implements those pieces against the
:class:`~arqueogal.xp_abundances.main.model.XpAbundanceModel` outputs
(``mu`` ∈ ℝ^{B×n}, ``L`` ∈ ℝ^{B×n×n}). Everything is post-hoc — no retraining.

Key objects:

- :func:`collect_predictions`: run a model over a DataLoader, return stacked
  (mu, L, y, sigma_Y) on CPU.
- :func:`bin_by_cells`: quantile-bin 2-D/3-D feature arrays into cell IDs —
  matches the stratified-split binning idiom.
- :func:`temperature_scaling_per_cell`: MLE-derived per-cell scalar ``s`` on
  the Cholesky factor; closed form ``s² = mean(mahal) / n_dims``.
- :func:`isotonic_per_label`: per-label monotone remap of predicted → empirical
  CDF, via sklearn's :class:`IsotonicRegression`.
- :func:`coverage_at_levels`: empirical coverage at 68/95/99% under both
  per-label Gaussian intervals and joint Mahalanobis (χ² quantile).
- :func:`conformal_nonconformity_scores`: split-conformal scores (whitened
  norm) for distribution-free intervals.
- :func:`fit_calibration` / :func:`apply_calibration`: round-trip dict whose
  shape matches the DESIGN §Checkpoint schema.

The module is deliberately numpy-heavy (not torch): calibration runs once per
ensemble member on a CPU-sized validation set, so autograd and GPU are
unnecessary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader

_DEFAULT_LEVELS: tuple[float, float, float] = (0.68, 0.95, 0.99)
"""Nominal credible-interval levels the release gate checks."""

_EPS: float = 1e-12


@dataclass
class CalibrationArtifacts:
    """Structured calibration bundle — what the checkpoint stores.

    Matches DESIGN §Checkpoint schema's ``calibration`` sub-dict keys so
    :func:`~.training.save_checkpoint` can slot it in directly.
    """

    temperature_per_cell: dict[int, float] = field(default_factory=dict)
    isotonic_per_label: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    conformal_scores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cell_definition: dict[str, Any] = field(default_factory=dict)

    def as_checkpoint_dict(self) -> dict[str, Any]:
        """Return a dict shaped exactly like DESIGN §Checkpoint's calibration key."""
        iso_serialisable = {
            int(k): {"X": v["X"].astype(np.float32), "y": v["y"].astype(np.float32)}
            for k, v in self.isotonic_per_label.items()
        }
        return {
            "temperature_per_cell": {
                int(k): float(v) for k, v in self.temperature_per_cell.items()
            },
            "isotonic_per_label": iso_serialisable,
            "conformal_scores": self.conformal_scores.astype(np.float32),
            "cell_definition": dict(self.cell_definition),
        }


# --- Prediction collection ---------------------------------------------------


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run ``model`` over ``loader`` and stack outputs into CPU numpy arrays.

    Returns a dict with keys ``"mu"``, ``"L"``, ``"y"``, and — if the loader
    yields 3-tuples — ``"sigma_Y"``. All arrays are float32 on CPU.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    mus: list[np.ndarray] = []
    Ls: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            mu, L, _h, _z = model(x)
            mus.append(mu.cpu().numpy())
            Ls.append(L.cpu().numpy())
            ys.append(y.numpy())
            if len(batch) >= 3:
                sigmas.append(batch[2].numpy())

    out: dict[str, np.ndarray] = {
        "mu": np.concatenate(mus, axis=0).astype(np.float32),
        "L": np.concatenate(Ls, axis=0).astype(np.float32),
        "y": np.concatenate(ys, axis=0).astype(np.float32),
    }
    if sigmas:
        out["sigma_Y"] = np.concatenate(sigmas, axis=0).astype(np.float32)
    return out


# --- Cell binning ------------------------------------------------------------


def bin_by_cells(
    features: np.ndarray,
    n_bins: tuple[int, ...] = (4, 4, 4),
    *,
    seed: int = 0,
    quantile_offset: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantile-bin each column of ``features`` into ``n_bins[j]`` bins and combine.

    Returns ``(cell_ids, cell_definition)`` where ``cell_ids`` is an int array
    of length ``N`` and ``cell_definition`` is a serialisable dict capturing
    the bin edges per column so the same binning can be reapplied at inference.

    Parameters
    ----------
    quantile_offset
        Shift applied to all internal quantile cut points. Default 0.0 places
        edges at ``[1/nb, 2/nb, …, (nb-1)/nb]``. A small offset (e.g.
        ``0.5/nb``) puts cuts at ``[1.5/nb, 2.5/nb, …]`` so cell boundaries
        do not coincide with the median (50 %ile) of the data — this avoids
        the visible "cell-edge cliff" at high-density modes (the disc peak in
        [M/H] sits at the 75 %ile of APOGEE training and would otherwise land
        exactly on a cell boundary, producing tier-filter discontinuities).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if features.shape[1] != len(n_bins):
        raise ValueError(
            f"n_bins length {len(n_bins)} does not match feature dim {features.shape[1]}"
        )
    if not -0.5 < quantile_offset < 0.5:
        raise ValueError(f"quantile_offset must be in (-0.5, 0.5), got {quantile_offset}")

    rng = np.random.default_rng(seed)
    n, _ = features.shape
    codes = np.zeros(n, dtype=np.int64)
    edges_per_col: list[list[float]] = []
    for j, nb in enumerate(n_bins):
        col = features[:, j].astype(np.float64)
        finite = np.isfinite(col)
        if finite.any():
            qs = np.linspace(0, 1, nb + 1)[1:-1] + quantile_offset / nb
            qs = np.clip(qs, 0.0, 1.0)
            edges = np.quantile(col[finite], qs)
        else:
            edges = np.zeros(nb - 1, dtype=np.float64)
        idx = np.zeros(n, dtype=np.int64)
        idx[finite] = np.digitize(col[finite], edges)
        idx[~finite] = rng.integers(0, nb, size=int((~finite).sum()))
        idx = np.clip(idx, 0, nb - 1)
        codes = codes * nb + idx
        edges_per_col.append([float(x) for x in edges])

    definition = {
        "n_bins": list(n_bins),
        "edges_per_col": edges_per_col,
        "quantile_offset": float(quantile_offset),
    }
    return codes, definition


# --- Temperature scaling -----------------------------------------------------


def _mahalanobis_per_star(mu: np.ndarray, L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-star Mahalanobis distance ``||y-μ||²_Σ⁻¹``. Numpy triangular solve."""
    diff = (y - mu)[..., None]  # (B, n, 1)
    # Solve L z = diff for z; then mahal = ||z||². Use scipy's solve_triangular.
    from scipy.linalg import solve_triangular

    mahal = np.empty(mu.shape[0], dtype=np.float64)
    for b in range(mu.shape[0]):
        z = solve_triangular(L[b], diff[b], lower=True)
        mahal[b] = float((z * z).sum())
    return mahal


def temperature_scaling_per_cell(
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
) -> dict[int, float]:
    """MLE temperature ``s`` per cell on the Cholesky factor.

    Closed form: for a MVN ``N(μ, s²·Σ)``, the NLL is minimised at
    ``s² = mean(mahal) / n``. Returns a mapping ``cell → s``. Cells with
    fewer than ``n+2`` stars inherit ``s = 1.0`` (under-determined fit).
    """
    if mu.ndim != 2:
        raise ValueError(f"mu must be 2D, got {mu.shape}")
    n = mu.shape[-1]
    mahal = _mahalanobis_per_star(mu, L, y)
    out: dict[int, float] = {}
    for c in np.unique(cell_ids):
        mask = cell_ids == c
        if int(mask.sum()) < n + 2:
            out[int(c)] = 1.0
            continue
        s_sq = max(float(mahal[mask].mean()) / n, _EPS)
        out[int(c)] = math.sqrt(s_sq)
    return out


# --- Shrunken per-cell-per-label scaling --------------------------------------


def shrunken_per_cell_per_label_scale(  # noqa: PLR0913
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
    *,
    tau: float = 50.0,
    min_cell_stars: int = 8,
    alpha_floor: float = 1e-4,
) -> dict[str, Any]:
    """Per-(cell, label) variance scale ``α_{c,l}`` with empirical-Bayes shrinkage.

    Addresses conditional heteroscedasticity across (Teff, log g, [M/H])
    cells when per-label marginal calibration is globally correct but fails
    within cells. Scalar per-cell temperature scaling cannot fix this because
    different labels need different scaling within the same cell — diagnosed
    in #140 follow-up as the root cause of the #135 gate failure.

    For each label ``j``:

    - Compute the unconditional ``α_j = √Var(z_{·,j})`` over all stars. This
      is the shrinkage target — since per-label marginal calibration is
      already good globally, ``α_j ≈ 1`` and sparse cells get pulled toward
      "don't scale".

    For each cell ``c`` and label ``j``:

    - ``α_{c,j}^{raw} = √Var(z_{within cell c, label j})``.
    - Shrink toward the global with ``λ_c = n_c / (n_c + τ)`` (empirical-Bayes
      form; cells with ``n_c = τ`` are half-way between raw and prior)::

          α_{c,j} = λ_c · α_{c,j}^{raw} + (1 − λ_c) · α_j

    Applied via ``L'_b = diag(α_{c(b), ·}) · L_b``, which yields
    ``Σ'_b = diag(α) Σ_b diag(α)``. This preserves PD-ness (``α > 0``) and
    the joint correlation structure, while rescaling marginal variances so
    ``Var(z) ≈ 1`` within each cell.

    Parameters
    ----------
    mu, L, y : arrays
        Ensemble moment-matched ``(μ, L)`` and truth ``y`` in raw units.
    cell_ids : int array
        From :func:`bin_by_cells` on ``(Teff, log g, [M/H])`` truth.
    tau : float
        Shrinkage pseudo-count. ``50.0`` default: a cell with 50 stars is
        half-way between raw and global; 500 stars ≈ 91 % raw; 5 stars ≈ 9 %
        raw. Given ~675 stars/cell on a 4³ grid over 42 k val stars, most
        cells use their raw fit, the sparse tail gets pulled to the global.
    min_cell_stars : int
        Below this, fall back to the global ``α_j`` for that (cell, label).
    alpha_floor : float
        Guard against degenerate zero scales; ``α`` is clipped from below.

    Returns
    -------
    dict with keys
        ``per_star_alpha`` : ``(B, n)`` float32 — per-star α, already
            broadcast across cells/labels. Multiply ``L`` by this along the
            row axis: ``L' = per_star_alpha[:, :, None] * L``.
        ``scales`` : dict ``{(cell_id, label_idx): α}``.
        ``global_alpha`` : ``(n,)`` per-label shrinkage target.
        ``n_per_cell`` : dict ``{cell_id: n}``.
        ``tau`` : echo of the input ``tau``.
    """
    if mu.ndim != 2:
        raise ValueError(f"mu must be 2D, got {mu.shape}")
    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L, L)).clip(_EPS, None)
    z = (y - mu) / sigma_diag
    n_dims = z.shape[1]

    global_alpha = np.ones(n_dims, dtype=np.float64)
    for j in range(n_dims):
        col = z[:, j]
        col = col[np.isfinite(col)]
        if col.size >= min_cell_stars:
            global_alpha[j] = float(np.sqrt(max(col.var(ddof=0), _EPS)))

    unique_cells = np.unique(cell_ids)
    per_star_alpha = np.ones((z.shape[0], n_dims), dtype=np.float32)
    scales_table: dict[tuple[int, int], float] = {}
    n_per_cell: dict[int, int] = {}

    for c in unique_cells:
        mask = cell_ids == c
        n_c = int(mask.sum())
        n_per_cell[int(c)] = n_c
        lam = n_c / (n_c + tau)
        for j in range(n_dims):
            col = z[mask, j]
            col = col[np.isfinite(col)]
            if col.size >= min_cell_stars:
                alpha_raw = float(np.sqrt(max(col.var(ddof=0), _EPS)))
            else:
                alpha_raw = float(global_alpha[j])
            alpha = lam * alpha_raw + (1.0 - lam) * global_alpha[j]
            alpha = max(alpha, alpha_floor)
            scales_table[(int(c), j)] = float(alpha)
            per_star_alpha[mask, j] = alpha

    return {
        "per_star_alpha": per_star_alpha,
        "scales": scales_table,
        "global_alpha": global_alpha.astype(np.float64),
        "n_per_cell": n_per_cell,
        "tau": float(tau),
    }


# --- GP-smoothed per-label scaling -------------------------------------------


@dataclass
class GpAlphaBundle:
    """Serialisable bundle of per-label GP α-smoothing artefacts.

    Stored inside :class:`CalibrationArtifacts` when ``fit_calibration`` is
    called with ``use_gp_smoothing=True``. Bundles the training data and
    fitted kernel hyperparameters per label so :func:`apply_gp_alpha`
    can refit cheaply at inference time (62-ish points, <1s/label).
    """

    feature_mean: np.ndarray  # (F,) standardiser mean
    feature_scale: np.ndarray  # (F,) standardiser scale
    cell_centers: np.ndarray  # (n_train_cells, F) standardised
    log_alpha_targets: np.ndarray  # (n_train_cells, n_labels) training targets
    log_alpha_noise: np.ndarray  # (n_train_cells,) per-cell noise std (log-space)
    kernel_params: list[dict[str, float]]  # per-label fitted kernel hyperparams
    global_log_alpha: np.ndarray  # (n_labels,) global fallback
    alpha_floor: float = 1e-4
    alpha_ceiling: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.astype(np.float32),
            "feature_scale": self.feature_scale.astype(np.float32),
            "cell_centers": self.cell_centers.astype(np.float32),
            "log_alpha_targets": self.log_alpha_targets.astype(np.float32),
            "log_alpha_noise": self.log_alpha_noise.astype(np.float32),
            "kernel_params": self.kernel_params,
            "global_log_alpha": self.global_log_alpha.astype(np.float32),
            "alpha_floor": float(self.alpha_floor),
            "alpha_ceiling": float(self.alpha_ceiling),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "GpAlphaBundle":
        return cls(
            feature_mean=np.asarray(blob["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(blob["feature_scale"], dtype=np.float64),
            cell_centers=np.asarray(blob["cell_centers"], dtype=np.float64),
            log_alpha_targets=np.asarray(blob["log_alpha_targets"], dtype=np.float64),
            log_alpha_noise=np.asarray(blob["log_alpha_noise"], dtype=np.float64),
            kernel_params=list(blob["kernel_params"]),
            global_log_alpha=np.asarray(blob["global_log_alpha"], dtype=np.float64),
            alpha_floor=float(blob.get("alpha_floor", 1e-4)),
            alpha_ceiling=float(blob.get("alpha_ceiling", 5.0)),
        )


def _make_gp(length_scale: float, constant: float, noise: float) -> GaussianProcessRegressor:
    kernel = ConstantKernel(constant, constant_value_bounds=(1e-3, 1e2)) * RBF(
        length_scale=length_scale, length_scale_bounds=(1e-1, 1e2)
    ) + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=2,
        random_state=0,
    )


def gp_smoothed_per_cell_per_label_scale(  # noqa: PLR0913, PLR0915
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    cell_ids: np.ndarray,
    *,
    min_cell_stars_for_training: int = 32,
    min_cell_stars: int = 8,
    alpha_floor: float = 1e-4,
    alpha_ceiling: float = 5.0,
) -> dict[str, Any]:
    """3-D Gaussian-process α-smoothing over the (Teff, log g, [M/H]) grid.

    Alternative to :func:`shrunken_per_cell_per_label_scale` that replaces
    discrete per-cell empirical-Bayes shrinkage with a GP fit across cells
    in feature space. Sparse-cell α is predicted by borrowing from well-
    populated neighbours rather than shrinking to a single global scalar.

    Per-star α is set by evaluating the GP at each star's **cell-center**
    coordinates (lookup via ``cell_ids``). This keeps per-cell semantics —
    all stars in a cell get the same α — so the reliability metric
    ``|Var(z)-1|`` remains well defined. Spatial smoothness is preserved
    *across* cells: cell-center GP predictions vary continuously with
    feature-space neighbours. Within-cell star-level continuity is
    secondary to across-cell smoothness, and using per-star features here
    empirically breaks per-cell reliability in well-populated interior
    cells by introducing within-cell α variance uncorrelated with residual
    structure.

    Parameters
    ----------
    mu, L, y : arrays
        Ensemble moment-matched ``(μ, L)`` and truth ``y`` in raw units.
    features : (B, F) array
        Per-star continuous features for GP coordinates — typically
        ``(Teff, log g, [M/H])`` (F = 3).
    cell_ids : int array
        Cell IDs from :func:`bin_by_cells` on the same features.
    min_cell_stars_for_training : int
        Cells with fewer stars than this are **not** used as GP training
        points — their raw α is too noisy. They still get predicted α_GP
        at inference.
    min_cell_stars : int
        Hard floor for computing any raw α. Below this, cell is skipped
        from training entirely.
    alpha_floor, alpha_ceiling : float
        Clip the GP-predicted α before return.

    Returns
    -------
    dict
        ``per_star_alpha`` : ``(B, n)`` float32 per-star α_GP.
        ``scales`` : dict ``{(cell_id, label_idx): α_GP(cell_center)}`` for
            reporting/adjacency diagnostics.
        ``global_alpha`` : ``(n,)`` global per-label α (from all stars).
        ``n_per_cell`` : dict ``{cell_id: n}``.
        ``gp_bundle`` : :class:`GpAlphaBundle` for serialisation.
        ``training_diagnostics`` : per-label fitted length scales, noise,
            residuals on training cells.
    """
    if mu.ndim != 2:
        raise ValueError(f"mu must be 2D, got {mu.shape}")
    if features.ndim != 2 or features.shape[0] != mu.shape[0]:
        raise ValueError(
            f"features must be (B, F), got {features.shape} for mu {mu.shape}",
        )

    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L, L)).clip(_EPS, None)
    z = (y - mu) / sigma_diag
    n_stars, n_labels = z.shape
    n_feat = features.shape[1]

    # Feature standardisation — RBF kernel is sensitive to input scale.
    finite_feat = np.isfinite(features).all(axis=1)
    feat_mean = np.nanmean(features[finite_feat], axis=0)
    feat_scale = np.nanstd(features[finite_feat], axis=0)
    feat_scale = np.where(feat_scale > _EPS, feat_scale, 1.0)
    feats_std = ((features - feat_mean) / feat_scale).astype(np.float64)

    # Global per-label α (also serves as fallback for rows with non-finite
    # features at prediction time).
    global_alpha = np.ones(n_labels, dtype=np.float64)
    for j in range(n_labels):
        col = z[:, j]
        col = col[np.isfinite(col)]
        if col.size >= min_cell_stars:
            global_alpha[j] = float(np.sqrt(max(col.var(ddof=0), _EPS)))
    global_log_alpha = np.log(np.clip(global_alpha, alpha_floor, alpha_ceiling))

    # Per-cell raw α + cell centers in standardised feature space.
    unique_cells = np.unique(cell_ids)
    raw_alpha_per_cell: dict[int, np.ndarray] = {}
    cell_centers: dict[int, np.ndarray] = {}
    cell_sizes: dict[int, int] = {}
    finite_row = np.isfinite(feats_std).all(axis=1)
    for c in unique_cells:
        mask = cell_ids == c
        mask_ok = mask & finite_row
        n_c = int(mask_ok.sum())
        cell_sizes[int(c)] = n_c
        if n_c < min_cell_stars:
            continue
        center = feats_std[mask_ok].mean(axis=0)
        if not np.isfinite(center).all():
            continue
        alpha_c = np.full(n_labels, np.nan, dtype=np.float64)
        for j in range(n_labels):
            col = z[mask, j]
            col = col[np.isfinite(col)]
            if col.size >= min_cell_stars:
                alpha_c[j] = float(np.sqrt(max(col.var(ddof=0), _EPS)))
        raw_alpha_per_cell[int(c)] = alpha_c
        cell_centers[int(c)] = center

    # Training set for the GP: cells with n >= min_cell_stars_for_training
    # AND finite α per the label under consideration. Selection per label.
    train_cell_ids = [c for c in raw_alpha_per_cell if cell_sizes[c] >= min_cell_stars_for_training]
    if not train_cell_ids:
        raise RuntimeError(
            "no cells pass min_cell_stars_for_training — GP fit requires "
            f"at least one cell with n >= {min_cell_stars_for_training}",
        )

    train_centers = np.vstack([cell_centers[c] for c in train_cell_ids])
    train_alpha = np.vstack([raw_alpha_per_cell[c] for c in train_cell_ids])
    train_sizes = np.array([cell_sizes[c] for c in train_cell_ids], dtype=np.float64)

    # Per-cell noise in log-α space. var(log α̂) ≈ 1/(2n) for sample variance
    # of sqrt-variance. We use noise_std = 1/sqrt(2 n_c) as an informative
    # per-cell prior on calibration-estimate uncertainty.
    log_alpha_noise = 1.0 / np.sqrt(np.maximum(2.0 * train_sizes, 1.0))

    # Fit one GP per label in log-α space. Predict at cell centers only;
    # broadcast per-star α via cell_id lookup (not per-star feature eval)
    # so within-cell α is constant and per-cell ``Var(z)=1`` stays defined.
    kernel_params: list[dict[str, float]] = []
    training_diagnostics: list[dict[str, Any]] = []
    log_alpha_targets = np.zeros_like(train_alpha)

    all_cell_list = sorted(cell_centers.keys())
    all_centers = (
        np.vstack([cell_centers[c] for c in all_cell_list])
        if all_cell_list
        else np.zeros((0, n_feat))
    )
    # α at each cell center, per label. Cells without a valid center
    # (n_c < min_cell_stars) fall back to global_alpha below.
    alpha_per_cell_center: dict[int, np.ndarray] = {c: global_alpha.copy() for c in all_cell_list}

    for j in range(n_labels):
        tgt_raw = train_alpha[:, j]
        finite = np.isfinite(tgt_raw)
        if finite.sum() < 4:
            kernel_params.append(
                {
                    "length_scale": float("nan"),
                    "constant": float("nan"),
                    "noise_level": float("nan"),
                }
            )
            training_diagnostics.append(
                {
                    "n_train_cells": int(finite.sum()),
                    "length_scale": None,
                    "constant": None,
                    "noise_level": None,
                    "train_log_alpha_rmse": None,
                }
            )
            log_alpha_targets[:, j] = np.log(
                np.clip(
                    np.where(finite, tgt_raw, global_alpha[j]),
                    alpha_floor,
                    alpha_ceiling,
                )
            )
            for c in all_cell_list:
                alpha_per_cell_center[c][j] = float(global_alpha[j])
            continue

        X_train = train_centers[finite]
        y_train = np.log(np.clip(tgt_raw[finite], alpha_floor, alpha_ceiling))
        y_centered = y_train - global_log_alpha[j]
        alpha_noise_sq = log_alpha_noise[finite] ** 2

        gp = _make_gp(length_scale=1.0, constant=0.1, noise=0.01)
        gp.alpha = alpha_noise_sq  # per-sample noise variance (sklearn API)
        gp.fit(X_train, y_centered)

        if len(all_cell_list) > 0:
            log_alpha_cells = gp.predict(all_centers) + global_log_alpha[j]
            alpha_cells = np.exp(
                np.clip(
                    log_alpha_cells,
                    np.log(alpha_floor),
                    np.log(alpha_ceiling),
                )
            )
            for c, a in zip(all_cell_list, alpha_cells):
                alpha_per_cell_center[int(c)][j] = float(a)

        # Extract fitted hyperparams.
        theta = gp.kernel_.get_params()
        const_val = float(theta["k1__k1__constant_value"])
        length_val = float(theta["k1__k2__length_scale"])
        noise_val = float(theta["k2__noise_level"])
        kernel_params.append(
            {"constant": const_val, "length_scale": length_val, "noise_level": noise_val}
        )

        # Training RMSE (in log-α space).
        y_pred_train = gp.predict(X_train) + global_log_alpha[j]
        train_rmse = float(np.sqrt(np.mean((y_pred_train - y_train) ** 2)))
        training_diagnostics.append(
            {
                "n_train_cells": int(finite.sum()),
                "constant": const_val,
                "length_scale": length_val,
                "noise_level": noise_val,
                "train_log_alpha_rmse": train_rmse,
            }
        )
        # Fill per-cell log α target array (include all cells; NaN imputed).
        log_alpha_targets[:, j] = np.log(
            np.clip(
                np.where(np.isfinite(tgt_raw), tgt_raw, global_alpha[j]),
                alpha_floor,
                alpha_ceiling,
            )
        )

    # Broadcast per-star α from cell-center predictions. Stars in cells
    # without a valid center fall back to the global per-label α.
    per_star_alpha = np.tile(global_alpha[None, :], (n_stars, 1)).astype(np.float64)
    for c in np.unique(cell_ids):
        mask = cell_ids == c
        if int(c) in alpha_per_cell_center:
            per_star_alpha[mask] = alpha_per_cell_center[int(c)]

    # Scales table for reporting/adjacency diagnostics.
    scales_table: dict[tuple[int, int], float] = {}
    for c in all_cell_list:
        alpha_c = alpha_per_cell_center[c]
        for j in range(n_labels):
            scales_table[(int(c), int(j))] = float(alpha_c[j])

    per_star_alpha = per_star_alpha.astype(np.float32)
    per_star_alpha = np.clip(per_star_alpha, alpha_floor, alpha_ceiling)

    bundle = GpAlphaBundle(
        feature_mean=feat_mean.astype(np.float64),
        feature_scale=feat_scale.astype(np.float64),
        cell_centers=train_centers.astype(np.float64),
        log_alpha_targets=log_alpha_targets[[i for i, c in enumerate(train_cell_ids)]].astype(
            np.float64
        )
        if len(train_cell_ids)
        else np.zeros((0, n_labels)),
        log_alpha_noise=log_alpha_noise.astype(np.float64),
        kernel_params=kernel_params,
        global_log_alpha=global_log_alpha.astype(np.float64),
        alpha_floor=float(alpha_floor),
        alpha_ceiling=float(alpha_ceiling),
    )

    return {
        "per_star_alpha": per_star_alpha,
        "scales": scales_table,
        "global_alpha": global_alpha,
        "n_per_cell": cell_sizes,
        "gp_bundle": bundle,
        "training_diagnostics": training_diagnostics,
        "train_cell_ids": [int(c) for c in train_cell_ids],
    }


def apply_gp_alpha(
    features: np.ndarray,
    bundle: GpAlphaBundle,
) -> np.ndarray:
    """Evaluate stored GP bundle at new star features; returns ``(B, n)`` α.

    Refits the per-label GPs from the stored training data + kernel
    hyperparams (a few hundred-µs op at n_train_cells ≈ 62), then predicts.
    This keeps the checkpoint schema a pure-data dict while recovering the
    GP posterior mean on demand.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    feats_std = (features - bundle.feature_mean) / np.where(
        bundle.feature_scale > _EPS,
        bundle.feature_scale,
        1.0,
    )
    row_ok = np.isfinite(feats_std).all(axis=1)

    n_stars = features.shape[0]
    n_labels = bundle.log_alpha_targets.shape[1]
    out = np.zeros((n_stars, n_labels), dtype=np.float64)

    global_alpha = np.exp(bundle.global_log_alpha)

    for j, kp in enumerate(bundle.kernel_params):
        if not np.isfinite(kp.get("length_scale", np.nan)):
            out[:, j] = global_alpha[j]
            continue
        gp = _make_gp(
            length_scale=kp["length_scale"],
            constant=kp["constant"],
            noise=kp["noise_level"],
        )
        y_train = bundle.log_alpha_targets[:, j] - bundle.global_log_alpha[j]
        gp.alpha = bundle.log_alpha_noise**2
        gp.fit(bundle.cell_centers, y_train)
        alpha = np.full(n_stars, global_alpha[j], dtype=np.float64)
        if row_ok.any():
            log_alpha = gp.predict(feats_std[row_ok]) + bundle.global_log_alpha[j]
            alpha_ok = np.exp(
                np.clip(
                    log_alpha,
                    np.log(bundle.alpha_floor),
                    np.log(bundle.alpha_ceiling),
                )
            )
            alpha[row_ok] = alpha_ok
        out[:, j] = alpha

    return out.astype(np.float32)


# --- Release-gate exclusion envelopes ----------------------------------------


@dataclass
class RegimeBEnvelope:
    """Galactic-plane warm-upper-RGB exclusion envelope.

    Diagnosed in the 5-label halt-cell analysis (2026-04-19): cells 34/49
    (Teff > 4820 K, log g < 2.05, |b| < 3°, A_V ≥ 0.38) show a ``+1σ`` Teff
    mean-bias driven by Galactic-plane extinction confounding that is not
    fixable within the 5 GB dust-map budget (research_brief §14 item 11).

    Stars inside this envelope are flagged ``tier1_release = False`` and
    carried as population-level-only. The envelope is specified in
    **predicted** (Teff, log g) plus Galactic latitude ``b`` so it is
    applicable at Stream 3 inference time where APOGEE truth is absent.

    Thresholds below are defined with a small buffer around the val-set
    halt cells (Teff 4820 → 4750, log g 2.05 → 2.10, |b| 3° → 5°) to avoid
    spillover through cell-boundary binning noise.
    """

    b_deg_max: float = 5.0
    teff_k_min: float = 4750.0
    logg_dex_max: float = 2.10

    def mask(
        self,
        teff_pred: np.ndarray,
        logg_pred: np.ndarray,
        b_deg: np.ndarray,
    ) -> np.ndarray:
        """Return ``True`` where a star falls *inside* the exclusion envelope."""
        teff_pred = np.asarray(teff_pred, dtype=np.float64)
        logg_pred = np.asarray(logg_pred, dtype=np.float64)
        b_deg = np.asarray(b_deg, dtype=np.float64)
        return (
            (np.abs(b_deg) < self.b_deg_max)
            & (teff_pred > self.teff_k_min)
            & (logg_pred < self.logg_dex_max)
        )

    def tier1_release_flag(
        self,
        teff_pred: np.ndarray,
        logg_pred: np.ndarray,
        b_deg: np.ndarray,
    ) -> np.ndarray:
        """Per-star boolean: ``True`` = eligible for Tier 1 per-star release."""
        return ~self.mask(teff_pred, logg_pred, b_deg)

    def to_dict(self) -> dict[str, float]:
        return {
            "b_deg_max": self.b_deg_max,
            "teff_k_min": self.teff_k_min,
            "logg_dex_max": self.logg_dex_max,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, float]) -> "RegimeBEnvelope":
        return cls(
            b_deg_max=float(blob["b_deg_max"]),
            teff_k_min=float(blob["teff_k_min"]),
            logg_dex_max=float(blob["logg_dex_max"]),
        )


# --- Isotonic per-label ------------------------------------------------------


def isotonic_per_label(
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Fit per-label isotonic CDF remaps on standardised residuals.

    For each label dim ``j``, compute the empirical CDF of the standardised
    residual ``(y_j - μ_j) / sqrt(Σ_jj)`` and fit an
    :class:`~sklearn.isotonic.IsotonicRegression` from the theoretical (std
    Normal) CDF to the empirical CDF. Returned as ``{j: {"X": ecdf_x,
    "y": ecdf_y}}`` — the fitted arrays from which
    :func:`apply_calibration` reconstructs the monotone map without pickling
    sklearn objects (keeps the checkpoint schema pure data).
    """
    n = mu.shape[-1]
    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L, L)).astype(np.float64)
    std_resid = (y - mu) / np.clip(sigma_diag, _EPS, None)

    out: dict[int, dict[str, np.ndarray]] = {}
    for j in range(n):
        r = np.sort(std_resid[:, j])
        emp_cdf = (np.arange(len(r), dtype=np.float64) + 0.5) / len(r)
        theor_cdf = norm.cdf(r)
        ir = IsotonicRegression(out_of_bounds="clip").fit(theor_cdf, emp_cdf)
        ir_x = np.asarray(ir.X_thresholds_, dtype=np.float64)
        ir_y = np.asarray(ir.y_thresholds_, dtype=np.float64)
        out[j] = {"X": ir_x, "y": ir_y}
    return out


# --- Coverage ----------------------------------------------------------------


def coverage_at_levels(
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...] = _DEFAULT_LEVELS,
) -> dict[str, dict[float, float | np.ndarray]]:
    """Empirical coverage for per-label and joint (Mahalanobis) intervals.

    Returns a dict with two sub-dicts ``"per_label"`` and ``"joint"``.
    ``per_label[level]`` is an array of shape ``(n,)`` of coverage per label;
    ``joint[level]`` is a scalar ``float`` — the fraction of stars with
    Mahalanobis² below the chi-squared quantile at ``level``.
    """
    n_dims = mu.shape[-1]
    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L, L)).astype(np.float64)
    abs_resid = np.abs(y - mu) / np.clip(sigma_diag, _EPS, None)
    mahal = _mahalanobis_per_star(mu, L, y)

    per_label: dict[float, np.ndarray] = {}
    joint: dict[float, float] = {}
    for lvl in levels:
        z = norm.ppf(0.5 + lvl / 2.0)
        per_label[float(lvl)] = (abs_resid <= z).mean(axis=0).astype(np.float64)
        q = chi2.ppf(lvl, df=n_dims)
        joint[float(lvl)] = float((mahal <= q).mean())
    return {"per_label": per_label, "joint": joint}


# --- Conformal ---------------------------------------------------------------


def conformal_nonconformity_scores(
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Split-conformal scores: Mahalanobis distance per star (``sqrt`` scale).

    The conformal interval at level ``α`` uses the ``⌈(1-α)(n+1)⌉``-th
    quantile of these scores as the radius. Monotone transform of
    Mahalanobis keeps the calibration ordering invariant; returning the
    unsquared distance makes the conformal radius interpretable as a σ.
    """
    mahal = _mahalanobis_per_star(mu, L, y)
    return np.sqrt(np.clip(mahal, 0.0, None)).astype(np.float32)


# --- Orchestration -----------------------------------------------------------


def fit_calibration(  # noqa: PLR0913 — orchestrator with distinct knobs
    mu: np.ndarray,
    L: np.ndarray,
    y: np.ndarray,
    *,
    cell_features: np.ndarray | None = None,
    cell_n_bins: tuple[int, ...] = (4, 4, 4),
    fit_isotonic: bool = True,
    fit_conformal: bool = True,
) -> CalibrationArtifacts:
    """Fit every calibration artefact in one shot.

    ``cell_features`` are the stratification columns used for per-cell
    temperature scaling (typically Teff / log g / [Fe/H] predictions from the
    same model, or frozen APOGEE labels on the val set). If ``None``, a
    single global cell is used.
    """
    art = CalibrationArtifacts()
    if cell_features is not None:
        cell_ids, cell_def = bin_by_cells(cell_features, cell_n_bins)
        art.cell_definition = cell_def
    else:
        cell_ids = np.zeros(mu.shape[0], dtype=np.int64)
        art.cell_definition = {"n_bins": [1], "edges_per_col": [[]]}
    art.temperature_per_cell = temperature_scaling_per_cell(mu, L, y, cell_ids)

    if fit_isotonic:
        art.isotonic_per_label = isotonic_per_label(mu, L, y)
    if fit_conformal:
        art.conformal_scores = conformal_nonconformity_scores(mu, L, y)
    return art


def apply_calibration(
    mu: np.ndarray,
    L: np.ndarray,
    art: CalibrationArtifacts,
    *,
    cell_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-cell temperature scaling to ``L``; means pass through.

    Isotonic remaps operate on CDFs, not directly on ``(μ, Σ)`` — callers
    who need probability corrections use
    :func:`isotonic_per_label` outputs at evaluation time. Conformal radii
    are applied by :func:`conformal_radius_at_level`.
    """
    if cell_ids is None:
        cell_ids = np.zeros(mu.shape[0], dtype=np.int64)
    scales = np.asarray(
        [art.temperature_per_cell.get(int(c), 1.0) for c in cell_ids],
        dtype=np.float32,
    )
    L_scaled = (L * scales[:, None, None]).astype(L.dtype)
    return mu.copy(), L_scaled


def conformal_radius_at_level(art: CalibrationArtifacts, level: float) -> float:
    """Quantile of nonconformity scores at ``level`` (e.g. 0.95)."""
    if art.conformal_scores.size == 0:
        raise ValueError("conformal_scores empty — fit_calibration with fit_conformal=True")
    n = art.conformal_scores.size
    q_idx = min(n - 1, int(math.ceil((n + 1) * level)) - 1)
    return float(np.sort(art.conformal_scores)[q_idx])


__all__ = [
    "CalibrationArtifacts",
    "GpAlphaBundle",
    "RegimeBEnvelope",
    "apply_calibration",
    "apply_gp_alpha",
    "bin_by_cells",
    "collect_predictions",
    "conformal_nonconformity_scores",
    "conformal_radius_at_level",
    "coverage_at_levels",
    "fit_calibration",
    "gp_smoothed_per_cell_per_label_scale",
    "isotonic_per_label",
    "shrunken_per_cell_per_label_scale",
    "temperature_scaling_per_cell",
]
