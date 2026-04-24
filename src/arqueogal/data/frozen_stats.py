"""Frozen z-score statistics for Stream-3 XP inference.

Pipeline 1 trains on Stream-1 (APOGEE × Gaia XP) with per-coefficient z-score
statistics computed **once** and then frozen. Applying the same model to
Stream-3 Gaia-RGB+RC coefficients requires applying those frozen stats — never
re-fitting on the inference set, which would shift the input distribution by
the Stream-1 vs Stream-3 mean offset and invalidate the network's learned
coefficient-deviation interpretation.

The source of truth is the provenance sidecar of
``data/processed/pipeline1_features_stream1.parquet`` — specifically the
``extra.c0_zscore_frozen`` and ``extra.coef_norm_zscore_frozen`` blocks plus
``extra.basis_fingerprint_sha256``. The Hermite basis fingerprint pins the
re-projection convention; mismatch between the basis used to project Stream-3
coefficients and the basis the frozen stats were fit on would silently yield
wrong z-scored values. :func:`verify_basis_fingerprint` raises on mismatch.

Public surface:

- :class:`FrozenZScoreStats` — typed container for the frozen tuple.
- :func:`load_frozen_zscore_stats` — read from provenance JSON.
- :func:`verify_basis_fingerprint` — raise on mismatch.
- :func:`apply_frozen_zscore` — apply stats to Stream-3 coefficients.
- :class:`FrozenStatsMismatchError` — raised on basis-fingerprint mismatch.

This module does **not** fit any statistics. Refitting on Stream-3 is a bug.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

XP_COEFF_LEN: Final[int] = 55
"""Length of each Hermite coefficient vector. Mirrors
:data:`arqueogal.data.gaia_xp.XP_COEFF_LEN` — duplicated here to avoid a
circular import for a module that is consumed by inference-time callers which
otherwise do not need the full ``gaia_xp`` dependency graph."""


class FrozenStatsMismatchError(RuntimeError):
    """Raised when the Hermite basis fingerprint does not match the frozen stats.

    Applying frozen z-score stats on top of a different Hermite basis is a
    silent correctness bug — the per-coefficient means and sigmas were fit on
    the original basis and will produce meaningless z-scores under a different
    basis. The pre-flight check must reject the configuration.
    """


@dataclass(frozen=True, slots=True)
class FrozenZScoreStats:
    """Frozen per-coefficient z-score stats for BP/RP Hermite ratios + c0.

    Attributes
    ----------
    basis_fingerprint
        SHA-256 of the Hermite basis the stats were fit on. See
        :data:`arqueogal.data.gaia_xp.HERMITE_REPROJECTION_VERSION` and the
        ``_build_hermite_basis`` fingerprint. Must match the basis used to
        project Stream-3 coefficients.
    c0_bp_mean_log10, c0_bp_sigma_log10
        Z-score stats for ``log10(bp_coef_0)``. Same for RP.
    coef_norm_bp_mean, coef_norm_bp_sigma
        Per-coefficient stats for ``bp_coef_norm_{1..54} = c_i / c_0`` — shape
        ``(XP_COEFF_LEN - 1,)``; index ``i`` corresponds to coefficient
        ``i + 1`` (since coefficient 0 has its own stats and is not a ratio).
        Same for RP.
    sigma_floor
        Minimum sigma allowed during fit; coefficients below this threshold
        had their sigma replaced by 1.0 to avoid division blow-up. Preserved
        here so the inference path can warn if it sees a zero input.
    n_reference_population
        Size of the reference population used to fit the stats (diagnostic
        only — not used by :func:`apply_frozen_zscore`).
    reference_population_description
        Human-readable description of the cuts defining the reference
        population (diagnostic only).
    """

    basis_fingerprint: str
    c0_bp_mean_log10: float
    c0_bp_sigma_log10: float
    c0_rp_mean_log10: float
    c0_rp_sigma_log10: float
    coef_norm_bp_mean: np.ndarray  # shape (XP_COEFF_LEN - 1,)
    coef_norm_bp_sigma: np.ndarray
    coef_norm_rp_mean: np.ndarray
    coef_norm_rp_sigma: np.ndarray
    sigma_floor: float
    n_reference_population: int
    reference_population_description: str

    def __post_init__(self) -> None:
        expected = XP_COEFF_LEN - 1
        for name in (
            "coef_norm_bp_mean",
            "coef_norm_bp_sigma",
            "coef_norm_rp_mean",
            "coef_norm_rp_sigma",
        ):
            arr = getattr(self, name)
            if arr.shape != (expected,):
                raise ValueError(
                    f"FrozenZScoreStats.{name}: expected shape ({expected},), got {arr.shape}"
                )


def load_frozen_zscore_stats(provenance_path: str | Path) -> FrozenZScoreStats:
    """Load frozen z-score statistics from a Pipeline-1 provenance JSON.

    Parameters
    ----------
    provenance_path
        Path to the ``*.provenance.json`` sidecar of the Stream-1 features
        Parquet. Expected layout (as emitted by
        ``scripts/emit_stream1_with_hermite.py``):

        ``extra.basis_fingerprint_sha256``: full 64-char SHA-256 hex.
        ``extra.c0_zscore_frozen``: ``{"bp": {"mu_log10", "sigma_log10"},
        "rp": {"mu_log10", "sigma_log10"}, "n_reference_population", ...}``.
        ``extra.coef_norm_zscore_frozen``: ``{"bp": {"1": {"mu", "sigma"},
        ...}, "rp": {...}, "sigma_floor", ...}``.

    Returns
    -------
    FrozenZScoreStats
        Typed container for downstream :func:`apply_frozen_zscore` calls.
    """
    path = Path(provenance_path)
    payload = json.loads(path.read_text())
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise KeyError(f"{path}: missing 'extra' block in provenance JSON")

    fp = extra.get("basis_fingerprint_sha256")
    if not isinstance(fp, str) or not fp:
        raise KeyError(
            f"{path}: missing or empty extra.basis_fingerprint_sha256",
        )

    c0_block = extra.get("c0_zscore_frozen")
    if not isinstance(c0_block, dict):
        raise KeyError(f"{path}: missing extra.c0_zscore_frozen")
    coef_block = extra.get("coef_norm_zscore_frozen")
    if not isinstance(coef_block, dict):
        raise KeyError(f"{path}: missing extra.coef_norm_zscore_frozen")

    try:
        c0_bp_mu = float(c0_block["bp"]["mu_log10"])
        c0_bp_sigma = float(c0_block["bp"]["sigma_log10"])
        c0_rp_mu = float(c0_block["rp"]["mu_log10"])
        c0_rp_sigma = float(c0_block["rp"]["sigma_log10"])
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"{path}: extra.c0_zscore_frozen missing expected {{bp,rp}}."
            f"{{mu_log10,sigma_log10}} keys ({exc})"
        ) from exc

    bp_mu, bp_sigma = _unpack_coef_band(coef_block, "bp", path=path)
    rp_mu, rp_sigma = _unpack_coef_band(coef_block, "rp", path=path)

    sigma_floor = float(coef_block.get("sigma_floor", 0.0))
    n_ref = int(
        coef_block.get(
            "n_reference_population",
            c0_block.get("n_reference_population", 0),
        ),
    )
    ref_desc = str(
        coef_block.get(
            "reference_population",
            c0_block.get("reference_population", ""),
        ),
    )

    return FrozenZScoreStats(
        basis_fingerprint=fp,
        c0_bp_mean_log10=c0_bp_mu,
        c0_bp_sigma_log10=c0_bp_sigma,
        c0_rp_mean_log10=c0_rp_mu,
        c0_rp_sigma_log10=c0_rp_sigma,
        coef_norm_bp_mean=bp_mu,
        coef_norm_bp_sigma=bp_sigma,
        coef_norm_rp_mean=rp_mu,
        coef_norm_rp_sigma=rp_sigma,
        sigma_floor=sigma_floor,
        n_reference_population=n_ref,
        reference_population_description=ref_desc,
    )


def _unpack_coef_band(
    block: dict,
    band: str,
    *,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (mu, sigma) arrays of length XP_COEFF_LEN - 1 for one band."""
    band_block = block.get(band)
    if not isinstance(band_block, dict):
        raise KeyError(f"{path}: coef_norm_zscore_frozen.{band} missing or malformed")
    mu = np.full(XP_COEFF_LEN - 1, np.nan, dtype=np.float64)
    sigma = np.full(XP_COEFF_LEN - 1, np.nan, dtype=np.float64)
    for i in range(1, XP_COEFF_LEN):
        key = str(i)
        if key not in band_block:
            raise KeyError(f"{path}: coef_norm_zscore_frozen.{band} missing coefficient {key}")
        entry = band_block[key]
        try:
            mu[i - 1] = float(entry["mu"])
            sigma[i - 1] = float(entry["sigma"])
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"{path}: coef_norm_zscore_frozen.{band}.{key} missing {{mu, sigma}} ({exc})"
            ) from exc
    return mu, sigma


def verify_basis_fingerprint(
    current_fingerprint: str,
    stats: FrozenZScoreStats,
) -> None:
    """Raise :class:`FrozenStatsMismatchError` if the current basis differs.

    Must be called before any :func:`apply_frozen_zscore` invocation. The
    current fingerprint is typically
    ``gaia_xp._build_hermite_basis()["fingerprint_sha256"]``.
    """
    if current_fingerprint != stats.basis_fingerprint:
        raise FrozenStatsMismatchError(
            "Hermite basis fingerprint mismatch: stream-3 basis "
            f"{current_fingerprint!r} != frozen-stats basis "
            f"{stats.basis_fingerprint!r}. Re-projecting Stream-3 with a "
            "different basis would invalidate the frozen z-score means/sigmas. "
            "Re-emit the Stream-3 Hermite coefficients on the original basis "
            "(see scripts/emit_stream1_with_hermite.py for the canonical "
            "construction) before applying frozen stats."
        )


def apply_frozen_zscore(
    bp_coef_norm: np.ndarray,
    rp_coef_norm: np.ndarray,
    bp_c0_log: np.ndarray,
    rp_c0_log: np.ndarray,
    stats: FrozenZScoreStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply frozen z-score stats to Stream-3 Hermite coefficients.

    Parameters
    ----------
    bp_coef_norm, rp_coef_norm
        Per-star normalised ratios ``c_i / c_0`` for ``i = 1..54``, shape
        ``(N, XP_COEFF_LEN - 1)`` or ``(N, XP_COEFF_LEN)``. In the former the
        54 entries correspond to coefficients 1..54. In the latter the array
        carries coefficient 0 in column 0 (typically log10(c_0) after
        :func:`gaia_xp.normalise_xp`); this column is left untouched here —
        use ``bp_c0_log`` / ``rp_c0_log`` for the c_0 z-scoring.
    bp_c0_log, rp_c0_log
        Per-star ``log10(c_0)`` values, shape ``(N,)``.
    stats
        Frozen statistics as returned by :func:`load_frozen_zscore_stats`.
        The caller must first verify the basis fingerprint via
        :func:`verify_basis_fingerprint` — this function does not re-check.

    Returns
    -------
    tuple of 4 ndarrays
        ``(bp_norm_z, rp_norm_z, bp_c0_z, rp_c0_z)`` where the ``_norm_z``
        arrays share the shape of the input ratios (coefficient 0 column, if
        present, is passed through unchanged) and the ``_c0_z`` arrays are
        1-D.

    Notes
    -----
    This function **does not refit** any statistic. It applies
    ``(x - mu) / sigma`` element-wise using the frozen ``stats``.
    """
    bp_norm_z = _apply_coef_zscore(
        bp_coef_norm,
        stats.coef_norm_bp_mean,
        stats.coef_norm_bp_sigma,
        band="bp",
    )
    rp_norm_z = _apply_coef_zscore(
        rp_coef_norm,
        stats.coef_norm_rp_mean,
        stats.coef_norm_rp_sigma,
        band="rp",
    )
    bp_c0_z = _apply_scalar_zscore(
        bp_c0_log,
        stats.c0_bp_mean_log10,
        stats.c0_bp_sigma_log10,
        name="bp_c0_log",
    )
    rp_c0_z = _apply_scalar_zscore(
        rp_c0_log,
        stats.c0_rp_mean_log10,
        stats.c0_rp_sigma_log10,
        name="rp_c0_log",
    )
    return bp_norm_z, rp_norm_z, bp_c0_z, rp_c0_z


def _apply_coef_zscore(
    ratios: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    band: str,
) -> np.ndarray:
    """Apply (x - mu) / sigma across coefficients 1..54.

    If ``ratios`` has shape ``(N, 55)``, column 0 is passed through unchanged
    (the emit path reserves it for log10(c_0)). If shape is ``(N, 54)``, all
    54 columns are z-scored.
    """
    arr = np.asarray(ratios, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{band}_coef_norm must be 2-D; got shape {arr.shape}")
    n_ratios = XP_COEFF_LEN - 1
    if arr.shape[1] == n_ratios:
        out = (arr - mu) / sigma
        return out
    if arr.shape[1] == XP_COEFF_LEN:
        out = arr.copy()
        out[:, 1:] = (arr[:, 1:] - mu) / sigma
        return out
    raise ValueError(
        f"{band}_coef_norm: expected last-axis length "
        f"{n_ratios} or {XP_COEFF_LEN}, got {arr.shape[1]}"
    )


def _apply_scalar_zscore(
    x: np.ndarray,
    mu: float,
    sigma: float,
    *,
    name: str,
) -> np.ndarray:
    """Apply (x - mu) / sigma to a 1-D array."""
    if sigma <= 0:
        raise ValueError(f"{name}: frozen sigma must be positive, got {sigma}")
    return (np.asarray(x, dtype=np.float64) - mu) / sigma


__all__ = [
    "FrozenStatsMismatchError",
    "FrozenZScoreStats",
    "XP_COEFF_LEN",
    "apply_frozen_zscore",
    "load_frozen_zscore_stats",
    "verify_basis_fingerprint",
]
