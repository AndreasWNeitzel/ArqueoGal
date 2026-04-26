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

FROZEN_V1_BASIS_FINGERPRINT: Final[str] = (
    "0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7"
)
"""Basis fingerprint for the frozen v1 Hermite z-score statistics.

This is the canonical basis used to fit the frozen stats loaded by
:func:`load_frozen_zscore_stats`. Stream-3 inference must project
coefficients on this exact basis or the z-scored values will be invalid.
"""

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

    Attributes
    ----------
    expected : str
        The fingerprint of the basis the frozen stats were fit on.
    observed : str
        The fingerprint of the current Hermite basis.
    """

    def __init__(
        self,
        expected: str,
        observed: str,
        path: str | None = None,
    ) -> None:
        self.expected = expected
        self.observed = observed
        self.path = path
        msg = (
            f"Hermite basis fingerprint mismatch: "
            f"current basis {observed!r} != "
            f"frozen-stats basis {expected!r}. "
            f"Re-projecting Stream-3 with a different basis would invalidate "
            f"the frozen z-score means/sigmas. "
            f"Re-emit the Stream-3 Hermite coefficients on the original basis "
            f"(see scripts/emit_stream1_with_hermite.py for the canonical "
            f"construction) before applying frozen stats."
        )
        if path:
            msg = f"{path}: {msg}"
        super().__init__(msg)


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

    Raises
    ------
    FrozenStatsMismatchError
        If the current basis fingerprint does not match the frozen stats.
    """
    if current_fingerprint != stats.basis_fingerprint:
        raise FrozenStatsMismatchError(
            expected=stats.basis_fingerprint,
            observed=current_fingerprint,
        )


def assert_frozen_stats_match(
    expected_fingerprint: str | None = None,
) -> None:
    """Assert that the frozen v1 Hermite z-score stats are available and match.

    This is a high-level gate for inference drivers (called early, before
    data loading). It loads the frozen stats from the canonical provenance
    sidecar and verifies the Hermite basis fingerprint against the expected
    value. If the stats are unavailable or the fingerprint mismatches, a
    :class:`FrozenStatsMismatchError` is raised.

    Parameters
    ----------
    expected_fingerprint : str, optional
        The Hermite basis fingerprint to verify. If None, defaults to
        :data:`FROZEN_V1_BASIS_FINGERPRINT`.

    Raises
    ------
    FrozenStatsMismatchError
        If the provenance sidecar is missing or the fingerprint mismatches.
    RuntimeError
        If the provenance sidecar cannot be found.

    Notes
    -----
    This function assumes the provenance sidecar is at
    ``data/processed/pipeline1_features_stream1.provenance.json`` relative
    to the repo root. If the path is different, callers should invoke
    :func:`load_frozen_zscore_stats` and :func:`verify_basis_fingerprint`
    directly.
    """
    from pathlib import Path

    if expected_fingerprint is None:
        expected_fingerprint = FROZEN_V1_BASIS_FINGERPRINT

    # Locate the provenance sidecar — traverse upward to repo root.
    cwd = Path.cwd()
    repo_root = None
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root is None:
        raise RuntimeError(
            "assert_frozen_stats_match: unable to locate .git repo root. "
            "Provide the provenance sidecar path explicitly or run from within "
            "the ArqueoGal repository."
        )

    prov_path = repo_root / "data" / "processed" / "pipeline1_features_stream1.provenance.json"
    if not prov_path.exists():
        raise RuntimeError(
            f"assert_frozen_stats_match: provenance sidecar not found at {prov_path}. "
            "This is required for Stream-3 inference. See docs/data_acquisition.md."
        )

    stats = load_frozen_zscore_stats(prov_path)
    verify_basis_fingerprint(expected_fingerprint, stats)
    logger.info(
        "Frozen v1 Hermite z-score stats verified (fingerprint matches). "
        "Stream-3 inference may proceed."
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


def verify_frozen_stats_match_parquet(
    stats: FrozenZScoreStats,
    parquet_path: str | Path,
    *,
    sample_n: int = 50_000,
    tolerance_mean: float = 0.05,
    tolerance_sigma_ratio: float = 0.10,
) -> dict[str, float | bool | str | int]:
    """Defensively check that frozen stats match the training parquet's sample statistics.

    The basis-fingerprint check (:func:`verify_basis_fingerprint`) ensures the Hermite
    basis used to project the coefficients matches the basis the stats were fit on.
    But a future v1.1 refactor could silently re-fit stats on a different reference
    population without updating the fingerprint string in the provenance sidecar.
    This function provides a *behavioural* check: re-compute the per-coefficient
    sample mean and standard deviation on a random subsample of the training parquet,
    and verify they fall within tolerance of the frozen values.

    The check is intentionally cheap (N=50k subsample by default) so it can be
    invoked at the inference driver entry point without slowing the run.

    Parameters
    ----------
    stats : FrozenZScoreStats
        The frozen stats loaded via :func:`load_frozen_zscore_stats`.
    parquet_path : str | Path
        Path to the training Parquet (typically Stream 1 with raw or
        log10-transformed XP coefficients).
    sample_n : int
        Random-sample size for the reverification (default 50_000).
    tolerance_mean : float
        Allowed absolute deviation between frozen and resampled means
        (default 0.05; in z-score units this corresponds to a 0.05σ shift).
    tolerance_sigma_ratio : float
        Allowed multiplicative drift in sigma; if the resampled sigma is
        outside ``frozen_sigma * (1 ± tolerance_sigma_ratio)`` the check fails.

    Returns
    -------
    dict
        Diagnostic payload:
        - ``ok``: bool, True if all checks pass.
        - ``c0_bp_mean_log10_drift``, ``c0_rp_mean_log10_drift``: max |Δμ|.
        - ``c0_bp_sigma_log10_ratio``, ``c0_rp_sigma_log10_ratio``: max σ_resample/σ_frozen.
        - ``sample_n_actual``: number of rows actually sampled (may be less if parquet small).
        - ``parquet_path``: input path.

    Raises
    ------
    FrozenStatsMismatchError
        If any check exceeds tolerance.

    Notes
    -----
    The function only checks c0 means and sigmas (the scalar pair per band) — it
    does not re-check all 108 per-coefficient means/sigmas because (a) those are
    high-dimensional and noisy at small subsample sizes, and (b) the c0 stats
    are the most likely to drift since they encode the magnitude scale.

    Source: feature_preprocessing.md MAJOR-3 (defensive supplement to fingerprint check).
    """
    import pandas as pd

    pq_path = Path(parquet_path)
    if not pq_path.exists():
        raise FileNotFoundError(f"training parquet not found: {pq_path}")

    df = pd.read_parquet(pq_path, columns=["bp_coef_0", "rp_coef_0"])
    n_total = len(df)
    if n_total == 0:
        raise ValueError(f"empty training parquet: {pq_path}")
    n_use = min(sample_n, n_total)
    if n_use < n_total:
        df = df.sample(n=n_use, random_state=20260424)

    bp0 = np.log10(df["bp_coef_0"].to_numpy(dtype=np.float64).clip(min=1e-30))
    rp0 = np.log10(df["rp_coef_0"].to_numpy(dtype=np.float64).clip(min=1e-30))

    bp_mean_drift = abs(float(np.nanmean(bp0) - stats.c0_bp_mean_log10))
    rp_mean_drift = abs(float(np.nanmean(rp0) - stats.c0_rp_mean_log10))
    bp_sigma_resample = float(np.nanstd(bp0, ddof=1))
    rp_sigma_resample = float(np.nanstd(rp0, ddof=1))
    bp_sigma_ratio = bp_sigma_resample / max(stats.c0_bp_sigma_log10, 1e-30)
    rp_sigma_ratio = rp_sigma_resample / max(stats.c0_rp_sigma_log10, 1e-30)

    failures: list[str] = []
    if bp_mean_drift > tolerance_mean:
        failures.append(
            f"BP c0 mean drift {bp_mean_drift:.4f} > tolerance {tolerance_mean}",
        )
    if rp_mean_drift > tolerance_mean:
        failures.append(
            f"RP c0 mean drift {rp_mean_drift:.4f} > tolerance {tolerance_mean}",
        )
    if not (1 - tolerance_sigma_ratio) <= bp_sigma_ratio <= (1 + tolerance_sigma_ratio):
        failures.append(
            f"BP c0 sigma ratio {bp_sigma_ratio:.4f} outside "
            f"[{1 - tolerance_sigma_ratio}, {1 + tolerance_sigma_ratio}]",
        )
    if not (1 - tolerance_sigma_ratio) <= rp_sigma_ratio <= (1 + tolerance_sigma_ratio):
        failures.append(
            f"RP c0 sigma ratio {rp_sigma_ratio:.4f} outside "
            f"[{1 - tolerance_sigma_ratio}, {1 + tolerance_sigma_ratio}]",
        )

    diagnostic: dict[str, float | bool | str | int] = {
        "ok": not failures,
        "c0_bp_mean_log10_drift": bp_mean_drift,
        "c0_rp_mean_log10_drift": rp_mean_drift,
        "c0_bp_sigma_log10_ratio": bp_sigma_ratio,
        "c0_rp_sigma_log10_ratio": rp_sigma_ratio,
        "sample_n_actual": int(n_use),
        "parquet_path": str(pq_path),
    }

    if failures:
        message = "; ".join(failures)
        raise FrozenStatsMismatchError(
            expected=f"frozen stats from {stats.reference_population_description}",
            observed=f"resampled from {pq_path.name}: {message}",
            path=str(pq_path),
        )
    logger.info(
        "verify_frozen_stats_match_parquet OK on %d rows: BP Δμ=%.4f σ_ratio=%.3f, RP Δμ=%.4f σ_ratio=%.3f",
        n_use,
        bp_mean_drift,
        bp_sigma_ratio,
        rp_mean_drift,
        rp_sigma_ratio,
    )
    return diagnostic


__all__ = [
    "FROZEN_V1_BASIS_FINGERPRINT",
    "FrozenStatsMismatchError",
    "FrozenZScoreStats",
    "XP_COEFF_LEN",
    "apply_frozen_zscore",
    "assert_frozen_stats_match",
    "load_frozen_zscore_stats",
    "verify_basis_fingerprint",
    "verify_frozen_stats_match_parquet",
]
