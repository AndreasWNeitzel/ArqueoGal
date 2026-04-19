"""Pipeline 2 feature vector construction — research_brief §10.2 / §10.3.

The main feature set replaces the Neitzel+2025 5-D vector
``(age, [Fe/H], [α/Fe], V_φ, √(U²+W²))`` with the 10–11-D chrono-chemo-kinematic
vector specified in research_brief §10.2:

    (age, [Fe/H], [Mg/Fe], [Al/Fe] where available, [C/N] RGB only,
     J_R, J_z, L_z, ecc, E)

plus optional ``r_peri / r_apo`` if both are non-null. The 5-D Neitzel+2025
feature set is kept as a backwards-compatibility preset so D5.1 reproduction
runs can be re-run without changing this module.

Design notes
------------
- **Evolutionary-stage gating on [C/N]**: red-clump and sub-giant stars
  violate the [C/N]-age calibration (research_brief §3.2). Stars
  outside the RGB validity window have their ``c_n`` feature masked out via
  ``evol_stage_probs``.
- **Standardisation**: a per-column ``StandardScaler`` fit on the training
  subset. Fit parameters are stored in :class:`FeatureMatrix` so inference
  applies the same transform.
- **Missing values**: rows with *any* required feature NaN are dropped. The
  ``include_mask`` in :class:`FeatureMatrix` records which input rows
  survived; downstream callers re-include excluded stars at end with
  posterior-probability dilution (research_brief §10.1).
- No ML, no GPU dependency — pure numpy/pandas so tests stay fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

FeatureSetName = Literal["main", "baseline_neitzel2025"]


# --- presets ---------------------------------------------------------------

MAIN_FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "fe_h",
    "mg_fe",
    "al_fe",
    "c_n",
    "J_R",
    "J_z",
    "L_z",
    "ecc",
    "E",
)
"""Research_brief §10.2 main-pipeline feature vector (10-D).

``r_peri``/``r_apo`` are optional extras the caller can append via
:class:`FeatureSpec`. Kept out of the default to avoid the redundancy with
``ecc`` (ε is a one-number summary of peri/apo).
"""

BASELINE_NEITZEL2025_COLUMNS: tuple[str, ...] = (
    "age",
    "fe_h",
    "alpha_m",
    "v_phi",
    "sqrt_u2_plus_w2",
)
"""Neitzel+2025 (A&A 695, A243) 5-D feature vector, for reproduction runs."""


@dataclass(frozen=True)
class FeatureSpec:
    """Which features to extract + which need evolutionary-stage gating.

    ``gated_columns`` lists feature-column names that are only valid for
    RGB stars and must be masked out / row-excluded otherwise. ``c_n`` is
    the canonical example per research_brief §3.2.
    """

    columns: tuple[str, ...]
    gated_columns: tuple[str, ...] = ("c_n",)
    name: str = "main"

    @classmethod
    def main(cls) -> FeatureSpec:
        return cls(columns=MAIN_FEATURE_COLUMNS, name="main")

    @classmethod
    def baseline_neitzel2025(cls) -> FeatureSpec:
        # No gating needed: Neitzel+2025 didn't use [C/N].
        return cls(
            columns=BASELINE_NEITZEL2025_COLUMNS,
            gated_columns=(),
            name="baseline_neitzel2025",
        )


# --- feature matrix --------------------------------------------------------

@dataclass
class FeatureMatrix:
    """Standardised feature tensor + the scaler that produced it.

    ``X`` is ``(N_included, D)`` float32; ``include_mask`` is a boolean
    array over the *input* rows (so ``df.iloc[include_mask]`` recovers the
    rows that made it through). ``mean`` / ``std`` are the per-column
    standardiser parameters — store them alongside any trained embedding so
    the scaler can be re-applied at inference.
    """

    X: np.ndarray
    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    include_mask: np.ndarray
    spec_name: str = "main"
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if self.X.shape[1] != len(self.columns):
            raise ValueError(
                f"X has {self.X.shape[1]} cols, columns spec has {len(self.columns)}",
            )
        if self.mean.shape != (len(self.columns),):
            raise ValueError(
                f"mean shape {self.mean.shape} != ({len(self.columns)},)",
            )
        if self.std.shape != (len(self.columns),):
            raise ValueError(
                f"std shape {self.std.shape} != ({len(self.columns)},)",
            )

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


# --- core routines ---------------------------------------------------------

def apply_c_n_gate(
    df: pd.DataFrame,
    *,
    evol_stage_probs: np.ndarray | None,
    rgb_prob_threshold: float = 0.5,
    cn_column: str = "c_n",
) -> pd.DataFrame:
    """Mask ``cn_column`` to NaN for non-RGB stars.

    ``evol_stage_probs`` is a ``(N,)`` array of P(RGB) from Pipeline 1's
    evolutionary-stage head. Stars with P(RGB) < ``rgb_prob_threshold`` have
    their [C/N] value replaced by NaN — downstream drop-na logic then
    excludes them cleanly.

    If ``evol_stage_probs`` is None the frame is returned unchanged (useful
    when the caller has *already* pre-gated, or when [C/N] isn't in the spec).
    """
    if evol_stage_probs is None or cn_column not in df.columns:
        return df
    if len(evol_stage_probs) != len(df):
        raise ValueError(
            f"evol_stage_probs length {len(evol_stage_probs)} != df length {len(df)}",
        )
    gated = df.copy()
    gated.loc[evol_stage_probs < rgb_prob_threshold, cn_column] = np.nan
    return gated


def standardize(
    X: np.ndarray, *, mean: np.ndarray | None = None, std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column standardisation. Fit if ``mean``/``std`` are None, else apply.

    Zero-variance columns get a ``std = 1`` fallback so they pass through
    unchanged instead of producing NaN.
    """
    if mean is None or std is None:
        mean = np.nanmean(X, axis=0).astype(np.float64)
        std = np.nanstd(X, axis=0).astype(np.float64)
    std_safe = np.where(std > 0.0, std, 1.0)
    X_scaled = ((X - mean) / std_safe).astype(np.float32)
    return X_scaled, mean.astype(np.float64), std.astype(np.float64)


def build_feature_matrix(  # noqa: PLR0913 — fit/apply paths share one orchestrator
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
    *,
    evol_stage_probs: np.ndarray | None = None,
    rgb_prob_threshold: float = 0.5,
    fit_scaler: bool = True,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> FeatureMatrix:
    """Extract + standardise the feature matrix from a catalogue frame.

    Parameters
    ----------
    df
        Catalogue frame — must contain every column in ``spec.columns``.
    spec
        Which feature set to pull. Defaults to the §10.2 main 10-D set.
    evol_stage_probs
        Optional ``(N,)`` P(RGB) array used to gate ``c_n`` if it's in the spec.
    fit_scaler
        If True (training), fit a new scaler from ``df``. If False (inference),
        ``mean`` and ``std`` must be supplied.
    """
    if spec is None:
        spec = FeatureSpec.main()
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if not fit_scaler and (mean is None or std is None):
        raise ValueError("fit_scaler=False requires mean and std")

    gated = apply_c_n_gate(
        df, evol_stage_probs=evol_stage_probs,
        rgb_prob_threshold=rgb_prob_threshold,
    )
    raw = gated.loc[:, list(spec.columns)].to_numpy(dtype=np.float64)
    include_mask = ~np.any(np.isnan(raw), axis=1)
    X_included = raw[include_mask]

    if fit_scaler:
        X_scaled, mu, sigma = standardize(X_included)
    else:
        X_scaled, mu, sigma = standardize(X_included, mean=mean, std=std)

    return FeatureMatrix(
        X=X_scaled,
        columns=tuple(spec.columns),
        mean=mu,
        std=sigma,
        include_mask=include_mask,
        spec_name=spec.name,
    )


__all__ = [
    "BASELINE_NEITZEL2025_COLUMNS",
    "MAIN_FEATURE_COLUMNS",
    "FeatureMatrix",
    "FeatureSpec",
    "apply_c_n_gate",
    "build_feature_matrix",
    "standardize",
]
