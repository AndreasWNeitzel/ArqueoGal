"""Master-catalog schemas — §10 of data_acquisition.md.

Defines the column contracts for the analysis-ready Parquet products
that live in ``data/processed/``:

- :data:`PIPELINE1_TRAINING_SCHEMA` — APOGEE × Gaia × XP supervised set.
- :data:`PIPELINE1_INFERENCE_SCHEMA` — Gaia RGB+RC + XP, no APOGEE labels.
- :data:`PIPELINE2_FEATURES_SCHEMA` — retained for historical schema
  compatibility. Population classification moved to the separate
  **Starfold** repository on 2026-04-22; Starfold is responsible for its
  own chrono-chemo-kinematic feature-matrix assembly on top of this
  repo's Pipeline 1 prediction parquets. The schema is kept in the
  registry as a reference contract and is not written to by any
  production driver in this repo.

Each schema is a frozen :class:`MasterSchema` carrying a flat set of
``required`` column names plus ``optional`` columns that downstream
consumers may exploit but not demand. :meth:`MasterSchema.validate` checks
a DataFrame against the contract and raises :class:`SchemaError` on
missing required columns (the typical downstream guard before model I/O).

The XP coefficient arrays (``bp_coeffs_norm``, ``rp_coeffs_norm``) are
stored as list-typed columns of length 55 per row — the array-length check
is opt-in because it's O(N) and only used at ingest time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

# --- Gaia DR3 astrometric correlations (§3.6 / gaia_enrich.py) ---------------

GAIA_ASTROMETRY_COV_COLS: Final[tuple[str, ...]] = (
    "ra_dec_corr",
    "ra_parallax_corr",
    "ra_pmra_corr",
    "ra_pmdec_corr",
    "dec_parallax_corr",
    "dec_pmra_corr",
    "dec_pmdec_corr",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
)
"""Ten upper-triangular correlation coefficients from ``gaiadr3.gaia_source``."""

# --- XP coefficient shape (§6) ------------------------------------------------

XP_N_COEFFS: Final[int] = 55
"""Number of Hermite coefficients per band in Gaia DR3 XP (§6.1)."""

XP_ARRAY_COLS: Final[tuple[str, ...]] = (
    "bp_coeffs_norm",
    "rp_coeffs_norm",
    "bp_coeff_errs_norm",
    "rp_coeff_errs_norm",
)
"""XP columns whose cells are length-55 float32 lists.

Names match :mod:`arqueogal.data.gaia_xp` (``normalise_xp`` output) and
``data_acquisition.md`` §6.4 step 5. Keep these in sync with the XP
preprocessing module — the master-catalog builders join on them by name.
"""

XP_SCALAR_COLS: Final[tuple[str, ...]] = ("bp_c0_z", "rp_c0_z")
"""Z-scored log first-coefficient scalars (§6.4)."""

# --- §10.1 pipeline1_training ------------------------------------------------

_TRAINING_IDENTIFIERS = ("source_id", "apogee_id", "sdss_id")
_TRAINING_ASTROMETRY = (
    "ra",
    "dec",
    "parallax_corr",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
)
_TRAINING_PHOTOMETRY = (
    "phot_g_mean_mag_corr",
    "bp_rp",
    "bp_g",
    "g_rp",
)
_TRAINING_DISTANCE = ("r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo")
_TRAINING_EXTINCTION = (
    "av_los",
    "av_los_source",
    "av_nbhd_median",
    "av_nbhd_std",
)
"""§8.2 composes Edenhofer+2024 / Lallement+2022 / SFD into one ``av_los``
column plus a categorical ``av_los_source`` ∈ {0, 1, 2, -1} telling the
ML which map was used. §8.3 neighborhood-median A_G features round out the
extinction representation without needing per-map columns (which would be
mostly NaN — each star falls in exactly one distance bin)."""

# APOGEE labels — §3 / apogee_dr19.ABUNDANCE_ELEMENTS.
_APOGEE_ATMOS_LABELS = (
    "teff_apogee",
    "e_teff_apogee",
    "logg_apogee",
    "e_logg_apogee",
    "mh_apogee",
    "e_mh_apogee",
    "alpha_m_apogee",
    "e_alpha_m_apogee",
)
APOGEE_ELEMENT_LABELS: Final[tuple[str, ...]] = tuple(
    col
    for el in (
        "c",
        "n",
        "o",
        "na",
        "mg",
        "al",
        "si",
        "s",
        "k",
        "ca",
        "ti",
        "v",
        "cr",
        "mn",
        "fe",
        "ni",
        "ce",
    )
    for col in (f"{el}_h_apogee", f"e_{el}_h_apogee")
)
"""Per-element [X/H]_APOGEE labels + their uncertainties (Mészáros+2025-corrected)."""

_TRAINING_FLAGS = ("flag_bad", "ruwe")
_TRAINING_OPTIONAL_FLAGS = ("dist_conflict",)
"""``dist_conflict`` is only present when StarHorse2 is merged (§7.3);
training pipelines that skip SH2 won't carry it."""


@dataclass(frozen=True, slots=True)
class MasterSchema:
    """Column contract for a master-catalog Parquet file.

    ``required`` columns must all be present. ``optional`` columns are
    allowed but not required. ``array_cols`` is a subset of ``required``
    whose cells should be length-``array_length`` sequences; the check is
    opt-in because it's O(N).
    """

    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    array_cols: tuple[str, ...] = ()
    array_length: int | None = None
    version: int = 2
    last_modified: str = "2026-04-24"

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.required + self.optional

    def validate(self, df: pd.DataFrame, *, check_array_lengths: bool = False) -> None:
        """Raise :class:`SchemaError` if ``df`` violates the contract.

        Parameters
        ----------
        df
            Frame to check.
        check_array_lengths
            If True, also verify that every ``array_cols`` cell has length
            :data:`array_length`. Default False (O(N) scan).
        """
        missing = [c for c in self.required if c not in df.columns]
        if missing:
            raise SchemaError(f"{self.name}: missing required columns {missing}")
        if check_array_lengths and self.array_length is not None:
            for col in self.array_cols:
                if col not in df.columns:
                    continue
                bad = df[col].map(lambda v, _n=self.array_length: v is None or len(v) != _n)
                if bad.any():
                    raise SchemaError(
                        f"{self.name}: column {col!r} has "
                        f"{int(bad.sum())} rows whose length != {self.array_length}"
                    )


class SchemaError(ValueError):
    """Raised when a DataFrame does not satisfy a :class:`MasterSchema`."""


PIPELINE1_TRAINING_SCHEMA: Final[MasterSchema] = MasterSchema(
    name="pipeline1_training",
    required=(
        *_TRAINING_IDENTIFIERS,
        *_TRAINING_ASTROMETRY,
        *GAIA_ASTROMETRY_COV_COLS,
        *_TRAINING_PHOTOMETRY,
        *XP_ARRAY_COLS,
        *XP_SCALAR_COLS,
        *_TRAINING_DISTANCE,
        *_TRAINING_EXTINCTION,
        *_APOGEE_ATMOS_LABELS,
        *APOGEE_ELEMENT_LABELS,
        *_TRAINING_FLAGS,
    ),
    optional=_TRAINING_OPTIONAL_FLAGS,
    array_cols=XP_ARRAY_COLS,
    array_length=XP_N_COEFFS,
    version=3,
    last_modified="2026-04-24",
)
"""§10.1 training-set schema (~700 k rows)."""

# --- §10.2 pipeline1_inference -----------------------------------------------

_ANDRAE_DIAG_COLS: Final[tuple[str, ...]] = (
    "teff_xgboost",
    "logg_xgboost",
    "mh_xgboost",
)
"""Andrae+2023 labels carried through as inference-time diagnostics (§10.2)."""

_ANDRAE_OPTIONAL_DIAG_COLS: Final[tuple[str, ...]] = ("evolutionary_stage_andrae",)
"""Andrae's published evolutionary-stage class label is not loaded by the
default :mod:`arqueogal.data.andrae2023` column set. Keep the schema slot
as optional so callers who opt into the extra column still validate."""

_PIPELINE1_RELEASE_COLS: Final[tuple[str, ...]] = (
    "release_tier",
    "release_tier__teff",
    "release_tier__logg",
    "release_tier__mh",
    "release_tier__alpha_m",
    "release_tier__mg_h",
    "xp_abundance_type__teff",
    "xp_abundance_type__logg",
    "xp_abundance_type__mh",
    "xp_abundance_type__alpha_m",
    "xp_abundance_type__mg_h",
    "kin_ood_flag",
    "g_mag_bin",
    "dist_prior_dominated",
    "ood_aux_mahalanobis_flag",
    # v4 (Phase A2-followup-2, 2026-04-25): per-element σ-inflation caveat.
    "prediction_sigma_inflated__teff",
    "prediction_sigma_inflated__logg",
    "prediction_sigma_inflated__mh",
    "prediction_sigma_inflated__alpha_m",
    "prediction_sigma_inflated__mg_h",
    "prediction_sigma_inflated_any",
)
"""Release-tier catalog columns added after inference.

v2 (Phase A2, 2026-04-24): release_tier, xp_abundance_type__<element>, kin_ood_flag, g_mag_bin.
v3 (Phase A2-followup, 2026-04-24): per-element release_tier__<element>, dist_prior_dominated,
ood_aux_mahalanobis_flag. The composite release_tier becomes the row-max of per-element
tiers (most conservative). The per-element tiers consume xp_abundance_type__<element> and
kin_ood_flag to demote aux-assisted elements when the kinematic-OOD flag fires (META_META
§14.3, outlier_flagging.md).
v4 (Phase A2-followup-2, 2026-04-25): per-element ``prediction_sigma_inflated__<elem>``
flag (5 columns) plus the ``prediction_sigma_inflated_any`` row-OR aggregate. When the
regression-head σ for an element exceeds its prior-collapse threshold (Teff: 150 K;
logg: 0.30 dex; [M/H], [Mg/H]: 0.20 dex; [α/M] originally 0.10 dex — see v5), that
element demotes to Tier 2 — see release.assign_prediction_sigma_inflated and
HIGH_SIGMA_RESCUE_REPORT.md.
v5 (per-cell-gate ablation, 2026-04-26): simplified release-tier composition; α/M
σ-threshold tightened 0.10 → 0.05 dex; ``mode_ambiguous_flag`` becomes a per-element
caveat on [α/M] only; ``regime_b_flag``, ``ood_disagreement_flag``, ``aux_missing_any``,
``dist_prior_dominated``, ``latent_support_flag``, ``ood_aux_mahalanobis_flag`` retired
from tier gating but kept as diagnostic columns. See ADR-0015 and
release/test_ablations_2026-04-26/REPORT.md."""

_PIPELINE1_KNN_RESCUE_COLS: Final[tuple[str, ...]] = (
    # Per-element neighbour-label statistics emitted by knn_rescue.summarize_neighbors.
    "knn_teff_med",
    "knn_teff_p25",
    "knn_teff_p75",
    "knn_teff_iqr",
    "knn_teff_std",
    "knn_logg_med",
    "knn_logg_p25",
    "knn_logg_p75",
    "knn_logg_iqr",
    "knn_logg_std",
    "knn_mh_med",
    "knn_mh_p25",
    "knn_mh_p75",
    "knn_mh_iqr",
    "knn_mh_std",
    "knn_alpha_m_med",
    "knn_alpha_m_p25",
    "knn_alpha_m_p75",
    "knn_alpha_m_iqr",
    "knn_alpha_m_std",
    "knn_mg_h_med",
    "knn_mg_h_p25",
    "knn_mg_h_p75",
    "knn_mg_h_iqr",
    "knn_mg_h_std",
    "knn_top_distance",
    "knn_median_distance",
)
"""Latent-kNN rescue columns (v4-followup, 2026-04-25).

For each Stream 3 star, ``knn_rescue.gpu_knn_search`` finds the K nearest training-set
neighbours in encoder-projection space (cosine distance, K=50) and ``summarize_neighbors``
reports their actual APOGEE label distribution. The kNN bypasses the regression head
entirely and is bounded by training-set support (no extrapolation), so it is the
preferred fallback for stars whose regression-head σ exceeds the prior-collapse
threshold (HIGH_SIGMA_RESCUE_REPORT.md, 2026-04-25). Hybrid composition with the
regression head is implemented separately in release_pipeline.run_hybrid_release_pipeline."""

_PIPELINE1_HYBRID_COLS: Final[tuple[str, ...]] = (
    # v5: hybrid (regressor + kNN) composed columns from
    # release_pipeline.attach_hybrid_columns. Per-element pred / sigma /
    # source-of-prediction / hybrid tier.
    "teff_hybrid_pred",
    "teff_hybrid_sigma",
    "teff_hybrid_source",
    "teff_hybrid_tier",
    "logg_hybrid_pred",
    "logg_hybrid_sigma",
    "logg_hybrid_source",
    "logg_hybrid_tier",
    "mh_hybrid_pred",
    "mh_hybrid_sigma",
    "mh_hybrid_source",
    "mh_hybrid_tier",
    "alpha_m_hybrid_pred",
    "alpha_m_hybrid_sigma",
    "alpha_m_hybrid_source",
    "alpha_m_hybrid_tier",
    "mg_h_hybrid_pred",
    "mg_h_hybrid_sigma",
    "mg_h_hybrid_source",
    "mg_h_hybrid_tier",
)
"""Hybrid composer columns (v5, 2026-04-25).

The hybrid surface composes the regressor and the latent-kNN: regressor is used
when its σ is below the per-element prior-collapse threshold; kNN-median
substitutes when σ is above-threshold AND kNN columns are populated;
``regressor_caveat`` is the fallback label when kNN is unavailable. See
``release_pipeline.attach_hybrid_columns``."""

PIPELINE1_INFERENCE_SCHEMA: Final[MasterSchema] = MasterSchema(
    name="pipeline1_inference",
    required=(
        "source_id",
        *_TRAINING_ASTROMETRY,
        *GAIA_ASTROMETRY_COV_COLS,
        *_TRAINING_PHOTOMETRY,
        *XP_ARRAY_COLS,
        *XP_SCALAR_COLS,
        *_TRAINING_DISTANCE,
        *_TRAINING_EXTINCTION,
        *_ANDRAE_DIAG_COLS,
        *_TRAINING_FLAGS,
    ),
    optional=(
        *_TRAINING_OPTIONAL_FLAGS,
        *_ANDRAE_OPTIONAL_DIAG_COLS,
        *_PIPELINE1_RELEASE_COLS,
        *_PIPELINE1_KNN_RESCUE_COLS,
        *_PIPELINE1_HYBRID_COLS,
    ),
    array_cols=XP_ARRAY_COLS,
    array_length=XP_N_COEFFS,
    version=5,
    last_modified="2026-04-25",
)
"""§10.2 inference-set schema (~1.5 M rows).

Same shape as :data:`PIPELINE1_TRAINING_SCHEMA` minus the APOGEE-sourced
labels; Andrae+2023 labels are kept as cross-reference diagnostics.
"""

# --- legacy population-classifier feature schema (historical) ---------------
# Kept as a reference contract; see module docstring. Not written by any
# production driver in this repo after 2026-04-22.

_PIPELINE2_CHEMO = (
    "fe_h",
    "fe_h_err",
    "mg_fe",
    "mg_fe_err",
    "al_fe",
    "al_fe_err",
    "c_n",
    "c_n_err",
)
_PIPELINE2_AGE = ("age", "age_err")
_PIPELINE2_KINEMATICS = (
    "J_R",
    "J_z",
    "L_z",
    "ecc",
    "r_peri",
    "r_apo",
    "z_max",
    "E",
)
_PIPELINE2_NEITZEL2025_COMPAT = ("v_phi", "sqrt_u2_plus_w2")

PIPELINE2_FEATURES_SCHEMA: Final[MasterSchema] = MasterSchema(
    name="pipeline2_features",
    required=(
        "source_id",
        *_PIPELINE2_CHEMO,
        *_PIPELINE2_KINEMATICS,
        *_PIPELINE2_NEITZEL2025_COMPAT,
        "evolutionary_stage_prob",
    ),
    optional=_PIPELINE2_AGE,
    # ``age`` / ``age_err`` are optional in the near-term product because
    # Task 4 (asteroseismic age pipeline) has not yet landed — schema
    # acceptance should not block downstream clustering development (now
    # in Starfold) on its absence.
)
"""Legacy chrono-chemo-kinematic feature-vector schema (~1.5 M rows).

Population classification was moved to the separate **Starfold** repository
on 2026-04-22. This schema is retained as a reference contract for any
downstream consumer that still wants to materialise the same shape; the
active feature-matrix definition lives in Starfold.
"""


SCHEMAS: Final[dict[str, MasterSchema]] = {
    s.name: s
    for s in (
        PIPELINE1_TRAINING_SCHEMA,
        PIPELINE1_INFERENCE_SCHEMA,
        PIPELINE2_FEATURES_SCHEMA,
    )
}
"""Name → schema registry, for provenance logging and CLI dispatch."""


__all__ = [
    "APOGEE_ELEMENT_LABELS",
    "GAIA_ASTROMETRY_COV_COLS",
    "PIPELINE1_INFERENCE_SCHEMA",
    "PIPELINE1_TRAINING_SCHEMA",
    "PIPELINE2_FEATURES_SCHEMA",
    "SCHEMAS",
    "XP_ARRAY_COLS",
    "XP_N_COEFFS",
    "XP_SCALAR_COLS",
    "MasterSchema",
    "SchemaError",
]
