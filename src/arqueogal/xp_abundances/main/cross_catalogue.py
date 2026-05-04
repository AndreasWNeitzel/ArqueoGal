"""Cross-catalogue consistency validation (§3.3 Test 6 / methods-paper).

This is the catalogue-level Test 6: a comprehensive comparison harness that
takes the ArqueoGal Pipeline-1 release and a set of external reference
catalogues (AspGap, SHBoost, Guiglion+2024, Andrae+2023, Zhang+2023,
Fallows-Sanders 2024, GALAH DR4) and produces

- per-(label, catalogue, magnitude bin) Bland-Altman statistics: mean bias,
  scatter, σ-ratio (joint-uncertainty calibration), MAD-robust scatter,
  coverage of the released σ at 68 / 95 / 99 % against the residual,
  Pearson correlation, n_overlap;
- per-(label, catalogue) metallicity-dependent and Teff-dependent bias
  curves;
- per-(label, catalogue) Teff×log g cell heatmaps of bias and scatter;
- a one-vs-many summary table with rank statistics (where ArqueoGal sits in
  the bias / scatter ordering across catalogues);
- the matched-σ subsample test: re-evaluate every metric with stars
  filtered down to a σ-percentile that matches the reference catalogue, so
  the σ-inflation Tier-2 demotion (see ``release.py`` selection-bias
  caveat) is unambiguous in the methods paper.

The module is **stats only** — the seven plot families live in
:mod:`arqueogal.xp_abundances.main.cross_catalogue_plots` and the CLI
driver in ``scripts/run_cross_catalogue_validation.py``. Cross-matching
is the caller's responsibility (each reference catalogue ships with its
own source-id column convention); this module operates on already-aligned
arrays.

References
----------
- ``docs/protocols/cross_catalogue_test6.md`` (acceptance gates).
- ``docs/plan/02_pipeline1_audit.md`` §6 (audit-state evolution).
- ``docs/research_brief.md`` §3.3 (six-test promotion protocol).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical ArqueoGal label set. Keys map element → (pred_col, sigma_col,
# unit, default APOGEE-σ scale used by the §3.3 test 5 acceptance gate).
# Extended to 21 elements in v6 (2026-04-28).
LABEL_SCHEMA: Final[dict[str, dict[str, object]]] = {
    "teff": {
        "pred": "teff_pred",
        "sigma": "teff_sigma",
        "unit": "K",
        "apogee_sigma": 80.0,
        "bias_limit": 50.0,
    },
    "logg": {
        "pred": "logg_pred",
        "sigma": "logg_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "mh": {
        "pred": "mh_pred",
        "sigma": "mh_sigma",
        "unit": "dex",
        "apogee_sigma": 0.05,
        "bias_limit": 0.05,
    },
    "fe_h": {
        "pred": "fe_h_pred",
        "sigma": "fe_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.05,
        "bias_limit": 0.05,
    },
    "alpha_m": {
        "pred": "alpha_m_pred",
        "sigma": "alpha_m_sigma",
        "unit": "dex",
        "apogee_sigma": 0.04,
        "bias_limit": 0.05,
    },
    "mg_h": {
        "pred": "mg_h_pred",
        "sigma": "mg_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.05,
        "bias_limit": 0.05,
    },
    "c_h": {
        "pred": "c_h_pred",
        "sigma": "c_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "n_h": {
        "pred": "n_h_pred",
        "sigma": "n_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "o_h": {
        "pred": "o_h_pred",
        "sigma": "o_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "na_h": {
        "pred": "na_h_pred",
        "sigma": "na_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "al_h": {
        "pred": "al_h_pred",
        "sigma": "al_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "si_h": {
        "pred": "si_h_pred",
        "sigma": "si_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "s_h": {
        "pred": "s_h_pred",
        "sigma": "s_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "k_h": {
        "pred": "k_h_pred",
        "sigma": "k_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "ca_h": {
        "pred": "ca_h_pred",
        "sigma": "ca_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "ti_h": {
        "pred": "ti_h_pred",
        "sigma": "ti_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "v_h": {
        "pred": "v_h_pred",
        "sigma": "v_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "cr_h": {
        "pred": "cr_h_pred",
        "sigma": "cr_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "mn_h": {
        "pred": "mn_h_pred",
        "sigma": "mn_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "ni_h": {
        "pred": "ni_h_pred",
        "sigma": "ni_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
    "ce_h": {
        "pred": "ce_h_pred",
        "sigma": "ce_h_sigma",
        "unit": "dex",
        "apogee_sigma": 0.10,
        "bias_limit": 0.10,
    },
}

# Magnitude bins for per-G stratified Bland-Altman analysis. Boundaries
# chosen to track Gaia DR3's optical-depth regime change at G ≈ 16
# (S/N drops from ~100 to ~40 in BP/RP).
DEFAULT_MAG_BINS: Final[tuple[tuple[float, float, str], ...]] = (
    (0.0, 15.0, "bright"),
    (15.0, 16.0, "intermediate"),
    (16.0, 21.0, "faint"),
)

# Coverage levels at which we evaluate the released σ (one-sided).
DEFAULT_COVERAGE_LEVELS: Final[tuple[float, ...]] = (0.68, 0.95, 0.99)


@dataclass(frozen=True, slots=True)
class CatalogueBinding:
    """Describes how to read one external catalogue into a comparison frame.

    The *binding* is just metadata; cross-matching to ``source_id`` is the
    caller's job (each reference catalogue carries a different source-id
    convention — DR3 ``source_id``, 2MASS ``designation``, etc.).

    Attributes
    ----------
    name
        Short tag used in plots, JSON, and table rows (``"aspgap"``,
        ``"shboost"``, ...).
    column_for
        Mapping from ArqueoGal canonical element key (one of
        :data:`LABEL_SCHEMA`) to the external-catalogue column name. Keys
        absent from this dict mean "this catalogue does not publish this
        element" and the comparison is skipped for the missing entries.
    sigma_for
        Optional mapping to per-star uncertainty columns in the external
        catalogue. Used for the σ-ratio diagnostic; if absent for an
        element, σ_external is treated as zero (i.e. all the residual
        scatter is attributed to ArqueoGal).
    citation
        Short citation tag (``"Li+2024 (AspGap)"``) for plot legends and
        the methods-paper table.
    """

    name: str
    column_for: dict[str, str]
    sigma_for: dict[str, str] = field(default_factory=dict)
    citation: str = ""


@dataclass(frozen=True, slots=True)
class BlandAltmanCell:
    """Bland-Altman statistics on one (label, catalogue, mag-bin) cell.

    ``residual`` is defined as ``arqueogal_pred − reference_value`` so a
    positive bias means ArqueoGal predicts higher than the reference.
    ``sigma_ratio`` is ``stddev(residual) / sqrt(σ_pipeline² + σ_ref²)``
    and tracks joint-σ calibration; values much larger than 1 indicate
    that the released σ is under-estimated relative to the empirical
    scatter.
    """

    label: str
    catalogue: str
    mag_bin: str
    n: int
    bias: float
    bias_se: float  # standard error of the mean residual
    scatter: float  # sample stddev of the residual
    mad_scatter: float  # 1.4826 × MAD, robust to outliers
    sigma_ratio: float
    coverage: dict[str, float]  # "0.68" -> empirical fraction within ±σ_pipeline
    pearson: float

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        # Non-finite floats break json.dump; coerce.
        for k, v in out.items():
            if isinstance(v, float) and not np.isfinite(v):
                out[k] = None
        return out


@dataclass(frozen=True, slots=True)
class CrossCatalogueReport:
    """Top-level Test-6 deliverable.

    Carries enough information to (a) drive the seven plot families in
    :mod:`.cross_catalogue_plots` without re-touching the underlying
    parquet files, and (b) emit a methods-paper-ready JSON sidecar.
    """

    # Per-cell statistics in long form for easy plotting and pandas pivot.
    cells: list[BlandAltmanCell]
    # Per-(label, catalogue) trend curves, keyed (label, catalogue):
    # bias_vs_x[(label, catalogue)] = {"x": ..., "bias": ..., "scatter": ...}
    bias_vs_mh: dict[tuple[str, str], dict[str, np.ndarray]]
    bias_vs_teff: dict[tuple[str, str], dict[str, np.ndarray]]
    # Per-(label, catalogue) Teff×log g heatmaps:
    cell_heatmaps: dict[tuple[str, str], dict[str, np.ndarray]]
    # Per-cell pass/fail against the §3.3 acceptance gate.
    passes: dict[tuple[str, str, str], bool]
    # Configuration echo for reproducibility.
    config: dict[str, object]

    def to_json(self, path: Path) -> Path:
        """Write the cell-level table + config echo to JSON.

        Trend curves and heatmaps are NumPy arrays and are written
        side-by-side as ``.npz`` to keep this JSON small enough to read
        in a paper appendix.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cells": [c.as_dict() for c in self.cells],
            "passes": {f"{label}|{cat}|{mag}": ok for (label, cat, mag), ok in self.passes.items()},
            "config": self.config,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        # Companion .npz with the curves and heatmaps.
        npz_path = path.with_suffix(".npz")
        flat: dict[str, np.ndarray] = {}
        for (label, cat), curves in self.bias_vs_mh.items():
            for k, v in curves.items():
                flat[f"mh|{label}|{cat}|{k}"] = v
        for (label, cat), curves in self.bias_vs_teff.items():
            for k, v in curves.items():
                flat[f"teff|{label}|{cat}|{k}"] = v
        for (label, cat), heatmap in self.cell_heatmaps.items():
            for k, v in heatmap.items():
                flat[f"heatmap|{label}|{cat}|{k}"] = v
        if flat:
            np.savez_compressed(npz_path, **flat)
        return path


@dataclass(frozen=True, slots=True)
class ApogeePipelineRanking:
    """Pipeline performance metrics and rank on APOGEE DR19 benchmark."""

    pipeline_name: str
    n_overlap: int
    bias: dict[str, float]  # label → mean residual
    scatter: dict[str, float]  # label → sample stddev
    rmse: dict[str, float]  # label → sqrt(bias^2 + scatter^2)
    sigma_ratio: dict[str, float]  # label → σ_resid / sqrt(σ_pipe^2 + σ_apogee^2)
    mad_scatter: dict[str, float]  # label → 1.4826*MAD (robust scatter)
    pearson: dict[str, float]  # label → Pearson correlation


@dataclass(frozen=True, slots=True)
class ApogeeBenchmarkReport:
    """APOGEE DR19 benchmark deliverable (methods-paper Figure 8).

    Compares ArqueoGal + external pipelines (AspGap, SHBoost, Guiglion+2024, etc.)
    on their common APOGEE DR19 overlap subset. Per-pipeline per-element metrics,
    global RMSE ranking, and configuration echo for reproducibility.

    The key insight: ArqueoGal is ranked by |bias| first, then scatter, so
    papers cite "Figure 8: ArqueoGal vs field" as the ground-truth comparison.
    """

    pipelines: list[ApogeePipelineRanking]
    # Per-(label, pipeline) detailed residual stats for plotting (per-element RMSE bar chart)
    per_label_stats: dict[str, dict[str, ApogeePipelineRanking]]
    config: dict[str, object]

    def to_json(self, path: Path) -> Path:
        """Write benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten per-pipeline metrics for JSON serialization
        pipeline_rows = []
        for pr in self.pipelines:
            row = {
                "pipeline_name": pr.pipeline_name,
                "n_overlap": int(pr.n_overlap),
                "bias": {k: (None if not np.isfinite(v) else float(v)) for k, v in pr.bias.items()},
                "scatter": {
                    k: (None if not np.isfinite(v) else float(v)) for k, v in pr.scatter.items()
                },
                "rmse": {k: (None if not np.isfinite(v) else float(v)) for k, v in pr.rmse.items()},
                "sigma_ratio": {
                    k: (None if not np.isfinite(v) else float(v)) for k, v in pr.sigma_ratio.items()
                },
                "mad_scatter": {
                    k: (None if not np.isfinite(v) else float(v)) for k, v in pr.mad_scatter.items()
                },
                "pearson": {
                    k: (None if not np.isfinite(v) else float(v)) for k, v in pr.pearson.items()
                },
            }
            pipeline_rows.append(row)

        payload = {
            "pipelines": pipeline_rows,
            "config": self.config,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path


# --- Statistics primitives ----------------------------------------------------


def _mad_scatter(residual: np.ndarray) -> float:
    """1.4826 × MAD scatter (robust σ estimator for Gaussian residuals)."""
    if residual.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def _coverage(
    residual: np.ndarray, sigma: np.ndarray, levels: tuple[float, ...]
) -> dict[str, float]:
    """Empirical fraction of residuals within ±z·σ for each level.

    Implemented as a marginal one-sided coverage so the result matches the
    methods-paper convention: fraction of stars where ``|y − μ| ≤ k·σ``
    for ``k ∈ {1, 2, 3}`` corresponding to 68 / 95 / 99 %.
    """
    if residual.size == 0:
        return {f"{lvl:.2f}": float("nan") for lvl in levels}
    # Convert one-sided coverage level (e.g. 0.68) to z-scale (≈1.0σ).
    out: dict[str, float] = {}
    for lvl in levels:
        # Two-sided coverage at level `lvl` corresponds to z = Φ⁻¹((1+lvl)/2).
        # For 0.68, 0.95, 0.99 this is 0.994, 1.960, 2.576.
        from scipy.stats import norm  # lazy: cross-cat module is import-light

        z = float(norm.ppf((1.0 + lvl) / 2.0))
        # Avoid divide-by-zero where σ = 0 by treating those stars as outside.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sigma > 0.0, np.abs(residual) / sigma, np.inf)
        out[f"{lvl:.2f}"] = float(np.mean(ratio <= z))
    return out


def _bland_altman_one_cell(  # noqa: PLR0913 — orthogonal scientific knobs
    pred: np.ndarray,
    ref: np.ndarray,
    sigma_pipeline: np.ndarray,
    sigma_ref: np.ndarray,
    *,
    label: str,
    catalogue: str,
    mag_bin: str,
    coverage_levels: tuple[float, ...],
) -> BlandAltmanCell:
    finite = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(sigma_pipeline)
    finite &= np.isfinite(sigma_ref) | (sigma_ref == 0.0)
    if not finite.any():
        return BlandAltmanCell(
            label=label,
            catalogue=catalogue,
            mag_bin=mag_bin,
            n=0,
            bias=float("nan"),
            bias_se=float("nan"),
            scatter=float("nan"),
            mad_scatter=float("nan"),
            sigma_ratio=float("nan"),
            coverage={f"{lvl:.2f}": float("nan") for lvl in coverage_levels},
            pearson=float("nan"),
        )
    p = pred[finite]
    r = ref[finite]
    sp = sigma_pipeline[finite]
    sr = sigma_ref[finite]
    residual = p - r
    n = int(residual.size)
    bias = float(residual.mean())
    bias_se = float(residual.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    scatter = float(residual.std(ddof=1)) if n > 1 else float("nan")
    joint_sigma = np.sqrt(sp**2 + sr**2)
    sigma_ratio = float(scatter / np.sqrt(np.mean(joint_sigma**2))) if n > 1 else float("nan")
    pearson = float(np.corrcoef(p, r)[0, 1]) if n > 1 else float("nan")
    coverage = _coverage(residual, sp, coverage_levels)
    return BlandAltmanCell(
        label=label,
        catalogue=catalogue,
        mag_bin=mag_bin,
        n=n,
        bias=bias,
        bias_se=bias_se,
        scatter=scatter,
        mad_scatter=_mad_scatter(residual),
        sigma_ratio=sigma_ratio,
        coverage=coverage,
        pearson=pearson,
    )


def _trend_curve(
    x: np.ndarray,
    residual: np.ndarray,
    *,
    bins: np.ndarray,
    min_per_bin: int,
) -> dict[str, np.ndarray]:
    """Per-bin median + 16-84 percentile of ``residual`` vs ``x``.

    Returns a dict with keys ``x_centre``, ``bias`` (median), ``p16``, ``p84``,
    ``scatter`` (16-84 half-width, robust σ proxy), ``n``.
    """
    finite = np.isfinite(x) & np.isfinite(residual)
    x = x[finite]
    residual = residual[finite]
    centres = 0.5 * (bins[:-1] + bins[1:])
    bias = np.full(centres.size, np.nan)
    p16 = np.full(centres.size, np.nan)
    p84 = np.full(centres.size, np.nan)
    scatter = np.full(centres.size, np.nan)
    n_per_bin = np.zeros(centres.size, dtype=np.int64)
    digit = np.digitize(x, bins) - 1
    for i in range(centres.size):
        mask = digit == i
        n_per_bin[i] = int(mask.sum())
        if n_per_bin[i] < min_per_bin:
            continue
        chunk = residual[mask]
        bias[i] = np.median(chunk)
        p16[i] = np.percentile(chunk, 16)
        p84[i] = np.percentile(chunk, 84)
        scatter[i] = 0.5 * (p84[i] - p16[i])
    return {
        "x_centre": centres,
        "bias": bias,
        "p16": p16,
        "p84": p84,
        "scatter": scatter,
        "n": n_per_bin,
    }


def _cell_heatmap(
    teff: np.ndarray,
    logg: np.ndarray,
    residual: np.ndarray,
    *,
    teff_bins: np.ndarray,
    logg_bins: np.ndarray,
    min_per_cell: int,
) -> dict[str, np.ndarray]:
    """Per-(Teff, log g) cell median bias and scatter heatmaps."""
    finite = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(residual)
    teff = teff[finite]
    logg = logg[finite]
    residual = residual[finite]
    n_t = teff_bins.size - 1
    n_g = logg_bins.size - 1
    bias = np.full((n_t, n_g), np.nan)
    scatter = np.full((n_t, n_g), np.nan)
    n_grid = np.zeros((n_t, n_g), dtype=np.int64)
    t_idx = np.digitize(teff, teff_bins) - 1
    g_idx = np.digitize(logg, logg_bins) - 1
    valid = (t_idx >= 0) & (t_idx < n_t) & (g_idx >= 0) & (g_idx < n_g)
    t_idx = t_idx[valid]
    g_idx = g_idx[valid]
    residual = residual[valid]
    # Use np.add.at to scatter, then divide by counts where applicable.
    for i in range(n_t):
        for j in range(n_g):
            mask = (t_idx == i) & (g_idx == j)
            n_grid[i, j] = int(mask.sum())
            if n_grid[i, j] < min_per_cell:
                continue
            chunk = residual[mask]
            bias[i, j] = float(np.median(chunk))
            p16, p84 = np.percentile(chunk, [16, 84])
            scatter[i, j] = 0.5 * (p84 - p16)
    return {
        "teff_edges": teff_bins,
        "logg_edges": logg_bins,
        "bias": bias,
        "scatter": scatter,
        "n": n_grid,
    }


# --- Public API ---------------------------------------------------------------


def compute_cross_catalogue_report(  # noqa: PLR0913 — orthogonal scientific knobs
    release: pd.DataFrame,
    catalogues: dict[str, pd.DataFrame],
    bindings: dict[str, CatalogueBinding],
    *,
    g_mag_col: str = "g_mag",
    mag_bins: tuple[tuple[float, float, str], ...] = DEFAULT_MAG_BINS,
    coverage_levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    min_per_bin: int = 100,
    min_per_cell: int = 25,
    teff_bin_edges: np.ndarray | None = None,
    logg_bin_edges: np.ndarray | None = None,
    mh_bin_edges: np.ndarray | None = None,
    teff_trend_edges: np.ndarray | None = None,
) -> CrossCatalogueReport:
    """Run Test 6 across every (label × catalogue × mag-bin) cell.

    Parameters
    ----------
    release
        Pipeline-1 release frame: must carry ``source_id``, the ``g_mag``
        column for binning, and the per-element ``*_pred`` / ``*_sigma``
        columns named in :data:`LABEL_SCHEMA`. The frame is *expected*
        already filtered to Tier 1 + Tier 2 (or Tier 1 only); the caller
        chooses the slice and the report carries the slice description in
        ``config["release_slice"]``.
    catalogues
        Mapping from short catalogue tag (matching ``bindings``) to the
        catalogue's already-cross-matched DataFrame, indexed-aligned with
        ``release`` on ``source_id``. Cross-matching is the caller's
        responsibility; we expect the rows to be in the same order as
        ``release``.
    bindings
        Per-catalogue :class:`CatalogueBinding` describing which columns
        carry which element.

    Other parameters
    ----------------
    g_mag_col
        Column in ``release`` to use for magnitude binning. Default
        ``"g_mag"``.
    mag_bins
        Tuples of ``(lo, hi, label)``; star is in bin if
        ``lo ≤ g < hi``. Default :data:`DEFAULT_MAG_BINS`.
    coverage_levels
        One-sided coverage targets at which to evaluate the released σ.
        Default 68 / 95 / 99 %.
    min_per_bin, min_per_cell
        Minimum overlap-row count below which a cell is reported as
        missing (``n=0``, NaN statistics). The 100-star floor for the
        Bland-Altman cells matches the §3.3 acceptance protocol.
    teff_bin_edges, logg_bin_edges, mh_bin_edges, teff_trend_edges
        Bin edges for the trend curves and heatmaps. Sensible defaults
        are supplied if ``None``.

    Returns
    -------
    CrossCatalogueReport
        Long-form statistics + trend curves + heatmaps + pass/fail map.
    """
    teff_bin_edges = (
        teff_bin_edges if teff_bin_edges is not None else np.linspace(3500.0, 7000.0, 8)
    )
    logg_bin_edges = logg_bin_edges if logg_bin_edges is not None else np.linspace(0.0, 5.0, 8)
    mh_bin_edges = mh_bin_edges if mh_bin_edges is not None else np.linspace(-2.0, 0.5, 11)
    teff_trend_edges = (
        teff_trend_edges if teff_trend_edges is not None else np.linspace(3500.0, 7000.0, 11)
    )

    cells: list[BlandAltmanCell] = []
    bias_vs_mh: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    bias_vs_teff: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    cell_heatmaps: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    passes: dict[tuple[str, str, str], bool] = {}

    g_mag = release[g_mag_col].to_numpy(dtype=np.float64, copy=False)

    for catalogue_name, binding in bindings.items():
        if catalogue_name not in catalogues:
            logger.info("cross_catalogue: skipping %s (no DataFrame supplied)", catalogue_name)
            continue
        cat_df = catalogues[catalogue_name]
        if len(cat_df) != len(release):
            raise ValueError(
                f"{catalogue_name}: cross-matched frame has {len(cat_df)} rows but release has "
                f"{len(release)}; cross-match should be performed before this function"
            )
        for label, schema in LABEL_SCHEMA.items():
            ref_col = binding.column_for.get(label)
            if ref_col is None or ref_col not in cat_df.columns:
                logger.debug(
                    "cross_catalogue: %s does not publish %s; skipping", catalogue_name, label
                )
                continue
            pred_col = str(schema["pred"])
            sigma_col = str(schema["sigma"])
            if pred_col not in release.columns or sigma_col not in release.columns:
                logger.warning(
                    "cross_catalogue: release lacks %s / %s; skipping element %s",
                    pred_col,
                    sigma_col,
                    label,
                )
                continue
            pred = release[pred_col].to_numpy(dtype=np.float64, copy=False)
            sigma_pipeline = release[sigma_col].to_numpy(dtype=np.float64, copy=False)
            ref = cat_df[ref_col].to_numpy(dtype=np.float64, copy=False)
            ref_sigma_col = binding.sigma_for.get(label)
            sigma_ref = (
                cat_df[ref_sigma_col].to_numpy(dtype=np.float64, copy=False)
                if ref_sigma_col is not None and ref_sigma_col in cat_df.columns
                else np.zeros_like(pred)
            )

            # Per-mag-bin Bland-Altman cells.
            for lo, hi, mag_label in mag_bins:
                mask = (g_mag >= lo) & (g_mag < hi)
                cell = _bland_altman_one_cell(
                    pred[mask],
                    ref[mask],
                    sigma_pipeline[mask],
                    sigma_ref[mask],
                    label=label,
                    catalogue=catalogue_name,
                    mag_bin=mag_label,
                    coverage_levels=coverage_levels,
                )
                cells.append(cell)
                if cell.n >= min_per_bin:
                    bias_ok = abs(cell.bias) <= float(schema["bias_limit"])
                    sigma_ratio_ok = 0.7 <= cell.sigma_ratio <= 1.4
                    passes[(label, catalogue_name, mag_label)] = bias_ok and sigma_ratio_ok
                else:
                    passes[(label, catalogue_name, mag_label)] = False

            # Trend curves over [M/H] and Teff (use the full release, not
            # mag-binned). Reference x is read from ``release``: the [M/H]
            # axis uses ArqueoGal's mh prediction (this is bookkeeping —
            # the residual is between Pipeline 1 and the comparison, but
            # the binning needs *some* axis and the released [M/H]
            # prediction is the consistent choice across catalogues).
            mh_axis = (
                release["mh_pred"].to_numpy(dtype=np.float64, copy=False)
                if "mh_pred" in release.columns
                else np.full_like(pred, np.nan)
            )
            teff_axis = (
                release["teff_pred"].to_numpy(dtype=np.float64, copy=False)
                if "teff_pred" in release.columns
                else np.full_like(pred, np.nan)
            )
            logg_axis = (
                release["logg_pred"].to_numpy(dtype=np.float64, copy=False)
                if "logg_pred" in release.columns
                else np.full_like(pred, np.nan)
            )
            residual = pred - ref
            bias_vs_mh[(label, catalogue_name)] = _trend_curve(
                mh_axis, residual, bins=mh_bin_edges, min_per_bin=min_per_bin // 4
            )
            bias_vs_teff[(label, catalogue_name)] = _trend_curve(
                teff_axis,
                residual,
                bins=teff_trend_edges,
                min_per_bin=min_per_bin // 4,
            )
            cell_heatmaps[(label, catalogue_name)] = _cell_heatmap(
                teff_axis,
                logg_axis,
                residual,
                teff_bins=teff_bin_edges,
                logg_bins=logg_bin_edges,
                min_per_cell=min_per_cell,
            )

    config = {
        "mag_bins": [list(b) for b in mag_bins],
        "coverage_levels": list(coverage_levels),
        "min_per_bin": min_per_bin,
        "min_per_cell": min_per_cell,
        "teff_bin_edges": teff_bin_edges.tolist(),
        "logg_bin_edges": logg_bin_edges.tolist(),
        "mh_bin_edges": mh_bin_edges.tolist(),
        "teff_trend_edges": teff_trend_edges.tolist(),
        "release_n_rows": int(len(release)),
        "catalogues": {name: b.citation for name, b in bindings.items()},
    }
    return CrossCatalogueReport(
        cells=cells,
        bias_vs_mh=bias_vs_mh,
        bias_vs_teff=bias_vs_teff,
        cell_heatmaps=cell_heatmaps,
        passes=passes,
        config=config,
    )


def report_to_long_dataframe(report: CrossCatalogueReport) -> pd.DataFrame:
    """Long-form pandas DataFrame of the per-cell statistics.

    One row per (label, catalogue, mag_bin); columns include ``bias``,
    ``scatter``, ``sigma_ratio``, ``mad_scatter``, ``pearson``, ``n``,
    ``coverage_0.68`` / ``0.95`` / ``0.99``, ``passed``.
    """
    rows: list[dict[str, object]] = []
    for cell in report.cells:
        row: dict[str, object] = {
            "label": cell.label,
            "catalogue": cell.catalogue,
            "mag_bin": cell.mag_bin,
            "n": cell.n,
            "bias": cell.bias,
            "bias_se": cell.bias_se,
            "scatter": cell.scatter,
            "mad_scatter": cell.mad_scatter,
            "sigma_ratio": cell.sigma_ratio,
            "pearson": cell.pearson,
            "passed": report.passes.get((cell.label, cell.catalogue, cell.mag_bin), False),
        }
        for level, frac in cell.coverage.items():
            row[f"coverage_{level}"] = frac
        rows.append(row)
    return pd.DataFrame(rows)


def rank_summary(report: CrossCatalogueReport) -> pd.DataFrame:
    """Per-(label, mag_bin) ranking of catalogues by |bias| and scatter.

    The methods-paper "where does ArqueoGal sit?" table is a pivot of the
    ranks — ArqueoGal is implicit (the residuals are *vs* each external
    catalogue), so the rank table is "across reference catalogues, who
    agrees with ArqueoGal best on each label-bin?" which doubles as a
    Pipeline-1-internal-consistency check (a single rogue catalogue is
    immediately visible).
    """
    long = report_to_long_dataframe(report)
    if long.empty:
        return long
    long["abs_bias"] = long["bias"].abs()
    long["bias_rank"] = long.groupby(["label", "mag_bin"])["abs_bias"].rank(method="dense")
    long["scatter_rank"] = long.groupby(["label", "mag_bin"])["scatter"].rank(method="dense")
    return long.drop(columns=["abs_bias"])


def matched_sigma_subsample(
    release: pd.DataFrame,
    *,
    sigma_quantile: float = 0.5,
) -> pd.Series:
    """Return a boolean mask selecting stars whose joint σ is below the
    per-element ``sigma_quantile`` of the release.

    This is the "matched-σ subsample" diagnostic: when comparing against an
    external catalogue that publishes only its low-σ tail (Andrae+2023's
    XGB-trustworthy slice, GALAH "flag_sp == 0"), restricting the ArqueoGal
    Tier-1 release to a comparable σ percentile makes the bias / scatter
    statistics match-up apples-to-apples and removes the σ-inflation
    selection-bias artefact discussed in ``release.py``.
    """
    if not 0.0 < sigma_quantile <= 1.0:
        raise ValueError(f"sigma_quantile must be in (0, 1], got {sigma_quantile}")
    mask = pd.Series(True, index=release.index)
    for schema in LABEL_SCHEMA.values():
        sigma_col = str(schema["sigma"])
        if sigma_col not in release.columns:
            continue
        sigma = release[sigma_col]
        threshold = float(sigma.quantile(sigma_quantile))
        mask &= sigma <= threshold
    return mask


def compute_apogee_benchmark_report(
    arqueogal_pred: pd.DataFrame,
    apogee_truth: pd.DataFrame,
    external_pipelines: dict[str, pd.DataFrame],
    *,
    label_schema: dict[str, dict[str, object]] = LABEL_SCHEMA,
) -> ApogeeBenchmarkReport:
    """Compute per-pipeline rankings on APOGEE DR19 overlap (methods-paper Figure 8).

    Compares ArqueoGal predictions against APOGEE DR19 ground truth, and optionally
    ranks external pipelines (AspGap, SHBoost, Guiglion+2024, etc.) on the same
    matched subset. The composite metric is: rank by |bias| (most important), then
    by scatter (secondary), per element. This is a strict but fair ranking that
    reflects the ability to reproduce the APOGEE DR19 ground truth without
    selection bias (unlike the σ-inflation Tier-1 subset, which removes 25 % of stars
    by design; see release.py selection-bias caveat).

    Parameters
    ----------
    arqueogal_pred : pd.DataFrame
        ArqueoGal predictions on the APOGEE-overlap subset. Must have columns
        matching LABEL_SCHEMA keys (e.g., "teff_pred", "teff_sigma", ...).
    apogee_truth : pd.DataFrame
        APOGEE DR19 ground truth on the same subset. Column names should match
        APOGEE convention (e.g., "teff_apogee", "e_teff_apogee", ...).
        Index must be aligned with arqueogal_pred.
    external_pipelines : dict[str, pd.DataFrame]
        Mapping pipeline name (e.g., "aspgap", "shboost") to predictions.
        Each DataFrame must have the same index as arqueogal_pred and apogee_truth.
    label_schema : dict
        Canonical element→(pred_col, sigma_col, unit, ...) mapping. Default is
        the 21-element v6 LABEL_SCHEMA from this module.

    Returns
    -------
    ApogeeBenchmarkReport
        Top-level deliverable with per-pipeline rankings, per-element metrics,
        and reproducibility config.

    References
    ----------
    - docs/plan/06_methods_paper.md: Figure 8 specification
    - release.py: selection-bias caveat (Tier 1+2 RMSE vs Tier-1-only)
    """
    pipelines: list[ApogeePipelineRanking] = []

    # Build APOGEE truth columns: expect "_apogee" and "e_<label>_apogee" suffixes
    apogee_pred_cols = {}
    apogee_sigma_cols = {}
    for label in label_schema:
        apogee_pred_cols[label] = f"{label}_apogee"
        apogee_sigma_cols[label] = f"e_{label}_apogee"

    # Process ArqueoGal first
    all_pipelines = {"ArqueoGal": arqueogal_pred}
    all_pipelines.update(external_pipelines)

    for pipe_name, pipe_df in all_pipelines.items():
        bias_dict = {}
        scatter_dict = {}
        rmse_dict = {}
        sigma_ratio_dict = {}
        mad_scatter_dict = {}
        pearson_dict = {}
        n_overlap_all = 0

        for label, schema_entry in label_schema.items():
            pred_col = schema_entry["pred"]
            sigma_col = schema_entry["sigma"]

            # Skip elements not in this pipeline's predictions
            if pred_col not in pipe_df.columns:
                bias_dict[label] = float("nan")
                scatter_dict[label] = float("nan")
                rmse_dict[label] = float("nan")
                sigma_ratio_dict[label] = float("nan")
                mad_scatter_dict[label] = float("nan")
                pearson_dict[label] = float("nan")
                continue

            # Skip elements not in APOGEE truth (unlikely, but handle gracefully)
            apogee_col = apogee_pred_cols.get(label)
            if apogee_col not in apogee_truth.columns:
                bias_dict[label] = float("nan")
                scatter_dict[label] = float("nan")
                rmse_dict[label] = float("nan")
                sigma_ratio_dict[label] = float("nan")
                mad_scatter_dict[label] = float("nan")
                pearson_dict[label] = float("nan")
                continue

            # Residual: predicted - truth
            pred = pipe_df[pred_col].values
            truth = apogee_truth[apogee_col].values
            sigma_pipe = (
                pipe_df[sigma_col].values if sigma_col in pipe_df.columns else np.zeros_like(pred)
            )
            sigma_apogee = (
                apogee_truth[apogee_sigma_cols[label]].values
                if apogee_sigma_cols[label] in apogee_truth.columns
                else np.zeros_like(truth)
            )

            # Finite filter
            finite = np.isfinite(pred) & np.isfinite(truth) & np.isfinite(sigma_pipe)
            finite &= np.isfinite(sigma_apogee) | (sigma_apogee == 0.0)

            if not finite.any():
                bias_dict[label] = float("nan")
                scatter_dict[label] = float("nan")
                rmse_dict[label] = float("nan")
                sigma_ratio_dict[label] = float("nan")
                mad_scatter_dict[label] = float("nan")
                pearson_dict[label] = float("nan")
                continue

            n_overlap = int(finite.sum())
            n_overlap_all = max(n_overlap_all, n_overlap)

            pred_fin = pred[finite]
            truth_fin = truth[finite]
            sigma_pipe_fin = sigma_pipe[finite]
            sigma_apogee_fin = sigma_apogee[finite]

            # Residual and scatter metrics
            residual = pred_fin - truth_fin
            bias = float(np.mean(residual))
            scatter = float(np.std(residual))
            mad = _mad_scatter(residual)
            rmse = float(np.sqrt(bias**2 + scatter**2))

            # Sigma ratio (joint uncertainty calibration)
            sigma_combined = np.sqrt(sigma_pipe_fin**2 + sigma_apogee_fin**2)
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma_ratio = float(
                    np.nanmean(
                        np.abs(residual) / np.where(sigma_combined > 0, sigma_combined, np.nan)
                    )
                )

            # Pearson correlation
            if len(pred_fin) > 1:
                pearson = float(np.corrcoef(pred_fin, truth_fin)[0, 1])
            else:
                pearson = float("nan")

            bias_dict[label] = bias
            scatter_dict[label] = scatter
            rmse_dict[label] = rmse
            sigma_ratio_dict[label] = sigma_ratio
            mad_scatter_dict[label] = mad
            pearson_dict[label] = pearson

        ranking = ApogeePipelineRanking(
            pipeline_name=pipe_name,
            n_overlap=n_overlap_all,
            bias=bias_dict,
            scatter=scatter_dict,
            rmse=rmse_dict,
            sigma_ratio=sigma_ratio_dict,
            mad_scatter=mad_scatter_dict,
            pearson=pearson_dict,
        )
        pipelines.append(ranking)

    # Build per_label_stats for per-element RMSE plotting
    per_label_stats: dict[str, dict[str, ApogeePipelineRanking]] = {}
    for label in label_schema:
        per_label_stats[label] = {pr.pipeline_name: pr for pr in pipelines}

    config = {
        "method": "APOGEE DR19 benchmark (methods-paper Figure 8)",
        "n_labels": len(label_schema),
        "ranking_metric": "|bias| primary, scatter secondary (per-element)",
        "note": "Unselected overlap (all matching APOGEE rows, not Tier-1 filtered)",
    }

    return ApogeeBenchmarkReport(
        pipelines=pipelines, per_label_stats=per_label_stats, config=config
    )


__all__ = [
    "DEFAULT_COVERAGE_LEVELS",
    "DEFAULT_MAG_BINS",
    "LABEL_SCHEMA",
    "BlandAltmanCell",
    "CatalogueBinding",
    "CrossCatalogueReport",
    "ApogeeBenchmarkReport",
    "ApogeePipelineRanking",
    "compute_cross_catalogue_report",
    "compute_apogee_benchmark_report",
    "matched_sigma_subsample",
    "rank_summary",
    "report_to_long_dataframe",
]
