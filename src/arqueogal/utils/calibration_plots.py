"""Reliability-diagram visualisations for Pipeline 1 calibration.

The methods paper requires per-element reliability diagrams (predicted σ
vs empirical σ) with explicit ±10 % bounds, both pre- and post-shrinkage.
The calibration code (``uncertainty.py``) computes per-cell (predicted_σ,
empirical_σ) pairs but emits no plotting function. This module fills that
gap (METRICS gap §1.5 in metrics_diagnostics.md, paper_external_evaluation.md
§9, hostile_referee_committee deliverable #2).

Design choices
--------------

- One scatter point per regime cell (Teff × logg × [M/H] grid) per element.
- Solid identity line ``y = x``; dashed lines at ±10 % bounds (the project's
  release calibration tolerance per research_brief §9.1).
- Marker size encodes the cell sample count (so the consumer's eye is drawn
  to well-populated cells).
- Marker colour encodes the per-cell χ² goodness-of-fit (viridis sequential).
- Per-element subplot in a 5-panel grid (Teff, logg, [M/H], [α/M], [Mg/H]).
- Output saved as PDF (publication-grade) with the rcParams from
  ``arqueogal.utils.plotting`` applied (so ``pdf.fonttype = 42`` etc.).

The function is ``pandas`` + ``matplotlib`` only; no astropy. Called by
``paper/methods_paper/make_figures.py`` as figure 3 (per the manifest in
that file). The Phase A2 v3 catalog schema's ``release_tier__<element>``
columns aren't required: this function consumes per-cell calibration
diagnostics emitted upstream by ``uncertainty.per_cell_calibration``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Element label conventions matching the project's canonical LaTeX strings.
# We don't import label_conventions here to keep this module standalone.
_ELEMENT_LATEX: dict[str, str] = {
    "teff": r"$T_{\rm eff}$",
    "logg": r"$\log g$",
    "mh": r"$[\mathrm{M/H}]$",
    "alpha_m": r"$[\alpha/\mathrm{M}]$",
    "mg_h": r"$[\mathrm{Mg/H}]$",
}

_ABUNDANCE_ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")


def reliability_diagram_per_label(
    calibration_df: pd.DataFrame,
    output_path: Path,
    *,
    pre_post_pair: tuple[str, str] | None = None,
    title_suffix: str = "",
    bounds_pct: float = 10.0,
    apply_aa_style: bool = True,
) -> dict[str, int | str]:
    """Plot a 5-panel reliability diagram (predicted σ vs empirical σ per element).

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Per-cell calibration diagnostics. Required columns:
        - ``element``: one of teff, logg, mh, alpha_m, mg_h.
        - ``predicted_sigma``: per-cell mean predicted σ.
        - ``empirical_sigma``: per-cell empirical σ from validation residuals.
        - ``n_cell``: per-cell sample count.
        Optional:
        - ``chi_squared`` for marker colouring.
    output_path : Path
        PDF output path; parent dir auto-created.
    pre_post_pair : tuple[str, str] | None
        If supplied, the calibration_df is expected to have a ``stage``
        column with two values matching the pair (e.g.
        ``("pre_shrinkage", "post_shrinkage")``). The plot becomes a
        2-row × 5-col grid: top row pre-, bottom row post-shrinkage.
    title_suffix : str
        Appended to each subplot title (e.g., release tag).
    bounds_pct : float
        ±% deviation bounds drawn as dashed lines (default 10).
    apply_aa_style : bool
        Apply the project's A&A rcParams via ``arqueogal.utils.plotting.set_aa_style()``.

    Returns
    -------
    dict
        Summary: ``{"output": ..., "n_panels": ..., "n_points": ...}``.

    Notes
    -----
    Figure dimensions: A&A double-column (180 mm = 7.087 in) for the 5-panel
    single-row variant; ratio 7.087 × 1.8 in. For the 2-row pre/post variant,
    height doubles to 7.087 × 3.6 in.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "calibration plots require matplotlib. Install via uv add matplotlib.",
        ) from e

    if apply_aa_style:
        try:
            from arqueogal.utils.plotting import set_aa_style

            set_aa_style()
        except ImportError:
            # Fall back to local rcParams (matches the Phase A1 plotting.py addition).
            plt.rcParams.update(
                {
                    "font.family": "serif",
                    "font.size": 9,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "savefig.bbox": "tight",
                    "savefig.dpi": 300,
                }
            )

    required_cols = {"element", "predicted_sigma", "empirical_sigma", "n_cell"}
    missing = required_cols - set(calibration_df.columns)
    if missing:
        raise ValueError(f"calibration_df missing required columns: {sorted(missing)}")

    if pre_post_pair is not None:
        if "stage" not in calibration_df.columns:
            raise ValueError(
                "pre_post_pair requested but 'stage' column missing from calibration_df",
            )
        n_rows = 2
        stages = pre_post_pair
        fig_height = 3.6
    else:
        n_rows = 1
        stages = ("",)
        fig_height = 1.8

    n_cols = len(_ABUNDANCE_ELEMENTS)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(7.087, fig_height),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    n_points_total = 0
    for r, stage in enumerate(stages):
        for c, elem in enumerate(_ABUNDANCE_ELEMENTS):
            ax = axes[r, c]
            sub = calibration_df[calibration_df["element"] == elem]
            if pre_post_pair is not None:
                sub = sub[sub["stage"] == stage]
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"no data: {elem}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x = sub["predicted_sigma"].to_numpy()
            y = sub["empirical_sigma"].to_numpy()
            sizes = np.clip(np.sqrt(sub["n_cell"].to_numpy()), 4, 60)
            colors = sub.get("chi_squared", pd.Series(1.0, index=sub.index)).to_numpy()

            scatter = ax.scatter(
                x,
                y,
                s=sizes,
                c=colors,
                cmap="viridis",
                alpha=0.7,
                edgecolors="none",
            )
            n_points_total += len(x)

            # Identity + ±bounds_pct lines.
            xy_max = float(np.nanmax([np.nanmax(x), np.nanmax(y), 1e-6]))
            xy_min = 0.0
            ax.plot(
                [xy_min, xy_max],
                [xy_min, xy_max],
                "k-",
                lw=0.6,
                label="y = x",
            )
            f = bounds_pct / 100.0
            ax.plot(
                [xy_min, xy_max],
                [xy_min, xy_max * (1 + f)],
                "k--",
                lw=0.4,
                alpha=0.5,
            )
            ax.plot(
                [xy_min, xy_max],
                [xy_min, xy_max * (1 - f)],
                "k--",
                lw=0.4,
                alpha=0.5,
            )

            ax.set_xlim(xy_min, xy_max)
            ax.set_ylim(xy_min, xy_max)
            label = _ELEMENT_LATEX.get(elem, elem)
            title = f"{label}"
            if stage:
                title = f"{label} ({stage})"
            if title_suffix:
                title = f"{title} {title_suffix}"
            ax.set_title(title, fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel(r"predicted $\sigma$", fontsize=8)
            if c == 0:
                ax.set_ylabel(r"empirical $\sigma$", fontsize=8)
            ax.tick_params(labelsize=7)

    # Colorbar for the χ² scale.
    if "chi_squared" in calibration_df.columns:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
        cbar.set_label(r"per-cell $\chi^2$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)

    return {
        "output": str(output_path),
        "n_panels": int(n_rows * n_cols),
        "n_points": int(n_points_total),
    }


__all__ = ["reliability_diagram_per_label"]
