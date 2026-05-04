"""Diagnostic plots for cross-catalogue Test-6 reports.

Seven plot families consume a :class:`CrossCatalogueReport` and emit
methods-paper-ready figures. Plots are assembled with the A&A-style
rcParams from :func:`arqueogal.utils.plotting.set_aa_style`. Every
figure is saved as PDF (paper) and PNG (gallery thumbnail) per the
project figure-format conventions in docs/context/conventions.md.

The seven families:

1. **Bland-Altman scatter panels**, ``arqueogal − reference`` vs the
   reference, with mean-bias horizontal line, ±1·σ_pipeline shaded band,
   per-mag-bin facet rows. One figure per (label, catalogue).
2. **Residual histogram + Gaussian overlay**, residual distribution
   against the released σ. Shows whether the joint σ is calibrated and
   whether the residual is heavy-tailed.
3. **Metallicity-dependent bias trend**, per-bin median residual vs
   ArqueoGal [M/H], one panel per element, all catalogues overlaid.
4. **Teff-dependent bias trend**, same idea, vs Teff.
5. **Teff × log g cell heatmap**, median bias and 16-84 scatter on a
   stellar-parameter grid; one heatmap pair per (label, catalogue).
6. **Coverage curves**, empirical fraction within ±k·σ_pipeline at the
   nominal levels, one panel per element, all catalogues overlaid.
7. **Rank summary heatmap**, per-(label, mag-bin) rank of each
   catalogue's |bias| and scatter, doubling as a one-glance methods-paper
   summary.

Each function accepts a :class:`CrossCatalogueReport`, a single output
directory, and optional kwargs. Each function is independent (no shared
mutable state) so the CLI driver can call any subset.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arqueogal.utils.plotting import save_figure, set_aa_style
from arqueogal.xp_abundances.main.cross_catalogue import (
    LABEL_SCHEMA,
    CrossCatalogueReport,
    rank_summary,
    report_to_long_dataframe,
)

logger = logging.getLogger(__name__)


def _label_pretty(label: str) -> str:
    """LaTeX-safe pretty label for axes."""
    return {
        "teff": r"$T_\mathrm{eff}$",
        "logg": r"$\log g$",
        "mh": r"[M/H]",
        "alpha_m": r"[$\alpha$/M]",
        "mg_h": r"[Mg/H]",
    }.get(label, label)


def _label_unit(label: str) -> str:
    return str(LABEL_SCHEMA[label]["unit"])


def _safe_savefig(fig: Any, path: Path) -> Path:
    """Save as PDF via :func:`save_figure`, drop a PNG sibling for gallery."""
    save_figure(fig, path)
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    return path


def _residual_arrays_from_release(
    release: pd.DataFrame,
    catalogues: dict[str, pd.DataFrame],
    label: str,
    catalogue_name: str,
    column_for: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull aligned (pred, ref, sigma_pipeline, g_mag) arrays.

    Used by plot families that need raw arrays (Bland-Altman scatter,
    residual histogram). The summary statistics in the report are
    pre-computed; the raw arrays are only loaded by these two families.
    """
    schema = LABEL_SCHEMA[label]
    pred = release[str(schema["pred"])].to_numpy(dtype=np.float64, copy=False)
    sigma = release[str(schema["sigma"])].to_numpy(dtype=np.float64, copy=False)
    ref = catalogues[catalogue_name][column_for[label]].to_numpy(dtype=np.float64, copy=False)
    g_mag = release["g_mag"].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(sigma) & np.isfinite(g_mag)
    return pred[finite], ref[finite], sigma[finite], g_mag[finite]


# --- Plot family 1: Bland-Altman per (label, catalogue) -----------------------


def plot_bland_altman_per_label(  # noqa: PLR0913, orthogonal scientific knobs
    report: CrossCatalogueReport,
    release: pd.DataFrame,
    catalogues: dict[str, pd.DataFrame],
    bindings: dict[str, Any],
    out_dir: Path,
    *,
    max_points: int = 50_000,
    rng_seed: int = 0,
) -> list[Path]:
    """Per-(label, catalogue) Bland-Altman scatter with mag-bin facets.

    Produces ``out_dir/bland_altman/<label>_<catalogue>.pdf``. Each figure
    has 3 panels (one per default magnitude bin); the y-axis is the
    residual ``arqueogal − reference``, the x-axis is the reference value,
    a horizontal red line marks the mean bias, and a shaded ±1·σ_pipeline
    band tracks the released uncertainty.
    """
    out_dir = Path(out_dir) / "bland_altman"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    written: list[Path] = []
    long = report_to_long_dataframe(report)
    for catalogue_name, binding in bindings.items():
        if catalogue_name not in catalogues:
            continue
        for label in LABEL_SCHEMA:
            ref_col = binding.column_for.get(label)
            if ref_col is None:
                continue
            pred, ref, sigma, g_mag = _residual_arrays_from_release(
                release, catalogues, label, catalogue_name, binding.column_for
            )
            if pred.size == 0:
                continue
            residual = pred - ref
            cells = long.query("label == @label and catalogue == @catalogue_name")
            n_panels = max(int(cells.shape[0]), 1)
            fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 3.0), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, (_, row) in zip(axes, cells.iterrows()):
                lo_hi = next((b for b in report.config["mag_bins"] if b[2] == row["mag_bin"]), None)
                if lo_hi is None:
                    ax.set_visible(False)
                    continue
                lo, hi, _ = lo_hi
                mask = (g_mag >= lo) & (g_mag < hi)
                if mask.sum() == 0:
                    ax.text(
                        0.5, 0.5, "(no overlap)", ha="center", va="center", transform=ax.transAxes
                    )
                    ax.set_title(f"{row['mag_bin']} (G ∈ [{lo}, {hi}))")
                    continue
                idx = np.flatnonzero(mask)
                if idx.size > max_points:
                    idx = rng.choice(idx, size=max_points, replace=False)
                ax.scatter(ref[idx], residual[idx], s=2, alpha=0.25, rasterized=True, color="0.3")
                ax.axhline(0.0, color="0.5", lw=0.8, linestyle=":")
                ax.axhline(row["bias"], color="C3", lw=1.0, label=f"bias = {row['bias']:.3f}")
                # ±σ_pipeline band: median σ in this bin.
                med_sigma = float(np.median(sigma[mask]))
                ax.axhspan(-med_sigma, med_sigma, alpha=0.15, color="C0", label="±median σ")
                ax.set_title(f"{row['mag_bin']} (n={int(row['n']):,})")
                ax.set_xlabel(f"{_label_pretty(label)} reference [{_label_unit(label)}]")
            axes[0].set_ylabel(
                f"residual ArqueoGal − {binding.citation or catalogue_name} [{_label_unit(label)}]"
            )
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, fontsize=7, loc="upper right")
            fig.suptitle(
                f"Bland–Altman: {_label_pretty(label)} vs {binding.citation or catalogue_name}",
                fontsize=10,
            )
            written.append(_safe_savefig(fig, out_dir / f"{label}_{catalogue_name}.pdf"))
            plt.close(fig)
    return written


# --- Plot family 2: residual histogram + Gaussian overlay ---------------------


def plot_residual_histograms(
    report: CrossCatalogueReport,
    release: pd.DataFrame,
    catalogues: dict[str, pd.DataFrame],
    bindings: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """One figure per label: histogram of standardised residual
    ``(arqueogal − reference) / σ_pipeline``, every catalogue overlaid,
    with the Standard Normal as reference.
    """
    out_dir = Path(out_dir) / "residual_hist"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    bin_edges = np.linspace(-5.0, 5.0, 61)
    for label in LABEL_SCHEMA:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        any_data = False
        for catalogue_name, binding in bindings.items():
            if catalogue_name not in catalogues or label not in binding.column_for:
                continue
            pred, ref, sigma, _ = _residual_arrays_from_release(
                release, catalogues, label, catalogue_name, binding.column_for
            )
            if pred.size == 0:
                continue
            standardised = (pred - ref) / np.where(sigma > 0, sigma, np.nan)
            standardised = standardised[np.isfinite(standardised)]
            if standardised.size == 0:
                continue
            ax.hist(
                standardised,
                bins=bin_edges,
                histtype="step",
                density=True,
                lw=1.2,
                label=binding.citation or catalogue_name,
            )
            any_data = True
        # Overlay N(0, 1).
        x = np.linspace(-5.0, 5.0, 401)
        ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), color="0.4", lw=1.0, label="N(0, 1)")
        ax.set_xlabel(
            rf"standardised residual $(\mathrm{{ArqueoGal}} - \mathrm{{ref}}) "
            rf"/ \sigma_\mathrm{{pipeline}}$ for {_label_pretty(label)}"
        )
        ax.set_ylabel("density")
        ax.set_title(f"σ-calibration of {_label_pretty(label)}")
        if any_data:
            # bbox_to_anchor parks the legend outside the axes so it can never
            # overlap title or histogram peaks regardless of how many catalogues
            # the user passes in.
            ax.legend(
                fontsize=7,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
            fig.tight_layout()
            written.append(_safe_savefig(fig, out_dir / f"{label}.pdf"))
        plt.close(fig)
    return written


# --- Plot family 3: metallicity-dependent bias trend --------------------------


def plot_bias_vs_mh(report: CrossCatalogueReport, out_dir: Path) -> list[Path]:
    """Per-element panel of median residual vs ArqueoGal [M/H], all
    catalogues overlaid. 16-84 percentile band per catalogue.
    """
    out_dir = Path(out_dir) / "bias_vs_mh"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    by_label: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {}
    for (label, catalogue), curves in report.bias_vs_mh.items():
        by_label.setdefault(label, []).append((catalogue, curves))
    for label, entries in by_label.items():
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        for catalogue, curves in entries:
            x = curves["x_centre"]
            bias = curves["bias"]
            p16 = curves["p16"]
            p84 = curves["p84"]
            (line,) = ax.plot(x, bias, lw=1.2, label=catalogue)
            ax.fill_between(x, p16, p84, alpha=0.15, color=line.get_color())
        ax.axhline(0.0, color="0.5", lw=0.8, linestyle=":")
        ax.set_xlabel(r"ArqueoGal [M/H] (dex)")
        ax.set_ylabel(f"median residual {_label_pretty(label)} [{_label_unit(label)}]")
        ax.set_title(f"{_label_pretty(label)} bias vs metallicity")
        ax.legend(fontsize=7, loc="best")
        written.append(_safe_savefig(fig, out_dir / f"{label}.pdf"))
        plt.close(fig)
    return written


# --- Plot family 4: Teff-dependent bias trend --------------------------------


def plot_bias_vs_teff(report: CrossCatalogueReport, out_dir: Path) -> list[Path]:
    """Like :func:`plot_bias_vs_mh` but vs ArqueoGal Teff."""
    out_dir = Path(out_dir) / "bias_vs_teff"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    by_label: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {}
    for (label, catalogue), curves in report.bias_vs_teff.items():
        by_label.setdefault(label, []).append((catalogue, curves))
    for label, entries in by_label.items():
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        for catalogue, curves in entries:
            x = curves["x_centre"]
            bias = curves["bias"]
            p16 = curves["p16"]
            p84 = curves["p84"]
            (line,) = ax.plot(x, bias, lw=1.2, label=catalogue)
            ax.fill_between(x, p16, p84, alpha=0.15, color=line.get_color())
        ax.axhline(0.0, color="0.5", lw=0.8, linestyle=":")
        ax.set_xlabel(r"ArqueoGal $T_\mathrm{eff}$ (K)")
        ax.set_ylabel(f"median residual {_label_pretty(label)} [{_label_unit(label)}]")
        ax.set_title(f"{_label_pretty(label)} bias vs $T_\\mathrm{{eff}}$")
        ax.invert_xaxis()
        ax.legend(fontsize=7, loc="best")
        written.append(_safe_savefig(fig, out_dir / f"{label}.pdf"))
        plt.close(fig)
    return written


# --- Plot family 5: Teff × log g cell heatmap --------------------------------


def plot_cell_heatmaps(report: CrossCatalogueReport, out_dir: Path) -> list[Path]:
    """Per-(label, catalogue) Teff × log g heatmap of bias and scatter.

    Two panels per figure: bias (diverging cmap centred at zero) and
    scatter (sequential cmap). Cells with insufficient stars are blanked.
    """
    out_dir = Path(out_dir) / "cell_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (label, catalogue), heatmap in report.cell_heatmaps.items():
        fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(8.5, 3.5))
        teff_edges = heatmap["teff_edges"]
        logg_edges = heatmap["logg_edges"]
        bias = heatmap["bias"]
        scatter = heatmap["scatter"]
        # Symmetric diverging scale for bias, anchored to the bias_limit.
        bias_lim = float(LABEL_SCHEMA[label]["bias_limit"])
        vmax_b = max(np.nanmax(np.abs(bias)) if np.isfinite(bias).any() else bias_lim, bias_lim)
        im_b = ax_b.pcolormesh(
            teff_edges, logg_edges, bias.T, cmap="RdBu_r", shading="auto", vmin=-vmax_b, vmax=vmax_b
        )
        ax_b.invert_xaxis()
        ax_b.invert_yaxis()
        ax_b.set_xlabel(r"$T_\mathrm{eff}$ (K)")
        ax_b.set_ylabel(r"$\log g$ (dex)")
        ax_b.set_title("median bias")
        cb = plt.colorbar(im_b, ax=ax_b)
        cb.set_label(f"bias {_label_pretty(label)} [{_label_unit(label)}]")
        # Scatter panel.
        if np.isfinite(scatter).any():
            vmax_s = float(np.nanmax(scatter))
        else:
            vmax_s = float(LABEL_SCHEMA[label]["apogee_sigma"])
        im_s = ax_s.pcolormesh(
            teff_edges, logg_edges, scatter.T, cmap="viridis", shading="auto", vmin=0.0, vmax=vmax_s
        )
        ax_s.invert_xaxis()
        ax_s.invert_yaxis()
        ax_s.set_xlabel(r"$T_\mathrm{eff}$ (K)")
        ax_s.set_ylabel(r"$\log g$ (dex)")
        ax_s.set_title("16–84 % scatter")
        cb2 = plt.colorbar(im_s, ax=ax_s)
        cb2.set_label(f"scatter {_label_pretty(label)} [{_label_unit(label)}]")
        fig.suptitle(f"Cell map of {_label_pretty(label)} residual vs {catalogue}", fontsize=10)
        written.append(_safe_savefig(fig, out_dir / f"{label}_{catalogue}.pdf"))
        plt.close(fig)
    return written


# --- Plot family 6: coverage curves -------------------------------------------


def plot_coverage_curves(report: CrossCatalogueReport, out_dir: Path) -> list[Path]:
    """One panel per element of empirical fraction within ±k·σ_pipeline at
    the nominal levels (68 / 95 / 99 %), with target horizontal lines.
    """
    out_dir = Path(out_dir) / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    long = report_to_long_dataframe(report)
    for label in LABEL_SCHEMA:
        sub = long.query("label == @label")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        for catalogue, group in sub.groupby("catalogue"):
            xs = []
            ys = []
            for level in DEFAULT_COVERAGE_LEVEL_KEYS:
                col = f"coverage_{level}"
                if col not in group.columns:
                    continue
                # Aggregate across mag bins with sample-size weighting.
                w = group["n"].to_numpy(dtype=np.float64)
                mask = w > 0
                if mask.sum() == 0:
                    continue
                empirical = float(np.average(group.loc[mask, col].to_numpy(), weights=w[mask]))
                xs.append(float(level))
                ys.append(empirical)
            if not xs:
                continue
            order = np.argsort(xs)
            xs_arr = np.array(xs)[order]
            ys_arr = np.array(ys)[order]
            ax.plot(xs_arr, ys_arr, "o-", lw=1.2, label=catalogue)
        # Diagonal target.
        levels_grid = np.linspace(0.0, 1.0, 51)
        ax.plot(levels_grid, levels_grid, color="0.4", lw=0.8, linestyle=":")
        ax.set_xlim(0.6, 1.0)
        ax.set_ylim(0.6, 1.0)
        ax.set_xlabel("nominal coverage level")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"{_label_pretty(label)} σ-coverage")
        ax.legend(fontsize=7, loc="best")
        written.append(_safe_savefig(fig, out_dir / f"{label}.pdf"))
        plt.close(fig)
    return written


# --- Plot family 7: rank summary heatmap --------------------------------------


def plot_rank_summary(report: CrossCatalogueReport, out_dir: Path) -> list[Path]:
    """Per-(label, mag_bin) rank of catalogues by |bias| and scatter.

    Produces two heatmaps (bias-rank, scatter-rank) saved as
    ``rank_summary_bias.pdf`` and ``rank_summary_scatter.pdf``. Rank 1 =
    best agreement; missing combinations stay NaN.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    long = rank_summary(report)
    if long.empty:
        return written
    for metric, fname in [
        ("bias_rank", "rank_summary_bias.pdf"),
        ("scatter_rank", "rank_summary_scatter.pdf"),
    ]:
        pivot = long.pivot_table(
            index="catalogue", columns=["label", "mag_bin"], values=metric, aggfunc="first"
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 3.0 + 0.3 * pivot.shape[0]))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(
            ["\n".join(map(str, c)) for c in pivot.columns], rotation=45, ha="right", fontsize=7
        )
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7, color="white")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("rank (1 = best agreement)")
        ax.set_title(metric.replace("_", " "))
        written.append(_safe_savefig(fig, out_dir / fname))
        plt.close(fig)
    return written


# --- Aggregator ---------------------------------------------------------------

DEFAULT_COVERAGE_LEVEL_KEYS: Sequence[str] = ("0.68", "0.95", "0.99")


def render_all(  # noqa: PLR0913, orthogonal scientific knobs
    report: CrossCatalogueReport,
    release: pd.DataFrame,
    catalogues: dict[str, pd.DataFrame],
    bindings: dict[str, Any],
    out_dir: Path,
    *,
    apply_aa_style: bool = True,
) -> dict[str, list[Path]]:
    """Run every plot family. Returns a mapping family→list of output paths."""
    if apply_aa_style:
        set_aa_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {
        "bland_altman": plot_bland_altman_per_label(report, release, catalogues, bindings, out_dir),
        "residual_hist": plot_residual_histograms(report, release, catalogues, bindings, out_dir),
        "bias_vs_mh": plot_bias_vs_mh(report, out_dir),
        "bias_vs_teff": plot_bias_vs_teff(report, out_dir),
        "cell_heatmaps": plot_cell_heatmaps(report, out_dir),
        "coverage": plot_coverage_curves(report, out_dir),
        "rank_summary": plot_rank_summary(report, out_dir),
    }
    return written


__all__ = [
    "plot_bias_vs_mh",
    "plot_bias_vs_teff",
    "plot_bland_altman_per_label",
    "plot_cell_heatmaps",
    "plot_coverage_curves",
    "plot_rank_summary",
    "plot_residual_histograms",
    "render_all",
]
