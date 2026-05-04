"""Methods-paper Figure-5 ablation: Av-as-feature vs hybrid-D dereddening.

This script produces the methods-paper figure that quantifies how much
explicit dereddening of the broadband auxiliary photometry actually buys,
relative to the "Av as a feature only" baseline. The deliverable is a
three-panel figure (per residual axis) plus a CSV table of the
quantitative slopes.

The harness is **prediction-frame-driven**: it takes two prediction
parquets — one from a model trained with the Av-as-feature recipe, one
from a model trained with the hybrid-D dereddening recipe — that are
indexed-aligned on ``source_id`` to a common APOGEE-truth frame. Panel
generation, slope fitting, and the residual-vs-Av tables operate on
real Stream-1 holdout data.

Outputs (under ``--out``):

- ``residual_vs_av_<element>.pdf/.png`` — five per-element residual
  plots, both recipes overlaid.
- ``residual_vs_quadrant_<element>.pdf/.png`` — five per-element
  per-Galactic-quadrant residual heatmaps.
- ``intrinsic_colour_vs_alpha_m.pdf/.png`` — the (BP−RP)_0 vs predicted
  [α/M] correlation plot (the smoking gun for under-dereddening).
- ``slopes.csv`` + ``slopes.json`` — per-element ``d(residual) / d(Av)``
  slope under each recipe.
- ``summary.json`` — pass/fail verdict ("hybrid-D wins iff every element's
  slope magnitude is at least 30 % lower under hybrid-D").

Usage on real Stream-1 holdout data:

    PYTHONPATH=src python scripts/ablation_extinction_recipe.py \\
        --baseline release/extinction_ablation_baseline.parquet \\
        --hybrid release/extinction_ablation_hybrid.parquet \\
        --truth release/stream1_holdout_truth.parquet \\
        --out reports/ablations/extinction_recipe/
"""

from __future__ import annotations

# isort: off
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# isort: on


logger = logging.getLogger(__name__)


LABEL_ELEMENTS: tuple[tuple[str, str, str], ...] = (
    ("teff", "Teff", "K"),
    ("logg", "log g", "dex"),
    ("mh", "[M/H]", "dex"),
    ("alpha_m", r"[$\alpha$/M]", "dex"),
    ("mg_h", "[Mg/H]", "dex"),
)


@dataclass
class AblationConfig:
    """Knobs that only the methods-paper figure cares about."""

    av_bins: int = 8
    av_max_quantile: float = 0.99
    quadrant_lat_bins: tuple[float, ...] = (-90, -30, -10, 10, 30, 90)
    slope_improvement_required: float = 0.30
    """``hybrid-D wins`` iff |slope_hybrid| ≤ (1 − this) · |slope_baseline|
    on every element. 0.30 = a 30 % reduction is the gate."""


def _residual_vs_av_slope(
    residual: np.ndarray,
    av: np.ndarray,
    *,
    n_bins: int,
    av_max_quantile: float,
) -> dict[str, np.ndarray | float]:
    """Per-Av-bin median + linear slope of the residual.

    Returns ``{x_centre, median, p16, p84, n, slope, slope_se}`` arrays.
    """
    finite = np.isfinite(residual) & np.isfinite(av)
    residual = residual[finite]
    av = av[finite]
    if residual.size < 30:
        return {
            "x_centre": np.array([]),
            "median": np.array([]),
            "p16": np.array([]),
            "p84": np.array([]),
            "n": np.array([]),
            "slope": float("nan"),
            "slope_se": float("nan"),
        }
    av_max = float(np.quantile(av, av_max_quantile))
    bins = np.linspace(0.0, av_max, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    median = np.full(n_bins, np.nan)
    p16 = np.full(n_bins, np.nan)
    p84 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    digit = np.digitize(av, bins) - 1
    for i in range(n_bins):
        mask = digit == i
        counts[i] = int(mask.sum())
        if counts[i] < 20:
            continue
        chunk = residual[mask]
        median[i] = float(np.median(chunk))
        p16[i] = float(np.percentile(chunk, 16))
        p84[i] = float(np.percentile(chunk, 84))
    # Linear slope of residual vs Av on the raw stars; report the std-error.
    slope, intercept = np.polyfit(av, residual, deg=1)
    pred = slope * av + intercept
    rss = float(np.sum((residual - pred) ** 2))
    se = float(np.sqrt(rss / max(av.size - 2, 1)) / max(np.std(av) * np.sqrt(av.size), 1e-12))
    return {
        "x_centre": centres,
        "median": median,
        "p16": p16,
        "p84": p84,
        "n": counts,
        "slope": float(slope),
        "slope_se": float(se),
    }


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_residual_vs_av(
    baseline: pd.DataFrame,
    hybrid: pd.DataFrame,
    truth: pd.DataFrame,
    out_dir: Path,
    cfg: AblationConfig,
) -> tuple[list[Path], dict[str, dict[str, float]]]:
    """Five per-element residual-vs-Av panels with both recipes overlaid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    slopes: dict[str, dict[str, float]] = {}
    av = truth["av_los"].to_numpy(dtype=np.float64, copy=False)
    for element, pretty, unit in LABEL_ELEMENTS:
        truth_col = f"{element}_truth"
        pred_col = f"{element}_pred"
        if truth_col not in truth.columns or pred_col not in baseline.columns:
            continue
        residual_b = baseline[pred_col].to_numpy() - truth[truth_col].to_numpy()
        residual_h = hybrid[pred_col].to_numpy() - truth[truth_col].to_numpy()

        stats_b = _residual_vs_av_slope(
            residual_b, av, n_bins=cfg.av_bins, av_max_quantile=cfg.av_max_quantile
        )
        stats_h = _residual_vs_av_slope(
            residual_h, av, n_bins=cfg.av_bins, av_max_quantile=cfg.av_max_quantile
        )
        slopes[element] = {
            "baseline_slope": stats_b["slope"],
            "baseline_slope_se": stats_b["slope_se"],
            "hybrid_slope": stats_h["slope"],
            "hybrid_slope_se": stats_h["slope_se"],
        }

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        if stats_b["x_centre"].size > 0:
            ax.fill_between(
                stats_b["x_centre"], stats_b["p16"], stats_b["p84"], alpha=0.2, color="C3"
            )
            ax.plot(
                stats_b["x_centre"],
                stats_b["median"],
                "-o",
                color="C3",
                lw=1.4,
                label=f"baseline (Av as feature) — slope {stats_b['slope']:+.4f}",
            )
        if stats_h["x_centre"].size > 0:
            ax.fill_between(
                stats_h["x_centre"], stats_h["p16"], stats_h["p84"], alpha=0.2, color="C0"
            )
            ax.plot(
                stats_h["x_centre"],
                stats_h["median"],
                "-o",
                color="C0",
                lw=1.4,
                label=f"hybrid-D (Yuan+2013 dered) — slope {stats_h['slope']:+.4f}",
            )
        ax.axhline(0.0, color="0.4", lw=0.8, linestyle=":")
        ax.set_xlabel(r"$A_V$ (mag)")
        ax.set_ylabel(f"{pretty} residual (pred − truth) [{unit}]")
        ax.set_title(f"{pretty}: residual vs $A_V$ — recipe ablation")
        ax.legend(fontsize=8, loc="best")
        written.append(_save(fig, out_dir / f"residual_vs_av_{element}.pdf"))
    return written, slopes


def _plot_residual_vs_quadrant(
    baseline: pd.DataFrame,
    hybrid: pd.DataFrame,
    truth: pd.DataFrame,
    out_dir: Path,
    cfg: AblationConfig,
) -> list[Path]:
    """Per-Galactic-quadrant residual heatmaps under each recipe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if "b_deg" not in truth.columns:
        logger.info("truth lacks b_deg; skipping quadrant heatmap")
        return written
    b = truth["b_deg"].to_numpy(dtype=np.float64, copy=False)
    bin_edges = np.array(cfg.quadrant_lat_bins, dtype=np.float64)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for element, pretty, unit in LABEL_ELEMENTS:
        truth_col = f"{element}_truth"
        pred_col = f"{element}_pred"
        if truth_col not in truth.columns or pred_col not in baseline.columns:
            continue
        residual_b = baseline[pred_col].to_numpy() - truth[truth_col].to_numpy()
        residual_h = hybrid[pred_col].to_numpy() - truth[truth_col].to_numpy()
        idx = np.digitize(b, bin_edges) - 1
        med_b = np.full(len(bin_centres), np.nan)
        med_h = np.full(len(bin_centres), np.nan)
        for i in range(len(bin_centres)):
            mask = idx == i
            if mask.sum() < 20:
                continue
            med_b[i] = float(np.nanmedian(residual_b[mask]))
            med_h[i] = float(np.nanmedian(residual_h[mask]))

        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.plot(bin_centres, med_b, "-o", color="C3", lw=1.4, label="baseline")
        ax.plot(bin_centres, med_h, "-o", color="C0", lw=1.4, label="hybrid-D")
        ax.axhline(0.0, color="0.4", lw=0.8, linestyle=":")
        ax.set_xlabel(r"Galactic latitude $b$ (deg)")
        ax.set_ylabel(f"median {pretty} residual [{unit}]")
        ax.set_title(f"{pretty}: residual vs Galactic latitude")
        ax.legend(fontsize=8)
        written.append(_save(fig, out_dir / f"residual_vs_quadrant_{element}.pdf"))
    return written


def _plot_intrinsic_colour_vs_alpha_m(
    baseline: pd.DataFrame,
    hybrid: pd.DataFrame,
    truth: pd.DataFrame,
    out_dir: Path,
) -> Path | None:
    """The (BP−RP)_0 vs predicted [α/M] correlation diagnostic.

    After dereddening, the intrinsic colour should be uncorrelated with
    [α/M]. Any residual correlation under the *baseline* recipe vs a
    flatter scatter under *hybrid-D* is the smoking gun that explicit
    dereddening removed an extinction-driven artefact.
    """
    if "alpha_m_pred" not in baseline.columns or "bp_rp_dered" not in truth.columns:
        logger.info("intrinsic-colour plot skipped (missing columns)")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    bp_rp = truth["bp_rp_dered"].to_numpy(dtype=np.float64, copy=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, frame, recipe, color in (
        (axes[0], baseline, "baseline (Av as feature)", "C3"),
        (axes[1], hybrid, "hybrid-D (dered)", "C0"),
    ):
        alpha_m = frame["alpha_m_pred"].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(bp_rp) & np.isfinite(alpha_m)
        ax.scatter(bp_rp[finite], alpha_m[finite], s=4, alpha=0.4, color=color, rasterized=True)
        # Slope-only fit, no intercept reporting.
        if finite.sum() > 2:
            slope, intercept = np.polyfit(bp_rp[finite], alpha_m[finite], deg=1)
            xs = np.array([np.nanmin(bp_rp), np.nanmax(bp_rp)])
            ax.plot(xs, slope * xs + intercept, color="0.2", lw=1.0, label=f"slope {slope:+.3f}")
        ax.set_xlabel(r"(BP − RP)$_0$ (dereddened)")
        ax.set_title(recipe)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"predicted [$\alpha$/M] (dex)")
    fig.suptitle(
        "Intrinsic-colour correlation: residual extinction signal in [α/M] prediction",
        fontsize=10,
    )
    return _save(fig, out_dir / "intrinsic_colour_vs_alpha_m.pdf")


def _verdict(slopes: dict[str, dict[str, float]], cfg: AblationConfig) -> dict[str, object]:
    """Pass/fail summary across the five elements.

    ``hybrid-D wins`` iff every element's slope magnitude has dropped by
    at least ``cfg.slope_improvement_required`` × the baseline magnitude.
    """
    per_element: dict[str, dict[str, float | bool]] = {}
    every_passes = True
    for element, vals in slopes.items():
        b = abs(vals["baseline_slope"])
        h = abs(vals["hybrid_slope"])
        if b == 0:
            improvement = 0.0
        else:
            improvement = (b - h) / b
        passes = improvement >= cfg.slope_improvement_required
        per_element[element] = {
            "baseline_slope": vals["baseline_slope"],
            "hybrid_slope": vals["hybrid_slope"],
            "improvement": improvement,
            "passes": bool(passes),
        }
        every_passes = every_passes and passes
    return {
        "per_element": per_element,
        "verdict": "hybrid-D wins" if every_passes else "inconclusive",
        "improvement_threshold": cfg.slope_improvement_required,
    }


def run_ablation(
    baseline: pd.DataFrame,
    hybrid: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    out_dir: Path,
    cfg: AblationConfig | None = None,
) -> dict[str, object]:
    """Top-level: produce all plots + tables; return the verdict."""
    cfg = cfg or AblationConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running residual-vs-Av ablation (n=%d)", len(truth))
    _, slopes = _plot_residual_vs_av(baseline, hybrid, truth, out_dir, cfg)
    _plot_residual_vs_quadrant(baseline, hybrid, truth, out_dir, cfg)
    _plot_intrinsic_colour_vs_alpha_m(baseline, hybrid, truth, out_dir)

    summary = _verdict(slopes, cfg)
    # Persist tables.
    rows = []
    for element, vals in summary["per_element"].items():
        rows.append({"element": element, **vals})
    pd.DataFrame(rows).to_csv(out_dir / "slopes.csv", index=False)
    with (out_dir / "slopes.json").open("w", encoding="utf-8") as f:
        json.dump(slopes, f, indent=2, sort_keys=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline predictions parquet (Av-as-feature recipe)",
    )
    parser.add_argument(
        "--hybrid",
        type=Path,
        required=True,
        help="Hybrid-D predictions parquet (dereddened recipe)",
    )
    parser.add_argument(
        "--truth", type=Path, required=True, help="Stream 1 holdout truth parquet (APOGEE labels)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    baseline = pd.read_parquet(args.baseline)
    hybrid = pd.read_parquet(args.hybrid)
    truth = pd.read_parquet(args.truth)

    summary = run_ablation(baseline, hybrid, truth, out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
