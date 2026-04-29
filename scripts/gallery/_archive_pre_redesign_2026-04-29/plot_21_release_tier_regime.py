"""Stage 19: release-tier regime composition.

What the deploy did: ``release.annotate_parquet`` (called inside
``release_pipeline.run_release_pipeline``) wrote per-element
``release_tier__<elem>``, ``prediction_sigma_inflated__<elem>``, and the
composite ``release_tier`` (row-max). Diagnostic columns
``dist_prior_dominated`` / ``ood_aux_mahalanobis_flag`` are still emitted
but no longer feed the tier (v5 schema, 2026-04-26 — see
``release/test_ablations_2026-04-26/REPORT.md``).

What we plot: regime composition of the release tier vs G, distance, and
on the sky; per-element tier counts; flag-firing breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/21_release_tier_regime"
PARQUET_S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"
PARQUET_S2 = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
ELEMENT_LABELS = {
    "teff": "Teff",
    "logg": "log g",
    "mh": "[M/H]",
    "alpha_m": "[α/M]",
    "mg_h": "[Mg/H]",
}


def _stack_by_bin(values: np.ndarray, tiers: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.zeros((3, len(bins) - 1))
    for ti in range(3):
        h, _ = np.histogram(values[tiers == ti + 1], bins=bins)
        out[ti] = h
    totals = out.sum(axis=0).clip(min=1)
    return out / totals * 100


def _render(parquet: Path, stream_label: str, out_png: Path) -> None:
    if not parquet.exists():
        return
    cols = (
        ["g_mag", "r_med_photogeo", "release_tier"]
        + [f"release_tier__{e}" for e in ELEMENTS]
        + [f"prediction_sigma_inflated__{e}" for e in ELEMENTS]
    )
    avail = pd.read_parquet(parquet, columns=None).columns
    cols = [c for c in cols if c in avail]
    df = pd.read_parquet(parquet, columns=cols)
    _do_render(df, stream_label, out_png)


def main() -> None:
    apply_style()
    _render(PARQUET_S3, "Stream 3", OUT / "release_tier_regime.png")
    _render(PARQUET_S2, "Stream 2", OUT / "release_tier_regime_stream2.png")


def _do_render(df: pd.DataFrame, stream_label: str, out_png: Path) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Tier vs G stacked
    if "g_mag" in df.columns:
        bins = np.linspace(8, 18, 41)
        stacks = _stack_by_bin(df["g_mag"].to_numpy(), df["release_tier"].to_numpy(), bins)
        centres = 0.5 * (bins[1:] + bins[:-1])
        bottom = np.zeros_like(centres)
        colors = ("#2ca02c", "#ff7f0e", "#d62728")
        for ti, color, label in zip(range(3), colors, ("Tier 1", "Tier 2", "Tier 3")):
            axes[0, 0].fill_between(
                centres,
                bottom,
                bottom + stacks[ti],
                color=color,
                alpha=0.7,
                label=label,
                step="mid",
            )
            bottom += stacks[ti]
        axes[0, 0].set_xlabel("G (mag)")
        axes[0, 0].set_ylabel("% per G bin")
        axes[0, 0].set_title("Release-tier composition vs G")
        axes[0, 0].legend(fontsize=8)

    # Tier vs distance stacked. r_med_photogeo is in PARSECS in this parquet
    # (range 32-22587 pc); convert to kpc for the binning.
    if "r_med_photogeo" in df.columns:
        bins = np.linspace(0, 10, 41)
        stacks = _stack_by_bin(
            (df["r_med_photogeo"].to_numpy() / 1000.0), df["release_tier"].to_numpy(), bins
        )
        centres = 0.5 * (bins[1:] + bins[:-1])
        bottom = np.zeros_like(centres)
        colors = ("#2ca02c", "#ff7f0e", "#d62728")
        for ti, color, label in zip(range(3), colors, ("Tier 1", "Tier 2", "Tier 3")):
            axes[0, 1].fill_between(
                centres,
                bottom,
                bottom + stacks[ti],
                color=color,
                alpha=0.7,
                label=label,
                step="mid",
            )
            bottom += stacks[ti]
        axes[0, 1].set_xlabel("Bailer-Jones distance (kpc)")
        axes[0, 1].set_ylabel("% per bin")
        axes[0, 1].set_title("Release-tier composition vs distance")
        axes[0, 1].legend(fontsize=8)

    # Per-element tier
    width = 0.25
    x = np.arange(len(ELEMENTS))
    for ti, color, label in zip(
        (1, 2, 3), ("#2ca02c", "#ff7f0e", "#d62728"), ("Tier 1", "Tier 2", "Tier 3")
    ):
        cnts = [int((df[f"release_tier__{e}"] == ti).sum()) for e in ELEMENTS]
        axes[1, 0].bar(x + (ti - 2) * width, cnts, width, color=color, label=label)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([ELEMENT_LABELS[e] for e in ELEMENTS])
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title("Per-element release_tier")
    axes[1, 0].legend(fontsize=8)

    # σ-inflation rate per element
    rates = [(df[f"prediction_sigma_inflated__{e}"].mean()) * 100 for e in ELEMENTS]
    bars = axes[1, 1].bar(x, rates, color="#d62728")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([ELEMENT_LABELS[e] for e in ELEMENTS])
    axes[1, 1].set_ylabel("% with σ-inflated flag")
    axes[1, 1].set_title("Per-element σ-inflation (v4 caveat)")
    for b, r in zip(bars, rates):
        axes[1, 1].text(
            b.get_x() + b.get_width() / 2, r + 0.3, f"{r:.1f}%", ha="center", fontsize=8
        )

    fig.suptitle(
        f"Release-tier regime composition ({stream_label} hybrid release, schema v5)", fontsize=11
    )
    fig.tight_layout()
    save_fig(fig, out_png, tight=False)


if __name__ == "__main__":
    main()
