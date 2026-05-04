"""H2: Out-of-distribution (OOD) gate diagnostics.

Real data visualization of XP-Mahalanobis distance distributions on Streams 2, 3.
Shows the OOD decision boundary (p=0.99 threshold from training data) and
per-stream fraction of stars flagged as OOD. Illustrates how the hybrid system
identifies and gates unreliable predictions.

Uses real Mahalanobis distances from predictions parquets for Streams 2 and 3.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/H_hybrid_release"


def main(argv: list[str] | None = None) -> int:
    apply_style()
    print("[H2] Loading real OOD gate data from Streams 2, 3")

    streams = {}
    for stream_id in (2, 3):
        try:
            pred = pd.read_parquet(
                REPO / f"data/processed/pipeline1_predictions_stream{stream_id}.parquet",
                columns=["source_id", "ood_mahalanobis_score"],
            )
            pred["stream"] = f"Stream {stream_id}"
            streams[f"s{stream_id}"] = pred
        except FileNotFoundError as e:
            print(f"  Warning: Stream {stream_id} not available: {e}")

    if not streams:
        print("Error: no streams could be loaded")
        return 1

    # Compute p=0.99 threshold from Stream 2 (proxy for training). Use
    # nanpercentile because a small fraction of XP rows fail Hermite
    # re-projection and surface as NaN in ``ood_mahalanobis_score`` —
    # plain np.percentile would return NaN for the threshold and silently
    # zero out the right-panel OOD fractions.
    s2_mahal = streams["s2"]["ood_mahalanobis_score"].to_numpy()
    threshold_p99 = float(np.nanpercentile(s2_mahal, 99.0))

    print("[H2] Rendering OOD gate diagnostics")
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: histograms
    ax = axes[0]
    colors = {
        "Stream 2": "#d62728",
        "Stream 3": "#ff7f0e",
    }

    for _stream_name, df in streams.items():
        color = colors.get(df["stream"].iloc[0], "#1f77b4")
        mahal = df["ood_mahalanobis_score"].to_numpy()
        mahal = mahal[~np.isnan(mahal)]  # Drop NaNs
        if len(mahal) > 0:
            ax.hist(
                mahal,
                bins=40,
                alpha=0.5,
                color=color,
                label=df["stream"].iloc[0],
                edgecolor=color,
                lw=0.5,
            )

    ax.axvline(
        threshold_p99,
        color="k",
        lw=2.0,
        ls="--",
        label=f"p=0.99 threshold: {threshold_p99:.2f}",
    )
    ax.set_xlabel(r"Mahalanobis distance")
    ax.set_ylabel("# stars")
    ax.set_title("Distribution by stream")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # Right panel: per-stream OOD fraction
    ax = axes[1]
    stream_names_avail = list(streams.keys())
    ood_fractions = []
    ood_counts = []
    total_counts = []
    colors_list = []

    for stream_key in stream_names_avail:
        df = streams[stream_key]
        mahal = df["ood_mahalanobis_score"].to_numpy()
        mahal = mahal[~np.isnan(mahal)]
        if len(mahal) > 0:
            n_total = len(mahal)
            n_ood = (mahal > threshold_p99).sum()
            ood_fractions.append(100.0 * n_ood / n_total)
            ood_counts.append(int(n_ood))
            total_counts.append(n_total)
            colors_list.append(colors.get(df["stream"].iloc[0], "#1f77b4"))

    stream_labels = [streams[k]["stream"].iloc[0] for k in stream_names_avail]
    bars = ax.bar(range(len(stream_labels)), ood_fractions, color=colors_list)
    ax.set_ylabel("OOD fraction (%)")
    ax.set_title("OOD-flagged stars (threshold p=0.99)")
    ax.set_xticks(range(len(stream_labels)))
    ax.set_xticklabels(stream_labels, rotation=15, ha="right")
    ax.set_ylim(0, max(ood_fractions) * 1.2 if ood_fractions else 1.0)

    for _i, (bar, _frac, count, total) in enumerate(
        zip(bars, ood_fractions, ood_counts, total_counts)
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{count}/{total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Streams 2, 3 — Mahalanobis OOD gate (p=0.99 threshold from Stream 2)",
        fontsize=11,
        fontweight="semibold",
    )
    fig.set_layout_engine("constrained")
    save_fig(fig, OUT / "H2_ood_gate_diagnostics.pdf", formats=("pdf", "png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
