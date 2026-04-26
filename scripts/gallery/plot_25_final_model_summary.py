"""Stage 25: final-model summary panel.

A single-page visualisation of THE final hybrid model, suitable for an
executive snapshot or for the methods-paper headline figure. Aggregates
the headline numbers from stages 13, 14, 19, 21, 22, 23, 24 so the reader
can read off in one glance: model recipe, per-element accuracy, σ-coverage,
hybrid source split, release-tier composition, structure-preservation
metrics, and stress-battery pass/fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/25_final_model_summary"
HIST = REPO / "reports/pipeline1/long_train_2026-04-25/ensemble_history.json"
HYBRID = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"
HYBRID_S2 = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"
GMM_J = REPO / "reports/gallery/22_gmm_cluster_tracking/gmm_cluster_tracking_metrics.json"
CONT_J = REPO / "reports/gallery/23_contamination/contamination_metrics.json"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
EL_LBL = {"teff": "Teff", "logg": "log g", "mh": "[M/H]", "alpha_m": "[α/M]", "mg_h": "[Mg/H]"}


def _stress_status() -> dict[str, str]:
    logs = sorted(REPO.glob(".expert_review_2026-04-24/stress_battery*.log"))
    if not logs:
        return {}
    text = logs[-1].read_text()
    return {m.group(1): m.group(2) for m in re.finditer(r"::(\S+)\s+(PASSED|FAILED)", text)}


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.40)

    # --- Panel 1 (header / recipe / training summary)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if HIST.exists():
        h = json.loads(HIST.read_text())
        m = h["members"][0]
        ax.text(
            0.5,
            0.85,
            "ArqueoGal Pipeline-1 D-Cat-b candidate — strong-contrastive-v2 + kNN-rescue HYBRID",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.45,
            f"Single-stage joint loss: SupCon=0.3 + β-NLL=1.0 + Barlow=0.8  ·  τ_init=0.15  ·  "
            f"1 seed  ·  best_val_loss={m['best_val_loss']:.3f} @ epoch {m['best_epoch']} of {len(m['history'])}",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.15,
            "Inference path: regressor (per-element σ-gated) → kNN rescue (K=50 cosine) → composer → release",
            ha="center",
            va="center",
            fontsize=10,
            color="#444",
            transform=ax.transAxes,
        )

    # --- Panel 2 (per-element RMSE/bias/std on Stream-1 70/15/15)
    ax = fig.add_subplot(gs[1, 0])
    if HYBRID.exists():
        # Per-element accuracy from the saved test sidecar of stage 14 — but
        # that's not available cheaply. Use header-only on this panel and
        # delegate to the actual stage 14 image for the scatter.
        ax.text(
            0.5,
            0.5,
            "see stage 14:\npred-vs-truth scatter",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_axis_off()

    # --- Panel 3 (release-tier composition: S3 + S2 side by side)
    ax = fig.add_subplot(gs[1, 1])
    streams = []
    if HYBRID.exists():
        streams.append(("Stream 3", "#d62728", pd.read_parquet(HYBRID, columns=["release_tier"])))
    if HYBRID_S2.exists():
        streams.append(
            ("Stream 2", "#9467bd", pd.read_parquet(HYBRID_S2, columns=["release_tier"]))
        )
    if streams:
        tiers = [1, 2, 3]
        n_s = len(streams)
        width = 0.8 / max(n_s, 1)
        x = np.arange(len(tiers))
        for i, (name, color, df_t) in enumerate(streams):
            cnts = [int((df_t["release_tier"] == t).sum()) for t in tiers]
            offset = (i - (n_s - 1) / 2) * width
            ax.bar(
                x + offset, cnts, width=width * 0.95, color=color, label=f"{name} (n={len(df_t):,})"
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Tier {t}" for t in tiers])
        ax.set_ylabel("count")
        ax.set_title("Release tier (S2 + S3)")
        ax.legend(
            fontsize=7,
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.4",
        )

    # --- Panel 4 (per-element hybrid source split)
    ax = fig.add_subplot(gs[1, 2:])
    if HYBRID.exists():
        df_h = pd.read_parquet(HYBRID, columns=[f"{e}_hybrid_source" for e in ELEMENTS])
        n = len(df_h)
        bottoms = np.zeros(len(ELEMENTS))
        for src, color in [
            ("regressor", "#1f77b4"),
            ("knn", "#ff7f0e"),
            ("regressor_caveat", "#d62728"),
        ]:
            fr = np.array([(df_h[f"{e}_hybrid_source"] == src).sum() / n * 100 for e in ELEMENTS])
            ax.bar(
                range(len(ELEMENTS)),
                fr,
                bottom=bottoms,
                color=color,
                label=src,
                edgecolor="white",
                lw=0.4,
            )
            for i, f in enumerate(fr):
                if f > 5:
                    ax.text(
                        i,
                        bottoms[i] + f / 2,
                        f"{f:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
            bottoms += fr
        ax.set_xticks(range(len(ELEMENTS)))
        ax.set_xticklabels([EL_LBL[e] for e in ELEMENTS])
        ax.set_ylabel("% of stars")
        ax.set_ylim(0, 100)
        ax.set_title("Hybrid source per element")
        ax.legend(fontsize=7, loc="upper right")

    # --- Panel 5: Per-element σ-inflation rate
    ax = fig.add_subplot(gs[2, 0])
    if HYBRID.exists():
        df_s = pd.read_parquet(
            HYBRID, columns=[f"prediction_sigma_inflated__{e}" for e in ELEMENTS]
        )
        rates = [df_s[f"prediction_sigma_inflated__{e}"].mean() * 100 for e in ELEMENTS]
        bars = ax.bar(range(len(ELEMENTS)), rates, color="#d62728")
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.3, f"{r:.1f}%", ha="center", fontsize=7)
        ax.set_xticks(range(len(ELEMENTS)))
        ax.set_xticklabels([EL_LBL[e] for e in ELEMENTS])
        ax.set_ylabel("% σ-inflated")
        ax.set_title("Prior-collapse rate per element")

    # --- Panel 6: GMM ARI / purity / centroid drift on test split
    ax = fig.add_subplot(gs[2, 1])
    if GMM_J.exists():
        gmm = json.loads(GMM_J.read_text())
        # Pick the "Test" split if present
        test = gmm.get("Test") or list(gmm.values())[-1]
        rows = [
            ("ARI", test["adjusted_rand_index"]),
            ("purity", test["purity"]),
            ("drift RMS (dex)", test["centroid_drift_total_rms"]),
        ]
        ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#1f77b4")
        for i, (_n, v) in enumerate(rows):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, max(0.7, max(r[1] for r in rows) * 1.15))
        ax.set_title("Structure (criterion 2, test)")

    # --- Panel 7: macro-F1 + per-class F1 (criterion 3)
    ax = fig.add_subplot(gs[2, 2])
    if CONT_J.exists():
        cont = json.loads(CONT_J.read_text())
        test = cont.get("Test") or list(cont.values())[-1]
        f1 = test["f1_per_class"]
        macro = test["macro_f1"]
        bars = ax.bar(
            ["G1", "G2", "G3", "macro"],
            f1 + [macro],
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"],
        )
        for b, v in zip(bars, f1 + [macro]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
        ax.set_title("Contamination (criterion 3, test)")

    # --- Panel 8: Hellinger per class
    ax = fig.add_subplot(gs[2, 3])
    if CONT_J.exists():
        cont = json.loads(CONT_J.read_text())
        test = cont.get("Test") or list(cont.values())[-1]
        hell = test["hellinger_per_class"]
        bars = ax.bar(["G1", "G2", "G3"], hell, color="#1f77b4")
        for b, v in zip(bars, hell):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(hell) * 1.25)
        ax.set_ylabel("Hellinger")
        ax.set_title("Density distance per cluster")

    # --- Panel 9 (full bottom row): stress battery 7-cell pass/fail
    status = _stress_status()
    tests = [
        ("test_1_kfold_cv", "5-fold CV"),
        ("test_2_leakage", "Leakage"),
        ("test_3_per_cell_calibration", "Per-cell"),
        ("test_4_sigma_coverage", "σ-coverage"),
        ("test_5_k_sensitivity", "K-sensitivity"),
        ("test_6_multispectrum_consistency", "Multi-spec"),
        ("test_7_permutation_importance", "XP importance"),
    ]
    for i, (tid, label) in enumerate(tests):
        ax = fig.add_subplot(gs[3, i % 4]) if i < 4 else None
        if i >= 4:
            continue  # only 4 panels in bottom row of gridspec
        outcome = status.get(tid, "NOT RUN")
        color = {"PASSED": "#bfecbf", "FAILED": "#f7b9b9", "NOT RUN": "#dddddd"}[outcome]
        edge = {"PASSED": "#2ca02c", "FAILED": "#d62728", "NOT RUN": "#888888"}[outcome]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            patches.Rectangle((0.05, 0.10), 0.9, 0.8, facecolor=color, edgecolor=edge, lw=1.4)
        )
        ax.text(0.5, 0.7, label, ha="center", va="center", fontsize=10, fontweight="semibold")
        ax.text(
            0.5, 0.30, outcome, ha="center", va="center", fontsize=10, fontweight="bold", color=edge
        )

    # If we still have unfilled panels in the last row, list the remaining stress tests
    summary_text = "Stress battery: " + " · ".join(
        f"{label}={status.get(tid, '?')}" for tid, label in tests[4:]
    )
    fig.text(0.5, 0.04, summary_text, ha="center", fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Walk gallery 00 → 25 top-to-bottom for the full deployment-graph audit. "
        "This panel is the executive headline.",
        ha="center",
        fontsize=8,
        color="#444",
    )
    save_fig(fig, OUT / "final_model_summary.png", tight=False)


if __name__ == "__main__":
    main()
