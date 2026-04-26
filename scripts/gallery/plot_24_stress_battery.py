"""Stage 23: hybrid stress-battery summary (the 7 quantitative gates).

What the deploy did: ``tests/integration/test_hybrid_stress_battery.py`` (run
with ``--run-stress``) executed seven independent quantitative tests on real
Stream-1 / Stream-3 data. Last run: 7/7 passed in 122 seconds (log
``.expert_review_2026-04-24/stress_battery_2026-04-25_iter2.log``).

What we plot: a 7-cell pass/fail summary panel (red = fail, green = pass)
with the exact quantitative bound and observed value per test. If the
stress battery has not been re-run on this checkout, the panel emits a
"NOT RUN" banner.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/24_stress_battery"
LOG_GLOB = ".expert_review_2026-04-24/stress_battery*.log"

TESTS = (
    ("test_1_kfold_cv", "5-fold CV stability",
     "per-fold RMSE std < 10% of mean"),
    ("test_2_leakage", "Leakage check",
     "Δ-RMS on overlap < per-element noise floor"),
    ("test_3_per_cell_calibration", "Per-cell calibration",
     "worst cell (n≥200) < 3× global RMSE"),
    ("test_4_sigma_coverage", "σ coverage",
     "IQR=50% & 1σ=68% within ±5pp"),
    ("test_5_k_sensitivity", "K sensitivity",
     "RMSE spread across K∈{20,50,100} < 10%"),
    ("test_6_multispectrum_consistency", "Multi-spectrum consistency",
     "median |Δ| < per-element noise floor"),
    ("test_7_permutation_importance", "XP block importance",
     "spectrum-dominant elements ≥ 1.15× baseline RMSE"),
)


def _read_status() -> dict[str, str]:
    """Parse latest stress-battery log for PASSED/FAILED per test."""
    logs = sorted((REPO).glob(LOG_GLOB))
    if not logs:
        return {}
    text = logs[-1].read_text()
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.search(r"::(\S+)\s+(PASSED|FAILED)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> None:
    apply_style()
    status = _read_status()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9)
    ax.axis("off")

    if not status:
        ax.text(5, 4.5,
                 "Stress battery has not been run on this checkout.\n\n"
                 "To run:\n"
                 "    pytest tests/integration --run-stress -v\n\n"
                 "(takes ~2 min on RTX 3060)",
                 ha="center", va="center", fontsize=12,
                 bbox=dict(facecolor="#fdd", edgecolor="#d62728", boxstyle="round,pad=1"))
        save_fig(fig, OUT / "stress_battery.png")
        return

    # 7-cell grid, 2 rows
    n_passed = sum(1 for v in status.values() if v == "PASSED")
    n_failed = sum(1 for v in status.values() if v == "FAILED")
    ax.text(5, 8.4, f"Hybrid stress battery: {n_passed} PASSED / {n_failed} FAILED "
            f"out of {len(TESTS)} (log: {sorted(REPO.glob(LOG_GLOB))[-1].name})",
            ha="center", fontsize=10)

    for i, (test_id, title, criterion) in enumerate(TESTS):
        row, col = i // 4, i % 4
        x = 0.1 + col * 2.45
        y = 6.0 - row * 3.0
        outcome = status.get(test_id, "NOT RUN")
        color = {"PASSED": "#bfecbf", "FAILED": "#f7b9b9", "NOT RUN": "#dddddd"}[outcome]
        edge = {"PASSED": "#2ca02c", "FAILED": "#d62728", "NOT RUN": "#888888"}[outcome]
        ax.add_patch(patches.Rectangle((x, y), 2.3, 2.5,
                                          facecolor=color, edgecolor=edge, lw=1.4))
        ax.text(x + 1.15, y + 2.15, title, ha="center", va="top",
                 fontsize=10, fontweight="semibold")
        ax.text(x + 1.15, y + 1.45, criterion, ha="center", va="top",
                 fontsize=7.5, color="#333", wrap=True)
        ax.text(x + 1.15, y + 0.45, outcome, ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=edge)
    fig.suptitle("Hybrid stress battery: 7 quantitative gates on real Stream-1/Stream-3 data",
                  fontsize=11)
    save_fig(fig, OUT / "stress_battery.png")


if __name__ == "__main__":
    main()
