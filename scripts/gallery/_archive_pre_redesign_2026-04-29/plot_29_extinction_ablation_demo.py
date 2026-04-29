"""Stage 29: extinction-recipe ablation gallery (synthetic demo).

Re-runs ``scripts.ablation_extinction_recipe.run_ablation`` on its
synthetic fixture so the methods-paper Figure-5 set is regenerated as
part of the gallery sweep. On real-data runs the same harness is
invoked directly via ``scripts/ablation_extinction_recipe.py
--baseline ... --hybrid ... --truth ...``; this gallery stage exists so
the visual contract is testable today on the maintainer's WSL2
environment without the v2 ensemble.

Outputs land at ``reports/gallery/29_extinction_ablation_demo/``:

- 5 ``residual_vs_av_<element>.pdf/.png`` panels.
- 5 ``residual_vs_quadrant_<element>.pdf/.png`` panels.
- 1 ``intrinsic_colour_vs_alpha_m.pdf/.png`` panel.
- ``slopes.csv``, ``slopes.json``, ``summary.json`` quantitative tables.
"""

from __future__ import annotations

# isort: off
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from ablation_extinction_recipe import (  # noqa: E402
    _build_synthetic_ablation_fixture,
    run_ablation,
)

# isort: on


OUT = REPO / "reports/gallery/29_extinction_ablation_demo"


def main() -> None:
    print("[plot_29] Synthetic ablation fixture (4000 stars × 2 recipes)")
    baseline, hybrid, truth = _build_synthetic_ablation_fixture()

    print("[plot_29] Running residual + quadrant + intrinsic-colour analyses")
    summary = run_ablation(baseline, hybrid, truth, out_dir=OUT)
    print(f"[plot_29] verdict: {summary['verdict']}")
    print(f"[plot_29] output: {OUT}")


if __name__ == "__main__":
    main()
