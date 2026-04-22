"""Run gallery batches 4, 5, 6:
  - batch 4: stages 10, 11 (training)
  - batch 5: stages 12, 13 (validation + uncertainty)
  - batch 6: stage 14 (inference)

Stages 15–16 (population-classifier σ-gate and HDBSCAN chemical-plane
evolution) were removed on 2026-04-22 when population classification moved
to the Starfold repository. Historical renders live under
``reports/gallery/archive/``.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

MODS = [
    "plot_10_contrastive",
    "plot_11_supervised",
    "plot_12_pipeline1_validation",
    "plot_13_ensemble_uncertainty",
    "plot_14_pipeline1_inference",
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for m in MODS:
        print(f"\n=== {m} ===")
        t0 = time.time()
        try:
            mod = importlib.import_module(m)
            mod.main()
            print(f"[ok]  {m}  ({time.time()-t0:.1f}s)")
        except Exception as exc:   # noqa: BLE001
            print(f"[FAIL]  {m}: {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
