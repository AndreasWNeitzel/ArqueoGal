"""Run gallery batch 3: stages 03 (missed) + 07, 08, 09.

$ python scripts/gallery/run_batch3.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

MODS = [
    "plot_03_hermite_reprojection",
    "plot_07_apogee_labels",
    "plot_08_kinematics",
    "plot_09_feature_matrix",
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
