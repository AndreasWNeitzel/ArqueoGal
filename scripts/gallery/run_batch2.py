"""Run all batch-2 gallery plot scripts (stages 00..06 except 03 which is existing).

One command produces every data-layer figure:
    PYTHONPATH=src python scripts/gallery/run_batch2.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gallery"))
sys.path.insert(0, str(ROOT / "src"))

STAGES = [
    ("plot_00_data_sources",     "00 data sources"),
    ("plot_01_gaia_xp_raw",      "01 raw XP"),
    ("plot_02_ye_correction",    "02 Ye correction"),
    ("plot_04_extinction",       "04 extinction"),
    ("plot_05_ir_photometry",    "05 IR photometry"),
    ("plot_06_selection_function","06 selection function"),
]


def main() -> None:
    total_start = time.time()
    for module_name, label in STAGES:
        print(f"\n=== {label} ===", flush=True)
        start = time.time()
        try:
            mod = __import__(module_name)
            mod.main()
        except Exception:
            traceback.print_exc()
            print(f"!!! {label} FAILED — continuing", flush=True)
        print(f"    {label} done in {time.time() - start:.1f}s", flush=True)
    print(f"\nTOTAL {time.time() - total_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
