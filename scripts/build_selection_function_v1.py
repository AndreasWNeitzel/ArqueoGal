"""Build the Ye+2024 ``NO_SYNTH_PHOT`` selection-function v1 artefact.

Consumes ``data/processed/pipeline1_features_stream1.parquet`` (columns
``b_deg``, ``g_mag``, ``ye2024_flag``) and writes:

- ``reports/selection_function/selection_function_v1.parquet`` — the 5×5
  ``(|b|, G)`` grid with per-cell counts, flagged counts, flag rates, and
  ``selection_prob = 1 − flag_rate`` clipped to ``[0.01, 1.0]``.
- ``reports/selection_function/selection_function_v1.provenance.json``.

The grid edges are hard-coded here so the artefact is reproducible and the
module-level :mod:`arqueogal.data.selection_function` scorer remains free
of the binning logic. See ``docs/data_acquisition.md`` §6.4 for the
scientific context and ``reports/selection_function/selection_function_v1.md``
for the full methodology narrative.

Read-only on the input Parquet. Writes atomically (tmp + rename).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar
from arqueogal.data.selection_function import SELECTION_PROB_CEIL, SELECTION_PROB_FLOOR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_selection_function_v1")

REPO: Final[Path] = Path(__file__).resolve().parents[1]
INPUT_PARQUET: Final[Path] = REPO / "data" / "processed" / "pipeline1_features_stream1.parquet"
OUTPUT_PARQUET: Final[Path] = (
    REPO / "reports" / "selection_function" / "selection_function_v1.parquet"
)

# 5×5 grid edges, Thread-1 informed. |b| edges cover plane / mid-plane / disc /
# high-latitude / cap; G edges span bright → XP-native faint end (17.65).
B_EDGES: Final[np.ndarray] = np.array([0.0, 5.0, 10.0, 20.0, 45.0, 90.0], dtype=np.float64)
G_EDGES: Final[np.ndarray] = np.array([2.0, 11.0, 12.5, 14.0, 15.5, 17.65], dtype=np.float64)

METHOD_VERSION: Final[str] = "v1-grid-5x5"
SPARSE_CELL_THRESHOLD: Final[int] = 200


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    logger.info("wrote %s (%d rows)", path, len(df))


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 5×5 grid as a tidy DataFrame (one row per cell)."""
    abs_b = np.abs(df["b_deg"].to_numpy(dtype=np.float64))
    g = df["g_mag"].to_numpy(dtype=np.float64)
    flag = df["ye2024_flag"].to_numpy(dtype=np.int8)

    n_b = len(B_EDGES) - 1
    n_g = len(G_EDGES) - 1

    rows: list[dict[str, float | int]] = []
    for ib in range(n_b):
        b_lo, b_hi = float(B_EDGES[ib]), float(B_EDGES[ib + 1])
        in_b = (abs_b >= b_lo) & (abs_b < b_hi if ib < n_b - 1 else abs_b <= b_hi)
        for ig in range(n_g):
            g_lo, g_hi = float(G_EDGES[ig]), float(G_EDGES[ig + 1])
            in_g = (g >= g_lo) & (g < g_hi if ig < n_g - 1 else g <= g_hi)
            mask = in_b & in_g
            n_total = int(mask.sum())
            n_flag = int(flag[mask].sum())
            flag_rate = float(n_flag) / n_total if n_total > 0 else 0.0
            sel_prob = float(np.clip(1.0 - flag_rate, SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL))
            rows.append(
                {
                    "b_lo": b_lo,
                    "b_hi": b_hi,
                    "g_lo": g_lo,
                    "g_hi": g_hi,
                    "n_total": n_total,
                    "n_flagged": n_flag,
                    "flag_rate": flag_rate,
                    "selection_prob": sel_prob,
                }
            )
    grid = pd.DataFrame(rows)
    logger.info(
        "built %d-cell grid (%d total rows, %d flagged, %d sparse cells n<%d)",
        len(grid),
        int(grid["n_total"].sum()),
        int(grid["n_flagged"].sum()),
        int((grid["n_total"] < SPARSE_CELL_THRESHOLD).sum()),
        SPARSE_CELL_THRESHOLD,
    )
    return grid


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise SystemExit(f"input parquet not found: {INPUT_PARQUET}")

    logger.info("loading %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET, columns=["b_deg", "g_mag", "ye2024_flag"])
    logger.info("loaded %d rows; %d flagged (%.2f%%)",
                len(df), int((df["ye2024_flag"] == 1).sum()),
                100.0 * float((df["ye2024_flag"] == 1).mean()))

    grid = build_grid(df)
    _atomic_write_parquet(grid, OUTPUT_PARQUET)

    input_sha = _sha256_of(INPUT_PARQUET)
    sparse_cells = int((grid["n_total"] < SPARSE_CELL_THRESHOLD).sum())
    prov = Provenance(
        output_file=str(OUTPUT_PARQUET.relative_to(REPO)),
        script="scripts/build_selection_function_v1.py",
        sources=[
            LocalSource(
                name="Stream 1 Pipeline 1 features (Thread-1 input)",
                path=str(INPUT_PARQUET.relative_to(REPO)),
                sha256=input_sha,
            ),
        ],
        cuts_applied=[],
        corrections=[],
        row_count_before=int(len(df)),
        row_count_after=int(grid["n_total"].sum()),
        notes=(
            "Ye+2024 NO_SYNTH_PHOT selection function on (|b|, G). "
            f"Grid: 5x5, edges b={B_EDGES.tolist()} deg, g={G_EDGES.tolist()}. "
            f"sparse cells (n < {SPARSE_CELL_THRESHOLD}): {sparse_cells}. "
            "Floor 0.01, ceil 1.0 on selection_prob."
        ),
        extra={
            "method_version": METHOD_VERSION,
            "b_edges_deg": B_EDGES.tolist(),
            "g_edges_mag": G_EDGES.tolist(),
            "selection_prob_floor": SELECTION_PROB_FLOOR,
            "selection_prob_ceil": SELECTION_PROB_CEIL,
            "n_sparse_cells": sparse_cells,
            "sparse_cell_threshold": SPARSE_CELL_THRESHOLD,
            "global_flag_rate": float((df["ye2024_flag"] == 1).mean()),
        },
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
