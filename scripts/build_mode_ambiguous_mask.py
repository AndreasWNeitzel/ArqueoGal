"""Precompute the (Teff, log g, [M/H]) bimodality grid for Pipeline-1 release.

Runs once against the Stream-1 training parquet and emits a serialisable
3-D mask + provenance sidecar. The inference driver loads this artefact
and queries it with the model's predicted (Teff, log g, [M/H]) to attach
``mode_ambiguous_flag`` per star.

Rationale — see ``bimodality.py`` docstring and ADR-0015. Per-star α/M is
not recoverable from XP alone where the training target is bimodal at the
star's cell, because Gaussian NLL collapses to the conditional mean (the
valley between the modes). Option-3 remediation: flag those stars out.

CLI
---

::

    PYTHONPATH=src python scripts/build_mode_ambiguous_mask.py \\
        --training-parquet data/processed/pipeline1_features_stream1.parquet \\
        --output data/processed/mode_ambiguous_grid.npz
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path

import pandas as pd

from arqueogal.xp_abundances.main.bimodality import fit_bimodality_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("build_mode_ambiguous_mask")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_OUT = REPO / "data/processed/mode_ambiguous_grid.npz"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-parquet", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-cell-n", type=int, default=50)
    parser.add_argument("--min-minor-weight", type=float, default=0.15)
    parser.add_argument("--min-mean-sep", type=float, default=0.08)
    parser.add_argument("--bic-delta-min", type=float, default=4.0)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    _LOG.info("reading %s", args.training_parquet)
    df = pd.read_parquet(
        args.training_parquet,
        columns=["teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"],
    )
    n_raw = len(df)
    df = df.dropna(subset=["teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"])
    _LOG.info("rows after dropna: %d / %d", len(df), n_raw)

    grid = fit_bimodality_grid(
        teff=df["teff_apogee"].to_numpy(),
        logg=df["logg_apogee"].to_numpy(),
        mh=df["mh_apogee"].to_numpy(),
        alpha_m=df["alpha_m_apogee"].to_numpy(),
        min_cell_n=args.min_cell_n,
        min_minor_weight=args.min_minor_weight,
        min_mean_sep=args.min_mean_sep,
        bic_delta_min=args.bic_delta_min,
        random_state=args.random_state,
    )

    n_cells_total = int(grid.is_bimodal.size)
    n_cells_eval = int((grid.n_per_cell >= grid.min_cell_n).sum())
    n_cells_bi = int(grid.is_bimodal.sum())
    _LOG.info(
        "cells: total=%d evaluated=%d bimodal=%d",
        n_cells_total,
        n_cells_eval,
        n_cells_bi,
    )
    stars_in_bi = int(grid.n_per_cell[grid.is_bimodal].sum())
    _LOG.info(
        "training stars in bimodal cells: %d / %d  (%.2f%%)",
        stars_in_bi,
        int(grid.n_per_cell.sum()),
        100.0 * stars_in_bi / max(1, int(grid.n_per_cell.sum())),
    )

    provenance = {
        "training_parquet": str(args.training_parquet),
        "training_parquet_sha256": _sha256(args.training_parquet),
        "training_n_rows_raw": n_raw,
        "training_n_rows_finite": int(len(df)),
        "script": "scripts/build_mode_ambiguous_mask.py",
        "git_sha": _git_sha(),
        "training_stars_in_bimodal_cells": stars_in_bi,
    }
    grid.save(args.output, provenance=provenance)
    _LOG.info("wrote %s", args.output)
    _LOG.info(
        "wrote %s",
        args.output.with_suffix(args.output.suffix + ".provenance.json"),
    )


if __name__ == "__main__":
    main()
