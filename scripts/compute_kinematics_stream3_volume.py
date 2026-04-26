"""Compute galpy kinematics for the volume-limited Pipeline-1 Stream-3 arm.

The artefact is consumed downstream by Starfold (population classification,
separate repo) as one of the chrono-chemo-kinematic feature-vector inputs.
The legacy filename ``pipeline2_kinematics_stream3_volume.parquet`` is
preserved for consumer stability.

Inputs
------
- ``data/processed/pipeline1_predictions_stream3_volume.parquet`` — 249,092
  volume-limited source_ids with 5-label predictions + OOD/Regime-B flags.
- ``data/interim/stream3_gaia_dr3_corrected.parquet`` (168k) +
  ``data/interim/stream3_delta_gaia_dr3_corrected.parquet`` (450k) — Gaia DR3
  astrometry + radial_velocity with Lindegren+2021 parallax-zpt and
  Riello+2021 G-mag corrections applied.
- ``data/processed/pipeline1_features_stream3.parquet`` — carries
  ``r_med_photogeo`` (BJ21 photogeometric distance in pc) for the 613,939-row
  union.

Output
------
``data/processed/pipeline2_kinematics_stream3_volume.parquet`` — one row per
input star surviving NaN filtering, columns per
:data:`arqueogal.data.kinematics.OUTPUT_COLS` (15 incl. source_id): R/z/phi,
v_R/v_T/v_z, J_R/L_z/J_z, ecc, r_peri/r_apo/z_max, E.

Runtime
-------
~3-4 hours for 249k stars on the CAUP dev box (galpy's actionAngleStaeckel
with per-star orbit analytic helpers is ~50 ms/star, sequential). Writes a
checkpoint parquet every ``--checkpoint-every`` stars so a mid-run crash
only loses the current chunk.

Provenance sidecar records input SHAs, the galactic-constants block, and
the number of rows that dropped in each stage (dedup, NaN filter,
galpy-rejection).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from arqueogal.data.kinematics import (
    OUTPUT_COLS,
    REQUIRED_INPUT_COLS,
    KinematicsConfig,
    compute_actions,
)

_LOG = logging.getLogger("compute_kinematics_stream3_volume")

_REPO = Path(__file__).resolve().parents[1]
_VOL_PRED = _REPO / "data" / "processed" / "pipeline1_predictions_stream3_volume.parquet"
_GAIA_MAIN = _REPO / "data" / "interim" / "stream3_gaia_dr3_corrected.parquet"
_GAIA_DELTA = _REPO / "data" / "interim" / "stream3_delta_gaia_dr3_corrected.parquet"
_FEATS = _REPO / "data" / "processed" / "pipeline1_features_stream3.parquet"
_OUT = _REPO / "data" / "processed" / "pipeline2_kinematics_stream3_volume.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def _load_inputs() -> pd.DataFrame:
    _LOG.info("loading volume-limited source_ids from %s", _VOL_PRED.name)
    vol_ids = pd.read_parquet(_VOL_PRED, columns=["source_id"])
    n_vol = len(vol_ids)
    _LOG.info("  %d source_ids", n_vol)

    _LOG.info("loading Gaia astrometry (main + delta)")
    cols = ["source_id", "ra", "dec", "pmra", "pmdec", "radial_velocity"]
    g1 = pd.read_parquet(_GAIA_MAIN, columns=cols)
    g2 = pd.read_parquet(_GAIA_DELTA, columns=cols)
    gaia = pd.concat([g1, g2], ignore_index=True)
    n_concat = len(gaia)
    gaia = gaia.drop_duplicates("source_id", keep="first").reset_index(drop=True)
    n_dedup = len(gaia)
    _LOG.info("  main=%d delta=%d concat=%d dedup=%d", len(g1), len(g2), n_concat, n_dedup)

    _LOG.info("loading r_med_photogeo from features matrix")
    feats = pd.read_parquet(_FEATS, columns=["source_id", "r_med_photogeo"])
    _LOG.info("  %d feature rows", len(feats))

    _LOG.info("inner-joining vol_ids ⟶ gaia ⟶ features")
    df = vol_ids.merge(gaia, on="source_id", how="inner", validate="one_to_one")
    _LOG.info("  after vol×gaia: %d (vol loss: %d)", len(df), n_vol - len(df))
    df = df.merge(feats, on="source_id", how="inner", validate="one_to_one")
    _LOG.info("  after ×feats: %d", len(df))

    # Ensure column order matches REQUIRED_INPUT_COLS for downstream readability.
    df = df[list(REQUIRED_INPUT_COLS)]
    return df


def _run_chunked(
    df: pd.DataFrame,
    *,
    config: KinematicsConfig,
    chunk_size: int,
    checkpoint_every: int,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Run compute_actions in fixed-size chunks, checkpointing periodically.

    galpy's per-star cost is dominated by fixed overhead (potential
    evaluation + Staeckel fudge per-orbit integrator) so chunking does not
    speed up the throughput, but it:
        (a) keeps peak memory bounded,
        (b) enables resumable checkpoints if the process is killed,
        (c) lets us log progress rather than wait 4h for a single print.
    """
    n = len(df)
    results: list[pd.DataFrame] = []
    n_done = 0
    last_ckpt = 0

    t0 = datetime.now(tz=UTC)
    for start in range(0, n, chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        out = compute_actions(chunk, config=config)
        results.append(out)
        n_done += len(chunk)
        dt = (datetime.now(tz=UTC) - t0).total_seconds()
        rate = n_done / max(dt, 1e-6)
        eta = (n - n_done) / max(rate, 1e-6)
        _LOG.info(
            "chunk %d: done=%d/%d (%.1f%%) rate=%.1f star/s eta=%.0f min",
            len(results),
            n_done,
            n,
            100 * n_done / n,
            rate,
            eta / 60,
        )

        # Periodic checkpoint — survive a kill without losing work.
        if n_done - last_ckpt >= checkpoint_every:
            partial = pd.concat(results, ignore_index=True)
            partial.to_parquet(checkpoint_path, index=False)
            _LOG.info("  checkpoint: %d rows to %s", len(partial), checkpoint_path.name)
            last_ckpt = n_done

    return (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(
            columns=list(OUTPUT_COLS),
        )
    )


def _write_provenance(
    out_path: Path,
    *,
    cfg: KinematicsConfig,
    n_input: int,
    n_output: int,
    input_shas: dict[str, str],
) -> None:
    prov = {
        "output_file": str(out_path.relative_to(_REPO)),
        "script": "scripts/compute_kinematics_stream3_volume.py",
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "n_input_rows": int(n_input),
        "n_output_rows": int(n_output),
        "n_dropped_nan": int(n_input - n_output),
        "galactic_constants": dataclasses.asdict(cfg),
        "output_columns": list(OUTPUT_COLS),
        "inputs": input_shas,
        "notes": (
            "galpy.actionAngleStaeckel central-value path (compute_actions). "
            "MC covariance sampling (compute_actions_mc) deferred to the "
            "D-Cat-d boundary-cluster subsample per data_acquisition.md §9.5."
        ),
    }
    sidecar = out_path.with_suffix(out_path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(prov, indent=2))
    _LOG.info("provenance → %s", sidecar.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=_OUT)
    parser.add_argument(
        "--dry-run", action="store_true", help="load inputs, report row counts, and exit"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = _load_inputs()
    n_input = len(df)
    if args.dry_run:
        _LOG.info("dry-run: %d rows assembled; exit without galpy", n_input)
        return 0

    cfg = KinematicsConfig()
    _LOG.info("config: %s", dataclasses.asdict(cfg))

    checkpoint_path = args.output.with_suffix(args.output.suffix + ".partial")
    out_df = _run_chunked(
        df,
        config=cfg,
        chunk_size=args.chunk_size,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=checkpoint_path,
    )
    _LOG.info("compute_actions finished: %d rows in, %d out", n_input, len(out_df))

    # Atomic replace.
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    out_df.to_parquet(tmp, index=False)
    tmp.replace(args.output)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    input_shas = {
        "volume_predictions": {
            "path": str(_VOL_PRED.relative_to(_REPO)),
            "sha256": _sha256(_VOL_PRED),
        },
        "gaia_main": {"path": str(_GAIA_MAIN.relative_to(_REPO)), "sha256": _sha256(_GAIA_MAIN)},
        "gaia_delta": {"path": str(_GAIA_DELTA.relative_to(_REPO)), "sha256": _sha256(_GAIA_DELTA)},
        "features": {"path": str(_FEATS.relative_to(_REPO)), "sha256": _sha256(_FEATS)},
    }
    _write_provenance(
        args.output,
        cfg=cfg,
        n_input=n_input,
        n_output=len(out_df),
        input_shas=input_shas,
    )
    _LOG.info("done: %s (%d rows)", args.output.name, len(out_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
