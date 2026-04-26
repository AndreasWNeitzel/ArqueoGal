"""Compute kin_ood_flag for Stream-3 stars.

Phase B Stage 1 of the operational rollout: replaces the v1 placeholder
``kin_ood_flag = False everywhere`` with a Mahalanobis-on-velocity OOD
detector (``xp_abundances.main.kinematic_ood``).

Approach:
1. Load Stream-3 kinematics from
   ``data/processed/pipeline2_kinematics_stream3_volume.parquet``
   (Galactocentric cylindrical velocities v_R, v_T = v_φ, v_z).
2. Select a "disc-like" subset by a coarse kinematic cut
   (|v_z| < 80 km/s AND v_T > 100 km/s). The cut captures the thin/thick
   disc and rejects halo / accreted-debris / counter-rotating stars by
   construction. The 99th-percentile Mahalanobis-distance threshold derived
   on this subset then flags everything outside the disc envelope.
3. Fit ``KinematicOODBundle`` on the disc subset.
4. Score the FULL Stream-3 kinematic set; flag stars whose Mahalanobis
   distance exceeds the bundle's threshold.
5. Emit a per-star table at
   ``data/processed/pipeline1_kin_ood_flag.parquet`` with
   ``[source_id, kin_ood_flag]``. Stars in the predictions parquet without
   kinematics (Stream-3 outside the kinematic-ready subset) are NOT
   included in this artefact; the release pipeline must left-join, leaving
   non-overlapping rows at ``False``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arqueogal.xp_abundances.main.kinematic_ood import (
    KinematicOODBundle,
    fit_kinematic_ood,
    flag_kinematic_ood,
    score_kinematic_ood,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("build_kin_ood_flag")

KIN_PARQUET = REPO / "data/processed/pipeline2_kinematics_stream3_volume.parquet"
OUT_PARQUET = REPO / "data/processed/pipeline1_kin_ood_flag.parquet"
BUNDLE_JSON = REPO / "data/processed/pipeline1_kin_ood_bundle.json"

# Disc-cut for envelope-fit subset selection.
DISC_VZ_MAX = 80.0   # km/s; thin + thick disc cap
DISC_VT_MIN = 100.0  # km/s; rejects retrograde / halo

# Mahalanobis distance threshold quantile.
P_THRESHOLD = 0.99


def main() -> None:
    if not KIN_PARQUET.exists():
        _LOG.error("kinematic parquet missing: %s", KIN_PARQUET)
        sys.exit(1)

    _LOG.info("loading %s", KIN_PARQUET)
    df = pd.read_parquet(
        KIN_PARQUET,
        columns=["source_id", "v_R_kms", "v_T_kms", "v_z_kms"],
    )
    n_total = len(df)
    _LOG.info("total Stream-3 kinematic rows: %d", n_total)

    V = df[["v_R_kms", "v_T_kms", "v_z_kms"]].to_numpy(dtype=np.float64)
    finite = np.isfinite(V).all(axis=1)
    n_finite = int(finite.sum())
    _LOG.info("finite velocity rows: %d (%.2f%%)", n_finite, 100 * n_finite / n_total)

    # Disc-cut: |v_z| < 80, v_T > 100. Rejects halo + counter-rotating.
    disc_mask = (
        finite
        & (np.abs(V[:, 2]) < DISC_VZ_MAX)
        & (V[:, 1] > DISC_VT_MIN)
    )
    n_disc = int(disc_mask.sum())
    _LOG.info(
        "disc-cut subset for envelope fit (|v_z|<%.0f, v_T>%.0f): %d (%.2f%%)",
        DISC_VZ_MAX, DISC_VT_MIN, n_disc, 100 * n_disc / n_total,
    )
    if n_disc < 10_000:
        _LOG.error("disc subset too small for stable envelope fit")
        sys.exit(2)

    bundle = fit_kinematic_ood(
        V[disc_mask],
        p_threshold=P_THRESHOLD,
        coordinate_system="galpy_galactocentric_cylindrical_v_LSR_230",
    )
    _LOG.info(
        "fit envelope: μ=%s, threshold (Mahalanobis)=%.4f at p=%.2f, n_train=%d",
        np.round(bundle.velocity_mean, 2).tolist(),
        bundle.threshold,
        bundle.p_threshold,
        bundle.n_training,
    )

    # Score the FULL set; non-finite velocities → NaN distance → flagged True.
    flags = flag_kinematic_ood(V, bundle)
    n_flagged = int(flags.sum())
    _LOG.info(
        "flagged %d / %d (%.2f%%) as kinematically OOD",
        n_flagged, n_total, 100 * n_flagged / n_total,
    )

    # Distance-only diagnostic on finite rows
    dists = score_kinematic_ood(V, bundle)
    finite_dists = dists[np.isfinite(dists)]
    _LOG.info(
        "Mahalanobis distance: p50=%.3f  p90=%.3f  p99=%.3f  max=%.3f (n=%d finite)",
        float(np.percentile(finite_dists, 50)),
        float(np.percentile(finite_dists, 90)),
        float(np.percentile(finite_dists, 99)),
        float(finite_dists.max()),
        len(finite_dists),
    )

    out = pd.DataFrame(
        {
            "source_id": df["source_id"].to_numpy(dtype=np.int64),
            "kin_ood_flag": flags.astype(bool),
            "kin_ood_score": dists.astype(np.float32),
        }
    )
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    _LOG.info("wrote %s (%d rows, %d cols)", OUT_PARQUET, len(out), len(out.columns))

    # Persist the bundle as JSON sidecar for reproducibility / downstream verify.
    bundle_blob = {
        "velocity_mean": bundle.velocity_mean.tolist(),
        "velocity_precision": bundle.velocity_precision.tolist(),
        "threshold": bundle.threshold,
        "p_threshold": bundle.p_threshold,
        "n_training": bundle.n_training,
        "regularization": bundle.regularization,
        "coordinate_system": bundle.coordinate_system,
        "disc_cut": {"v_z_max_kms": DISC_VZ_MAX, "v_T_min_kms": DISC_VT_MIN},
        "n_total": n_total,
        "n_finite": n_finite,
        "n_flagged": n_flagged,
        "rate_flagged_pct": float(100 * n_flagged / n_total),
        "source_kinematics_parquet": str(KIN_PARQUET),
    }
    BUNDLE_JSON.write_text(json.dumps(bundle_blob, indent=2))
    _LOG.info("wrote bundle sidecar %s", BUNDLE_JSON)


if __name__ == "__main__":
    main()
