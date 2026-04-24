"""Phase 3b self-critique battery.

Runs checks that the Phase 3b report asserts but didn't individually
verify:

1.  ood_joint_flag = Mahalanobis-or-ensemble-or-NaN. The 33,717
    NaN-Mahalanobis rows MUST land in ood_joint_flag=True, otherwise
    downstream consumers get silent pass-through on reprojection-
    failing stars.
2.  uniform ∩ volume_limited source_id sets are disjoint (per-arm
    splits are non-overlapping by construction — sanity only).
3.  Prediction-NaN rows (the ~615 across the five label columns) are
    all either NaN Hermite reprojection or flagged via aux-missing.
4.  HR-diagram sanity on volume-arm predictions: Teff_pred vs logg_pred
    should stay inside the RGB/RC band (no thin-disc dwarfs, no
    supergiants).
5.  Uniform arm [M/H] shift diagnosis: fraction of uniform arm with
    [M/H]_pred < −1 (halo-candidate rate).

Prints a concise PASS/FAIL per check. Exit 0 iff all PASS.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("selfcheck_phase3b")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    pred_path = repo / "data" / "processed" / "pipeline1_predictions_stream3.parquet"
    u_path = repo / "data" / "processed" / "pipeline1_predictions_stream3_uniform.parquet"
    v_path = repo / "data" / "processed" / "pipeline1_predictions_stream3_volume.parquet"

    logger.info("loading predictions")
    df = pd.read_parquet(pred_path)
    u = pd.read_parquet(u_path)
    v = pd.read_parquet(v_path)
    logger.info("union=%d uniform=%d volume=%d", len(df), len(u), len(v))

    failures: list[str] = []

    # --- Check 1: NaN Mahalanobis → ood_joint_flag=True ----------------------
    mahal_nan = df["ood_mahalanobis_score"].isna()
    joint_true_on_nan_mahal = (df.loc[mahal_nan, "ood_joint_flag"] == True).mean()
    logger.info(
        "check 1: NaN Mahalanobis → ood_joint=True fraction = %.4f (expected 1.0, n_nan=%d)",
        joint_true_on_nan_mahal,
        int(mahal_nan.sum()),
    )
    if joint_true_on_nan_mahal < 0.999:
        failures.append(
            f"NaN-Mahalanobis stars are NOT all marked ood_joint_flag=True "
            f"(actual {joint_true_on_nan_mahal:.4f}). Silent pass-through "
            f"risk for reprojection-failing rows downstream."
        )
    else:
        logger.info("  PASS")

    # --- Check 2: uniform/volume disjoint -----------------------------------
    u_ids = set(u["source_id"].to_numpy().tolist())
    v_ids = set(v["source_id"].to_numpy().tolist())
    overlap = u_ids & v_ids
    logger.info(
        "check 2: uniform ∩ volume overlap = %d (expected 0, |U|=%d, |V|=%d, "
        "|U|+|V|=%d vs |union|=%d)",
        len(overlap),
        len(u_ids),
        len(v_ids),
        len(u_ids) + len(v_ids),
        len(df),
    )
    if len(overlap) > 0:
        failures.append(
            f"uniform and volume_limited source_id sets overlap by "
            f"{len(overlap)} stars — Stream-3 splitter is leaking"
        )
    else:
        logger.info("  PASS")

    # --- Check 3: prediction-NaN rows are all ood_joint=True ---------------
    # Any row where the forward pass produced NaN predictions MUST be gated
    # out via ood_joint_flag. Otherwise a downstream Tier-1 release would
    # ship NaN labels — the precise failure mode the flag system exists to
    # prevent. We don't care HOW the NaN arose (failed reprojection, c0_log
    # NaN propagating despite nan_to_num, forward-pass overflow, etc.) — we
    # only care that every NaN row is gated.
    pred_nan = df["teff_pred"].isna()
    n_pred_nan = int(pred_nan.sum())
    if n_pred_nan:
        gated = int((df.loc[pred_nan, "ood_joint_flag"] == True).sum())
        logger.info(
            "check 3: pred_NaN = %d; of those, ood_joint=True = %d (%.4f)",
            n_pred_nan,
            gated,
            gated / n_pred_nan,
        )
        if gated < n_pred_nan:
            failures.append(
                f"{n_pred_nan - gated} pred_NaN rows are NOT gated via "
                f"ood_joint_flag — Tier-1 release leak risk"
            )
        else:
            logger.info("  PASS")
    else:
        logger.info("check 3: no pred_NaN rows — PASS trivially")

    # --- Check 4: HR-diagram sanity on volume arm ---------------------------
    v_clean = v.dropna(subset=["teff_pred", "logg_pred"])
    v_rgb = (
        (v_clean["teff_pred"] >= 3900)
        & (v_clean["teff_pred"] <= 5400)
        & (v_clean["logg_pred"] >= 1.0)
        & (v_clean["logg_pred"] <= 3.7)
    )
    frac_in_rgb = float(v_rgb.mean())
    logger.info(
        "check 4: volume-arm stars inside (Teff in [3900,5400], logg in [1.0,3.7]) = %.4f",
        frac_in_rgb,
    )
    if frac_in_rgb < 0.97:
        failures.append(
            f"only {frac_in_rgb:.3f} of volume-arm predictions inside nominal "
            f"RGB/RC box — predictions may be drifting (should be ~0.99 given "
            f"Andrae+2023 input catalogue is RGB by construction)"
        )
    else:
        logger.info("  PASS")

    # --- Check 5: Uniform arm halo-candidate rate ---------------------------
    u_clean = u.dropna(subset=["mh_pred"])
    halo_rate = float((u_clean["mh_pred"] < -1.0).mean())
    logger.info(
        "check 5: uniform arm [M/H] < -1.0 fraction = %.4f "
        "(training [M/H]<-1 coverage ~5%%; uniform-arm ~20-30%% is expected "
        "and drives OOD overshoot)",
        halo_rate,
    )
    # Informational — not a halt, just prints the rate. Fail only if the rate
    # is implausibly high (would indicate model collapse to metal-poor).
    if halo_rate > 0.6:
        failures.append(
            f"uniform-arm [M/H]<-1 rate = {halo_rate:.3f} looks model-collapse-y "
            f"(>60% metal-poor predictions)"
        )
    else:
        logger.info("  PASS")

    # --- summary ------------------------------------------------------------
    logger.info("-" * 50)
    if failures:
        logger.error("SELF-CRITIQUE FAILED — %d check(s):", len(failures))
        for i, f in enumerate(failures, 1):
            logger.error("  [%d] %s", i, f)
        return 1

    logger.info("SELF-CRITIQUE PASSED — all 5 checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
