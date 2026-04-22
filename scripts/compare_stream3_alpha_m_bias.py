"""Stream-3 [α/M]-vs-[M/H] comparison: v1 (unweighted) vs v1.1 (weighted).

The val-set diagnostic is half the story — the bias that triggered #198 was
observed on **Stream-3 inference**, not the val set. This script joins v1 and
v1.1 Stream-3 prediction Parquets on ``source_id`` and reports per-[M/H]-bin
mean [α/M] for both, so the user can see whether the metal-poor regression-to-
mean is actually fixed in deployment.

Stratifies by **predicted** [M/H] (Stream-3 has no ground truth).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("stream3_compare")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_V1 = REPO / "data/processed/pipeline1_predictions_stream3.parquet"
DEFAULT_V11 = REPO / "data/processed/pipeline1_predictions_stream3_v11.parquet"
DEFAULT_OUT = REPO / "reports/pipeline1/run_a_v11/stream3_alpha_m_bias.json"

MH_EDGES = (-1.5, -1.0, -0.5, 0.0)
BIN_LABELS = ["(-inf, -1.50)", "[-1.50, -1.00)", "[-1.00, -0.50)", "[-0.50, 0.00)", "[0.00, +inf)"]


def _stratify(df: pd.DataFrame, tag: str) -> list[dict]:
    mh = df["mh_pred"].to_numpy()
    am = df["alpha_m_pred"].to_numpy()
    am_sigma = df["alpha_m_sigma"].to_numpy() if "alpha_m_sigma" in df.columns else None
    bin_idx = np.digitize(mh, MH_EDGES)

    rows = []
    for b in range(len(MH_EDGES) + 1):
        m = (bin_idx == b) & np.isfinite(mh) & np.isfinite(am)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": BIN_LABELS[b], "n": 0})
            continue
        row = {
            "bin": BIN_LABELS[b],
            "n": n,
            f"{tag}_alpha_mean": float(np.mean(am[m])),
            f"{tag}_alpha_median": float(np.median(am[m])),
            f"{tag}_alpha_std": float(np.std(am[m])),
            f"{tag}_mh_mean": float(np.mean(mh[m])),
        }
        if am_sigma is not None:
            row[f"{tag}_alpha_sigma_mean"] = float(np.mean(am_sigma[m]))
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1", type=Path, default=DEFAULT_V1)
    p.add_argument("--v11", type=Path, default=DEFAULT_V11)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    _LOG.info("v1  Parquet: %s", args.v1)
    _LOG.info("v11 Parquet: %s", args.v11)
    cols = ["source_id", "mh_pred", "alpha_m_pred", "alpha_m_sigma"]
    df1 = pd.read_parquet(args.v1, columns=cols)
    df11 = pd.read_parquet(args.v11, columns=cols)
    _LOG.info("v1 rows=%d, v11 rows=%d", len(df1), len(df11))

    merged = df1.merge(df11, on="source_id", suffixes=("_v1", "_v11"))
    _LOG.info("inner-join rows=%d", len(merged))

    # Stratify independently on each dataset's own predicted [M/H].
    v1_rows = _stratify(df1.rename(columns={"mh_pred": "mh_pred",
                                            "alpha_m_pred": "alpha_m_pred",
                                            "alpha_m_sigma": "alpha_m_sigma"}), tag="v1")
    v11_rows = _stratify(df11.rename(columns={"mh_pred": "mh_pred",
                                              "alpha_m_pred": "alpha_m_pred",
                                              "alpha_m_sigma": "alpha_m_sigma"}), tag="v11")

    # Pairwise table on the inner join, bin by v1's predicted [M/H] so the
    # "same stars" interpretation stays clean.
    mh_v1 = merged["mh_pred_v1"].to_numpy()
    am_v1 = merged["alpha_m_pred_v1"].to_numpy()
    am_v11 = merged["alpha_m_pred_v11"].to_numpy()
    bin_idx = np.digitize(mh_v1, MH_EDGES)

    print()
    print(f"{'[M/H] bin (v1)':<20} {'n':>8}  "
          f"{'v1_α_mean':>10} {'v11_α_mean':>11}  "
          f"{'v1_α_med':>10} {'v11_α_med':>11}  "
          f"{'Δα_mean':>10}  {'v1_σ_mean':>10} {'v11_σ_mean':>11}")
    print("-" * 126)
    pair_rows = []
    for b in range(len(MH_EDGES) + 1):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            continue
        row = {
            "bin": BIN_LABELS[b],
            "n": n,
            "v1_alpha_mean": float(np.mean(am_v1[m])),
            "v11_alpha_mean": float(np.mean(am_v11[m])),
            "v1_alpha_median": float(np.median(am_v1[m])),
            "v11_alpha_median": float(np.median(am_v11[m])),
            "delta_alpha_mean": float(np.mean(am_v11[m] - am_v1[m])),
        }
        if "alpha_m_sigma_v1" in merged.columns:
            row["v1_sigma_mean"] = float(np.mean(merged["alpha_m_sigma_v1"].to_numpy()[m]))
            row["v11_sigma_mean"] = float(np.mean(merged["alpha_m_sigma_v11"].to_numpy()[m]))
        print(
            f"{row['bin']:<20} {row['n']:>8d}  "
            f"{row['v1_alpha_mean']:>10.4f} {row['v11_alpha_mean']:>11.4f}  "
            f"{row['v1_alpha_median']:>10.4f} {row['v11_alpha_median']:>11.4f}  "
            f"{row['delta_alpha_mean']:>+10.4f}  "
            f"{row.get('v1_sigma_mean', float('nan')):>10.4f} "
            f"{row.get('v11_sigma_mean', float('nan')):>11.4f}"
        )
        pair_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "mh_bin_edges": list(MH_EDGES),
            "v1_parquet": str(args.v1),
            "v11_parquet": str(args.v11),
            "n_v1": len(df1),
            "n_v11": len(df11),
            "n_inner_join": len(merged),
            "v1_standalone": v1_rows,
            "v11_standalone": v11_rows,
            "paired_by_v1_mh_bin": pair_rows,
        }, f, indent=2)
    _LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
