"""Stage 03: Hermite reprojection + per-coefficient z-score.

Outputs:
  - reports/gallery/03_hermite_reprojection/hermite_zscore_per_coef.png
  - reports/gallery/03_hermite_reprojection/frozen_stats_v1_vs_stream3.png

Per DESIGN.md §ML-input primary:
  bp_coef_norm_k = bp_coef_k / bp_coef_0 at emit time.
  Per-coef z-scoring is then applied using frozen v1 stats (basis fingerprint
  0d34b565...). Stream 3 MUST reuse v1's stats and not refit.

This script shows (a) pre- vs post-z-score per-coef histograms on Stream 1, and
(b) that Stream 3 inference really did inherit v1's per-coef stats (empirical
mean/std of bp_coef_norm_k on the two streams should match to ~0.01 when the
stat-freeze contract is respected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_PROCESSED,
    GALLERY,
    apply_style,
    save_fig,
)

OUT = GALLERY / "03_hermite_reprojection"


def _have(schema, name: str) -> bool:
    return name in {f.name for f in schema}


def _load_norm_coefs(path: Path, n_rows: int = 120_000) -> dict:
    """Load normalised Hermite coefs and their raw ratios (raw_k / raw_0)."""
    schema = pq.read_schema(path)
    # normalised (post-z-score) columns
    norm_cols = [
        f"bp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"bp_coef_norm_{k}")
    ] + [f"rp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"rp_coef_norm_{k}")]
    # raw columns for the pre-z-score ratio
    raw_cols = (
        ["bp_coef_0"]
        + [f"bp_coef_{k}" for k in range(1, 55) if _have(schema, f"bp_coef_{k}")]
        + ["rp_coef_0"]
        + [f"rp_coef_{k}" for k in range(1, 55) if _have(schema, f"rp_coef_{k}")]
    )
    raw_cols = [c for c in raw_cols if _have(schema, c)]

    tbl = pq.read_table(path, columns=norm_cols + raw_cols)
    df = tbl.to_pandas()

    # subsample
    if len(df) > n_rows:
        rng = np.random.default_rng(11)
        df = df.iloc[rng.choice(len(df), size=n_rows, replace=False)].reset_index(drop=True)

    # pre-z ratios
    pre = {}
    for band in ("bp", "rp"):
        c0 = df.get(f"{band}_coef_0")
        if c0 is None:
            continue
        for k in range(1, 55):
            ck = df.get(f"{band}_coef_{k}")
            if ck is None:
                continue
            ratio = ck.to_numpy(dtype=float) / np.where(
                np.abs(c0.to_numpy(dtype=float)) > 1e-30, c0.to_numpy(dtype=float), np.nan
            )
            pre[f"{band}_{k}"] = ratio

    post = {
        col.replace("_coef_norm_", "_").replace("bp_", "bp_").replace("rp_", "rp_"): df[
            col
        ].to_numpy(dtype=float)
        for col in norm_cols
    }
    # rename keys consistently: "bp_coef_norm_12" -> "bp_12"
    post = {
        k.replace("bp_coef_norm_", "bp_").replace("rp_coef_norm_", "rp_"): v
        for k, v in {col: df[col].to_numpy(dtype=float) for col in norm_cols}.items()
    }

    return {"pre": pre, "post": post}


def hermite_zscore_per_coef() -> None:
    path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    data = _load_norm_coefs(path)
    pre, post = data["pre"], data["post"]

    # 10 rows × 11 cols = 110 coefs; top 5 rows BP_1..54, bottom 5 rows RP_1..54 (k up to 54).
    # We'll do 10 rows × 11 cols => 110 subpanels. Use k-index 1..54 for BP in rows 0-4,
    # RP in rows 5-9 (each band 54 coefs → place them in the first 54 of 55 slots, last column
    # row 4 and row 9 will be "empty"/skipped).
    fig, axes = plt.subplots(10, 11, figsize=(22, 18), sharex=False, sharey=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    def _bandrow(band: str, row0: int) -> None:
        for k in range(1, 55):
            slot = k - 1
            r, c = divmod(slot, 11)
            ax = axes[row0 + r, c]
            pre_vals = pre.get(f"{band}_{k}")
            post_vals = post.get(f"{band}_{k}")
            if pre_vals is None and post_vals is None:
                ax.set_visible(False)
                continue
            # symmetric finite range; clip for tails
            if pre_vals is not None:
                pv = pre_vals[np.isfinite(pre_vals)]
                if len(pv) > 10:
                    lo, hi = np.percentile(pv, [0.5, 99.5])
                    if hi > lo:
                        ax.hist(
                            pv,
                            bins=np.linspace(lo, hi, 40),
                            color="#9e9e9e",
                            alpha=0.6,
                            density=True,
                            label="pre",
                        )
            if post_vals is not None:
                qv = post_vals[np.isfinite(post_vals)]
                if len(qv) > 10:
                    lo, hi = np.percentile(qv, [0.5, 99.5])
                    if hi > lo:
                        ax.hist(
                            qv,
                            bins=np.linspace(lo, hi, 40),
                            color="#d62728",
                            alpha=0.6,
                            density=True,
                            label="post",
                        )
            ax.set_title(f"{band.upper()} {k}", fontsize=7, pad=1)
            ax.tick_params(axis="both", labelsize=6)

    _bandrow("bp", 0)
    _bandrow("rp", 5)

    # Hide the last column on rows 4 and 9 (k=55 slot) — only 54 coefs per band
    for row0 in (0, 5):
        # The 55th slot would be at slot index 54 → divmod(54, 11) = (4, 10)
        axes[row0 + 4, 10].set_visible(False)

    # Shared legend
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, color="#9e9e9e", alpha=0.6, label="pre z-score (raw $c_k/c_0$)"
        ),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=0.6, label="post z-score (emit-time)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.995)
    )
    fig.suptitle(
        r"Hermite coefficients — pre- vs post-z-score  —  10×11 panel, 110 coefs "
        r"(rows 1-5 BP, rows 6-10 RP)",
        fontsize=13,
        fontweight="bold",
        y=0.975,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_fig(fig, OUT / "hermite_zscore_per_coef.png", tight=False)


def frozen_stats_v1_vs_stream3() -> None:
    """Compare empirical mean/std of bp/rp_coef_norm_k on Stream 1 vs Stream 3.

    The contract: Stream 3 MUST reuse v1's per-coef stats at emit time. If the
    contract is honoured, the empirical post-z-score mean and std on Stream 3
    are ~(0, 1) within sampling and drift, matching Stream 1 to ~0.01.
    """
    s1_path = DATA_PROCESSED / "pipeline1_features_stream1.parquet"
    s3_path = DATA_PROCESSED / "pipeline1_features_stream3.parquet"

    def _stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
        schema = pq.read_schema(path)
        cols = [f"bp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"bp_coef_norm_{k}")] + [
            f"rp_coef_norm_{k}" for k in range(1, 55) if _have(schema, f"rp_coef_norm_{k}")
        ]
        df = pq.read_table(path, columns=cols).to_pandas()
        # Subsample if huge
        if len(df) > 200_000:
            df = df.sample(200_000, random_state=13)
        mu = df.mean(axis=0, skipna=True).to_numpy(dtype=float)
        sd = df.std(axis=0, skipna=True).to_numpy(dtype=float)
        return mu, sd

    s1_mu, s1_sd = _stats(s1_path)
    s3_mu, s3_sd = _stats(s3_path)

    idx = np.arange(len(s1_mu))
    # k=1..54 for BP then k=1..54 for RP (order returned by _stats, defensive if some absent)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(idx, s1_mu, "o-", color="#1f77b4", lw=1.2, ms=3, label="Stream 1 (v1 fit)")
    axes[0].plot(
        idx, s3_mu, "s--", color="#d62728", lw=1.2, ms=3, label="Stream 3 (frozen v1 stats)"
    )
    axes[0].axhline(0, color="k", lw=0.5, ls=":")
    axes[0].set_ylabel(r"empirical mean of $c_k^{\rm norm, z}$")
    axes[0].set_title(
        "Per-coefficient post-z-score empirical mean  —  should be ~0 on both streams "
        "if frozen-stats contract is honoured",
        fontsize=11,
        fontweight="semibold",
    )
    axes[0].legend()

    axes[1].plot(idx, s1_sd, "o-", color="#1f77b4", lw=1.2, ms=3, label="Stream 1 (v1 fit)")
    axes[1].plot(
        idx, s3_sd, "s--", color="#d62728", lw=1.2, ms=3, label="Stream 3 (frozen v1 stats)"
    )
    axes[1].axhline(1, color="k", lw=0.5, ls=":")
    axes[1].set_xlabel("coefficient index  (0-53: BP 1..54, 54-107: RP 1..54)")
    axes[1].set_ylabel(r"empirical std of $c_k^{\rm norm, z}$")
    axes[1].set_title(
        "Per-coefficient post-z-score empirical std  —  should be ~1 on Stream 1;"
        " Stream 3 drift is a diagnostic",
        fontsize=11,
        fontweight="semibold",
    )
    axes[1].legend()

    # Write a small text box with max |mu| and |std-1|
    bbox = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9)
    msg = (
        rf"Stream-1 max $|\mu|$ = {np.nanmax(np.abs(s1_mu)):.3f},  "
        rf"max $|\sigma-1|$ = {np.nanmax(np.abs(s1_sd - 1)):.3f}"
        "\n"
        rf"Stream-3 max $|\mu|$ = {np.nanmax(np.abs(s3_mu)):.3f},  "
        rf"max $|\sigma-1|$ = {np.nanmax(np.abs(s3_sd - 1)):.3f}"
    )
    axes[0].text(
        0.01, 0.97, msg, transform=axes[0].transAxes, ha="left", va="top", fontsize=9, bbox=bbox
    )

    save_fig(fig, OUT / "frozen_stats_v1_vs_stream3.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    hermite_zscore_per_coef()
    frozen_stats_v1_vs_stream3()


if __name__ == "__main__":
    main()
