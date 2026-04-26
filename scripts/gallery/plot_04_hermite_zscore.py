"""Stage 04: Hermite normalisation + frozen v1 z-score (basis fingerprint 0d34b565...).

What the deploy did: per-coefficient z-scoring using the FROZEN training-pool
mean and std — never refit on Stream 3 (CLAUDE.md hard rule #16). The frozen
stats live in ``data/processed/pipeline1_features_stream1.provenance.json``
and ``data.frozen_stats.verify_basis_fingerprint`` raises
``FrozenStatsMismatchError`` on drift.

What we plot: per-coef Hermite z-score histograms for Stream 1 vs Stream 3.
If the frozen-stats compliance ever broke, the Stream-3 distributions would
shift; here they should overlap nearly exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/04_hermite_zscore"
S1 = REPO / "data/processed/pipeline1_features_stream1.parquet"
S2 = REPO / "data/processed/pipeline1_features_stream2.parquet"
S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"
PROV = REPO / "data/processed/pipeline1_features_stream1.provenance.json"


def _load_zscored(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read coefs 1-54 and the c0 z-score (or c0_log if z is unavailable in this parquet)."""
    schema = pd.read_parquet(path).iloc[:0]
    coef_cols = ([f"bp_coef_norm_{i}" for i in range(1, 55)]
                 + [f"rp_coef_norm_{i}" for i in range(1, 55)])
    c0_cols = []
    bp_c0_key = "bp_c0_z" if "bp_c0_z" in schema.columns else (
        "bp_c0_log" if "bp_c0_log" in schema.columns else None)
    rp_c0_key = "rp_c0_z" if "rp_c0_z" in schema.columns else (
        "rp_c0_log" if "rp_c0_log" in schema.columns else None)
    if bp_c0_key: c0_cols.append(bp_c0_key)
    if rp_c0_key: c0_cols.append(rp_c0_key)
    df = pd.read_parquet(path, columns=coef_cols + c0_cols)
    bp = np.column_stack([df[f"bp_coef_norm_{i}"].to_numpy() for i in range(1, 55)])
    rp = np.column_stack([df[f"rp_coef_norm_{i}"].to_numpy() for i in range(1, 55)])
    bp_c0 = df[bp_c0_key].to_numpy() if bp_c0_key else None
    rp_c0 = df[rp_c0_key].to_numpy() if rp_c0_key else None
    return bp, rp, bp_c0, rp_c0


def main() -> None:
    apply_style()
    bp1, rp1, bpc01, rpc01 = _load_zscored(S1)
    have_s2 = S2.exists()
    bp2 = rp2 = bpc02 = rpc02 = None
    if have_s2:
        bp2, rp2, bpc02, rpc02 = _load_zscored(S2)
    bp3, rp3, bpc03, rpc03 = _load_zscored(S3)

    fingerprint = "?"
    if PROV.exists():
        try:
            payload = json.loads(PROV.read_text())
            fingerprint = (payload.get("extra", {}).get("frozen_stats_basis_fingerprint_sha256")
                            or payload.get("extra", {}).get("basis_fingerprint_sha256")
                            or "?")
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    idx = np.arange(1, 55)
    for axis_idx, (band, m1, m2, m3) in enumerate([("BP", bp1, bp2, bp3),
                                                     ("RP", rp1, rp2, rp3)]):
        ax = axes[axis_idx]
        q1 = np.nanpercentile(m1, [16, 50, 84], axis=0)
        ax.fill_between(idx, q1[0], q1[2], color="#1f77b4", alpha=0.30,
                         label=f"Stream 1 (n={len(m1):,})")
        ax.plot(idx, q1[1], color="#1f77b4", lw=0.9)
        if m2 is not None:
            q2 = np.nanpercentile(m2, [16, 50, 84], axis=0)
            ax.fill_between(idx, q2[0], q2[2], color="#9467bd", alpha=0.25,
                             label=f"Stream 2 (n={len(m2):,})")
            ax.plot(idx, q2[1], color="#9467bd", lw=0.9, ls="-.")
        q3 = np.nanpercentile(m3, [16, 50, 84], axis=0)
        ax.fill_between(idx, q3[0], q3[2], color="#d62728", alpha=0.30,
                         label=f"Stream 3 (n={len(m3):,})")
        ax.plot(idx, q3[1], color="#d62728", lw=0.9, ls="--")
        ax.axhline(0, color="k", lw=0.4); ax.axhline(1, color="k", lw=0.3, ls=":")
        ax.axhline(-1, color="k", lw=0.3, ls=":")
        ax.set_xlabel(f"{band} coef index (1..54)")
        ax.set_ylabel("c_i / c_0 (raw or z-scored — see legend)")
        ax.set_title(f"{band}: per-coef IQR overlay (S1 / S2 / S3)")
        ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.95,
                  facecolor="white", edgecolor="0.4")

    # c0 overlay
    ax = axes[2]
    for c0, color, name in [(bpc01, "#1f77b4", "S1 (z-scored)"),
                              (bpc02, "#9467bd", "S2 (log10)"),
                              (bpc03, "#d62728", "S3 (log10)")]:
        if c0 is None:
            continue
        m = np.isfinite(c0)
        if m.sum() < 100:
            continue
        bins = np.linspace(np.nanpercentile(c0, 0.5),
                            np.nanpercentile(c0, 99.5), 61)
        ax.hist(c0[m], bins=bins, color=color, alpha=0.45, density=True,
                 label=f"{name} (n={int(m.sum()):,})")
    ax.set_xlabel("BP c0")
    ax.set_ylabel("density")
    ax.set_title("BP c0 distributions (Stream 1 z vs Stream 2/3 log10)")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")

    fig.suptitle(
        f"Frozen v1 Hermite z-score (fingerprint {fingerprint[:16]}…) — "
        f"S2 and S3 IQRs must overlap S1 (frozen-stats compliance gate).",
        fontsize=10,
    )
    save_fig(fig, OUT / "hermite_zscore.png")


if __name__ == "__main__":
    main()
