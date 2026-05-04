"""B8: Frozen Hermite z-score basis fingerprint visualisation.

What this shows:
- Per-coefficient retained variance from PCA on the Stream 1 training pool.
- Two panels: BP (54 coeffs) and RP (54 coeffs).
- Demonstrates that lower-order Hermite coefficients retain signal while
  higher-order coefficients become noise.
- Basis fingerprint hash annotated in the suptitle.

What it reads:
- data/processed/pipeline1_features_stream1.provenance.json
  (extracts frozen_stats_basis_fingerprint_sha256 from extra).

The frozen stats are computed on the Stream 1 training set and applied
consistently to all streams (1, 2, 3) during preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig


def _load_frozen_stats():
    """Load per-coefficient z-score stats from the Stream-1 provenance file.

    The provenance JSON stores under ``extra.coef_norm_zscore_frozen.{bp,rp}``
    a per-coefficient ``mu`` / ``sigma`` (length 54) computed on the
    normal-population subset (ye2024_flag == 0 AND xp_fit_flag_residual_high
    == 0 AND bp_coef_0 > 0 AND rp_coef_0 > 0). The σ array is the meaningful
    quantity for this plot: it is the standard deviation of c_i/c_0 across
    the normal population, BEFORE z-scoring. High-order coefficients have
    σ → 0 (noise floor); low-order coefficients carry the bulk of the
    physical SED variance.
    """
    prov_file = REPO / "data/processed/pipeline1_features_stream1.provenance.json"
    if not prov_file.exists():
        return None
    payload = json.loads(prov_file.read_text())
    extra = payload.get("extra", {})
    fingerprint = (
        extra.get("frozen_stats_basis_fingerprint_sha256")
        or extra.get("basis_fingerprint_sha256")
        or "unknown"
    )
    block = extra.get("coef_norm_zscore_frozen", {})
    bp = block.get("bp", {})
    rp = block.get("rp", {})
    # Per-coefficient stats are keyed by string indices "1".."54"; each value
    # is a dict {"mu": ..., "sigma": ...} on the normal-population subset.
    if not bp or not rp:
        return None
    indices = sorted((int(k) for k in bp), key=int)
    bp_sigma = np.asarray([bp[str(i)]["sigma"] for i in indices], dtype=float)
    rp_sigma = np.asarray([rp[str(i)]["sigma"] for i in indices], dtype=float)
    if bp_sigma.size == 0 or rp_sigma.size == 0:
        return None
    return {
        "bp_sigma": bp_sigma,
        "rp_sigma": rp_sigma,
        "fingerprint": fingerprint,
        "n_ref": int(block.get("n_reference_population", 0)),
        "sigma_floor": float(block.get("sigma_floor", 0.0)),
    }


def main() -> None:
    apply_style()

    stats = _load_frozen_stats()
    if stats is None:
        raise SystemExit(
            "Stream-1 provenance file is missing the per-coefficient z-score "
            "block (extra.coef_norm_zscore_frozen). Re-run "
            "scripts/emit_stream1_with_hermite.py to regenerate."
        )

    bp_sigma = stats["bp_sigma"]
    rp_sigma = stats["rp_sigma"]
    fingerprint = stats["fingerprint"]
    n_ref = stats["n_ref"]
    sigma_floor = stats["sigma_floor"]

    # Coefficient index: stats are stored for indices 1..54 (the trivial
    # bp_coef_norm_0 = 1 is excluded by construction).
    coef_idx = np.arange(1, len(bp_sigma) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Compute a sensible y-axis range: the sigma_floor (1e-30 by construction)
    # is informational only; if we let the log axis extend to it, the actual
    # data (typically 1e-2..1e1) collapses to a thin strip at the top.
    sig_all = np.concatenate([bp_sigma, rp_sigma])
    sig_pos = sig_all[sig_all > 0]
    if sig_pos.size:
        y_lo = max(np.min(sig_pos) / 3.0, 1e-4)
        y_hi = np.max(sig_pos) * 3.0
    else:
        y_lo, y_hi = 1e-3, 1e1
    floor_in_view = sigma_floor > 0 and sigma_floor >= y_lo

    # BP panel — per-coefficient σ(c_i/c_0) on log scale
    ax = axes[0]
    ax.semilogy(
        coef_idx, bp_sigma, "-o", color="tab:blue", lw=1.5, ms=4, label=r"$\sigma(c_i/c_0)$"
    )
    if floor_in_view:
        ax.axhline(
            sigma_floor,
            color="red",
            lw=0.8,
            ls="--",
            alpha=0.5,
            label=f"sigma floor ({sigma_floor:.0e})",
        )
    ax.set_xlabel("BP coefficient index (1-54)")
    ax.set_ylabel(r"$\sigma(c_i / c_0)$ on normal-population subset")
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("BP: per-coefficient σ(c_i/c_0) (frozen v1 basis)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, which="both")

    # RP panel
    ax = axes[1]
    ax.semilogy(
        coef_idx, rp_sigma, "-o", color="tab:orange", lw=1.5, ms=4, label=r"$\sigma(c_i/c_0)$"
    )
    if floor_in_view:
        ax.axhline(
            sigma_floor,
            color="red",
            lw=0.8,
            ls="--",
            alpha=0.5,
            label=f"sigma floor ({sigma_floor:.0e})",
        )
    ax.set_xlabel("RP coefficient index (1-54)")
    ax.set_ylabel(r"$\sigma(c_i / c_0)$ on normal-population subset")
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("RP: per-coefficient σ(c_i/c_0) (frozen v1 basis)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, which="both")

    # Truncate fingerprint for display
    fp_short = fingerprint[:16] + "..." if len(fingerprint) > 16 else fingerprint

    # Reader-facing caption: what this plot answers and how to read it
    caption = (
        f"What this shows: σ(c_i/c_0) per Hermite coefficient on the Stream-1 normal-population subset\n"
        f"(n_ref = {n_ref:,} stars; frozen v1). Low-order coefficients carry strong SED signal; high-order\n"
        f"coefficients fall toward the sigma floor and contribute marginally to label inference. The basis\n"
        f"fingerprint above is the inference-time contract: any Stream-2/3 inference must apply these exact\n"
        f"per-coefficient stats — divergence breaks the release."
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        va="top",
        fontsize=8,
        color="#333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa", edgecolor="#bbb"),
    )

    fig.suptitle(
        f"B8 — Streams 1, 2, 3: Frozen v1 Hermite basis (per-coefficient σ, fingerprint: {fp_short})",
        fontsize=10,
        fontweight="semibold",
    )
    save_fig(
        fig, REPO / "reports/gallery/B_preprocessing" / "B8_hermite_zscore", formats=("pdf", "png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot B8: frozen Hermite z-score basis fingerprint."
    )
    args = parser.parse_args()
    main()
