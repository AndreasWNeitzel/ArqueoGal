"""F03: BP/RP sampled flux + Hermite coefficients (slide 4)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from arqueogal.style import (  # noqa: E402
    LABELS, OKABE_ITO, apply_style, save,
)

XP = REPO / "data/interim/xp_sampled_corrected.parquet"
FEAT = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
N_STARS = 200
N_COEF = 55
SAMPLING_NM = np.geomspace(360.0, 990.0, 330)
RNG = 0


def _load_flux(target_ids):
    target = pa.array(sorted(int(x) for x in target_ids))
    opts = pc.SetLookupOptions(value_set=target)
    pf = pq.ParquetFile(XP)
    kept = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=["source_id", "corrected_flux", "ye2024_flag"])
        m = pc.is_in(chunk.column("source_id"), options=opts)
        ok = pc.equal(chunk.column("ye2024_flag"), 0)
        sub = chunk.filter(pc.and_(m, ok))
        if sub.num_rows:
            kept.append(sub)
    if not kept:
        return np.empty((0, len(SAMPLING_NM)), dtype=np.float32)
    return np.asarray(
        pa.concat_tables(kept).column("corrected_flux").to_pylist(),
        dtype=np.float32,
    )


def _normalise_band(flux, lo, hi):
    mask = (SAMPLING_NM >= lo) & (SAMPLING_NM <= hi)
    f = flux[:, mask].astype(np.float64)
    med = np.nanmedian(f, axis=1, keepdims=True)
    med = np.where(med > 0, med, np.nan)
    return f / med, SAMPLING_NM[mask]


def _draw_spectrum(ax, flux, lo, hi, color):
    f, lam = _normalise_band(flux, lo, hi)
    for row in f:
        ax.plot(lam, row, color=color, lw=0.4, alpha=0.05)
    p16 = np.nanpercentile(f, 16, axis=0)
    p50 = np.nanpercentile(f, 50, axis=0)
    p84 = np.nanpercentile(f, 84, axis=0)
    ax.fill_between(lam, p16, p84, color=color, alpha=0.20, lw=0,
                    label=r"16-84 envelope")
    ax.plot(lam, p50, color=color, lw=2.0, label=r"median")
    ax.plot([], [], color=color, lw=0.4, alpha=0.5, label=r"individual stars")
    ax.set_xlim(lo, hi)
    # Brief: top BP/RP panels share the same ylim, 0 to 2 in band-median
    # units (per-star spectra are normalised so the median is 1.0).
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(LABELS["wavelength"])
    ax.set_ylabel(LABELS["flux_norm"])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.30)


def _draw_coefs(ax, coefs, color):
    n_c = coefs.shape[1]
    x = np.arange(1, n_c + 1)
    for row in coefs:
        ax.plot(x, row, color=color, lw=0.4, alpha=0.05)
    p16 = np.nanpercentile(coefs, 16, axis=0)
    p50 = np.nanpercentile(coefs, 50, axis=0)
    p84 = np.nanpercentile(coefs, 84, axis=0)
    ax.fill_between(x, p16, p84, color=color, alpha=0.20, lw=0,
                    label="16-84 envelope")
    ax.plot(x, p50, color=color, lw=2.0, label="median")
    ax.plot([], [], color=color, lw=0.4, alpha=0.5, label="individual stars")
    ax.axhline(0.0, color="#5C6378", lw=0.6, ls=":", alpha=0.6)
    p1 = float(np.nanpercentile(coefs, 1.0))
    p99 = float(np.nanpercentile(coefs, 99.0))
    pad = 0.08 * (p99 - p1) if p99 > p1 else 1e-3
    ax.set_xlim(1, n_c)
    ax.set_ylim(p1 - pad, p99 + pad)
    ax.set_xlabel(LABELS["hermite_idx"])
    ax.set_ylabel(LABELS["amp_norm"])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.30)


def main() -> int:
    apply_style()
    feat = pd.read_parquet(FEAT, columns=["source_id"]).drop_duplicates("source_id")
    rng = np.random.default_rng(RNG)
    target_ids = rng.choice(feat["source_id"].to_numpy(),
                             size=min(N_STARS, len(feat)), replace=False)
    flux = _load_flux(target_ids)
    if flux.size == 0:
        return 1

    bp_cols = [f"bp_coef_norm_{i}" for i in range(1, N_COEF)]
    rp_cols = [f"rp_coef_norm_{i}" for i in range(1, N_COEF)]
    df = pd.read_parquet(FEAT, columns=["source_id"] + bp_cols + rp_cols
                          ).drop_duplicates("source_id")
    sub = df.iloc[rng.choice(len(df), size=N_STARS, replace=False)]
    bp = sub[bp_cols].to_numpy(dtype=np.float32)
    rp = sub[rp_cols].to_numpy(dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.5),
                              layout="constrained")
    bp_color = OKABE_ITO["blue"]
    rp_color = OKABE_ITO["vermillion"]

    _draw_spectrum(axes[0, 0], flux, 360.0, 660.0, bp_color)
    axes[0, 0].set_title(r"BP")
    _draw_spectrum(axes[0, 1], flux, 670.0, 990.0, rp_color)
    axes[0, 1].set_title(r"RP")
    _draw_coefs(axes[1, 0], bp, bp_color)
    _draw_coefs(axes[1, 1], rp, rp_color)

    save(fig, "F03_xp_feature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
