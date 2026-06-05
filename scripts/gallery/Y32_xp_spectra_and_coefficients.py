"""Y32: Gaia BP/RP XP, sampled-flux spectra and Hermite coefficient form.

Slide-friendly 4-panel layout (top: BP and RP sampled flux on a common
wavelength grid; bottom: BP and RP normalised Hermite coefficients 1..54).
For each panel we draw a faint per-star ribbon (~200 randomly selected
Stream-1 stars), plus the median + 16-84 percentile envelope.

Wider-than-tall figure (16:7) sized for a half-slide insert.
"""

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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import OKABE_ITO, PALETTE, apply_style, headline, save  # noqa: E402

XP_PATH = REPO / "data/interim/xp_sampled_corrected.parquet"
FEAT_PATH = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

N_STARS = 200
N_COEFS = 55
SAMPLING_NM = np.geomspace(360.0, 990.0, 330)
RNG_SEED = 0


def _load_xp_flux(target_ids: np.ndarray) -> np.ndarray:
    target = pa.array(sorted(int(x) for x in target_ids.tolist()))
    opts = pc.SetLookupOptions(value_set=target)
    pf = pq.ParquetFile(XP_PATH)
    kept: list[pa.Table] = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(
            rg, columns=["source_id", "corrected_flux", "ye2024_flag"]
        )
        m = pc.is_in(chunk.column("source_id"), options=opts)
        ok = pc.equal(chunk.column("ye2024_flag"), 0)
        sub = chunk.filter(pc.and_(m, ok))
        if sub.num_rows:
            kept.append(sub)
    if not kept:
        return np.empty((0, len(SAMPLING_NM)), dtype=np.float32)
    flux_list = pa.concat_tables(kept).column("corrected_flux").to_pylist()
    return np.asarray(flux_list, dtype=np.float32)


def _load_coefs() -> tuple[np.ndarray, np.ndarray]:
    bp_cols = [f"bp_coef_norm_{i}" for i in range(1, N_COEFS)]
    rp_cols = [f"rp_coef_norm_{i}" for i in range(1, N_COEFS)]
    df = pd.read_parquet(FEAT_PATH, columns=["source_id"] + bp_cols + rp_cols)
    df = df.drop_duplicates("source_id", keep="first")
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(df), size=min(N_STARS, len(df)), replace=False)
    sub = df.iloc[idx]
    bp = sub[bp_cols].to_numpy(dtype=np.float32)
    rp = sub[rp_cols].to_numpy(dtype=np.float32)
    return bp, rp


_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


_BP_RANGE = (330.0, 680.0)   # Gaia BP nominal coverage (Riello+2021)
_RP_RANGE = (640.0, 1050.0)  # Gaia RP nominal coverage (Riello+2021)


def _normalise_band(flux: np.ndarray, lam_lo: float, lam_hi: float) -> np.ndarray:
    """Per-star, divide the spectrum by its median over a single band's window.

    BP and RP do NOT share absolute calibration: they are two independent
    photometers with different effective areas, so the raw flux levels in
    the BP/RP overlap (640-680 nm) genuinely disagree. Normalising each
    band by its own per-star median puts the two bands on a self-consistent
    in-band scale (the median of each band is 1 by construction), which is
    what we want when we want to read absorption features inside a band.
    """
    mask = (SAMPLING_NM >= lam_lo) & (SAMPLING_NM <= lam_hi)
    flux_b = flux[:, mask].astype(np.float64, copy=True)
    med = np.nanmedian(flux_b, axis=1, keepdims=True)
    med = np.where(med > 0, med, np.nan)
    return flux_b / med, SAMPLING_NM[mask]


def _draw_spectrum(ax, flux_full: np.ndarray, lam_lo: float, lam_hi: float, color: str, title: str) -> None:
    """Per-star band-local median normalisation, ribbons + median (16-84)."""
    f, lam = _normalise_band(flux_full, lam_lo, lam_hi)
    for row in f:
        ax.plot(lam, row, color=color, lw=0.4, alpha=0.05)
    p16 = np.nanpercentile(f, 16, axis=0)
    p50 = np.nanpercentile(f, 50, axis=0)
    p84 = np.nanpercentile(f, 84, axis=0)
    ax.fill_between(lam, p16, p84, color=color, alpha=0.25, lw=0)
    ax.plot(lam, p50, color=color, lw=2.0, label="median")
    ax.set_xlim(lam_lo, lam_hi)
    p1 = float(np.nanpercentile(f, 1.0))
    p99 = float(np.nanpercentile(f, 99.0))
    pad = 0.08 * (p99 - p1) if p99 > p1 else 1e-3
    ax.set_ylim(p1 - pad, p99 + pad)
    ax.set_xlabel(r"wavelength [nm]")
    ax.set_ylabel("flux / band-median")
    ax.set_title(title, **_TITLE_KW)


def _draw_coefs(ax, coefs: np.ndarray, color: str, title: str) -> None:
    """Per-star polylines over Hermite index 1..54 + median + 16-84 envelope."""
    n_coef = coefs.shape[1]
    x = np.arange(1, n_coef + 1)
    for row in coefs:
        ax.plot(x, row, color=color, lw=0.4, alpha=0.05)
    p16 = np.nanpercentile(coefs, 16, axis=0)
    p50 = np.nanpercentile(coefs, 50, axis=0)
    p84 = np.nanpercentile(coefs, 84, axis=0)
    ax.fill_between(x, p16, p84, color=color, alpha=0.25, lw=0)
    ax.plot(x, p50, color=color, lw=2.0, label="median")
    ax.axhline(0.0, color=PALETTE["ash"], lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(1, n_coef)
    p1 = float(np.nanpercentile(coefs, 1.0))
    p99 = float(np.nanpercentile(coefs, 99.0))
    pad = 0.08 * (p99 - p1) if p99 > p1 else 1e-3
    ax.set_ylim(p1 - pad, p99 + pad)
    ax.set_xlabel("Hermite coefficient index")
    ax.set_ylabel("normalised amplitude")
    ax.set_title(title, **_TITLE_KW)


def main() -> int:
    apply_style()

    feat = pd.read_parquet(FEAT_PATH, columns=["source_id"]).drop_duplicates("source_id")
    rng = np.random.default_rng(RNG_SEED)
    target_ids = rng.choice(
        feat["source_id"].to_numpy(),
        size=min(N_STARS, len(feat)),
        replace=False,
    )
    flux = _load_xp_flux(target_ids)
    if flux.size == 0:
        print("[Y32] no XP rows matched, aborting")
        return 1
    bp, rp = _load_coefs()

    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    bp_color = OKABE_ITO[0]   # blue
    rp_color = OKABE_ITO[1]   # vermillion

    _draw_spectrum(axes[0, 0], flux, _BP_RANGE[0], _BP_RANGE[1], bp_color, "BP, sampled flux")
    _draw_spectrum(axes[0, 1], flux, _RP_RANGE[0], _RP_RANGE[1], rp_color, "RP, sampled flux")
    _draw_coefs(axes[1, 0], bp, bp_color, "BP, normalised Hermite coefficients")
    _draw_coefs(axes[1, 1], rp, rp_color, "RP, normalised Hermite coefficients")

    for ax in axes.ravel():
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.10,
                        hspace=0.45, wspace=0.22)
    headline(
        fig,
        "Gaia BP/RP XP: sampled-flux spectra and Hermite coefficient form",
        f"Stream 1, n = {flux.shape[0]:,} stars; per-star ribbons + median (16-84 envelope).",
        top=0.78,
    )
    save(fig, "Y32_xp_spectra_and_coefficients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
