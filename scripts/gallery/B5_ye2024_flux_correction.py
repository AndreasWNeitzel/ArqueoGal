"""B5: Ye+2024 corrected XP sampled-flux SEDs at three G-magnitude bins.

What this shows (real data only):
- Mean ± 16-84 percentile band of the Ye+2024-corrected sampled flux for a
  random subset of stars in three G-magnitude bins (G ≈ 13, 15, 17).
- One panel per bin, x-axis = wavelength (nm) on the 330-point geometric
  grid (360--990 nm), y-axis = corrected flux on internal Gaia XP units.
- Demonstrates how the SED shape and signal-to-noise change with apparent
  magnitude — faint stars get noisier flux; bright stars saturate the
  SED locus.

What it reads:
- data/interim/xp_sampled_corrected.parquet (corrected_flux: list<float32>(330)).
- data/processed/pipeline1_features_stream1_kiel.parquet (g_mag for binning).

Note: the raw-vs-corrected-pair view is not constructable from on-disk
artefacts because raw GaiaXPy samples are not cached. This plot therefore
shows only the post-Ye SED locus, which is what enters the Hermite
re-projection downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

# Matches gaia_xp.YE2024_SAMPLING_NM exactly: 330 geometric-spaced wavelengths.
SAMPLING_NM = np.geomspace(360.0, 990.0, 330)
MAG_BINS = [(12.5, 13.5), (14.5, 15.5), (16.5, 17.5)]
BIN_LABELS = ["G in [12.5, 13.5)", "G in [14.5, 15.5)", "G in [16.5, 17.5)"]
BIN_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]
SAMPLE_PER_BIN = 5000


def main() -> None:
    apply_style()
    rng = np.random.default_rng(0)

    s1 = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
        columns=["source_id", "g_mag"],
    )
    print(f"[B5] loaded {len(s1):,} Stream-1 rows for G binning")

    xp_pf = pq.ParquetFile(REPO / "data/interim/xp_sampled_corrected.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    for ax, (g_lo, g_hi), label, color in zip(axes, MAG_BINS, BIN_LABELS, BIN_COLORS):
        bin_mask = (s1["g_mag"] >= g_lo) & (s1["g_mag"] < g_hi)
        candidates = s1.loc[bin_mask, "source_id"].to_numpy()
        n_bin_total = int(len(candidates))
        if n_bin_total == 0:
            ax.text(0.5, 0.5, "no stars in bin", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        target = rng.choice(candidates,
                            size=min(SAMPLE_PER_BIN, n_bin_total),
                            replace=False)
        target_arr = pa.array(sorted(int(x) for x in target.tolist()))
        opts = pc.SetLookupOptions(value_set=target_arr)
        kept = []
        for rg_idx in range(xp_pf.metadata.num_row_groups):
            rg = xp_pf.read_row_group(rg_idx, columns=["source_id", "corrected_flux", "ye2024_flag"])
            mask = pc.is_in(rg.column("source_id"), options=opts)
            chunk = rg.filter(mask)
            if chunk.num_rows:
                ok_mask = pc.equal(chunk.column("ye2024_flag"), 0)
                chunk_ok = chunk.filter(ok_mask)
                if chunk_ok.num_rows:
                    kept.append(chunk_ok)
        if not kept:
            ax.text(0.5, 0.5, "no Ye-OK XP for bin", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        kept_table = pa.concat_tables(kept)
        flux_list = kept_table.column("corrected_flux").to_pylist()
        flux = np.asarray(flux_list, dtype=np.float32)
        n_kept = flux.shape[0]
        if flux.size == 0:
            ax.text(0.5, 0.5, "no flux", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        median = np.median(flux, axis=0)
        p16 = np.percentile(flux, 16, axis=0)
        p84 = np.percentile(flux, 84, axis=0)
        ax.fill_between(SAMPLING_NM, p16, p84, alpha=0.3, color=color, lw=0)
        ax.plot(SAMPLING_NM, median, "-", color=color, lw=1.4)
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("Ye-corrected flux (Gaia internal units)")
        ax.set_title(
            f"{label}\n"
            f"bin total: {n_bin_total:,} stars  ·  SED sample: {n_kept:,}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "B5 — Stream 1: Ye+2024 NN-corrected XP sampled flux at three G-magnitude bins.\n"
        "Median ± 16-84 percentile; 330-point geometric grid 360-990 nm.",
        fontsize=10, fontweight="semibold",
    )
    save_fig(fig, REPO / "reports/gallery/B_preprocessing" / "B5_ye2024_flux_correction",
             formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B5: Ye+2024 corrected XP SED locus.")
    args = parser.parse_args()
    main()
