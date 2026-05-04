"""B10: 2MASS + AllWISE IR photometry coverage and colour-colour locus per stream.

What this shows (real data only):
- Panel 1: per-stream Ks-band detection fraction vs G (real 2MASS coverage drops
  at the faint end).
- Panel 2: real (J − Ks) vs (W1 − W2) colour-colour scatter, three streams overlaid.
- Panel 3: per-band median magnitude per stream (J, H, Ks, W1, W2).

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (Stream 1,
  Kiel-bounded RGB pool: logg ∈ [1.0, 3.5], Teff ∈ [4000, 5500] K)
- data/processed/pipeline1_features_stream{2,3}.parquet
  (columns: g_mag, j_mag, h_mag, k_mag, w1_mag, w2_mag).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/B_preprocessing"
STREAMS = (1, 2, 3)
PALETTE = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
# Stream 1 = Kiel-bounded training pool. Streams 2 and 3 keep their full
# inference cohorts (no Kiel mask — the bbox is a training-time decision).
STREAM_PARQUET = {
    1: "pipeline1_features_stream1_kiel.parquet",
    2: "pipeline1_features_stream2.parquet",
    3: "pipeline1_features_stream3.parquet",
}


def _load(stream_id: int) -> pd.DataFrame:
    path = REPO / "data/processed" / STREAM_PARQUET[stream_id]
    cols = ["g_mag", "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
    df = pd.read_parquet(path, columns=cols)
    df["stream"] = stream_id
    return df


def main(max_per_stream: int | None = None) -> None:
    apply_style()
    streams = {sid: _load(sid) for sid in STREAMS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")

    # Panel 1: per-stream Ks detection fraction vs G
    bins = np.linspace(8, 17, 13)
    centres = 0.5 * (bins[:-1] + bins[1:])
    for sid, df in streams.items():
        completeness = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (df["g_mag"] >= lo) & (df["g_mag"] < hi)
            if mask.sum() == 0:
                completeness.append(np.nan)
            else:
                completeness.append(df.loc[mask, "k_mag"].notna().mean())
        axes[0].plot(centres, completeness, "-o", color=PALETTE[sid],
                     label=f"Stream {sid} (n={len(df):,})", lw=1.5)
    axes[0].set_xlabel("G magnitude")
    axes[0].set_ylabel("2MASS Ks-band detection fraction")
    axes[0].set_title("IR completeness vs G")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    # Panel 2: J - Ks vs W1 - W2 (real)
    for sid, df in streams.items():
        m = df[["j_mag", "k_mag", "w1_mag", "w2_mag"]].notna().all(axis=1)
        sub = df.loc[m]
        if max_per_stream is not None and len(sub) > max_per_stream:
            sub = sub.sample(n=max_per_stream, random_state=42)
        axes[1].scatter(
            sub["j_mag"] - sub["k_mag"],
            sub["w1_mag"] - sub["w2_mag"],
            s=2, alpha=0.30, color=PALETTE[sid],
            label=f"Stream {sid}",
            rasterized=True,
        )
    axes[1].axhline(0, color="k", lw=0.5, ls=":", alpha=0.6)
    axes[1].set_xlabel("J − Ks (mag)")
    axes[1].set_ylabel("W1 − W2 (mag)")
    axes[1].set_title("IR colour-colour (real 2MASS × AllWISE)")
    axes[1].legend(loc="upper right")
    # tight bounds around the bulk to make the locus readable
    axes[1].set_xlim(-0.2, 1.8)
    axes[1].set_ylim(-0.4, 0.6)

    # Panel 3: per-band median magnitude per stream
    bands = ["j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
    band_labels = ["J", "H", "Ks", "W1", "W2"]
    x = np.arange(len(bands))
    width = 0.27
    for i, sid in enumerate(STREAMS):
        df = streams[sid]
        medians = [df[b].median() for b in bands]
        axes[2].bar(x + (i - 1) * width, medians, width,
                    color=PALETTE[sid], label=f"Stream {sid}")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(band_labels)
    axes[2].set_ylabel("median magnitude")
    axes[2].set_title("Per-band median magnitudes")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "B10 — Streams 1, 2, 3: real 2MASS + AllWISE IR photometry",
        fontsize=11, fontweight="semibold",
    )
    save_fig(fig, OUT / "B10_ir_photometry", formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B10: IR photometry coverage (real data).")
    parser.add_argument(
        "--max-per-stream", type=int, default=50000,
        help="Optional: per-stream scatter cap for panel 2 (default 50000)",
    )
    args = parser.parse_args()
    main(max_per_stream=args.max_per_stream)
