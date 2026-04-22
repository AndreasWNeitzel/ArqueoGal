"""Stage 00: data-source maps.

Outputs:
  - reports/gallery/00_data_sources/stream_sky_maps.png   (Galactic Mollweide)
  - reports/gallery/00_data_sources/stream_row_counts.png

Stream 1 (APOGEE × Gaia training)     — data/processed/pipeline1_features_stream1.parquet
Stream 2 (TESS Hon+2021 audit)        — data/interim/stream2_tess_gaia.parquet (if available)
Stream 3 (Andrae+2023 RGB deployment) — data/processed/pipeline1_features_stream3.parquet

Galactic-coord sky maps follow the convention l=0 at centre, longitude increasing
right-to-left (see _common.galactic_mollweide).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_INTERIM,
    DATA_PROCESSED,
    GALLERY,
    PALETTE,
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = GALLERY / "00_data_sources"


def _load_stream(path: Path, cols: list[str], n_sample: int) -> dict[str, np.ndarray]:
    t = pq.read_table(path, columns=cols)
    n = t.num_rows
    idx = sample_index(n, n_sample)
    return {c: t.column(c).to_numpy(zero_copy_only=False)[idx] for c in cols}


def sky_maps() -> None:
    s1 = _load_stream(DATA_PROCESSED / "pipeline1_features_stream1.parquet",
                      ["ra_deg", "dec_deg"], 60_000)
    s3 = _load_stream(DATA_PROCESSED / "pipeline1_features_stream3.parquet",
                      ["ra_deg", "dec_deg", "sample"], 80_000)
    s2_path = DATA_INTERIM / "stream2_tess_gaia.parquet"
    s2 = None
    if s2_path.exists():
        try:
            s2 = _load_stream(s2_path, ["ra", "dec"], 30_000)
            s2["ra_deg"] = s2.pop("ra"); s2["dec_deg"] = s2.pop("dec")
        except Exception:
            s2 = None

    fig = plt.figure(figsize=(14, 10))
    axes = [
        fig.add_subplot(221, projection="mollweide"),
        fig.add_subplot(222, projection="mollweide"),
        fig.add_subplot(223, projection="mollweide"),
        fig.add_subplot(224, projection="mollweide"),
    ]

    x, y = radec_to_galactic_mollweide(s1["ra_deg"], s1["dec_deg"])
    axes[0].scatter(x, y, s=0.4, alpha=0.4, color=PALETTE["apogee"], rasterized=True)
    axes[0].set_title(f"Stream 1 — APOGEE × Gaia training  (n={s1['ra_deg'].size:,} plotted)")

    if s2 is not None:
        x, y = radec_to_galactic_mollweide(s2["ra_deg"], s2["dec_deg"])
        axes[1].scatter(x, y, s=0.4, alpha=0.4, color=PALETTE["tess"], rasterized=True)
        axes[1].set_title(f"Stream 2 — TESS Hon+2021 audit  (n={s2['ra_deg'].size:,} plotted)")
    else:
        axes[1].text(0, 0, "Stream 2 not found in interim/", ha="center", va="center")
        axes[1].set_title("Stream 2 — TESS Hon+2021 audit  (missing)")

    # Stream 3: values are "uniform" and "volume_limited" (build_stream3_expansion_union.py)
    vol_mask = s3["sample"] == "volume_limited"
    uni_mask = s3["sample"] == "uniform"
    x, y = radec_to_galactic_mollweide(s3["ra_deg"][vol_mask], s3["dec_deg"][vol_mask])
    axes[2].scatter(x, y, s=0.4, alpha=0.35, color=PALETTE["andrae_volume"], rasterized=True)
    axes[2].set_title(f"Stream 3 — volume-limited arm  (n={int(vol_mask.sum()):,} plotted)")

    x, y = radec_to_galactic_mollweide(s3["ra_deg"][uni_mask], s3["dec_deg"][uni_mask])
    axes[3].scatter(x, y, s=0.4, alpha=0.35, color=PALETTE["andrae_uniform"], rasterized=True)
    axes[3].set_title(f"Stream 3 — uniform arm  (n={int(uni_mask.sum()):,} plotted)")

    for ax in axes:
        style_galactic_mollweide(ax)

    fig.suptitle("Stream footprints on the sky  —  Galactic coordinates (l increasing right-to-left)",
                 y=1.00, fontsize=13, fontweight="bold")
    save_fig(fig, OUT / "stream_sky_maps.png")


def row_counts() -> None:
    counts = {
        "Stream 1\nAPOGEE × Gaia\ntraining": 324_054,
        "Stream 2\nTESS Hon+2021\naudit (held-out)": 66_000,
        "Stream 3\nAndrae+2023 RGB\ndeployment (union)": 613_939,
        "Pipeline-1\npredicted\n(volume arm)": 249_092,
        "σ-gated\n(downstream input)": 211_739,
    }
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(counts.keys(), counts.values(),
                  color="#4c78a8", edgecolor="#222", linewidth=0.6)
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width()/2, v * 1.04, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("row count")
    ax.set_yscale("log")
    ax.set_title("Row-count waterfall across the pipeline  (log y)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)
    save_fig(fig, OUT / "stream_row_counts.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    sky_maps()
    row_counts()


if __name__ == "__main__":
    main()
