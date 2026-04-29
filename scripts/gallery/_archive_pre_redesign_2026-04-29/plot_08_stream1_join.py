"""Stage 08: Stream-1 APOGEE × Gaia join with multi-spectrum dedup.

What the deploy did: ``data.ingest_stream1`` joined APOGEE DR19 onto
Gaia DR3 with ``validate="many_to_one"`` and a Δmag tie-break on the
``dr2_neighbourhood`` (300 mas / 0.1 mag, ADR-0001). ``data.dedup``
collapsed multi-ASPCAP-spectrum entries onto a single source_id.

What we plot: source_id duplicate-count distribution before vs after dedup;
post-dedup row count vs raw APOGEE count; verifies the join didn't quietly
inflate or shrink.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/08_stream1_join"


def main() -> None:
    apply_style()
    raw = REPO / "data/raw/apogee_dr19/aspcap_rgb.parquet"
    s1 = REPO / "data/processed/pipeline1_features_stream1.parquet"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    if raw.exists():
        ar = pd.read_parquet(raw, columns=["source_id"]).dropna()
        sid_counts = ar["source_id"].value_counts()
        bins = np.arange(1, sid_counts.max() + 2)
        axes[0].hist(
            sid_counts.values,
            bins=bins - 0.5,
            color="#d62728",
            alpha=0.7,
            label=f"raw APOGEE\n({len(ar):,} rows, {ar['source_id'].nunique():,} unique)",
        )
    else:
        axes[0].text(
            0.5,
            0.7,
            "raw APOGEE parquet not present\nshowing only post-dedup",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=8,
        )

    if s1.exists():
        post = pd.read_parquet(s1, columns=["source_id"])
        post_counts = post["source_id"].value_counts()
        bins = np.arange(1, max(post_counts.max(), 4) + 2)
        axes[0].hist(
            post_counts.values,
            bins=bins - 0.5,
            color="#1f77b4",
            alpha=0.7,
            label=f"post-dedup Stream 1\n({len(post):,} rows, "
            f"{post['source_id'].nunique():,} unique)",
        )
    axes[0].set_xlabel("# spectra per source_id")
    axes[0].set_ylabel("count")
    axes[0].set_yscale("log")
    axes[0].set_title("Multi-spectrum dedup")
    axes[0].legend(fontsize=8)

    # APOGEE label sanity post-dedup
    if s1.exists():
        post = pd.read_parquet(s1, columns=["teff_apogee", "logg_apogee", "mh_apogee"])
        n = len(post)
        n_finite = post.dropna().shape[0]
        bars = axes[1].bar(
            ["raw rows", "post-dedup unique sources", "complete labels (3-tier)"],
            [n, post["teff_apogee"].notna().sum(), n_finite],
            color=["#9467bd", "#1f77b4", "#2ca02c"],
        )
        for b, v in zip(bars, [n, post["teff_apogee"].notna().sum(), n_finite]):
            axes[1].text(
                b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8
            )
        axes[1].set_ylabel("rows / sources")
        axes[1].set_title("Stream-1 row inventory")
        axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle("APOGEE DR19 × Gaia DR3 join + multi-spectrum dedup", fontsize=11)
    save_fig(fig, OUT / "stream1_join.png")


if __name__ == "__main__":
    main()
