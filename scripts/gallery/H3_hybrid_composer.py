"""H3: Hybrid composer — per-element source attribution from real Stream 3 hybrid output.

What this shows:
- For each of 5 elements (Teff, log g, [M/H], [alpha/M], [Mg/H]), a pie chart
  of the fraction of stars whose final hybrid prediction came from each source
  (regressor / kNN rescue / regressor with caveat / etc).
- Source labels are taken verbatim from the *_hybrid_source string column in
  data/processed/pipeline1_predictions_stream3_hybrid.parquet.

What it reads:
- data/processed/pipeline1_predictions_stream3_hybrid.parquet (Stream 3,
  613,939 stars, columns: <elem>_hybrid_source for each element).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/H_hybrid_release"
ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
ELEMENT_LABELS = ("Teff", r"$\log g$", "[M/H]", r"[$\alpha$/M]", "[Mg/H]")


def main(argv: list[str] | None = None) -> int:
    apply_style()
    print("[H3] Loading real Stream 3 hybrid composer output")

    pq_path = REPO / "data/processed/pipeline1_predictions_stream3_hybrid.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"missing real hybrid output: {pq_path}")

    src_cols = [f"{elem}_hybrid_source" for elem in ELEMENTS]
    df = pd.read_parquet(pq_path, columns=src_cols)
    print(f"[H3] Loaded {len(df):,} stars; counting source attribution per element")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    palette = {
        "regressor": "#2ca02c",
        "knn_rescue": "#d62728",
        "knn": "#d62728",
        "regressor_caveat": "#ff7f0e",
        "regressor+caveat": "#ff7f0e",
        "drop": "#7f7f7f",
        "skip": "#7f7f7f",
    }

    for idx, (elem, elem_label) in enumerate(zip(ELEMENTS, ELEMENT_LABELS)):
        ax = axes[idx]
        col = f"{elem}_hybrid_source"
        counts = df[col].value_counts(dropna=False)
        labels = [str(s) for s in counts.index.tolist()]
        # value_counts on a pyarrow-backed string column returns an
        # ArrowExtensionArray for .values; cast to a numpy int64 array so
        # downstream .sum()/.tolist() are guaranteed.
        import numpy as np

        sizes = np.asarray(counts.to_numpy(), dtype=np.int64)
        colors = [palette.get(label, "#1f77b4") for label in labels]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(8)
            autotext.set_fontweight("bold")
        for text in texts:
            text.set_fontsize(7)
        ax.set_title(f"{elem_label} (n={int(sizes.sum()):,})")

    fig.suptitle(
        "Stream 3 hybrid composer — per-element source attribution (real)",
        fontsize=11,
        fontweight="semibold",
    )
    fig.set_layout_engine("constrained")
    save_fig(fig, OUT / "H3_hybrid_composer.pdf", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
