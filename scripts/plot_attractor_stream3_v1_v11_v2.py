"""Side-by-side [α/M]_pred vs [M/H]_pred density on Stream-3 volume for v1, v1.1, v2.

Visualises whether the prototype attractor at [α/M]≈+0.1 (first reported on v1)
is still present in v1.1 and v2 — and if so, where it sits.

All three panels share the same axes and colourmap so the attractor can be
compared by eye. Stars flagged OOD (ood_joint_flag) or inside the Regime-B
envelope are excluded, matching release selection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("attractor_plot")

REPO = Path(__file__).resolve().parent.parent
INPUTS = {
    "v1":   REPO / "data/processed/pipeline1_predictions_stream3_volume.parquet",
    "v1.1": REPO / "data/processed/pipeline1_predictions_stream3_volume_v11.parquet",
    "v2":   REPO / "data/processed/pipeline1_predictions_stream3_volume_v2.parquet",
}
OUT_PNG = REPO / "reports/pipeline1/run_a_v2/attractor_stream3_v1_v11_v2.png"

MH_RANGE = (-2.5, 0.6)
ALPHA_RANGE = (-0.1, 0.5)
BINS = (180, 120)


def _load_release_ok(
    p: Path, *, apply_mode_ambiguous: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    cols = ["mh_pred", "alpha_m_pred", "ood_joint_flag", "regime_b_flag"]
    if apply_mode_ambiguous:
        cols.append("mode_ambiguous_flag")
    df = pd.read_parquet(p, columns=cols)
    total = len(df)
    ok = (~df["ood_joint_flag"].astype(bool)) & (~df["regime_b_flag"].astype(bool))
    if apply_mode_ambiguous:
        ok &= ~df["mode_ambiguous_flag"].astype(bool)
    sub = df.loc[ok]
    return (
        sub["mh_pred"].to_numpy(),
        sub["alpha_m_pred"].to_numpy(),
        total,
        int(ok.sum()),
    )


def main() -> None:
    # Four panels: v1, v1.1, v2 raw, v2 after mode_ambiguous_flag
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.045, right=0.92, top=0.80, bottom=0.12, wspace=0.08)
    vmax = 0

    spec = [
        ("v1",   INPUTS["v1"],   False),
        ("v1.1", INPUTS["v1.1"], False),
        ("v2",   INPUTS["v2"],   False),
        ("v2 (release: ¬mode_ambiguous)", INPUTS["v2"], True),
    ]
    panels = []
    for ax, (tag, path, apply_ma) in zip(axes, spec, strict=True):
        if not path.is_file():
            _LOG.warning("missing %s — skipping %s panel", path, tag)
            ax.text(0.5, 0.5, f"{tag}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        mh, am, total, n_ok = _load_release_ok(path, apply_mode_ambiguous=apply_ma)
        finite = np.isfinite(mh) & np.isfinite(am)
        mh, am = mh[finite], am[finite]
        H, xed, yed = np.histogram2d(
            mh, am, bins=BINS,
            range=[MH_RANGE, ALPHA_RANGE],
        )
        panels.append((ax, tag, H, xed, yed, total, n_ok))
        vmax = max(vmax, float(np.percentile(H, 99.5)))

    for ax, tag, H, xed, yed, total, n_ok in panels:
        im = ax.imshow(
            H.T, origin="lower", aspect="auto",
            extent=(xed[0], xed[-1], yed[0], yed[-1]),
            cmap="magma", vmin=0, vmax=vmax,
        )
        ax.axhline(0.11, color="cyan", lw=0.6, ls="--", alpha=0.6)
        ax.axvline(-1.0, color="cyan", lw=0.6, ls="--", alpha=0.6)
        ax.set_xlabel("[M/H]_pred (dex)")
        ax.set_title(
            f"{tag}\nrelease_ok {n_ok:,} / {total:,}  ({100*n_ok/total:.1f}%)",
            fontsize=10,
        )
    axes[0].set_ylabel("[α/M]_pred (dex)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("stars / 2-D bin  (counts, clipped at p99.5)")
    fig.suptitle(
        "Stream-3 volume: [α/M]_pred vs [M/H]_pred — v1 → v1.1 → v2 → v2 released\n"
        "dashed guide at [α/M]=+0.11, [M/H]=-1.0 (reported attractor)",
        fontsize=12, y=0.98,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    _LOG.info("wrote %s", OUT_PNG)


if __name__ == "__main__":
    main()
