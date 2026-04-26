"""
ArqueoGal methods-paper figure regeneration.

Single-entry-point script that regenerates every figure in the methods paper
from pinned data sources, with deterministic seeds, git SHA logging, input
SHA-256 capture, and a manifest output.

Usage:
    python paper/methods_paper/make_figures.py [--fig N | --all]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# from arqueogal.utils.plotting import set_aa_style
# from arqueogal.utils.label_conventions import TEFF, LOGG, M_H, MG_H, ALPHA_M, FE_H, BP_RP, G_MAG

PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
DATA_DIR = PAPER_DIR.parents[1] / "data" / "processed"


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def setup_style() -> None:
    """Apply A&A figure conventions including pdf.fonttype=42."""
    # set_aa_style()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# Manifest: figure number -> (caption stub, list of input data Parquets, output filename)
MANIFEST: dict[int, tuple[str, list[str], str]] = {
    1: ("Pipeline schematic (DAG)", [], "fig01_pipeline_schematic.pdf"),
    2: ("Per-element residual scatter", ["pipeline1_predictions.parquet"], "fig02_residuals.pdf"),
    3: ("Reliability diagrams pre/post shrinkage", ["calibration.parquet"], "fig03_reliability.pdf"),
    4: ("Information-content audit (CMI per label per regime)", ["audit_cmi.parquet"], "fig04_cmi.pdf"),
    5: ("Aux-only baseline RMSE comparison", ["aux_only_baseline.parquet"], "fig05_aux_baseline.pdf"),
    6: ("HRD colored by predicted [Fe/H]", ["pipeline1_predictions.parquet"], "fig06_hrd.pdf"),
    7: ("Cross-catalogue Bland-Altman vs GALAH DR4", ["xcat_galah.parquet"], "fig07_bland_altman.pdf"),
    8: ("Mahalanobis OOD distribution Stream 1 vs Stream 3", ["ood_distribution.parquet"], "fig08_ood.pdf"),
    9: ("Regime B Teff bias diagnostic", ["regime_b_residuals.parquet"], "fig09_regime_b.pdf"),
    10: ("Per-magnitude RMSE", ["pipeline1_predictions.parquet"], "fig10_mag_stratified.pdf"),
    11: ("CMI decomposition: parallax-only vs full aux conditioning", ["audit_cmi.parquet"], "fig11_cmi_decomp.pdf"),
    12: ("Methods-paper headline: per-label CMI showing aux-assisted labels", ["audit_cmi.parquet"], "fig12_headline.pdf"),
}


def make_figure_01_pipeline_schematic() -> Path:
    """Pipeline DAG schematic. Phase B fills in."""
    fig, ax = plt.subplots(figsize=(7.087, 4.0))
    ax.text(0.5, 0.5, "TODO: Pipeline schematic (Phase B)", ha="center", va="center")
    ax.set_axis_off()
    out = FIG_DIR / MANIFEST[1][2]
    fig.savefig(out)
    plt.close(fig)
    return out


def make_figure_02_residuals() -> Path:
    """Per-element residual scatter. Phase B fills in."""
    fig, ax = plt.subplots(figsize=(3.464, 2.5))
    ax.text(0.5, 0.5, "TODO: residuals (Phase B)", ha="center", va="center")
    out = FIG_DIR / MANIFEST[2][2]
    fig.savefig(out)
    plt.close(fig)
    return out


# Stubs for figures 3..12 follow the same pattern.

FIGURE_BUILDERS = {
    1: make_figure_01_pipeline_schematic,
    2: make_figure_02_residuals,
    # Phase B adds 3..12.
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(20260424)

    sha = git_sha()
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    targets = list(FIGURE_BUILDERS.keys()) if args.all else (
        [args.fig] if args.fig else list(FIGURE_BUILDERS.keys())
    )

    output_manifest = {
        "git_sha": sha,
        "timestamp_utc": timestamp,
        "figures": {},
    }

    for fig_n in targets:
        builder = FIGURE_BUILDERS.get(fig_n)
        if builder is None:
            continue
        path = builder()
        output_manifest["figures"][fig_n] = {
            "caption_stub": MANIFEST[fig_n][0],
            "output": str(path.relative_to(PAPER_DIR)),
            "sha256": file_sha256(path) if path.exists() else None,
            "input_data": MANIFEST[fig_n][1],
        }

    (FIG_DIR / "manifest.json").write_text(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
