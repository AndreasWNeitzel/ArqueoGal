"""B15: Wavelength coverage of XP, broadband, and auxiliary inputs.

Shows the wavelength ranges of every photometric input the encoder
consumes, overlaid against the major stellar absorption features used
for chemical-abundance discrimination. Answers: do our inputs cover
the wavelengths where Fe, Mg, alpha-element, CN, etc. lines live?

Top panel: a median Ye+2024-corrected XP sampled flux on the 330-point
geometric grid (360-990 nm), with named lines / molecular bands
overplotted.

Bottom panel: a horizontal-bar diagram of the photometric passbands
(Gaia BP, G, RP; 2MASS J, H, Ks; AllWISE W1, W2) with their published
FWHM ranges.
"""

from __future__ import annotations

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
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/B_preprocessing"
SAMPLING_NM = np.geomspace(360.0, 990.0, 330)


# Wavelengths in nm. Sources: Hayden+15 disc α-lines reference, APOGEE-DR17
# atomic-line list, Mészáros+25 [X/M] correction paper, Mészáros+13 Gaia-ESO
# survey calibration, plus standard textbook positions.
ABSORPTION_LINES = [
    # (wavelength_nm, label, family)
    (393.4, "Ca II K", "Ca"),
    (396.8, "Ca II H", "Ca"),
    (430.8, "G band (CH)", "C"),
    (486.1, r"H$\beta$", "H"),
    (513.0, "Mg b triplet", "Mg"),
    (516.7, "Mg b", "Mg"),
    (517.3, "Mg b", "Mg"),
    (518.4, "Mg b", "Mg"),
    (527.0, "Fe I (MILES)", "Fe"),
    (561.0, "Fe I", "Fe"),
    (588.9, "Na D", "Na"),
    (589.6, "Na D", "Na"),
    (610.4, "Ca I", "Ca"),
    (615.2, "Fe I", "Fe"),
    (617.4, "Ca I", "Ca"),
    (656.3, r"H$\alpha$", "H"),
    (706.5, "MgH", "Mg"),
    (760.0, "O$_2$ (telluric)", "telluric"),
    (820.0, "Ca II IR", "Ca"),
    (854.2, "Ca II IR", "Ca"),
    (866.2, "Ca II IR", "Ca"),
    (881.0, "Fe I", "Fe"),
    (912.0, "Paschen jump", "H"),
    (939.0, "TiO band head", "TiO"),
    (970.0, "TiO band head", "TiO"),
]
FAMILY_COLOR = {
    "Mg": "#d62728",
    "Fe": "#1f77b4",
    "Ca": "#9467bd",
    "Na": "#ff7f0e",
    "C": "#2ca02c",
    "H": "#7f7f7f",
    "TiO": "#8c564b",
    "telluric": "#bcbd22",
}

# Gaia EDR3 + 2MASS + AllWISE filter ranges, central wavelengths in nm.
# Sources: Riello+2021 (Gaia), Cohen+2003 (2MASS), Wright+2010 (WISE).
PASSBANDS = [
    # (label, λ_lo, λ_hi, λ_central, family)
    ("Gaia BP", 330.0, 680.0, 511.0, "gaia"),
    ("Gaia G", 330.0, 1050.0, 673.0, "gaia"),
    ("Gaia RP", 640.0, 1050.0, 783.0, "gaia"),
    ("2MASS J", 1080.0, 1410.0, 1235.0, "2mass"),
    ("2MASS H", 1500.0, 1820.0, 1662.0, "2mass"),
    ("2MASS Ks", 2030.0, 2370.0, 2159.0, "2mass"),
    ("AllWISE W1", 2750.0, 3870.0, 3353.0, "wise"),
    ("AllWISE W2", 4030.0, 5340.0, 4603.0, "wise"),
]
PASSBAND_COLOR = {"gaia": "#1f77b4", "2mass": "#d62728", "wise": "#9467bd"}


def main() -> int:
    apply_style()

    # Sample 5000 stars from the Ye-corrected sampled-flux parquet to compute
    # a median XP SED on the 330-point grid. Stream-1 training pool only.
    xp_path = REPO / "data/interim/xp_sampled_corrected.parquet"
    feat_path = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not xp_path.exists() or not feat_path.exists():
        print(f"Error: required input missing\n  xp:  {xp_path}\n  feat:{feat_path}")
        return 1

    feat = pd.read_parquet(feat_path, columns=["source_id"])
    feat = feat.drop_duplicates("source_id", keep="first")
    rng = np.random.default_rng(0)
    target_ids = rng.choice(feat["source_id"].to_numpy(), size=min(5000, len(feat)), replace=False)
    target = pa.array(sorted(int(x) for x in target_ids.tolist()))
    opts = pc.SetLookupOptions(value_set=target)

    pf = pq.ParquetFile(xp_path)
    kept = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=["source_id", "corrected_flux", "ye2024_flag"])
        m = pc.is_in(chunk.column("source_id"), options=opts)
        ok = pc.equal(chunk.column("ye2024_flag"), 0)
        sel = pc.and_(m, ok)
        sub = chunk.filter(sel)
        if sub.num_rows:
            kept.append(sub)
    if not kept:
        print("Error: no XP rows match Stream-1 sample")
        return 1
    flux_list = pa.concat_tables(kept).column("corrected_flux").to_pylist()
    flux = np.asarray(flux_list, dtype=np.float32)
    median = np.nanmedian(flux, axis=0)
    p16 = np.nanpercentile(flux, 16, axis=0)
    p84 = np.nanpercentile(flux, 84, axis=0)
    print(f"[B15] median XP SED computed from {flux.shape[0]} Stream-1 stars")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.35)

    # --- Top panel: XP wavelength coverage with absorption lines ---
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(
        SAMPLING_NM, p16, p84, color="#1f77b4", alpha=0.22, label="Stream-1 16-84 percentile"
    )
    ax.plot(
        SAMPLING_NM, median, "-", color="#1f77b4", lw=1.4, label="Stream-1 median Ye-corrected flux"
    )

    ax.axvspan(
        SAMPLING_NM[0], SAMPLING_NM[-1], color="green", alpha=0.04, label="XP coverage 360-990 nm"
    )
    ax.axvline(SAMPLING_NM[0], color="green", lw=0.6, ls=":", alpha=0.5)
    ax.axvline(SAMPLING_NM[-1], color="green", lw=0.6, ls=":", alpha=0.5)

    # Plot absorption-line markers; group nearby labels vertically to avoid
    # overlap. Cycle through y-offsets within a wavelength window of 12 nm.
    sorted_lines = sorted(ABSORPTION_LINES, key=lambda r: r[0])
    last_lambda = -100.0
    cycle = 0
    y_positions = [0.95, 0.86, 0.77, 0.68]
    flux_max = float(np.nanpercentile(p84, 99))
    for lam, label, family in sorted_lines:
        if lam < SAMPLING_NM[0] - 5 or lam > SAMPLING_NM[-1] + 5:
            continue
        color = FAMILY_COLOR.get(family, "k")
        ax.axvline(lam, color=color, lw=0.8, ls="--", alpha=0.55)
        if lam - last_lambda < 12.0:
            cycle = (cycle + 1) % len(y_positions)
        else:
            cycle = 0
        y = y_positions[cycle]
        ax.text(
            lam,
            flux_max * y,
            label,
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            rotation=90,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
        )
        last_lambda = lam

    # Family legend.
    handles_family = [
        plt.Line2D([0], [0], color=c, ls="--", lw=1.2, label=f"{f} lines")
        for f, c in FAMILY_COLOR.items()
    ]
    leg1 = ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=handles_family, loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(340, 1010)
    ax.set_ylim(0, flux_max * 1.05)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("Ye-corrected flux (Gaia internal units)")
    ax.set_title("Gaia BP/RP XP coverage 360-990 nm with stellar absorption features overlaid")
    ax.grid(alpha=0.25)

    # --- Bottom panel: passband bars + XP coverage marker ---
    ax = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(PASSBANDS))
    for i, (label, lo, hi, central, family) in enumerate(PASSBANDS):
        c = PASSBAND_COLOR[family]
        ax.barh(i, hi - lo, left=lo, color=c, alpha=0.55, edgecolor=c, lw=1.0, height=0.75)
        ax.plot(
            central, i, "k|", ms=12, mew=1.5, label=r"$\lambda_\mathrm{eff}$" if i == 0 else None
        )
        ax.text(
            lo + (hi - lo) / 2,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="k",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )

    # XP coverage range as a separate band
    ax.axvspan(SAMPLING_NM[0], SAMPLING_NM[-1], color="green", alpha=0.10)
    ax.text(
        (SAMPLING_NM[0] + SAMPLING_NM[-1]) / 2,
        len(PASSBANDS) - 0.3,
        "Gaia BP/RP XP\n(360-990 nm,\n330 sampled points)",
        ha="center",
        va="bottom",
        fontsize=10,
        color="green",
        fontweight="semibold",
    )
    # Vertical lines for the same absorption features
    for lam, _label, family in sorted_lines:
        ax.axvline(lam, color=FAMILY_COLOR.get(family, "k"), lw=0.6, ls=":", alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlim(300, 6000)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in PASSBANDS], fontsize=9)
    ax.set_xlabel("wavelength [nm, log scale]")
    ax.set_title("Photometric input passband coverage (Gaia BP/G/RP + 2MASS JHKs + AllWISE W1/W2)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, which="both")

    fig.suptitle(
        "B15 - Wavelength coverage of encoder inputs versus stellar "
        "absorption features used for chemical-abundance discrimination",
        fontsize=12,
        fontweight="semibold",
        y=0.995,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B15_wavelength_coverage", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
