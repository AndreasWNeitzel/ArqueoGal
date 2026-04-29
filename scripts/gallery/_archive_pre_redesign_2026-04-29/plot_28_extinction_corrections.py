"""Stage 28: Extinction-correction diagnostic gallery.

Five plot families that the methods-paper §5 must carry:

1. **A_V provenance map** — distance × |b| scatter coloured by which
   dust-map layer fired (Edenhofer / Lallement / SFD / neighbourhood-median
   / missing). Confirms the fusion priority is implemented correctly.
2. **Dereddening lever-arm panel** — per-band raw vs dereddened mag with
   the Yuan+2013 reference slope overlaid. Reads "the recipe was applied".
3. **Intrinsic-magnitude recovery** — per-band median + 16-84% envelope
   vs A_V for raw and dereddened mags. The dereddened median should be
   *flat* in A_V; the raw median should track the Yuan+2013 slope. Reads
   "the recipe collapsed the extinction signal correctly".
4. **Residual-vs-A_V diagnostic** — using the synthetic-fixture truth
   ``av_intrinsic``, residual = ``mag_dered − mag_truth`` plotted vs A_V.
   Median should sit on zero; per-band RMS reports the propagated error
   from the dust-map-fusion noise. Reads "the recipe is unbiased".
5. **A_V trust-flag breakdown** — stacked bar of the three trust flags'
   firing rates.

The script runs on a synthetic-but-realistic Stream-1-shaped fixture so the
plots are visually inspectable today. Production runs (against the real
Stream-1 / Stream-2 / Stream-3 parquets) reuse the same code path by
swapping the fixture for a parquet read.
"""

from __future__ import annotations

# isort: off
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.data.extinction import (  # noqa: E402
    AV_SOURCE_CODES,
    AV_SOURCE_NAMES,
    DEFAULT_EXTINCTION_LAW,
    apply_extinction_corrections,
)
from arqueogal.utils.plotting import set_aa_style  # noqa: E402

# isort: on


OUT = REPO / "reports/gallery/28_extinction_corrections"


def _build_fixture(n: int = 8000, seed: int = 20260429) -> pd.DataFrame:
    """Stream-1-shaped fixture: disc giants over 0-5 kpc, A_V realistic."""
    rng = np.random.default_rng(seed)

    # Distances: log-uniform 0.05-5 kpc to populate every dust-map regime.
    distance = rng.uniform(0.05, 5.0, n)

    # Galactic latitude: uniform in sin(b) so the high-extinction plane
    # gets fair sampling.
    sin_b = rng.uniform(-1.0, 1.0, n)
    b_deg = np.degrees(np.arcsin(sin_b))
    l_deg = rng.uniform(0.0, 360.0, n)

    # Per-sightline Av: high in the plane, low at the poles, scaled by
    # distance. Layered so different dust-map columns fire in different
    # regimes.
    av_intrinsic = 0.4 * (1 - np.abs(sin_b)) * np.minimum(
        distance, 3.0
    ) + 0.05 * rng.standard_normal(n)
    av_intrinsic = np.clip(av_intrinsic, 0.0, 4.0)

    # Each map sees the same underlying av but with its own noise + drop-out.
    eden_mask = (distance <= 1.25) & (rng.random(n) > 0.05)
    lall_mask = (distance > 0.5) & (distance <= 3.5) & (rng.random(n) > 0.10)
    sfd_mask = rng.random(n) > 0.15
    nbhd_mask = rng.random(n) > 0.30

    av_eden = np.where(eden_mask, av_intrinsic + 0.03 * rng.standard_normal(n), np.nan)
    av_lall = np.where(lall_mask, av_intrinsic + 0.06 * rng.standard_normal(n), np.nan)
    av_sfd = np.where(sfd_mask, av_intrinsic + 0.10 * rng.standard_normal(n), np.nan)
    av_nbhd = np.where(nbhd_mask, av_intrinsic + 0.15 * rng.standard_normal(n), np.nan)
    av_nbhd_std = np.clip(
        0.05 + 0.4 * np.abs(av_intrinsic) + 0.05 * rng.standard_normal(n), 0.01, 1.5
    )

    # Broadband photometry: 13 mag fiducial + colour spread + extinction in
    # each band per Yuan+2013 (so the fixture is internally consistent with
    # the law we will recover).
    j_mag = 13.0 + 0.4 * rng.standard_normal(n) + av_intrinsic * 0.276
    h_mag = 12.6 + 0.4 * rng.standard_normal(n) + av_intrinsic * 0.176
    k_mag = 12.4 + 0.4 * rng.standard_normal(n) + av_intrinsic * 0.112
    w1_mag = 12.3 + 0.4 * rng.standard_normal(n) + av_intrinsic * 0.063
    w2_mag = 12.25 + 0.4 * rng.standard_normal(n) + av_intrinsic * 0.050

    parallax_over_error = np.clip(20.0 / np.maximum(distance, 0.1), 0.5, 200.0)
    parallax_over_error += 1.5 * rng.standard_normal(n)

    return pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "l_deg": l_deg,
            "b_deg": b_deg,
            "r_med_photogeo": distance,
            "parallax_over_error": parallax_over_error,
            "av_edenhofer": av_eden,
            "av_lallement": av_lall,
            "av_sfd": av_sfd,
            "av_nbhd_median": av_nbhd,
            "av_nbhd_std": av_nbhd_std,
            "j_mag": j_mag,
            "h_mag": h_mag,
            "k_mag": k_mag,
            "w1_mag": w1_mag,
            "w2_mag": w2_mag,
            "av_intrinsic": av_intrinsic,
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_av_provenance(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: distance vs |b|, coloured by source code.
    ax = axes[0]
    palette = {
        AV_SOURCE_CODES["edenhofer"]: "#1f77b4",
        AV_SOURCE_CODES["lallement"]: "#2ca02c",
        AV_SOURCE_CODES["sfd"]: "#d62728",
        AV_SOURCE_CODES["neighborhood_median"]: "#ff7f0e",
        AV_SOURCE_CODES["missing"]: "0.6",
    }
    src = df["av_los_source"].to_numpy()
    for code, color in palette.items():
        mask = src == code
        if not mask.any():
            continue
        label = f"{AV_SOURCE_NAMES[code]} (n={int(mask.sum()):,})"
        ax.scatter(
            df.loc[mask, "r_med_photogeo"],
            np.abs(df.loc[mask, "b_deg"]),
            s=4,
            alpha=0.4,
            color=color,
            label=label,
            rasterized=True,
        )
    ax.axvline(1.25, color="0.4", linestyle=":", lw=0.8)
    ax.axvline(3.0, color="0.4", linestyle=":", lw=0.8)
    ax.set_xlabel("Bailer-Jones distance (kpc)")
    ax.set_ylabel(r"$|b|$ (deg)")
    ax.set_title("A$_\\mathrm{V}$ source by distance × |b|")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.95)

    # Right: A_V vs distance per source.
    ax = axes[1]
    for code, color in palette.items():
        mask = src == code
        if mask.sum() == 0:
            continue
        ax.scatter(
            df.loc[mask, "r_med_photogeo"],
            df.loc[mask, "av_los"],
            s=4,
            alpha=0.4,
            color=color,
            rasterized=True,
            label=AV_SOURCE_NAMES[code],
        )
    ax.set_xlabel("Bailer-Jones distance (kpc)")
    ax.set_ylabel(r"A$_\mathrm{V}$ (mag)")
    ax.set_title("Fused A$_\\mathrm{V}$ vs distance")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Extinction provenance — {DEFAULT_EXTINCTION_LAW.name}",
        fontsize=10,
    )
    return _save(fig, out_dir / "av_provenance.pdf")


def _plot_dereddening_lever_arms(df: pd.DataFrame, out_dir: Path) -> Path:
    """Per-band raw-vs-dereddened scatter with 1:1 and Yuan+2013 reference.

    Reads as: a star with A_V = 0 sits on the 1:1 line; a star with
    A_V = 1 sits on the red dashed reference (offset by the Yuan+2013
    ratio); higher A_V slides further down. A clean diagonal track
    coloured smoothly by A_V means the recipe is applied; the
    *recovery* diagnostic in :func:`_plot_intrinsic_recovery` is what
    proves it works.
    """
    bands = (
        ("j_mag", 0.276, "J"),
        ("h_mag", 0.176, "H"),
        ("k_mag", 0.112, r"K_s"),
        ("w1_mag", 0.063, "W1"),
        ("w2_mag", 0.050, "W2"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()
    axes[-1].set_visible(False)
    sc = None
    finite = df["av_los"].notna()
    for ax, (band, ratio, pretty) in zip(axes[:5], bands):
        raw = df.loc[finite, band].to_numpy()
        dered = df.loc[finite, f"{band}_dered"].to_numpy()
        av = df.loc[finite, "av_los"].to_numpy()
        sc = ax.scatter(raw, dered, s=4, alpha=0.5, c=av, cmap="viridis", rasterized=True)
        lo = float(np.nanmin(raw))
        hi = float(np.nanmax(raw))
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.9, linestyle=":", label="1:1 (A_V = 0)")
        ax.plot(
            [lo, hi],
            [lo - ratio, hi - ratio],
            color="C3",
            lw=1.0,
            linestyle="--",
            label=f"A_V = 1·{ratio:.3f}",
        )
        ax.set_xlabel(f"raw {pretty} (mag)")
        ax.set_ylabel(f"dered {pretty} (mag)")
        ax.set_title(f"{pretty}: $A_\\lambda / A_V = {ratio:.3f}$")
        ax.legend(fontsize=8, loc="upper left")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[:5].tolist(), shrink=0.7, pad=0.02)
        cbar.set_label(r"$A_V$ (mag)")
    fig.suptitle(
        "Per-band dereddening lever arms — Yuan+2013 ratios, CCM89 R_V=3.1",
        fontsize=11,
    )
    return _save(fig, out_dir / "dereddening_lever_arms.pdf")


def _plot_intrinsic_recovery(df: pd.DataFrame, out_dir: Path) -> Path:
    """Methods-paper-grade "did the recipe actually work?" diagnostic.

    For each band, plot the dispersion of (mag_raw, mag_dered) as a
    function of A_V. If the recipe works, ``mag_dered`` at fixed
    intrinsic colour is *independent* of A_V — i.e. the dispersion of
    ``mag_dered`` vs A_V is flat at the intrinsic scatter floor (~0.4
    mag in the synthetic fixture, set by stellar-population colour
    spread), while ``mag_raw`` rises steeply with A_V at the
    Yuan+2013 slope.

    This figure is the single best summary of "the dereddening reached
    intrinsic colour, not just shifted the magnitudes by a constant".
    """
    bands = (
        ("j_mag", 0.276, "J"),
        ("h_mag", 0.176, "H"),
        ("k_mag", 0.112, "K_s"),
        ("w1_mag", 0.063, "W1"),
        ("w2_mag", 0.050, "W2"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()
    axes[-1].set_visible(False)

    finite = df["av_los"].notna()
    av = df.loc[finite, "av_los"].to_numpy()
    av_bins = np.linspace(0.0, np.nanpercentile(av, 99), 9)
    centres = 0.5 * (av_bins[:-1] + av_bins[1:])

    for ax, (band, ratio, pretty) in zip(axes[:5], bands):
        raw = df.loc[finite, band].to_numpy()
        dered = df.loc[finite, f"{band}_dered"].to_numpy()
        # For each Av bin: median + 16/84 percentile of raw and dered.
        med_raw, p16_raw, p84_raw = [], [], []
        med_d, p16_d, p84_d = [], [], []
        for lo, hi in zip(av_bins[:-1], av_bins[1:]):
            mask = (av >= lo) & (av < hi)
            if mask.sum() < 30:
                med_raw.append(np.nan)
                p16_raw.append(np.nan)
                p84_raw.append(np.nan)
                med_d.append(np.nan)
                p16_d.append(np.nan)
                p84_d.append(np.nan)
                continue
            r = raw[mask]
            d = dered[mask]
            med_raw.append(np.median(r))
            p16_raw.append(np.percentile(r, 16))
            p84_raw.append(np.percentile(r, 84))
            med_d.append(np.median(d))
            p16_d.append(np.percentile(d, 16))
            p84_d.append(np.percentile(d, 84))
        med_raw = np.array(med_raw)
        med_d = np.array(med_d)
        p16_raw = np.array(p16_raw)
        p84_raw = np.array(p84_raw)
        p16_d = np.array(p16_d)
        p84_d = np.array(p84_d)

        ax.fill_between(centres, p16_raw, p84_raw, alpha=0.25, color="C3", label="raw 16-84%")
        ax.plot(centres, med_raw, "-o", color="C3", lw=1.5, label="raw median")
        ax.fill_between(centres, p16_d, p84_d, alpha=0.25, color="C0", label="dered 16-84%")
        ax.plot(centres, med_d, "-o", color="C0", lw=1.5, label="dered median")

        # Reference: where the raw median *should* sit if it followed
        # exactly mag(A_V=0) + A_V·ratio. Fixture intrinsic median ≈ raw
        # median in the lowest Av bin.
        intrinsic = float(np.nanmedian(med_d))
        ax.plot(
            centres,
            intrinsic + centres * ratio,
            color="0.3",
            lw=0.9,
            linestyle=":",
            label=f"Yuan+2013 model (slope {ratio:.3f})",
        )

        ax.set_xlabel(r"$A_V$ (mag)")
        ax.set_ylabel(f"{pretty} (mag)")
        ax.set_title(f"{pretty}: extinction recovery")
        ax.legend(fontsize=7, loc="upper left")
        ax.invert_yaxis()  # mag axis astronomical convention

    fig.suptitle(
        "Intrinsic-magnitude recovery — dered should be flat vs $A_V$, "
        "raw should track Yuan+2013 slope",
        fontsize=11,
    )
    return _save(fig, out_dir / "intrinsic_recovery.pdf")


def _plot_residual_vs_av(df: pd.DataFrame, out_dir: Path) -> Path:
    """Bias-vs-A_V residual: ``mag_dered − mag_dered_truth`` vs A_V.

    The synthetic fixture has a known intrinsic magnitude (the
    ``av_intrinsic = 0`` baseline). After dereddening with the *fitted*
    Av, the residual ``mag_dered − intrinsic`` should be zero-mean
    across A_V; any slope is the systematic the recipe failed to remove.
    On the synthetic fixture the residual exposes (a) the noise floor
    of the dust-map-fusion column (Edenhofer ≈ 0.03, Lallement ≈ 0.06,
    SFD ≈ 0.10, neighbourhood ≈ 0.15 mag in our injection) and (b) the
    Yuan+2013 ratio precision.
    """
    bands = (
        ("j_mag", 0.276, "J"),
        ("h_mag", 0.176, "H"),
        ("k_mag", 0.112, "K_s"),
        ("w1_mag", 0.063, "W1"),
        ("w2_mag", 0.050, "W2"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()
    axes[-1].set_visible(False)

    finite = df["av_los"].notna() & df["av_intrinsic"].notna()
    av_los = df.loc[finite, "av_los"].to_numpy()
    av_true = df.loc[finite, "av_intrinsic"].to_numpy()
    av_bins = np.linspace(0.0, np.nanpercentile(av_los, 99), 9)
    centres = 0.5 * (av_bins[:-1] + av_bins[1:])

    for ax, (band, ratio, pretty) in zip(axes[:5], bands):
        raw = df.loc[finite, band].to_numpy()
        dered = df.loc[finite, f"{band}_dered"].to_numpy()
        # Truth intrinsic magnitude in the fixture: raw − A_V_true · ratio.
        intrinsic = raw - av_true * ratio
        residual = dered - intrinsic  # = ratio · (av_true − av_los)

        med, p16, p84 = [], [], []
        for lo, hi in zip(av_bins[:-1], av_bins[1:]):
            mask = (av_los >= lo) & (av_los < hi)
            if mask.sum() < 30:
                med.append(np.nan)
                p16.append(np.nan)
                p84.append(np.nan)
                continue
            r = residual[mask]
            med.append(np.median(r))
            p16.append(np.percentile(r, 16))
            p84.append(np.percentile(r, 84))
        med = np.array(med)
        p16 = np.array(p16)
        p84 = np.array(p84)

        ax.fill_between(centres, p16, p84, alpha=0.30, color="C0", label="16-84% residual")
        ax.plot(centres, med, "-o", color="C0", lw=1.5, label="median residual")
        ax.axhline(0.0, color="0.3", lw=0.8, linestyle=":")

        # Annotate the typical residual at this band.
        rms = float(np.sqrt(np.nanmean(residual**2)))
        ax.text(
            0.04,
            0.95,
            f"RMS = {rms:.3f} mag",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="0.4", alpha=0.85, pad=2),
        )

        ax.set_xlabel(r"$A_V$ (fused, mag)")
        ax.set_ylabel(f"{pretty} dered residual (mag)")
        ax.set_title(f"{pretty}: bias vs $A_V$")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Dereddening bias diagnostic — residual median should sit on zero",
        fontsize=11,
    )
    return _save(fig, out_dir / "residual_vs_av.pdf")


def _plot_trust_flags(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = pd.Series(
        {
            "neighbourhood\nfallback": int(df["av_is_neighborhood_fallback"].sum()),
            "distance prior\ndominated": int(df["av_distance_prior_dominated"].sum()),
            "high-dispersion\nsightline": int(df["av_neighbourhood_high_dispersion"].sum()),
        }
    )
    total = len(df)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(counts.index, counts.values, color="#1f77b4")
    ax.set_ylabel("# stars carrying the flag")
    ax.set_title(f"A_V trust-flag breakdown (n_total = {total:,})")
    for bar, val in zip(bars, counts.values):
        pct = 100.0 * val / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,} ({pct:.1f} %)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(counts.values) * 1.15 if counts.values.max() > 0 else 1.0)
    return _save(fig, out_dir / "av_trust_flags.pdf")


def main() -> None:
    set_aa_style(usetex=False)
    print("[plot_28] Building synthetic Stream-1-shaped fixture (8000 stars)")
    df_raw = _build_fixture()
    print("[plot_28] Applying CCM89 R_V=3.1 + Yuan+2013 dereddening")
    df = apply_extinction_corrections(df_raw)
    OUT.mkdir(parents=True, exist_ok=True)

    print("[plot_28] Rendering plot families")
    written = [
        _plot_av_provenance(df, OUT),
        _plot_dereddening_lever_arms(df, OUT),
        _plot_intrinsic_recovery(df, OUT),
        _plot_residual_vs_av(df, OUT),
        _plot_trust_flags(df, OUT),
    ]
    for path in written:
        print(f"  - {path}")

    # Write the law fingerprint as a sidecar so methods-paper consumers can
    # cite the exact ratios used.
    import json

    sidecar = OUT / "extinction_law.json"
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_EXTINCTION_LAW.fingerprint(), f, indent=2, sort_keys=True)
    print(f"  - {sidecar}")


if __name__ == "__main__":
    main()
