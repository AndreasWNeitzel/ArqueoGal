"""Stage 01: raw Gaia XP spectra + coefficient distributions.

Outputs:
  - reports/gallery/01_gaia_xp_raw/xp_sed_atlas_by_hrd.png
  - reports/gallery/01_gaia_xp_raw/xp_coef_distributions.png
  - reports/gallery/01_gaia_xp_raw/xp_example_stars.png

Uses data/interim/xp_sampled_corrected.parquet (corrected_flux: 330 samples per star,
the continuous SED on the Gaia XP wavelength grid) JOINed on source_id with
pipeline1_features_stream1.parquet for Teff/logg/[M/H] labels and A_V/G for filtering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import DATA_INTERIM, DATA_PROCESSED, GALLERY, apply_style, sample_index, save_fig  # noqa: E402

OUT = GALLERY / "01_gaia_xp_raw"

XP_WAVELENGTHS_NM = np.linspace(336.0, 1020.0, 330)  # gaiaxpy default external wave grid


def _load_xp_subset(n_target: int = 8_000, seed: int = 17) -> dict[str, np.ndarray]:
    """Load a random subset of corrected XP SEDs with matching HRD labels + A_V + G."""
    xp = pq.read_table(
        DATA_INTERIM / "xp_sampled_corrected.parquet",
        columns=["source_id", "corrected_flux"],
    )
    feat_cols = ["source_id", "teff_apogee", "logg_apogee", "mh_apogee",
                 "g_mag", "bp_rp", "av_sfd"]
    feat = pq.read_table(
        DATA_PROCESSED / "pipeline1_features_stream1.parquet",
        columns=feat_cols,
    )
    xp_df = xp.to_pandas()
    feat_df = feat.to_pandas().dropna(subset=["teff_apogee", "logg_apogee", "mh_apogee"])
    merged = xp_df.merge(feat_df, on="source_id", how="inner")
    if len(merged) == 0:
        raise RuntimeError("no XP × features overlap — check source_id ranges")
    idx = sample_index(len(merged), n_target, np.random.default_rng(seed))
    merged = merged.iloc[idx].reset_index(drop=True)
    flux = np.vstack(merged["corrected_flux"].apply(np.asarray).to_list())
    return {
        "flux": flux,
        "teff": merged["teff_apogee"].to_numpy(),
        "logg": merged["logg_apogee"].to_numpy(),
        "mh": merged["mh_apogee"].to_numpy(),
        "g_mag": merged["g_mag"].to_numpy(),
        "bp_rp": merged["bp_rp"].to_numpy(),
        "av_sfd": merged["av_sfd"].to_numpy(),
        "source_id": merged["source_id"].to_numpy(),
    }


def sed_atlas_by_hrd(data: dict[str, np.ndarray]) -> None:
    """3×3 grid over (Teff, logg) bins. Each SED peak-normalised for shape comparison."""
    # Bins sized to where APOGEE-RGB/subgiant density actually lives.
    teff_edges = np.array([4000, 4500, 5000, 5800])   # 3 cols
    logg_edges = np.array([1.0, 2.0, 3.0, 4.5])       # 3 rows, high→low

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")

    for i in range(3):   # rows: logg high → low (giant → dwarf → subgiant)
        logg_lo, logg_hi = logg_edges[2 - i], logg_edges[2 - i + 1]
        for j in range(3):  # cols: Teff cool → warm
            teff_lo, teff_hi = teff_edges[j], teff_edges[j + 1]
            mask = (
                (data["teff"] >= teff_lo) & (data["teff"] < teff_hi)
                & (data["logg"] >= logg_lo) & (data["logg"] < logg_hi)
            )
            ax = axes[i, j]
            n = int(mask.sum())
            if n == 0:
                ax.text(0.5, 0.5, "empty cell", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
            else:
                sub_idx = np.where(mask)[0]
                if len(sub_idx) > 25:
                    sub_idx = np.random.default_rng(int(teff_lo + logg_lo * 100)).choice(
                        sub_idx, size=25, replace=False
                    )
                mh_vals = data["mh"][sub_idx]
                mh_norm = np.clip((mh_vals + 1.5) / 2.0, 0, 1)
                # per-star peak-normalisation so SED shape (not absolute flux scale)
                # is what the reader sees — otherwise fluxes span 5 orders of magnitude
                # and render as near-horizontal lines.
                for k, ii in enumerate(sub_idx):
                    f = data["flux"][ii]
                    pk = np.nanmax(np.abs(f))
                    if pk > 0 and np.isfinite(pk):
                        ax.plot(XP_WAVELENGTHS_NM, f / pk,
                                color=cmap(mh_norm[k]), alpha=0.55, lw=0.7)
                ax.text(0.97, 0.93, f"n={n}", transform=ax.transAxes, ha="right",
                        va="top", fontsize=8, color="#333")
            ax.set_title(
                rf"$T_{{\rm eff}} \in [{teff_lo},{teff_hi}]\,$K,  "
                rf"$\log g \in [{logg_lo},{logg_hi}]$",
                fontsize=10,
            )
            ax.grid(alpha=0.2)
            ax.set_ylim(-0.05, 1.15)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\lambda$ [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("flux (per-star peak-normalised)")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=-1.5, vmax=0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.01, aspect=30)
    cbar.set_label(r"[M/H]$_{\mathrm{APOGEE}}$")

    fig.suptitle(
        r"Gaia XP corrected SEDs by $(T_{\rm eff},\,\log g)$ cell"
        "\n"
        r"per-star peak-normalised, coloured by [M/H]",
        fontsize=12, fontweight="bold", y=1.00,
    )
    save_fig(fig, OUT / "xp_sed_atlas_by_hrd.png", tight=False)


def coef_distributions() -> None:
    """Distributions of the 108 normalised Hermite coefficients (c_1..c_54 for BP & RP).

    coef_0 is the absolute flux scale (~1e4 e/s) and would dominate any joint axis.
    The gallery uses the normalised-by-c0 coefs `bp_coef_norm_k`, `rp_coef_norm_k`
    (dimensionless shape coefficients) and adds a separate log-histogram for c_0.
    """
    norm_cols_bp = [f"bp_coef_norm_{i}" for i in range(1, 55)]
    norm_cols_rp = [f"rp_coef_norm_{i}" for i in range(1, 55)]
    # `bp_c0_z` / `rp_c0_z` are the log-taken, z-scored flux scale (emit-time
    # transform). `bp_coef_0` / `rp_coef_0` are raw and contain catastrophic
    # outliers (max ~1e36) that make a histogram meaningless.
    feat = pq.read_table(
        DATA_PROCESSED / "pipeline1_features_stream1.parquet",
        columns=norm_cols_bp + norm_cols_rp + ["bp_c0_z", "rp_c0_z"],
    )
    df = feat.to_pandas()
    if len(df) > 30_000:
        df = df.sample(30_000, random_state=11)
    bp = df[norm_cols_bp].to_numpy()
    rp = df[norm_cols_rp].to_numpy()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.55], hspace=0.38, wspace=0.12)
    ax_bp = fig.add_subplot(gs[0, :])
    ax_rp = fig.add_subplot(gs[1, :], sharex=ax_bp)
    ax_c0_bp = fig.add_subplot(gs[2, 0])
    ax_c0_rp = fig.add_subplot(gs[2, 1])

    for ax, coefs, label in ((ax_bp, bp, "BP"), (ax_rp, rp, "RP")):
        p02 = np.nanpercentile(coefs, 2, axis=0)
        p16 = np.nanpercentile(coefs, 16, axis=0)
        p50 = np.nanpercentile(coefs, 50, axis=0)
        p84 = np.nanpercentile(coefs, 84, axis=0)
        p98 = np.nanpercentile(coefs, 98, axis=0)
        x = np.arange(1, 55)
        ax.fill_between(x, p02, p98, color="#c8d6e5", alpha=0.8, label="2–98%")
        ax.fill_between(x, p16, p84, color="#6b8cbe", alpha=0.85, label="16–84%")
        ax.plot(x, p50, color="#1a2c55", lw=1.3, label="median")
        ax.axhline(0, color="#666", lw=0.5, ls="--")
        ax.set_title(rf"{label} normalised coefficients  $c_k / c_0$  (k=1..54,  n={len(df):,})")
        ax.set_ylabel("coef value  (normalised)")
        ax.set_xlim(0.5, 54.5)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax_rp.set_xlabel("coefficient index $k$")

    # z(log c_0) — the log-taken, z-scored absolute flux scale (emit-time transform,
    # docs/data_acquisition.md §6.4 preprocessing step 3). Unit-variance zero-mean.
    for ax, col, band in ((ax_c0_bp, "bp_c0_z", "BP"), (ax_c0_rp, "rp_c0_z", "RP")):
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        v = v[(v > -8) & (v < 8)]
        ax.hist(v, bins=60, color="#6b8cbe", edgecolor="#1a2c55", lw=0.4)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(rf"{band} $c_0$  (log-taken, z-scored flux scale)", fontsize=10)
        ax.set_xlabel(r"$z(\log c_0)$")
        ax.set_ylabel("count")

    fig.suptitle(
        "Gaia XP Hermite coefficients — shape (normalised $c_k/c_0$) + absolute scale $c_0$",
        y=0.995, fontsize=13, fontweight="bold",
    )
    save_fig(fig, OUT / "xp_coef_distributions.png", tight=False)


def _pick_nearest(data: dict[str, np.ndarray], ref: dict, mask: np.ndarray) -> int:
    idx_pool = np.where(mask)[0]
    if len(idx_pool) == 0:
        return -1
    d = (
        ((data["teff"][idx_pool] - ref["teff"]) / 300.0) ** 2
        + ((data["logg"][idx_pool] - ref["logg"]) / 0.5) ** 2
        + ((data["mh"][idx_pool] - ref["mh"]) / 0.15) ** 2
    )
    return int(idx_pool[int(np.argmin(d))])


def example_stars(data: dict[str, np.ndarray]) -> None:
    """3×3 grid: giant-type (cool / normal / warm) × metallicity (low / solar / high).

    Filter pool to low-extinction, moderate-magnitude stars so SED-shape differences
    reflect photosphere + [M/H], not A_V or SNR.
    """
    pool_mask = (
        (data["av_sfd"] < 0.3)
        & (data["g_mag"] > 12.0) & (data["g_mag"] < 14.0)
        & (data["logg"] < 3.5) & (data["logg"] > 1.0)
    )
    if pool_mask.sum() < 9:
        # fall back to full set if the tight filter is too aggressive for the 10k subsample
        pool_mask = np.ones_like(pool_mask, dtype=bool)

    types = [
        ("cool giant",   dict(teff=4200, logg=1.8)),
        ("normal giant", dict(teff=4650, logg=2.4)),
        ("warm giant",   dict(teff=5100, logg=2.8)),
    ]
    mhs = [
        ("metal-poor", -1.0),
        ("solar",      +0.0),
        ("metal-rich", +0.25),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=False)

    for i, (row_label, tref) in enumerate(types):
        for j, (col_label, mh) in enumerate(mhs):
            ref = dict(tref, mh=mh)
            best = _pick_nearest(data, ref, pool_mask)
            ax = axes[i, j]
            if best < 0:
                ax.text(0.5, 0.5, "no match in pool", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
            else:
                f = data["flux"][best]
                pk = np.nanmax(np.abs(f))
                ax.plot(XP_WAVELENGTHS_NM, f / (pk if pk > 0 else 1.0),
                        color="#1a2c55", lw=1.1)
                ax.set_ylim(-0.05, 1.15)
                info = (
                    rf"$T_{{\rm eff}}={data['teff'][best]:.0f}$ K,  "
                    rf"$\log g={data['logg'][best]:.2f}$"
                    + "\n"
                    + rf"[M/H]$={data['mh'][best]:+.2f}$,  "
                    rf"$G={data['g_mag'][best]:.2f}$,  "
                    rf"$A_V^{{\rm SFD}}={data['av_sfd'][best]:.2f}$"
                )
                ax.text(0.03, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.92))
            if i == 0:
                ax.set_title(col_label + rf"  ([M/H]$\approx{mh:+.2f}$)", fontsize=10)
            if j == 0:
                ax.set_ylabel(row_label + "\nflux (peak-norm)", fontsize=10)
            if i == 2:
                ax.set_xlabel(r"$\lambda$ [nm]")
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Example Stream-1 XP SEDs  —  3 giant types × 3 metallicities, "
        r"$A_V^{\rm SFD}<0.3$, $12<G<14$",
        fontsize=13, fontweight="bold", y=1.00,
    )
    save_fig(fig, OUT / "xp_example_stars.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = _load_xp_subset(n_target=10_000)
    print(f"[stage01] loaded {data['flux'].shape[0]:,} SEDs, {data['flux'].shape[1]} samples each")
    sed_atlas_by_hrd(data)
    coef_distributions()
    example_stars(data)


if __name__ == "__main__":
    main()
