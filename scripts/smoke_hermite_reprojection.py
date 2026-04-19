"""Smoke test for the §6.4 step-2 Hermite re-projection of Ye+2024 sampled flux.

Draws a 1000-star stratified sample from
``data/processed/pipeline1_features_stream1.parquet`` with mandatory
oversampling of adversarial sub-populations, re-projects each star's Ye
corrected flux onto the 55+55 orthonormal Hermite basis defined in
:mod:`arqueogal.data.gaia_xp`, and emits the four required diagnostics:

1. Residual RMS distribution — overall plus per-sub-population overlays.
2. ``c_0`` (log-scaled) vs G per band — probes whether the integrated-flux
   proxy tracks brightness monotonically.
3. PCA of the 110-coefficient vector — first two PCs coloured by Teff,
   [Fe/H], log g, A_V, G.
4. Per-coefficient spread vs Hermite mode index — identifies the noise
   floor beyond which coefficients stop carrying information (expected
   to flatten around n=45 for BP and n=30 for RP per data_acquisition.md
   §6.1).

A markdown summary is written alongside the figures. This is **decision
support only**: the feature matrix is NOT re-emitted until the user
greenlights the noise-floor threshold and the RESIDUAL_HIGH p99 cutoff.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u_astro
from astropy.coordinates import SkyCoord
from sklearn.decomposition import PCA

from arqueogal.data.gaia_xp import (
    HERMITE_N_BASIS,
    HERMITE_REPROJECTION_VERSION,
    YE2024_N_OUTPUT,
    YE2024_SAMPLING_NM,
    reproject_ye_to_hermite,
)
from arqueogal.utils.plotting import (
    AA_DOUBLE_COLUMN_IN,
    WONG_PALETTE,
    save_figure,
    set_aa_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_hermite_reprojection")

# The Ye grid is geomspace(360, 990, 330). "Blue" = λ < 560 nm picks out roughly
# the bluest sixth of the grid, where Ye's calibration is weakest (negative
# flux excursions in low-S/N stars).
BLUE_CUTOFF_NM = 560.0
SFD_TO_AV = 2.742  # Schlafly & Finkbeiner 2011, R_V = 3.1
# Ye+2024 nominal flux magnitudes are ~10^-14; values 1e-10 already flag a
# NN failure. We treat rows whose reprojection_residual_rms exceeds this as
# catastrophic Ye failures — plotted separately and excluded from PCA so
# they don't saturate the variance structure of the normal population.
YE_CATASTROPHIC_RESIDUAL = 1e-10


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/pipeline1_features_stream1.parquet"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/hermite_smoke"),
    )
    p.add_argument("--n-sample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260418)
    return p.parse_args()


def _load(path: Path) -> pd.DataFrame:
    cols = [
        "source_id", "corrected_flux", "ye2024_flag",
        "g_mag", "bp_mag", "rp_mag",
        "teff_gspphot", "logg_gspphot",
        "fe_h_atm", "m_h_atm",
        "ebv_sfd", "av_nbhd_median",
        "ra_deg", "dec_deg",
    ]
    df = pd.read_parquet(path, columns=cols)
    # Drop rows Ye refused to emit (flag=1 ⇒ NO_SYNTH_PHOT, flux is NaN-filled).
    df = df[df["ye2024_flag"] == 0].reset_index(drop=True)
    # Need teff for stratification; drop NaN GSP-Phot (BHB contaminants etc).
    df = df[df["teff_gspphot"].notna()].reset_index(drop=True)
    logger.info("Loaded %d Ye-OK stars with GSP-Phot Teff", len(df))
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach |b|, A_V_SFD, and a blue-negative-flux flag."""
    coord = SkyCoord(
        df["ra_deg"].to_numpy() * u_astro.deg,
        df["dec_deg"].to_numpy() * u_astro.deg,
        frame="icrs",
    ).galactic
    df["b_deg"] = coord.b.deg
    df["av_sfd"] = SFD_TO_AV * df["ebv_sfd"]
    blue_mask = YE2024_SAMPLING_NM < BLUE_CUTOFF_NM
    # Bool column: does the Ye-corrected blue flux dip negative anywhere?
    df["blue_neg"] = df["corrected_flux"].apply(
        lambda f: bool((np.asarray(f)[blue_mask] < 0).any())
    )
    return df


def _stratified_indices(
    df: pd.DataFrame, n_sample: int, rng: np.random.Generator,
) -> np.ndarray:
    """Stratify on (Teff, [Fe/H], G) via a coarse 3D histogram.

    Stars with NaN APOGEE [Fe/H] fall into the ``nan`` bin but are still
    eligible for sampling — many cool-giant metallicities are uncensored.
    """
    teff = df["teff_gspphot"].to_numpy()
    feh = df["fe_h_atm"].to_numpy()
    g = df["g_mag"].to_numpy()

    teff_bins = np.array([0, 4200, 4600, 4900, 5200, 5500, np.inf])
    feh_bins = np.array([-np.inf, -1.0, -0.5, -0.25, 0.0, 0.25, np.inf])
    g_bins = np.array([0, 11, 12, 13, 14, 15, 16, np.inf])

    teff_i = np.digitize(teff, teff_bins) - 1
    feh_i = np.digitize(np.nan_to_num(feh, nan=0.0), feh_bins) - 1
    g_i = np.digitize(g, g_bins) - 1
    keys = teff_i * 100 + feh_i * 10 + g_i

    uniq, inv = np.unique(keys, return_inverse=True)
    # Approximate per-cell quota.
    per_cell = max(1, int(np.ceil(n_sample / len(uniq))))
    picks: list[int] = []
    for k in range(len(uniq)):
        cell_idx = np.where(inv == k)[0]
        take = min(per_cell, cell_idx.size)
        picks.extend(rng.choice(cell_idx, size=take, replace=False))
    return np.array(picks, dtype=np.int64)


def _forced_indices(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Mandatory inclusion of adversarial sub-populations."""
    pools: dict[str, tuple[np.ndarray, int]] = {
        "blue_neg": (df.index[df["blue_neg"]].to_numpy(), 100),
        "fe_h_lt_-1.5": (
            df.index[(df["fe_h_atm"] < -1.5) & df["fe_h_atm"].notna()].to_numpy(), 50,
        ),
        "teff_gt_6000": (df.index[df["teff_gspphot"] > 6000].to_numpy(), 50),
        "av_sfd_gt_3": (df.index[df["av_sfd"] > 3.0].to_numpy(), 50),
        "abs_b_lt_15": (df.index[df["b_deg"].abs() < 15.0].to_numpy(), 100),
    }
    forced: dict[str, np.ndarray] = {}
    for name, (pool, quota) in pools.items():
        take = min(quota, pool.size)
        if take == 0:
            logger.warning("OOD sub-pop %s is empty; skipping", name)
            forced[name] = np.empty(0, dtype=np.int64)
            continue
        forced[name] = rng.choice(pool, size=take, replace=False)
        logger.info("OOD forced %-16s %4d / pool %d", name, take, pool.size)
    return forced


def _build_sample(
    df: pd.DataFrame, n_sample: int, seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    strat = _stratified_indices(df, n_sample, rng)
    forced = _forced_indices(df, rng)
    all_idx = np.unique(np.concatenate([strat, *forced.values()]))
    sample = df.loc[all_idx].copy().reset_index(drop=True)
    # Remap forced indices into the sample frame for later labelling.
    pos = {src: i for i, src in enumerate(all_idx)}
    sub_masks = {
        name: np.array([pos[s] for s in idx], dtype=np.int64)
        for name, idx in forced.items()
    }
    logger.info(
        "Sample size: %d (stratified %d + forced %d, union %d)",
        len(sample), strat.size, sum(v.size for v in forced.values()), all_idx.size,
    )
    return sample, sub_masks


def _reproject_sample(sample: pd.DataFrame) -> dict:
    """Stack flux, run re-projection, return the raw dict + fused coefficient matrix."""
    flux = np.stack([np.asarray(f, dtype=np.float32) for f in sample["corrected_flux"]])
    assert flux.shape == (len(sample), YE2024_N_OUTPUT), flux.shape
    out = reproject_ye_to_hermite(flux)
    out["coeffs"] = np.concatenate([out["bp_coeffs"], out["rp_coeffs"]], axis=1)
    return out


def _plot_residual_distribution(
    residuals: np.ndarray,
    sub_masks: dict[str, np.ndarray],
    out_path: Path,
) -> dict:
    """Histogram of residual RMS: overall + per-sub-pop."""
    set_aa_style()
    fig, axes = plt.subplots(
        2, 3, figsize=(AA_DOUBLE_COLUMN_IN, 5.0), sharex=True, sharey=False,
    )
    log_r = np.log10(np.maximum(residuals, 1e-30))
    # Compute p99 on the "normal" population (excluding catastrophic Ye failures)
    # — that's the threshold we actually want for RESIDUAL_HIGH.
    normal = residuals < YE_CATASTROPHIC_RESIDUAL
    p_overall_normal = np.percentile(residuals[normal], [50, 90, 95, 99])
    p_overall_all = np.percentile(residuals, [50, 90, 95, 99])
    bins = np.linspace(log_r.min(), log_r.max(), 60)

    # Overall.
    ax = axes[0, 0]
    ax.hist(log_r, bins=bins, color=WONG_PALETTE[0], alpha=0.8)
    ax.axvline(
        np.log10(p_overall_normal[3]), color="k", lw=0.8, ls="--",
        label=f"normal p99 = {p_overall_normal[3]:.1e}",
    )
    ax.axvline(
        np.log10(YE_CATASTROPHIC_RESIDUAL), color="r", lw=0.8, ls=":",
        label=f"catastrophic = {YE_CATASTROPHIC_RESIDUAL:.0e}",
    )
    n_cat = int((~normal).sum())
    ax.set_title(f"overall (N={residuals.size}, catastrophic={n_cat})")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)

    names = list(sub_masks.keys())
    per_subpop: dict[str, np.ndarray] = {}
    per_subpop_n_cat: dict[str, int] = {}
    for i, name in enumerate(names, start=1):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        idx = sub_masks[name]
        if idx.size == 0:
            ax.set_title(f"{name} (empty)")
            continue
        sub_r = residuals[idx]
        per_subpop[name] = sub_r
        per_subpop_n_cat[name] = int((sub_r >= YE_CATASTROPHIC_RESIDUAL).sum())
        ax.hist(log_r, bins=bins, color="0.8", alpha=0.6, label="all")
        ax.hist(
            np.log10(np.maximum(sub_r, 1e-30)),
            bins=bins, color=WONG_PALETTE[1 + i % (len(WONG_PALETTE) - 1)],
            alpha=0.9, label=name,
        )
        ax.set_title(f"{name} (N={sub_r.size}, cat={per_subpop_n_cat[name]})")
        ax.legend(fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}$ residual RMS")
    fig.suptitle("Hermite re-projection residual RMS (flux units)")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)

    def _normal_percentiles(v: np.ndarray) -> tuple[float, float]:
        n = v[v < YE_CATASTROPHIC_RESIDUAL]
        if n.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(n, 50)), float(np.percentile(n, 99))

    return {
        "p50_all": float(p_overall_all[0]),
        "p99_all": float(p_overall_all[3]),
        "p50_normal": float(p_overall_normal[0]),
        "p90_normal": float(p_overall_normal[1]),
        "p95_normal": float(p_overall_normal[2]),
        "p99_normal": float(p_overall_normal[3]),
        "n_catastrophic": n_cat,
        "per_subpop_normal_p50": {
            k: _normal_percentiles(v)[0] for k, v in per_subpop.items()
        },
        "per_subpop_normal_p99": {
            k: _normal_percentiles(v)[1] for k, v in per_subpop.items()
        },
        "per_subpop_n_catastrophic": per_subpop_n_cat,
    }


def _plot_c0_vs_g(
    sample: pd.DataFrame,
    bp_coeffs: np.ndarray,
    rp_coeffs: np.ndarray,
    out_path: Path,
) -> None:
    set_aa_style()
    fig, axes = plt.subplots(1, 2, figsize=(AA_DOUBLE_COLUMN_IN, 3.2))
    g = sample["g_mag"].to_numpy()
    for ax, c0, title in [
        (axes[0], bp_coeffs[:, 0], "BP $c_0$"),
        (axes[1], rp_coeffs[:, 0], "RP $c_0$"),
    ]:
        # Keep sign explicit; c_0 can go negative if the Ye flux dips.
        sign = np.sign(c0)
        mag = np.log10(np.abs(c0).clip(min=1e-40))
        sc = ax.scatter(
            g, sign * mag, c=sign, s=8, cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0,
        )
        ax.set_xlabel("G (mag)")
        ax.set_ylabel(r"$\mathrm{sign}(c_0)\,\log_{10}|c_0|$")
        ax.set_title(title)
        ax.invert_xaxis()
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("sign($c_0$)")
    fig.suptitle("Integrated-flux proxy $c_0$ vs Gaia G — expect monotone decrease")
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_pca(
    sample: pd.DataFrame, coeffs: np.ndarray, residuals: np.ndarray, out_path: Path,
) -> dict:
    set_aa_style()
    # Exclude catastrophic-Ye rows so the PCA describes the normal population
    # rather than being saturated by a handful of 10^-4-scale outliers.
    keep = residuals < YE_CATASTROPHIC_RESIDUAL
    X_full = coeffs[keep]
    sample_k = sample.loc[keep].reset_index(drop=True)
    logger.info(
        "PCA on %d / %d rows (dropped %d catastrophic-Ye rows)",
        keep.sum(), keep.size, (~keep).sum(),
    )
    # Robust standardisation: median + MAD*1.4826 (≈σ under Gaussian) so a
    # handful of moderate outliers don't blow up the column scales. Then
    # winsorise at ±8 "robust σ" — keeps the PCA geometry diagnostic of the
    # bulk population rather than saturated by a few anomalous rows.
    med = np.median(X_full, axis=0)
    mad = np.median(np.abs(X_full - med), axis=0)
    sd = 1.4826 * mad
    sd[sd < 1e-30] = 1.0
    X = (X_full - med) / sd
    X = np.clip(X, -8.0, 8.0)
    pca = PCA(n_components=5)
    pcs = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    color_cols = [
        ("teff_gspphot", "Teff (K)", "inferno"),
        ("fe_h_atm", "[Fe/H]", "viridis"),
        ("logg_gspphot", "log g", "cividis"),
        ("av_nbhd_median", r"$A_V$ (nbhd)", "plasma"),
        ("g_mag", "G (mag)", "magma"),
    ]
    fig, axes = plt.subplots(
        1, len(color_cols), figsize=(AA_DOUBLE_COLUMN_IN, 3.0), sharex=True, sharey=True,
    )
    for ax, (col, label, cmap) in zip(axes, color_cols):
        c = sample_k[col].to_numpy()
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, s=6, cmap=cmap, linewidths=0)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        if ax is axes[0]:
            ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_title(label, fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle("PCA of 110-dim Hermite coefficient vector (standardised)")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return {f"pc{i + 1}_var": float(v) for i, v in enumerate(explained)}


def _plot_noise_floor(
    bp_coeffs: np.ndarray, rp_coeffs: np.ndarray, out_path: Path,
) -> dict:
    """Per-mode robust spread (MAD→σ) and per-mode median — log-scaled."""
    set_aa_style()
    fig, axes = plt.subplots(1, 2, figsize=(AA_DOUBLE_COLUMN_IN, 3.2), sharey=True)
    stats: dict[str, np.ndarray] = {}
    for ax, coeffs, band, cutoff in [
        (axes[0], bp_coeffs, "BP", 45),
        (axes[1], rp_coeffs, "RP", 30),
    ]:
        med = np.median(coeffs, axis=0)
        mad = np.median(np.abs(coeffs - med), axis=0)
        sig_mad = 1.4826 * mad
        stats[f"{band}_sigma_mad"] = sig_mad
        n = np.arange(HERMITE_N_BASIS)
        ax.semilogy(n, np.abs(med), color=WONG_PALETTE[0], marker=".", label="|median|")
        ax.semilogy(n, sig_mad, color=WONG_PALETTE[1], marker="x", label=r"$\sigma_{\mathrm{MAD}}$")
        ax.axvline(cutoff, color="k", ls=":", lw=0.8, label=f"expected floor n={cutoff}")
        ax.set_xlabel("Hermite mode $n$")
        ax.set_title(f"{band} per-mode spread")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("|value| (flux units)")
    fig.suptitle("Noise-floor check: does spread flatten at the expected n?")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)

    def _floor_bin(s: np.ndarray, a: int, b: int) -> float:
        return float(np.median(s[a:b]))

    return {
        "bp_sigma_median_n0_10": _floor_bin(stats["BP_sigma_mad"], 0, 10),
        "bp_sigma_median_n40_55": _floor_bin(stats["BP_sigma_mad"], 40, 55),
        "rp_sigma_median_n0_10": _floor_bin(stats["RP_sigma_mad"], 0, 10),
        "rp_sigma_median_n25_55": _floor_bin(stats["RP_sigma_mad"], 25, 55),
    }


def _write_summary(
    out_dir: Path,
    sample: pd.DataFrame,
    sub_masks: dict[str, np.ndarray],
    residual_stats: dict,
    pca_stats: dict,
    floor_stats: dict,
    basis_version: str,
    basis_fingerprint: str,
) -> None:
    md = out_dir / "SUMMARY.md"
    lines = [
        "# Hermite re-projection smoke test — SUMMARY",
        "",
        "**Purpose.** Validate §6.4 step 2 before re-emitting "
        "`pipeline1_features_stream1.parquet` with 55+55 Hermite coefficients.",
        "",
        f"- Basis version: `{basis_version}`",
        f"- Basis fingerprint (SHA-256): `{basis_fingerprint}`",
        f"- Sample size: **{len(sample)} stars** drawn from "
        "`pipeline1_features_stream1.parquet` (Ye flag=0 only).",
        f"- Catastrophic-Ye rows (residual RMS ≥ {YE_CATASTROPHIC_RESIDUAL:.0e}): "
        f"**{residual_stats['n_catastrophic']}** — these are Ye+2024 NN "
        "failure modes, not a reprojection bug. They must be flagged at "
        "materialisation time (`xp_fit_flag = RESIDUAL_HIGH`), not dropped.",
        "",
        "## Forced sub-populations",
        "",
        "| sub-pop | N | catastrophic | normal p50 | normal p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, idx in sub_masks.items():
        n = idx.size
        if n == 0:
            lines.append(f"| `{name}` | 0 | 0 | — | — |")
            continue
        ncat = residual_stats["per_subpop_n_catastrophic"][name]
        p50 = residual_stats["per_subpop_normal_p50"][name]
        p99 = residual_stats["per_subpop_normal_p99"][name]
        lines.append(f"| `{name}` | {n} | {ncat} | {p50:.3e} | {p99:.3e} |")

    lines += [
        "",
        "## Residual RMS — NORMAL population (RMS < catastrophic threshold)",
        "",
        f"- p50: **{residual_stats['p50_normal']:.3e}**",
        f"- p90: {residual_stats['p90_normal']:.3e}",
        f"- p95: {residual_stats['p95_normal']:.3e}",
        f"- p99 (⇒ `XP_FIT_FLAG_RESIDUAL_HIGH` threshold candidate): "
        f"**{residual_stats['p99_normal']:.3e}**",
        "",
        "## Residual RMS — including catastrophic Ye failures",
        "",
        f"- p50: {residual_stats['p50_all']:.3e}",
        f"- p99: {residual_stats['p99_all']:.3e}  "
        "(pulled far above normal p99 by the catastrophic tail)",
        "",
        "## PCA explained variance (110-dim standardised coeffs, catastrophic rows removed)",
        "",
    ] + [f"- PC{i + 1}: {pca_stats[f'pc{i + 1}_var']:.3%}" for i in range(5)] + [
        "",
        "## Noise-floor check",
        "",
        f"- BP σ_MAD median, modes 0–9:   {floor_stats['bp_sigma_median_n0_10']:.3e}",
        f"- BP σ_MAD median, modes 40–54: {floor_stats['bp_sigma_median_n40_55']:.3e}",
        f"- RP σ_MAD median, modes 0–9:   {floor_stats['rp_sigma_median_n0_10']:.3e}",
        f"- RP σ_MAD median, modes 25–54: {floor_stats['rp_sigma_median_n25_55']:.3e}",
        "",
        "## Figures",
        "",
        "- `residual_rms.png` — residual RMS histograms overall + per sub-pop, "
        "with the catastrophic cutoff and normal-p99 marked.",
        "- `c0_vs_g.png` — signed log|c₀| vs G per band.",
        "- `pca_110d.png` — PC1/PC2 of the 110-coefficient vector coloured by "
        "Teff, [Fe/H], log g, A_V, G (catastrophic rows excluded).",
        "- `noise_floor.png` — per-mode |median| and σ_MAD for BP and RP.",
        "",
    ]
    md.write_text("\n".join(lines))
    logger.info("Wrote %s", md)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _enrich(_load(args.features))
    sample, sub_masks = _build_sample(df, args.n_sample, args.seed)

    repro = _reproject_sample(sample)
    bp, rp = repro["bp_coeffs"], repro["rp_coeffs"]
    residuals = repro["reprojection_residual_rms"]
    coeffs = repro["coeffs"]

    residual_stats = _plot_residual_distribution(
        residuals, sub_masks, args.out_dir / "residual_rms.png",
    )
    _plot_c0_vs_g(sample, bp, rp, args.out_dir / "c0_vs_g.png")
    pca_stats = _plot_pca(sample, coeffs, residuals, args.out_dir / "pca_110d.png")
    floor_stats = _plot_noise_floor(bp, rp, args.out_dir / "noise_floor.png")

    _write_summary(
        args.out_dir, sample, sub_masks,
        residual_stats, pca_stats, floor_stats,
        basis_version=repro["basis_version"],
        basis_fingerprint=repro["basis_fingerprint_sha256"],
    )

    logger.info("Done. Outputs under %s", args.out_dir)


if __name__ == "__main__":
    main()
