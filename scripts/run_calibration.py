"""Run A — calibration harness (#135).

Loads the 5-member ensemble, collects per-member predictions on the val split,
moment-matches them into a single ensemble Gaussian (μ̄, Σ̄), and runs the
calibration harness from :mod:`xp_abundances.main.uncertainty` — reliability
diagrams per (Teff, log g, [M/H]) cell, temperature scaling if any cell
exceeds 15% reliability error, empirical coverage at 68/95/99%, and split-
conformal radii at the same levels.

Pass criteria (DESIGN + phase directive):
- Global reliability error ≤ 10%.
- Per-cell reliability error ≤ 15%.
- Coverage within 5 percentage points of nominal at 68/95/99%.

Halt: reliability error > 30% in any cell **after** temperature scaling.

Run: ``PYTHONPATH=src python scripts/run_calibration.py --ensemble <dir>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import (
    build_dataloaders,
    load_checkpoint,
)
from arqueogal.xp_abundances.main.uncertainty import (
    RegimeBEnvelope,
    bin_by_cells,
    conformal_nonconformity_scores,
    coverage_at_levels,
    gp_smoothed_per_cell_per_label_scale,
    shrunken_per_cell_per_label_scale,
    temperature_scaling_per_cell,
)
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_calibration")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/run_a"
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"


def _build_cfg_for_val_loader(
    parquet: Path, pretrained_ckpt: Path, batch_size: int, seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        train_parquet=parquet,
        output_dir=REPO_ROOT / "tmp_calibration",
        epochs=1, batch_size=batch_size, num_workers=2,
        amp_dtype="bfloat16",
        use_c0_scalars=True,
        encoder_lr_ratio=0.1,
        pretrained_encoder_ckpt=pretrained_ckpt,
        reload_head_from_pretrained=False,
        split_seed=seed,
        output_prefix="xp_abundances_main_calibration",
        loss_weights=LossWeights(supcon=0.0, beta_nll=1.0, beta=0.5),
        temperature_init=0.10,
        ensemble_seeds=(0,),
    )


def _reconstruct_model(
    blob: dict, layout: FeatureLayout, block_layout: CovarianceBlockLayout,
    device: torch.device,
) -> tuple[XpAbundanceModel, XpFeatureAdapter]:
    cfg_yaml = json.loads(blob["config_yaml"])
    use_c0 = bool(cfg_yaml.get("use_c0_scalars", True))
    latent_dim = int(cfg_yaml.get("latent_dim", 32))
    trunk_hidden = tuple(cfg_yaml.get("trunk_hidden", (256, 128)))
    head_hidden = int(cfg_yaml.get("head_hidden", 128))
    dropout = float(cfg_yaml.get("dropout", 0.10))

    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=latent_dim, trunk_hidden=trunk_hidden,
            head_hidden=head_hidden, dropout=dropout,
        ),
    ).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["regressor"])
    adapter = XpFeatureAdapter(layout, use_c0_scalars=use_c0).to(device)
    return model, adapter


def _collect_member_preds(
    model: XpAbundanceModel, adapter: XpFeatureAdapter,
    loader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `model(adapter(x))` over `loader`; return (mu, L, y) as CPU float32 arrays."""
    model.eval()
    mus: list[np.ndarray] = []
    Ls: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]
            mu, L, _h, _z = model(adapter(x))
            mus.append(mu.float().cpu().numpy())
            Ls.append(L.float().cpu().numpy())
            ys.append(y.numpy())
    return (
        np.concatenate(mus, axis=0).astype(np.float32),
        np.concatenate(Ls, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.float32),
    )


def _moment_match(mus: np.ndarray, Ls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine ensemble-member (mu_k, Sigma_k=L_k L_k^T) into (mu_bar, L_bar).

    ``Sigma_bar = mean(Sigma_k) + between-member-var(mu_k)`` — the standard
    mixture-of-Gaussians moment match. The diagonal of ``Sigma_bar`` gives the
    released per-label σ; :func:`torch.linalg.cholesky` with small jitter
    recovers the full Cholesky the downstream harness needs.
    """
    k, n_stars, n_dim = mus.shape
    mu_bar = mus.mean(axis=0)
    aleatoric = np.einsum("kbij,kblj->bil", Ls, Ls) / k  # avg(L L^T)
    diff = mus - mu_bar[None]
    epistemic = np.einsum("kbi,kbj->bij", diff, diff) / k
    sigma = aleatoric + epistemic
    # Jitter until PD — cholesky fallback for any ill-conditioned star.
    jitter = 1e-6
    for _ in range(6):
        try:
            t = torch.from_numpy(sigma + jitter * np.eye(n_dim)[None])
            L_bar = torch.linalg.cholesky(t).numpy().astype(np.float32)
            break
        except RuntimeError:
            jitter *= 10.0
    else:
        raise RuntimeError(f"ensemble Σ̄ not PD even at jitter={jitter}")
    return mu_bar.astype(np.float32), L_bar


def _mahal_bulk(mu: np.ndarray, L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-star Mahalanobis² using scipy's triangular solve (bulk loop)."""
    from scipy.linalg import solve_triangular
    diff = (y - mu)
    out = np.empty(mu.shape[0], dtype=np.float64)
    for b in range(mu.shape[0]):
        z = solve_triangular(L[b], diff[b], lower=True)
        out[b] = float((z * z).sum())
    return out


def _variance_decomposition(
    mu: np.ndarray, L: np.ndarray, y: np.ndarray, cell_ids: np.ndarray,
) -> dict:
    """Decompose E[z²] per label into within-cell and between-cell components.

    For each label ``j`` define ``z_{b,j} = (y-μ)/σ_diag``. The law of total
    variance gives ``E[z²] = E[Var(z|cell)] + Var(E[z|cell])``. The first
    term is within-cell σ-miscalibration (what the shrunken α correction
    targets); the second is per-cell μ bias — systematic drift of the mean
    prediction across parameter-space cells.

    β-NLL at β=0.5 can trade the second term for the first by inflating σ
    to absorb a biased μ. β=0 (pure Gaussian NLL) disallows that trade.
    A large drop in ``Var(E[z|cell])`` between a β=0.5 run and a β=0 retrain
    is the signature that β was the cause of the calibration failure.

    Returns per-label arrays: ``total`` (=E[z²]), ``within`` (=E[Var(z|cell)]),
    ``between`` (=Var(E[z|cell])).
    """
    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L, L)).clip(1e-8, None)
    z = (y - mu) / sigma_diag
    n_dim = z.shape[1]
    total = np.zeros(n_dim)
    within = np.zeros(n_dim)
    between = np.zeros(n_dim)
    cells = np.unique(cell_ids)
    for j in range(n_dim):
        col = z[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 8:
            total[j] = within[j] = between[j] = float("nan")
            continue
        total[j] = float(col[finite].var())
        # Within = weighted mean of cell-wise variances; between = variance of cell means.
        cell_means: list[float] = []
        cell_vars: list[float] = []
        cell_weights: list[float] = []
        for c in cells:
            mask = (cell_ids == c) & finite
            n_c = int(mask.sum())
            if n_c < 8:
                continue
            col_c = col[mask]
            cell_means.append(float(col_c.mean()))
            cell_vars.append(float(col_c.var()))
            cell_weights.append(float(n_c))
        if not cell_means:
            within[j] = between[j] = float("nan")
            continue
        w = np.asarray(cell_weights); w = w / w.sum()
        means = np.asarray(cell_means)
        vars_ = np.asarray(cell_vars)
        within[j] = float((w * vars_).sum())
        between[j] = float((w * (means - (w * means).sum()) ** 2).sum())
    return {
        "total_Ez2": total,
        "within_cell_var": within,
        "between_cell_var": between,
    }


def _reliability_per_cell(
    mu: np.ndarray, L: np.ndarray, y: np.ndarray, cell_ids: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-cell reliability error via standardised-residual variance.

    For each star and each label, compute ``z_{b,j} = (y - μ) / σ_diag``. A
    well-calibrated marginal means ``Var(z_{·,j}) ≈ 1`` within any cell. We
    report ``err_cell = mean_j |Var(z) − 1|``. This is the correct diagnostic:
    it is invariant to whether ``σ_pred`` or ``σ_obs`` is "within-cell" vs
    "over-all", and it's directly tied to coverage (Var(z)≈1 ⇒ 1σ coverage≈0.68).

    Returns ``(per_cell_err, global_err, per_label_Var_z)`` where
    ``per_label_Var_z`` is the unconditional Var(z) per label dim, for the
    per-label reliability report line.
    """
    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L, L)).clip(1e-8, None)
    z = (y - mu) / sigma_diag
    cells = np.unique(cell_ids)
    per_cell = np.zeros(cells.size, dtype=np.float64)
    for i, c in enumerate(cells):
        mask = cell_ids == c
        if mask.sum() < 8:
            per_cell[i] = np.nan
            continue
        z_cell = z[mask]
        var_z = np.zeros(z.shape[1])
        for j in range(z.shape[1]):
            col = z_cell[:, j]
            col = col[np.isfinite(col)]
            var_z[j] = col.var() if col.size > 8 else np.nan
        per_cell[i] = np.nanmean(np.abs(var_z - 1.0))
    global_err = float(np.nanmean(per_cell))
    # Unconditional per-label Var(z)
    per_label_var = np.zeros(z.shape[1])
    for j in range(z.shape[1]):
        col = z[:, j]
        col = col[np.isfinite(col)]
        per_label_var[j] = col.var() if col.size > 8 else np.nan
    return per_cell, global_err, per_label_var


def _adjacent_cell_smoothness(
    scales: dict[tuple[int, int], float],
    n_bins: list[int],
) -> dict:
    """α-ratio smoothness across cells differing by ±1 index in one axis.

    Hard cell boundaries are an acceptable v1 simplification iff adjacent
    cells fit similar α. If not, two stars straddling a cell edge with
    nearly-identical spectra will see visibly different σ. We report
    ``max_{(c,c',l)} α_{c,l} / α_{c',l}`` plus the distribution; reviewer-
    threshold expectations per DESIGN §9.1: < 1.5 accept, > 2.0 flag for v2
    smoothing.
    """
    n_axes = len(n_bins)

    def decode(c: int) -> tuple[int, ...]:
        idx: list[int] = []
        for nb in reversed(n_bins):
            idx.append(c % nb)
            c //= nb
        return tuple(reversed(idx))

    def encode(idx: tuple[int, ...]) -> int:
        c = 0
        for i, nb in zip(idx, n_bins):
            c = c * nb + i
        return c

    cells = sorted({c for (c, _) in scales.keys()})
    cell_set = set(cells)
    labels = sorted({j for (_, j) in scales.keys()})

    log_ratios: list[float] = []
    max_log = 0.0
    max_info: dict | None = None
    for c in cells:
        idx = decode(c)
        for axis in range(n_axes):
            if idx[axis] + 1 >= n_bins[axis]:
                continue
            idx2 = list(idx)
            idx2[axis] += 1
            c2 = encode(tuple(idx2))
            if c2 not in cell_set:
                continue
            for j in labels:
                a1 = scales.get((c, j))
                a2 = scales.get((c2, j))
                if a1 is None or a2 is None or a1 <= 0 or a2 <= 0:
                    continue
                lr = abs(math.log(a1 / a2))
                log_ratios.append(lr)
                if lr > max_log:
                    max_log = lr
                    max_info = {
                        "cell_a": int(c), "cell_b": int(c2),
                        "axis": int(axis), "label_idx": int(j),
                        "alpha_a": float(a1), "alpha_b": float(a2),
                    }

    arr = np.asarray(log_ratios, dtype=np.float64)
    if arr.size == 0:
        return {"n_pairs": 0, "max_ratio": 1.0, "median_ratio": 1.0,
                "p90_ratio": 1.0, "max_pair_info": None}
    ratios = np.exp(arr)
    return {
        "n_pairs": int(arr.size),
        "max_ratio": float(ratios.max()),
        "median_ratio": float(np.median(ratios)),
        "p90_ratio": float(np.quantile(ratios, 0.90)),
        "max_pair_info": max_info,
    }


def _plot_reliability(
    per_cell: np.ndarray, out_path: Path,
    title: str, threshold: float = 0.15,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(per_cell.size)
    colors = ["tab:green" if (not np.isnan(v) and v <= threshold)
              else ("tab:orange" if (not np.isnan(v) and v <= 2 * threshold)
                    else "tab:red")
              for v in per_cell]
    ax.bar(x, np.where(np.isnan(per_cell), 0, per_cell),
           color=colors, edgecolor="k", linewidth=0.3)
    ax.axhline(threshold, linestyle="--", color="k", linewidth=0.8,
               label=f"pass threshold = {threshold}")
    ax.set_xlabel("(Teff × log g × [M/H]) cell id")
    ax.set_ylabel("reliability error (mean across labels)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ensemble", type=Path,
                     help="Ensemble directory holding member_seed{N}/ sub-dirs.")
    src.add_argument("--checkpoint", type=Path,
                     help="Single-member checkpoint .pt file. Used for the #135 "
                          "β=0 canary — one member trained end-to-end.")
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--tag", type=str, default=None,
                   help="Optional suffix for report filenames (e.g. 'beta0'). "
                        "When set, outputs go to calibration_report_<tag>.json etc.")
    p.add_argument("--apply-gp-smoothing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Fit 3-D GP α over (Teff, log g, [M/H]) and replace the "
                        "discrete shrunken per-cell α with continuous predictions. "
                        "Default on — disable with --no-apply-gp-smoothing.")
    p.add_argument("--apply-regime-b", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute regime-B exclusion envelope (|b|<5°, Teff>4750 K, "
                        "log g<2.1) and report calibration with those stars excluded.")
    args = p.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ensemble is not None:
        _LOG.info("device=%s ensemble=%s", device, args.ensemble)
        member_ckpts: list[Path] = sorted(
            [p for p in args.ensemble.glob(
                "member_seed*/xp_abundances_main_ensemble*_seed*_best.pt",
            )],
        )
        if not member_ckpts:
            raise FileNotFoundError(f"no member checkpoints under {args.ensemble}")
        source_label = str(args.ensemble)
    else:
        _LOG.info("device=%s single-checkpoint=%s", device, args.checkpoint)
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
        member_ckpts = [args.checkpoint]
        source_label = str(args.checkpoint)
    _LOG.info("found %d member(s)", len(member_ckpts))

    layout = FeatureLayout()

    # Reload the first member's cfg so we rebuild a val loader with the same split.
    first_blob = load_checkpoint(member_ckpts[0], map_location="cpu")
    first_cfg_yaml = json.loads(first_blob["config_yaml"])
    pretrained_ckpt = Path(first_cfg_yaml["pretrained_encoder_ckpt"])
    split_seed = int(first_cfg_yaml.get("split_seed", 0))

    # Reconstruct tiers + block_layout from the checkpoint so the 5-label and
    # 21-label variants both load without hardcoded assumptions.
    ckpt_label_names = tuple(first_blob["label_names"])
    tier_map = first_blob.get("tier_map", {})
    tier1 = tuple(name for name in ckpt_label_names if tier_map.get(name) == 1)
    tier2 = tuple(name for name in ckpt_label_names if tier_map.get(name) == 2)
    tier3 = tuple(name for name in ckpt_label_names if tier_map.get(name) == 3)
    tiers = LabelTiers(tier1=tier1, tier2=tier2, tier3=tier3)
    if tiers.all_labels != ckpt_label_names:
        raise RuntimeError(
            "checkpoint label_names do not match reconstructed tier ordering — "
            f"{ckpt_label_names} vs {tiers.all_labels}",
        )
    block_layout = CovarianceBlockLayout.from_dict(first_blob["block_layout"])
    _LOG.info(
        "tiers: t1=%d t2=%d t3=%d  block_sizes=%s",
        len(tier1), len(tier2), len(tier3), block_layout.block_sizes,
    )

    cfg = _build_cfg_for_val_loader(
        parquet=args.parquet, pretrained_ckpt=pretrained_ckpt,
        batch_size=args.batch_size, seed=split_seed,
    )
    _, val_loader, _, _scaler_from_loader = build_dataloaders(
        cfg, layout, tiers, seed=split_seed,
    )
    _LOG.info("val loader built, batches=%d", len(val_loader))

    # Pull b_deg for val stars — needed for the regime B exclusion envelope.
    # val_loader has shuffle=False, so val_ds.source_id order matches prediction
    # order in _collect_member_preds.
    val_source_ids = np.asarray(val_loader.dataset.source_id)
    import pandas as pd  # local import avoids top-level coupling.
    b_deg_lookup = pd.read_parquet(
        args.parquet, columns=["source_id", "b_deg"],
    ).drop_duplicates(subset="source_id", keep="first")
    b_deg_by_sid = dict(
        zip(b_deg_lookup["source_id"].to_numpy(), b_deg_lookup["b_deg"].to_numpy()),
    )
    b_deg_val = np.asarray(
        [b_deg_by_sid.get(int(sid), np.nan) for sid in val_source_ids],
        dtype=np.float64,
    )
    _LOG.info(
        "joined b_deg for %d/%d val stars (%d NaN)",
        int(np.isfinite(b_deg_val).sum()), b_deg_val.size,
        int((~np.isfinite(b_deg_val)).sum()),
    )

    # Collect per-member predictions (μ_k, L_k) on the val split.
    per_mu: list[np.ndarray] = []
    per_L: list[np.ndarray] = []
    y_cpu: np.ndarray | None = None
    scaler_block: LabelScaler | None = None
    for ckpt in member_ckpts:
        _LOG.info("collecting predictions from %s", ckpt.name)
        blob = load_checkpoint(ckpt, map_location=device)
        ckpt_layout = CovarianceBlockLayout.from_dict(blob["block_layout"])
        if ckpt_layout.label_order_block != block_layout.label_order_block:
            raise RuntimeError(
                f"member {ckpt.name} has a different block layout than the first"
                " — ensemble is inconsistent",
            )
        model, adapter = _reconstruct_model(blob, layout, ckpt_layout, device)
        mu, L, y_human = _collect_member_preds(model, adapter, val_loader, device)
        per_mu.append(mu)
        per_L.append(L)
        if y_cpu is None:
            # Reorder y from human to block order to match μ / L.
            perm = block_layout.human_to_block_perm.cpu().numpy()
            y_cpu = y_human[:, perm]
            # Load the fit scaler from the checkpoint and permute to block
            # order so it aligns with μ / L / y_cpu. The scaler is written
            # in ``tiers.all_labels`` order (= block_layout.label_order_human).
            ckpt_scaler = LabelScaler(
                mean=np.asarray(blob["label_scaler_mean"], dtype=np.float32),
                scale=np.asarray(blob["label_scaler_scale"], dtype=np.float32),
                label_names=tuple(blob["label_names"]),
            )
            if ckpt_scaler.is_default():
                raise RuntimeError(
                    f"checkpoint {ckpt} has the zeros/ones placeholder scaler; "
                    "retrain with the fitted scaler before running calibration.",
                )
            scaler_block = ckpt_scaler.reorder_to(block_layout.label_order_block)

    mus = np.stack(per_mu, axis=0)  # (K, B, n)
    Ls = np.stack(per_L, axis=0)
    mu_bar_scaled, L_bar_scaled = _moment_match(mus, Ls)
    _LOG.info(
        "moment-matched ensemble (scaled): μ̄ shape=%s  L̄ shape=%s",
        mu_bar_scaled.shape, L_bar_scaled.shape,
    )

    # Un-scale μ̄, L̄, and y back to raw physical units so every downstream
    # diagnostic (per-cell binning, reliability, coverage, conformal radii)
    # reports in Kelvin / dex rather than standardised units. Moment-matching
    # commutes with linear label standardisation, so un-scaling post-mixture
    # is mathematically equivalent to un-scaling each member first.
    assert scaler_block is not None
    mu_bar = scaler_block.inverse_mean(mu_bar_scaled).astype(np.float32)
    L_bar = scaler_block.inverse_L(L_bar_scaled).astype(np.float32)
    y_cpu = scaler_block.inverse_mean(y_cpu).astype(np.float32)
    _LOG.info(
        "un-scaled to raw units — Teff scale=%.1f [M/H] scale=%.2f",
        float(scaler_block.scale[0]), float(scaler_block.scale[2]),
    )

    # Cell binning on APOGEE truth (Teff, logg, [M/H]) — block indices 0,1,2.
    cell_features = y_cpu[:, :3].copy()
    cell_ids, cell_def = bin_by_cells(cell_features, n_bins=(4, 4, 4))
    _LOG.info("binned into %d cells (4x4x4 nominal)", len(np.unique(cell_ids)))

    # Pre-calibration reliability + coverage. Coverage and Mahalanobis can't
    # tolerate NaN labels, so we impute missing y with μ_bar (zero contribution
    # to the Mahalanobis solve). Per-cell reliability uses the original y_cpu
    # so its observed-σ is computed only on finite residuals.
    y_clean = np.where(np.isfinite(y_cpu), y_cpu, mu_bar)
    per_cell_raw, global_raw, var_z_raw = _reliability_per_cell(
        mu_bar, L_bar, y_cpu, cell_ids,
    )
    cov_raw = coverage_at_levels(mu_bar, L_bar, y_clean, levels=(0.68, 0.95, 0.99))
    _LOG.info("pre-calibration global reliability err=%.4f", global_raw)

    # Variance decomposition: E[z²] = E[Var(z|cell)] + Var(E[z|cell]).
    # The between-cell component is the per-cell μ-bias signal the user wants
    # to compare against the β=0.5 baseline (Teff=0.48, [M/H]=0.62).
    var_decomp_pre = _variance_decomposition(mu_bar, L_bar, y_cpu, cell_ids)
    _LOG.info(
        "pre-cal variance decomp (Teff/logg/[M/H] between-cell Var(E[z|c])): "
        "%.3f / %.3f / %.3f",
        var_decomp_pre["between_cell_var"][0],
        var_decomp_pre["between_cell_var"][1],
        var_decomp_pre["between_cell_var"][2],
    )

    # Off-diagonal diagnostic: what does E[mahal²] look like if we zero the
    # off-diagonal Cholesky entries? If diagonal-only recovers n_dims (=21)
    # while full-Σ is much larger, the block-Cholesky off-diagonals are the
    # miscalibration source.
    n_dims = mu_bar.shape[-1]
    L_diag = np.zeros_like(L_bar)
    diag_idx = np.arange(n_dims)
    L_diag[:, diag_idx, diag_idx] = L_bar[:, diag_idx, diag_idx]
    mahal_full = _mahal_bulk(mu_bar, L_bar, y_clean)
    mahal_diag = _mahal_bulk(mu_bar, L_diag, y_clean)
    offdiag_diag = {
        "n_dims": n_dims,
        "expected_mean_mahal_sq": n_dims,
        "full_Sigma_mean_mahal_sq": float(mahal_full.mean()),
        "full_Sigma_median_mahal_sq": float(np.median(mahal_full)),
        "diag_only_mean_mahal_sq": float(mahal_diag.mean()),
        "diag_only_median_mahal_sq": float(np.median(mahal_diag)),
    }
    _LOG.info(
        "off-diagonal diagnostic: E[mahal²] full=%.2f diag=%.2f (target=%d)",
        offdiag_diag["full_Sigma_mean_mahal_sq"],
        offdiag_diag["diag_only_mean_mahal_sq"], n_dims,
    )

    need_cal = bool((np.nan_to_num(per_cell_raw, nan=0.0) > 0.15).any() or global_raw > 0.10)

    # Comparator only — we no longer *apply* scalar per-cell temperature scaling
    # (it can't fix anisotropic per-label miscalibration and empirically makes
    # coverage worse in this regime; see #135 diagnostic). Retained in the
    # report for completeness.
    temp_scale_map = temperature_scaling_per_cell(mu_bar, L_bar, y_clean, cell_ids)

    # Primary calibration: shrunken per-cell-per-label α, applied via
    # L' = diag(α) L so Σ' = diag(α) Σ diag(α) — preserves PD and the joint
    # correlation structure, rescales marginal variances per (cell, label).
    shrunk = shrunken_per_cell_per_label_scale(
        mu_bar, L_bar, y_clean, cell_ids, tau=50.0, min_cell_stars=8,
    )
    per_star_alpha = shrunk["per_star_alpha"]  # (B, n)
    L_calibrated = (per_star_alpha[:, :, None] * L_bar).astype(np.float32)

    per_cell_post, global_post, var_z_post = _reliability_per_cell(
        mu_bar, L_calibrated, y_cpu, cell_ids,
    )
    cov_post = coverage_at_levels(mu_bar, L_calibrated, y_clean,
                                  levels=(0.68, 0.95, 0.99))
    _LOG.info("post-shrinkage global reliability err=%.4f", global_post)

    # Joint-covariance preservation check: diag(α) Σ diag(α) is still PD and
    # should keep E[Mahal²] near n_dims if α's are near 1. Since shrinkage
    # target α_j ≈ 1 (global per-label already calibrated), deviations track
    # purely the per-cell heteroscedasticity correction.
    mahal_shrunk = _mahal_bulk(mu_bar, L_calibrated, y_clean)
    joint_preservation = {
        "n_dims": n_dims,
        "pre_shrink_mean_mahal_sq": float(mahal_full.mean()),
        "post_shrink_mean_mahal_sq": float(mahal_shrunk.mean()),
        "post_shrink_median_mahal_sq": float(np.median(mahal_shrunk)),
    }
    _LOG.info(
        "joint preservation: pre E[mahal²]=%.2f post=%.2f (target=%d)",
        joint_preservation["pre_shrink_mean_mahal_sq"],
        joint_preservation["post_shrink_mean_mahal_sq"], n_dims,
    )

    # Adjacent-cell smoothness diagnostic: max α-ratio between cells that
    # differ by ±1 index in exactly one axis of the (Teff, log g, [M/H])
    # grid. Large ratios ⇒ hard cell edges will produce visible σ
    # discontinuities at cell boundaries — flag for v2 GP smoothing.
    adjacency_stats = _adjacent_cell_smoothness(
        shrunk["scales"], cell_def["n_bins"],
    )

    # --- GP α-smoothing (regime A remediation) ---
    gp_stats: dict = {}
    per_cell_gp: np.ndarray | None = None
    global_gp: float | None = None
    cov_gp: dict[str, dict] | None = None
    L_gp: np.ndarray | None = None
    if args.apply_gp_smoothing:
        _LOG.info("fitting 3-D GP α-smoothing over (Teff, log g, [M/H])")
        gp_out = gp_smoothed_per_cell_per_label_scale(
            mu_bar, L_bar, y_clean, cell_features, cell_ids,
            min_cell_stars_for_training=32, min_cell_stars=8,
        )
        alpha_gp = gp_out["per_star_alpha"]  # (B, n)
        L_gp = (alpha_gp[:, :, None] * L_bar).astype(np.float32)
        per_cell_gp, global_gp, var_z_gp = _reliability_per_cell(
            mu_bar, L_gp, y_cpu, cell_ids,
        )
        cov_gp = coverage_at_levels(
            mu_bar, L_gp, y_clean, levels=(0.68, 0.95, 0.99),
        )
        gp_adjacency = _adjacent_cell_smoothness(
            gp_out["scales"], cell_def["n_bins"],
        )
        _LOG.info(
            "GP calibration: global err=%.4f  cov95=%.3f  max α-ratio=%.3f",
            global_gp, cov_gp["joint"][0.95], gp_adjacency["max_ratio"],
        )
        gp_stats = {
            "global_alpha": [float(x) for x in gp_out["global_alpha"]],
            "n_train_cells": len(gp_out["train_cell_ids"]),
            "training_diagnostics": gp_out["training_diagnostics"],
            "adjacency": gp_adjacency,
            "alpha_per_cell_per_label": {
                f"{c}_{j}": float(v) for (c, j), v in gp_out["scales"].items()
            },
            "per_label_var_z": [float(x) if not np.isnan(x) else None
                                for x in var_z_gp],
        }

    # --- Regime B exclusion envelope (per-star tier1 release flag) ---
    envelope_stats: dict = {}
    tier1_release: np.ndarray | None = None
    if args.apply_regime_b:
        envelope = RegimeBEnvelope()
        # Uses *predicted* Teff / log g (block indices 0, 1) so the envelope is
        # applicable at Stream 3 inference where APOGEE truth is absent. b_deg
        # has no predicted counterpart (it's a sky coordinate).
        teff_pred = mu_bar[:, 0]
        logg_pred = mu_bar[:, 1]
        inside_mask = envelope.mask(teff_pred, logg_pred, b_deg_val)
        tier1_release = ~inside_mask

        # Overlap with halt-cell-derived set {34, 49}.
        regimeB_cells = {34, 49}
        halt_mask_truth = np.isin(cell_ids, np.array(list(regimeB_cells), dtype=np.int64))
        envelope_excludes_halt = int((inside_mask & halt_mask_truth).sum())
        halt_in_val = int(halt_mask_truth.sum())
        envelope_spillover = int((inside_mask & ~halt_mask_truth).sum())

        # Calibration metrics on the Tier 1 subset (excluded stars removed).
        if tier1_release.any():
            L_use = L_gp if L_gp is not None else L_calibrated
            idx = np.where(tier1_release)[0]
            per_cell_tier1, global_tier1, var_z_tier1 = _reliability_per_cell(
                mu_bar[idx], L_use[idx], y_cpu[idx], cell_ids[idx],
            )
            cov_tier1 = coverage_at_levels(
                mu_bar[idx], L_use[idx], y_clean[idx], levels=(0.68, 0.95, 0.99),
            )
            halt_cells_tier1 = [
                int(c) for c, e in zip(np.unique(cell_ids[idx]), per_cell_tier1)
                if (not np.isnan(e) and e > 0.30)
            ]
            cells_over_15_tier1 = [
                int(c) for c, e in zip(np.unique(cell_ids[idx]), per_cell_tier1)
                if (not np.isnan(e) and e > 0.15)
            ]
        else:
            per_cell_tier1 = np.array([])
            global_tier1 = float("nan")
            var_z_tier1 = np.array([])
            cov_tier1 = {"per_label": {}, "joint": {}}
            halt_cells_tier1 = []
            cells_over_15_tier1 = []

        envelope_stats = {
            "envelope": envelope.to_dict(),
            "n_excluded": int(inside_mask.sum()),
            "n_released": int(tier1_release.sum()),
            "n_halt_truth_cells_{34,49}": halt_in_val,
            "halt_cells_captured_by_envelope": envelope_excludes_halt,
            "envelope_spillover_outside_halts": envelope_spillover,
            "post_gp_and_release": {
                "global_reliability_err": (
                    float(global_tier1) if not math.isnan(global_tier1) else None
                ),
                "per_cell_reliability_err": [
                    float(x) if not np.isnan(x) else None for x in per_cell_tier1
                ],
                "per_label_var_z": [
                    float(x) if not np.isnan(x) else None for x in var_z_tier1
                ],
                "coverage_joint": {str(k): float(v)
                                   for k, v in cov_tier1["joint"].items()},
                "coverage_per_label": {
                    str(k): [float(x) for x in v]
                    for k, v in cov_tier1["per_label"].items()
                },
                "halt_cells_over_30pct": halt_cells_tier1,
                "cells_over_15pct": cells_over_15_tier1,
            },
        }
        _LOG.info(
            "regime B envelope: excluded=%d/%d (%.3f%%)  captures cells{34,49}: %d/%d",
            envelope_stats["n_excluded"], tier1_release.size,
            100.0 * envelope_stats["n_excluded"] / tier1_release.size,
            envelope_excludes_halt, halt_in_val,
        )
    _LOG.info(
        "adjacent-cell α-ratio: max=%.3f  median=%.3f  p90=%.3f over %d pairs",
        adjacency_stats["max_ratio"], adjacency_stats["median_ratio"],
        adjacency_stats["p90_ratio"], adjacency_stats["n_pairs"],
    )

    # Conformal non-conformity scores (on calibrated L if applied).
    conf_scores = conformal_nonconformity_scores(mu_bar, L_calibrated, y_clean)
    conf_radii = {
        float(lvl): float(np.quantile(conf_scores, lvl))
        for lvl in (0.68, 0.95, 0.99)
    }

    tag_suffix = f"_{args.tag}" if args.tag else ""
    _plot_reliability(
        per_cell_raw, args.report_dir / f"reliability_precal{tag_suffix}.png",
        title=f"Reliability error per cell — pre-calibration{tag_suffix}",
    )
    _plot_reliability(
        per_cell_post, args.report_dir / f"reliability_postcal{tag_suffix}.png",
        title=f"Reliability error per cell — post shrunken α_{{c,l}}{tag_suffix}",
    )
    if per_cell_gp is not None:
        _plot_reliability(
            per_cell_gp, args.report_dir / f"reliability_gp{tag_suffix}.png",
            title=f"Reliability error per cell — post GP α-smoothing{tag_suffix}",
        )

    per_cell_post_clean = np.nan_to_num(per_cell_post, nan=0.0)
    cells_unique = np.unique(cell_ids)
    halt_cells = [int(c) for c, e in zip(cells_unique, per_cell_post)
                  if (not np.isnan(e) and e > 0.30)]
    cells_over_15 = [int(c) for c, e in zip(cells_unique, per_cell_post)
                     if (not np.isnan(e) and e > 0.15)]

    # Pass-gate decision (DESIGN + phase directive).
    pass_global = bool(global_post <= 0.10)
    pass_per_cell = bool((per_cell_post_clean <= 0.15).all())
    cov_deltas = {
        float(lvl): float(cov_post["joint"][float(lvl)] - lvl)
        for lvl in (0.68, 0.95, 0.99)
    }
    pass_coverage = all(abs(d) <= 0.05 for d in cov_deltas.values())

    # Escalation signals: if more than 5 cells still exceed
    # 15% or cov95 is off by > 3pp after shrinkage, post-hoc calibration
    # cannot rescue this run — candidates are training-data expansion, loss
    # change, or richer covariance structure.
    escalate_n_cells = len(cells_over_15) > 5
    escalate_cov95 = abs(cov_deltas[0.95]) > 0.03
    escalate = bool(escalate_n_cells or escalate_cov95)

    # Smoothness decision: < 1.5 accept, > 2.0 flag for v2.
    smooth_accept = adjacency_stats["max_ratio"] < 1.5
    smooth_flag_v2 = adjacency_stats["max_ratio"] > 2.0

    # Release-target gate — the one the ensemble retrain must pass:
    # global err ≤ 10%, joint cov95 within 5 pp, all remaining cells ≤ 15% post-GP
    # with regime B stars excluded.
    release_gate: dict = {"applied": False}
    if args.apply_gp_smoothing and args.apply_regime_b and envelope_stats:
        post_block = envelope_stats["post_gp_and_release"]
        rel_global = post_block["global_reliability_err"]
        rel_cells = post_block["per_cell_reliability_err"]
        rel_cov = post_block["coverage_joint"]
        rel_cells_clean = [x for x in rel_cells if x is not None]
        pass_rel_global = rel_global is not None and rel_global <= 0.10
        pass_rel_cells = all(x <= 0.15 for x in rel_cells_clean) if rel_cells_clean else False
        pass_rel_cov = True
        for lvl in (0.68, 0.95, 0.99):
            got = rel_cov.get(str(lvl))
            if got is None or abs(got - lvl) > 0.05:
                pass_rel_cov = False
                break
        release_gate = {
            "applied": True,
            "spec": "global≤0.10 AND all cells≤0.15 AND |cov-nominal|≤0.05 "
                    "AND GP α + regime B applied",
            "pass_global": bool(pass_rel_global),
            "pass_per_cell": bool(pass_rel_cells),
            "pass_coverage": bool(pass_rel_cov),
            "pass": bool(pass_rel_global and pass_rel_cells and pass_rel_cov),
            "n_halt_cells_post": len(post_block["halt_cells_over_30pct"]),
            "n_cells_over_15_post": len(post_block["cells_over_15pct"]),
            "n_excluded": envelope_stats["n_excluded"],
            "frac_excluded": envelope_stats["n_excluded"] / max(tier1_release.size, 1),
        }
        _LOG.info(
            "release gate: global=%s cells=%s cov=%s → PASS=%s",
            pass_rel_global, pass_rel_cells, pass_rel_cov,
            release_gate["pass"],
        )

    summary = {
        "source": source_label,
        "source_is_single_member": args.checkpoint is not None,
        "n_members": len(member_ckpts),
        "n_val_stars": int(y_cpu.shape[0]),
        "n_labels": int(y_cpu.shape[1]),
        "cell_definition": cell_def,
        "n_cells": int(np.unique(cell_ids).size),
        "reliability_metric": "mean_j |Var((y-mu)/sigma_diag)_j - 1|",
        "off_diagonal_diagnostic": offdiag_diag,
        "variance_decomposition": {
            "total_Ez2": [float(x) if not np.isnan(x) else None
                          for x in var_decomp_pre["total_Ez2"]],
            "within_cell_var": [float(x) if not np.isnan(x) else None
                                for x in var_decomp_pre["within_cell_var"]],
            "between_cell_var": [float(x) if not np.isnan(x) else None
                                 for x in var_decomp_pre["between_cell_var"]],
            "notes": (
                "E[z²] = E[Var(z|cell)] + Var(E[z|cell]). Between-cell var is"
                " the per-cell μ-bias signal that β=0.5 can absorb by"
                " inflating σ. Compare to β=0.5 baseline: Teff 0.48, [M/H] 0.62."
            ),
        },
        "pre_calibration": {
            "global_reliability_err": global_raw,
            "per_cell_reliability_err": [float(x) if not np.isnan(x) else None
                                         for x in per_cell_raw],
            "per_label_var_z": [float(x) if not np.isnan(x) else None
                                for x in var_z_raw],
            "coverage_joint": {str(k): float(v)
                               for k, v in cov_raw["joint"].items()},
            "coverage_per_label": {
                str(k): [float(x) for x in v]
                for k, v in cov_raw["per_label"].items()
            },
        },
        "calibration_method": "shrunken_per_cell_per_label_alpha",
        "needed_calibration": need_cal,
        "temperature_per_cell_comparator": {
            str(k): float(v) for k, v in temp_scale_map.items()
        },
        "shrinkage": {
            "tau": shrunk["tau"],
            "global_alpha": [float(x) for x in shrunk["global_alpha"]],
            "n_per_cell": {str(k): int(v) for k, v in shrunk["n_per_cell"].items()},
            "alpha_per_cell_per_label": {
                f"{c}_{j}": float(v) for (c, j), v in shrunk["scales"].items()
            },
        },
        "joint_preservation": joint_preservation,
        "adjacent_cell_smoothness": adjacency_stats,
        "gp_smoothing": (
            {
                **gp_stats,
                "global_reliability_err": float(global_gp) if global_gp is not None else None,
                "per_cell_reliability_err": (
                    [float(x) if not np.isnan(x) else None for x in per_cell_gp]
                    if per_cell_gp is not None else None
                ),
                "coverage_joint": (
                    {str(k): float(v) for k, v in cov_gp["joint"].items()}
                    if cov_gp is not None else None
                ),
                "coverage_per_label": (
                    {str(k): [float(x) for x in v]
                     for k, v in cov_gp["per_label"].items()}
                    if cov_gp is not None else None
                ),
            }
            if args.apply_gp_smoothing else None
        ),
        "regime_b_envelope": envelope_stats if args.apply_regime_b else None,
        "post_calibration": {
            "global_reliability_err": float(global_post),
            "per_cell_reliability_err": [float(x) if not np.isnan(x) else None
                                         for x in per_cell_post],
            "per_label_var_z": [float(x) if not np.isnan(x) else None
                                for x in var_z_post],
            "coverage_joint": {str(k): float(v) for k, v in cov_post["joint"].items()},
            "coverage_per_label": {
                str(k): [float(x) for x in v]
                for k, v in cov_post["per_label"].items()
            },
            "coverage_deltas_vs_nominal": {str(k): float(v) for k, v in cov_deltas.items()},
        },
        "conformal_radii": {str(k): v for k, v in conf_radii.items()},
        "gate": {
            "global_within_10pct": pass_global,
            "per_cell_within_15pct": pass_per_cell,
            "coverage_within_5pp": pass_coverage,
            "halt_cells_over_30pct": halt_cells,
            "cells_over_15pct": cells_over_15,
            "escalate_more_than_5_bad_cells": bool(escalate_n_cells),
            "escalate_cov95_off_by_gt_3pp": bool(escalate_cov95),
            "escalate": bool(escalate),
            "smoothness_accept_v1": bool(smooth_accept),
            "smoothness_flag_v2": bool(smooth_flag_v2),
            "pass": pass_global and pass_per_cell and pass_coverage,
        },
        "release_gate_gp_plus_envelope": release_gate,
    }
    report_path = args.report_dir / f"calibration_report{tag_suffix}.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _LOG.info("wrote %s", report_path)
    _LOG.info(
        "gate: global=%s cells=%s cov=%s halt=%d escalate=%s smooth_v2=%s",
        pass_global, pass_per_cell, pass_coverage,
        len(halt_cells), escalate, smooth_flag_v2,
    )


if __name__ == "__main__":
    main()
