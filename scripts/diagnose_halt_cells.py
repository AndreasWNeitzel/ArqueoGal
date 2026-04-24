"""Halt-cell diagnosis for the 5-label z-scored calibration report.

For each halt cell (reliability err > 30 %) and each err>15 % cell in the
(Teff, log g, [M/H]) 4×4×4 grid, report:

1. Cell coordinates (Teff / log g / [M/H] range).
2. Val-split star count.
3. Per-label Var(z) and E[z] within the cell (identifies which labels
   drive the cell err).
4. Overlap with known-problematic populations: low |b|, high A_V (Edenhofer
   LOS), Ye+2024 OOD flag, metal-poor, cool-RGB.

Grid-edge vs interior classification is emitted for each cell.

Run: ``PYTHONPATH=src python scripts/diagnose_halt_cells.py``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint
from arqueogal.xp_abundances.main.uncertainty import bin_by_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("diagnose_halt_cells")

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / (
    "models/main/xp_abundances/"
    "20260419_nogit_859afab_finetune_5label/"
    "xp_abundances_main_finetune_5label_seed0_best.pt"
)
REPORT_PATH = REPO_ROOT / "reports/pipeline1/run_a/calibration_report_zscore_5label.json"
OUT_PATH = REPO_ROOT / "reports/pipeline1/run_a/halt_cell_diagnosis_5label.md"
PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"

LABELS = ("teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee")

# Context columns we want for population-overlap checks.
CONTEXT_COLS_WANTED = [
    "source_id",
    "ra",
    "dec",
    "b_deg",
    "l_deg",
    "phot_g_mean_mag_corrected",
    "bp_rp",
    "av_edenhofer",
    "av_nbhd_median",
    "av_nbhd_std",
    "ye2024_flag",
    "fe_h_apogee",
]


def decode_cell(cell_id: int, n_bins: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of bin_by_cells encoding: cell = ((t*nb1)+g)*nb2 + m."""
    idx: list[int] = []
    for nb in reversed(n_bins):
        idx.append(cell_id % nb)
        cell_id //= nb
    return tuple(reversed(idx))


def cell_range(axis_idx: int, axis_edges: list[float], axis_label: str) -> str:
    """Human-readable range for a given bin index along an axis."""
    nb = len(axis_edges) + 1
    if axis_idx == 0:
        return f"{axis_label} < {axis_edges[0]:.3f}"
    if axis_idx == nb - 1:
        return f"{axis_label} > {axis_edges[-1]:.3f}"
    lo, hi = axis_edges[axis_idx - 1], axis_edges[axis_idx]
    return f"{lo:.3f} ≤ {axis_label} < {hi:.3f}"


def main() -> None:
    with REPORT_PATH.open() as f:
        report = json.load(f)

    cell_def = report["cell_definition"]
    n_bins = tuple(cell_def["n_bins"])
    edges = cell_def["edges_per_col"]
    halt_cells = list(report["gate"]["halt_cells_over_30pct"])
    over_15 = list(report["gate"]["cells_over_15pct"])
    over_15_only = [c for c in over_15 if c not in halt_cells]
    target_cells = halt_cells + over_15_only
    _LOG.info("halt cells: %s", halt_cells)
    _LOG.info("err>15 only: %s", over_15_only)

    # Load checkpoint + reconstruct model with layout from blob.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = load_checkpoint(CKPT, map_location=device)
    block_layout = CovarianceBlockLayout.from_dict(blob["block_layout"])
    ckpt_label_names = tuple(blob["label_names"])
    tier_map = blob.get("tier_map", {})
    tier1 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 1)
    tier2 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 2)
    tier3 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 3)
    tiers = LabelTiers(tier1=tier1, tier2=tier2, tier3=tier3)
    assert tiers.all_labels == ckpt_label_names

    cfg_yaml = json.loads(blob["config_yaml"])
    split_seed = int(cfg_yaml.get("split_seed", 0))
    pretrained_ckpt = Path(cfg_yaml["pretrained_encoder_ckpt"])

    cfg = TrainingConfig(
        train_parquet=PARQUET,
        output_dir=REPO_ROOT / "tmp_halt_diag",
        epochs=1,
        batch_size=1024,
        num_workers=2,
        amp_dtype="bfloat16",
        use_c0_scalars=True,
        encoder_lr_ratio=0.1,
        pretrained_encoder_ckpt=pretrained_ckpt,
        reload_head_from_pretrained=False,
        split_seed=split_seed,
        output_prefix="halt_diag",
        loss_weights=LossWeights(supcon=0.0, beta_nll=1.0, beta=0.5),
        temperature_init=0.10,
        ensemble_seeds=(0,),
    )
    layout = FeatureLayout()
    _, val_loader, _, _scaler = build_dataloaders(cfg, layout, tiers, seed=split_seed)
    _LOG.info("val loader built, batches=%d", len(val_loader))

    # Build model and collect predictions + attach the val source_ids so we
    # can cross-reference population flags.
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=int(cfg_yaml.get("latent_dim", 32)),
            trunk_hidden=tuple(cfg_yaml.get("trunk_hidden", (256, 128))),
            head_hidden=int(cfg_yaml.get("head_hidden", 128)),
            dropout=float(cfg_yaml.get("dropout", 0.10)),
        ),
    ).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["regressor"])
    model.eval()
    adapter = XpFeatureAdapter(layout, use_c0_scalars=True).to(device)

    ckpt_scaler = LabelScaler(
        mean=np.asarray(blob["label_scaler_mean"], dtype=np.float32),
        scale=np.asarray(blob["label_scaler_scale"], dtype=np.float32),
        label_names=tuple(blob["label_names"]),
    )
    scaler_block = ckpt_scaler.reorder_to(block_layout.label_order_block)

    mus_s: list[np.ndarray] = []
    Ls_s: list[np.ndarray] = []
    ys_h: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]
            mu, L, _h, _z = model(adapter(x))
            mus_s.append(mu.float().cpu().numpy())
            Ls_s.append(L.float().cpu().numpy())
            ys_h.append(y.numpy())
    mu_s = np.concatenate(mus_s, axis=0)
    L_s = np.concatenate(Ls_s, axis=0)
    y_human = np.concatenate(ys_h, axis=0)

    # Reorder y from human → block order.
    perm = block_layout.human_to_block_perm.cpu().numpy()
    y_b = y_human[:, perm]
    mu_raw = scaler_block.inverse_mean(mu_s)
    L_raw = scaler_block.inverse_L(L_s)
    y_raw = scaler_block.inverse_mean(y_b)

    cell_ids, cell_def2 = bin_by_cells(y_raw[:, :3].copy(), n_bins=n_bins)
    assert cell_def2["edges_per_col"][0] == cell_def["edges_per_col"][0]

    sigma_diag = np.sqrt(np.einsum("bij,bij->bi", L_raw, L_raw)).clip(1e-8, None)
    z = (y_raw - mu_raw) / sigma_diag

    # Val DataLoader is shuffle=False; dataset.source_id is the same order as
    # the batch iteration above, so we can directly index it to join metadata.
    source_ids = getattr(val_loader.dataset, "source_id", None)
    if source_ids is not None:
        source_ids = np.asarray(source_ids)
        if source_ids.shape[0] != y_raw.shape[0]:
            _LOG.warning(
                "source_id count %d != y %d — skipping metadata join",
                source_ids.shape[0],
                y_raw.shape[0],
            )
            source_ids = None

    # Load context columns for the val stars we care about.
    meta_df: pd.DataFrame | None = None
    if source_ids is not None:
        cols_present = _parquet_columns_present(PARQUET, CONTEXT_COLS_WANTED)
        if cols_present:
            _LOG.info("loading context columns: %s", cols_present)
            full = pd.read_parquet(PARQUET, columns=cols_present)
            full = full.drop_duplicates("source_id", keep="first")
            meta_df = full.set_index("source_id").reindex(source_ids).reset_index()
            _LOG.info("meta_df shape=%s", meta_df.shape)

    # Per-cell reporting.
    lines: list[str] = []
    lines.append("# Halt-cell diagnosis — 5-label z-scored main pipeline\n")
    lines.append(f"Source ckpt: `{CKPT.relative_to(REPO_ROOT)}`  \n")
    lines.append(f"Val stars total: **{y_raw.shape[0]}**  \n")
    lines.append(
        f"Halt cells (err > 30 %): `{halt_cells}`  \nOver 15 % (warning): `{over_15_only}`  \n\n",
    )
    lines.append(
        "Axis edges (quantile-binned on val):\n\n"
        f"- Teff:  {edges[0]}  → 4 bins\n"
        f"- log g: {edges[1]}  → 4 bins\n"
        f"- [M/H]: {edges[2]}  → 4 bins\n\n",
    )
    lines.append("Grid is 4×4×4 = 64 nominal cells.\n\n")

    lines.append("## Per-cell breakdown (halt + warn)\n\n")
    headers = (
        "| cell | severity | Teff bin | log g bin | [M/H] bin | n | edge? "
        "| per-label Var(z) (Te/lg/M/αM/MgH) | per-label E[z] (same) "
        "| median \\|b\\| | median A_V | OOD frac | metal-poor frac |"
    )
    ruler = "|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|"
    lines.append(headers + "\n")
    lines.append(ruler + "\n")

    cells_sorted = sorted(
        target_cells,
        key=lambda c: (0 if c in halt_cells else 1, c),
    )
    for c in cells_sorted:
        idx3 = decode_cell(c, n_bins)
        in_cell = cell_ids == c
        n = int(in_cell.sum())
        severity = "HALT" if c in halt_cells else "warn"
        is_edge = any(i == 0 or i == nb - 1 for i, nb in zip(idx3, n_bins))
        edge_str = "edge" if is_edge else "interior"
        teff_range = cell_range(idx3[0], edges[0], "Teff")
        logg_range = cell_range(idx3[1], edges[1], "log g")
        mh_range = cell_range(idx3[2], edges[2], "[M/H]")

        var_z = [float(np.nanvar(z[in_cell, j])) if n >= 8 else float("nan") for j in range(5)]
        mean_z = [float(np.nanmean(z[in_cell, j])) if n >= 8 else float("nan") for j in range(5)]

        b_med = float("nan")
        av_med = float("nan")
        ood_frac = float("nan")
        mp_frac = float("nan")
        if meta_df is not None:
            sub = meta_df.loc[in_cell]
            if "b_deg" in sub.columns:
                b_med = float(np.nanmedian(np.abs(sub["b_deg"].to_numpy())))
            if "av_edenhofer" in sub.columns:
                av = sub["av_edenhofer"].to_numpy()
                av_med = float(np.nanmedian(av))
            if "ye2024_flag" in sub.columns:
                flag = sub["ye2024_flag"].to_numpy()
                # ye2024_flag is int8: 0 = OK, 1 = flagged (NO_SYNTH_PHOT etc.).
                ood_frac = float(np.nanmean(flag.astype(float))) if flag.size else float("nan")
            if "fe_h_apogee" in sub.columns:
                feh = sub["fe_h_apogee"].to_numpy()
                mp_frac = float(np.nanmean(feh < -1.0)) if feh.size else float("nan")

        def _fmt_vec(v: list[float]) -> str:
            return " / ".join(f"{x:+.2f}" if not np.isnan(x) else "nan" for x in v)

        lines.append(
            f"| {c} | {severity} | {teff_range} | {logg_range} | {mh_range} "
            f"| {n} | {edge_str} | {_fmt_vec(var_z)} | {_fmt_vec(mean_z)} "
            f"| {b_med:.1f}° | {av_med:.2f} | {ood_frac:.2f} | {mp_frac:.2f} |\n",
        )

    # Summaries.
    lines.append("\n## Summary\n\n")
    n_halt_edge = sum(
        1
        for c in halt_cells
        if any(i == 0 or i == nb - 1 for i, nb in zip(decode_cell(c, n_bins), n_bins))
    )
    total_halt_stars = int(sum((cell_ids == c).sum() for c in halt_cells))
    n_halt_sparse = sum(1 for c in halt_cells if int((cell_ids == c).sum()) < 50)
    lines.append(f"- Halt cells at grid edges: **{n_halt_edge} / {len(halt_cells)}**\n")
    lines.append(f"- Halt cells with n < 50 val stars: **{n_halt_sparse} / {len(halt_cells)}**\n")
    lines.append(
        f"- Total val stars in halt cells: **{total_halt_stars} / {y_raw.shape[0]}** "
        f"({100.0 * total_halt_stars / max(y_raw.shape[0], 1):.2f} %)\n",
    )

    with OUT_PATH.open("w") as f:
        f.writelines(lines)
    _LOG.info("wrote %s", OUT_PATH)

    # Stdout recap.
    for c in cells_sorted[: len(halt_cells)]:
        idx3 = decode_cell(c, n_bins)
        n = int((cell_ids == c).sum())
        edge = any(i == 0 or i == nb - 1 for i, nb in zip(idx3, n_bins))
        _LOG.info(
            "cell %d  t=%d g=%d m=%d  n=%d  edge=%s",
            c,
            idx3[0],
            idx3[1],
            idx3[2],
            n,
            edge,
        )


def _parquet_columns_present(path: Path, wanted: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    present = set(schema.names)
    return [c for c in wanted if c in present]


if __name__ == "__main__":
    main()
