"""Run A — halfway-checkpoint UMAP gate driver (#132).

Loads the contrastive-pretrain best-val checkpoint, samples a held-out slice
of the Stream-1 parquet, runs the pretrained trunk (adapter → encoder) on it,
UMAPs the hidden state ``h``, and writes three scatter plots coloured by
``teff_apogee`` / ``mh_apogee`` / ``logg_apogee``.

Halt-the-phase criterion: if any of the three Tier-1 labels shows shattered or
discontinuous structure on the 2-D projection, the operator halts Run A before
supervised fine-tune. The script records the file paths and a compact summary
in ``reports/pipeline1/halfway/halfway_summary.json`` so the operator's verdict
can be attached to the audit trail.

Run: ``PYTHONPATH=src python scripts/run_halfway_umap.py --ckpt <path>``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelTiers,
    load_arrays,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.halfway_umap import (
    compute_halfway_embedding,
    save_halfway_plots,
)
from arqueogal.xp_abundances.main.model import (
    ModelConfig,
    XpAbundanceModel,
    default_pipeline1_layout,
)
from arqueogal.xp_abundances.main.training import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_halfway_umap")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/halfway"

TIER1_COLUMNS = ("teff_apogee", "mh_apogee", "logg_apogee")


def _load_pretrained_model(
    ckpt_path: Path, layout: FeatureLayout, device: torch.device,
) -> tuple[XpAbundanceModel, XpFeatureAdapter, dict]:
    """Reconstruct the pretrained model + adapter from a v2 checkpoint.

    Reads ``use_c0_scalars`` from the checkpoint's flattened config to match
    the adapter state used during pretraining.
    """
    blob = load_checkpoint(ckpt_path, map_location=device)
    cfg_yaml = json.loads(blob["config_yaml"])
    use_c0 = bool(cfg_yaml.get("use_c0_scalars", True))
    latent_dim = int(cfg_yaml.get("latent_dim", 32))
    trunk_hidden = tuple(cfg_yaml.get("trunk_hidden", (256, 128)))
    head_hidden = int(cfg_yaml.get("head_hidden", 128))
    dropout = float(cfg_yaml.get("dropout", 0.10))

    block_layout = default_pipeline1_layout()  # DESIGN-fixed for Run A
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=latent_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        ),
    ).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    if blob.get("regressor"):
        model.head.load_state_dict(blob["regressor"])
    adapter = XpFeatureAdapter(layout, use_c0_scalars=use_c0).to(device)
    return model, adapter, blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Pretrained best-val checkpoint.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--n-stars", type=int, default=10_000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--umap-seed", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)

    layout = FeatureLayout()
    tiers = LabelTiers()

    _LOG.info("loading pretrained checkpoint %s", args.ckpt)
    model, adapter, blob = _load_pretrained_model(args.ckpt, layout, device)

    # Build the same held-out split the pretrain used: fracs from the
    # checkpoint's config; sample from val partition so no train leakage.
    cfg_yaml = json.loads(blob["config_yaml"])
    fracs = tuple(cfg_yaml.get("fracs", (0.70, 0.15, 0.15)))
    split_seed = int(cfg_yaml.get("split_seed", 0))

    _LOG.info("loading feature matrix + running stratified split")
    df_ids = pd.read_parquet(
        args.parquet, columns=["source_id", "fe_h_apogee", "teff_apogee", "b_deg"],
    )
    df_ids = df_ids.drop_duplicates(subset="source_id", keep="first").reset_index(drop=True)
    split_ids = stratified_split_ids(df_ids, fracs=fracs, seed=split_seed)

    arrs = load_arrays(args.parquet, layout, tiers, include_label_errors=False)
    _, first_idx = np.unique(arrs["source_id"], return_index=True)
    first_idx = np.sort(first_idx)
    for k in ("X", "Y", "source_id"):
        arrs[k] = arrs[k][first_idx]

    # Apply same NaN filter/imputation used by training.build_dataloaders so the
    # pretrained trunk sees the inputs it was trained on.
    n_xp = (
        len(layout.bp_coef_cols) + len(layout.rp_coef_cols)
        + len(layout.xp_scalar_cols)
    )
    xp_finite = np.isfinite(arrs["X"][:, :n_xp]).all(axis=1)
    if not xp_finite.all():
        _LOG.info("dropping %d/%d rows with NaN in XP features",
                  int((~xp_finite).sum()), len(xp_finite))
        for k in ("X", "Y", "source_id"):
            arrs[k] = arrs[k][xp_finite]
    np.nan_to_num(arrs["X"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    val_mask = np.isin(arrs["source_id"], split_ids["val"])
    X_val = arrs["X"][val_mask]
    Y_val = arrs["Y"][val_mask]

    rng = np.random.default_rng(args.split_seed)
    idx = rng.choice(len(X_val), size=min(args.n_stars, len(X_val)), replace=False)
    X_sub = X_val[idx]
    Y_sub = Y_val[idx]

    # Build the per-column label dict (human-ordered Y — tiers.all_labels).
    name_to_idx = {name: i for i, name in enumerate(tiers.all_labels)}
    labels = {col: Y_sub[:, name_to_idx[col]].astype(np.float32) for col in TIER1_COLUMNS}

    _LOG.info("embedding %d stars through trunk on %s", len(X_sub), device)
    he = compute_halfway_embedding(
        model, adapter, X_sub, labels,
        device=device,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        umap_seed=args.umap_seed, batch_size=args.batch_size,
    )
    plot_paths = save_halfway_plots(he, args.report_dir, prefix="halfway")

    summary = {
        "ckpt": str(args.ckpt),
        "parquet": str(args.parquet),
        "device": str(device),
        "n_stars": he.n_stars,
        "n_finite_per_label": he.n_finite,
        "umap": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "seed": args.umap_seed,
        },
        "plots": [str(p) for p in plot_paths],
        "verdict": "PENDING_VISUAL_INSPECTION",
    }
    summary_path = args.report_dir / "halfway_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _LOG.info("wrote %d plots + summary to %s", len(plot_paths), args.report_dir)


if __name__ == "__main__":
    main()
