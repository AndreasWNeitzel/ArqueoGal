"""PCA-CMI across all 5 Pipeline 1 labels — methodology-consistency follow-up.

The three-question diagnostic (``run_three_question_diagnostic.py``) computed
``I(XP; label | aux)`` via KSG with a 7-component PCA summary (95.8% variance)
for Teff and log g only. This driver extends that estimator to the three
info-rich chemistry labels ([M/H], [α/M], [Mg/H]) so the §9.2 final audit
uses the same estimator across all released labels.

Scope: Q1 only. Q2 (permutation importance) and Q3 (aux-only baseline) are
already tabulated in ``three_question_diagnostic.json`` for Teff + log g and
do not need to be re-run for this consistency pass.

Outputs (under ``reports/pipeline1/audit/``):

- ``pca_cmi_all_labels.json`` — per-label PCA-CMI + 2-D rerun + existing 2-D.
- Appends a consolidated table to ``three_question_diagnostic.md`` (as a new
  section, leaving the original Teff/log g narrative intact).

PCA basis and estimator settings match the Teff/log g driver exactly:
variance threshold 0.95 → 7 components, KSG k=8, 8000-sample cap, 4-D aux
conditioning (bp_rp, g_mag, parallax, av_sfd).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from arqueogal.xp_abundances.main.audit import conditional_mi_ksg
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.model import CovarianceBlockLayout
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("pca_cmi_all_labels")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENSEMBLE = REPO_ROOT / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/audit"

ALL_LABELS = (
    "teff_apogee",
    "logg_apogee",
    "mh_apogee",
    "alpha_m_apogee",
    "mg_h_apogee",
)


def _build_cfg_for_val_loader(
    parquet: Path,
    split_seed: int,
    batch_size: int,
) -> TrainingConfig:
    return TrainingConfig(
        train_parquet=parquet,
        output_dir=REPO_ROOT / "tmp_audit",
        epochs=1,
        batch_size=batch_size,
        num_workers=2,
        amp_dtype="bfloat16",
        use_c0_scalars=True,
        split_seed=split_seed,
        pretrained_encoder_ckpt=None,
        output_prefix="xp_abundances_main_audit",
        loss_weights=LossWeights(supcon=0.0, beta_nll=1.0, beta=0.5),
        ensemble_seeds=(0,),
    )


def _feature_family_indices(layout: FeatureLayout) -> dict[str, list[int]]:
    i = 0
    families: dict[str, list[int]] = {}
    n_bp = len(layout.bp_coef_cols)
    families["bp_shape"] = list(range(i, i + n_bp))
    i += n_bp
    n_rp = len(layout.rp_coef_cols)
    families["rp_shape"] = list(range(i, i + n_rp))
    i += n_rp
    n_c0 = len(layout.xp_scalar_cols)
    families["xp_c0"] = list(range(i, i + n_c0))
    i += n_c0
    n_res = len(layout.residual_cols)
    families["residual"] = list(range(i, i + n_res))
    i += n_res
    n_aux = len(layout.aux_cols)
    families["aux"] = list(range(i, i + n_aux))
    i += n_aux
    assert i == layout.input_dim, (i, layout.input_dim)
    return families


def _pca_xp_summary(
    X_val_xp: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, int, float]:
    """Return PCA projection of the XP block retaining ``variance_threshold`` var."""
    Xc = X_val_xp - X_val_xp.mean(axis=0, keepdims=True)
    U, s, _Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    var = (s**2) / max(Xc.shape[0] - 1, 1)
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, variance_threshold) + 1)
    k = max(k, 2)
    proj = U[:, :k] * s[:k]
    return proj.astype(np.float64), k, float(cum[k - 1])


def _aux_conditioning(
    parquet_path: Path,
    source_ids: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    import pandas as pd

    cond_cols = ("source_id", "bp_rp", "g_mag", "parallax", "av_sfd")
    df = pd.read_parquet(parquet_path, columns=list(cond_cols))
    df = df.drop_duplicates(subset="source_id", keep="first")
    sid_to_row = {int(sid): i for i, sid in enumerate(df["source_id"].to_numpy())}
    rows = np.asarray([sid_to_row.get(int(s), -1) for s in source_ids])
    cond_names = tuple(c for c in cond_cols if c != "source_id")
    Z = np.full((len(source_ids), len(cond_names)), np.nan, dtype=np.float64)
    valid = rows >= 0
    for j, c in enumerate(cond_names):
        col = df[c].to_numpy(dtype=np.float64)
        Z[valid, j] = col[rows[valid]]
    return Z, cond_names


def _cmi_for_labels(
    xp_summary: np.ndarray,
    Y_raw: np.ndarray,
    Z: np.ndarray,
    label_names: list[str],
    target_indices: list[int],
    *,
    max_samples: int = 8000,
    k: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finite_row = np.isfinite(xp_summary).all(axis=1) & np.isfinite(Z).all(axis=1)
    out: dict[str, float] = {}
    for j in target_indices:
        name = label_names[j]
        y = Y_raw[:, j]
        mask = finite_row & np.isfinite(y)
        idx = np.flatnonzero(mask)
        if idx.size < 200:  # noqa: PLR2004
            out[name] = float("nan")
            continue
        if idx.size > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        try:
            cmi = conditional_mi_ksg(
                xp_summary[idx],
                y[idx],
                Z[idx],
                k=k,
            )
        except ValueError as exc:
            _LOG.warning("CMI failed for %s: %s", name, exc)
            out[name] = float("nan")
            continue
        out[name] = float(cmi)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mi-max-samples", type=int, default=8000)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s ensemble=%s", device, args.ensemble)

    member_ckpts = sorted(
        args.ensemble.glob("member_seed*/xp_abundances_main_ensemble*_seed*_best.pt"),
    )
    if len(member_ckpts) != 5:  # noqa: PLR2004
        raise FileNotFoundError(
            f"expected 5 member ckpts under {args.ensemble}, found {len(member_ckpts)}",
        )

    layout = FeatureLayout()
    families = _feature_family_indices(layout)

    # Reconstruct block layout + tier map from first checkpoint so the val
    # loader matches the production audit exactly.
    first_blob = load_checkpoint(member_ckpts[0], map_location="cpu")
    ckpt_label_names = tuple(first_blob["label_names"])
    tier_map = first_blob.get("tier_map", {})
    tier1 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 1)
    tier2 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 2)
    tier3 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 3)
    tiers = LabelTiers(tier1=tier1, tier2=tier2, tier3=tier3)
    block_layout = CovarianceBlockLayout.from_dict(first_blob["block_layout"])
    split_seed = int(json.loads(first_blob["config_yaml"]).get("split_seed", 0))

    cfg = _build_cfg_for_val_loader(
        parquet=args.parquet,
        split_seed=split_seed,
        batch_size=args.batch_size,
    )
    _train_loader, val_loader, _split_ids, scaler_human = build_dataloaders(
        cfg,
        layout,
        tiers,
        seed=split_seed,
    )

    # Collect X_val and Y_val (raw-unit, block-order) --------------------------
    val_ds = val_loader.dataset
    X_val = np.asarray(val_ds.X).astype(np.float32)
    Y_val_human_scaled = np.asarray(val_ds.Y).astype(np.float32)
    Y_val_human_raw = scaler_human.inverse_mean(Y_val_human_scaled)
    human_to_block = block_layout.human_to_block_perm.cpu().numpy()
    Y_val = Y_val_human_raw[:, human_to_block]
    _LOG.info("X_val=%s Y_val=%s", X_val.shape, Y_val.shape)

    # PCA basis on XP block (same as Teff/log g driver) -----------------------
    xp_idx = np.asarray(families["bp_shape"] + families["rp_shape"], dtype=np.int64)
    X_val_xp = X_val[:, xp_idx].astype(np.float64)
    xp_pca, k_pca, var_at_k = _pca_xp_summary(X_val_xp, args.pca_variance)
    _LOG.info(
        "PCA: k=%d components, cumulative variance at k = %.4f",
        k_pca,
        var_at_k,
    )

    xp_2d = np.column_stack(
        [
            np.abs(X_val[:, families["bp_shape"]]).sum(axis=1),
            np.abs(X_val[:, families["rp_shape"]]).sum(axis=1),
        ]
    ).astype(np.float64)

    val_source_ids = np.asarray(val_loader.dataset.source_id)
    Z_cond, cond_names = _aux_conditioning(args.parquet, val_source_ids)

    target_indices = [ckpt_label_names.index(n) for n in ALL_LABELS]
    cmi_2d = _cmi_for_labels(
        xp_2d,
        Y_val,
        Z_cond,
        list(ckpt_label_names),
        target_indices,
        max_samples=args.mi_max_samples,
        k=8,
        seed=0,
    )
    cmi_pca = _cmi_for_labels(
        xp_pca,
        Y_val,
        Z_cond,
        list(ckpt_label_names),
        target_indices,
        max_samples=args.mi_max_samples,
        k=8,
        seed=0,
    )

    # Load existing 2-D CMI from audit_payload for the full cross-check table.
    existing_audit = json.loads(
        (args.report_dir / "audit_payload.json").read_text(),
    )
    existing_cmi = {name: float(existing_audit["per_label"][name]["cmi"]) for name in ALL_LABELS}

    payload = {
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "scope": "PCA-CMI across all 5 Pipeline 1 labels (methodology-consistency pass)",
        "ensemble_dir": str(args.ensemble),
        "parquet": str(args.parquet),
        "split_seed": split_seed,
        "n_val": int(X_val.shape[0]),
        "label_names_block_order": list(ckpt_label_names),
        "target_labels": list(ALL_LABELS),
        "pca_components": int(k_pca),
        "pca_variance_retained": float(var_at_k),
        "pca_variance_threshold": float(args.pca_variance),
        "cmi_estimator_k": 8,
        "max_samples": int(args.mi_max_samples),
        "conditioning_columns": list(cond_names),
        "cmi_release_floor_nats": 0.02,
        "existing_audit_2d_summary_cmi": existing_cmi,
        "rerun_2d_summary_cmi": cmi_2d,
        "pca_summary_cmi": cmi_pca,
    }

    out_path = args.report_dir / "pca_cmi_all_labels.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    _LOG.info("wrote %s", out_path)

    _append_markdown_table(args.report_dir / "three_question_diagnostic.md", payload)
    _LOG.info("appended consolidated PCA-CMI table to three_question_diagnostic.md")


def _fmt(x: float, precision: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{precision}f}"


def _append_markdown_table(path: Path, payload: dict[str, Any]) -> None:
    """Append a consolidated PCA-CMI table across all 5 labels.

    Idempotent: if the section already exists it is replaced in place.
    """
    marker = "## PCA-CMI across all 5 labels (methodology-consistency pass)"
    text = path.read_text() if path.exists() else ""
    if marker in text:
        head, _tail = text.split(marker, 1)
        text = head.rstrip() + "\n\n"

    lines: list[str] = []
    lines.append(marker)
    lines.append("")
    lines.append(
        f"_Appended {payload['timestamp']} · "
        f"Same val split (seed {payload['split_seed']}, N_val={payload['n_val']}), "
        f"same PCA basis ({payload['pca_components']} components, "
        f"{payload['pca_variance_retained'] * 100:.2f}% variance), same KSG "
        f"(k={payload['cmi_estimator_k']}, {payload['max_samples']}-sample cap)._",
    )
    lines.append("")
    lines.append(
        "Extends Q1 to the three info-rich chemistry labels so the final audit "
        "uses the same CMI estimator across all released labels.",
    )
    lines.append("")
    lines.append(
        "| label | CMI (2-D, original audit) | CMI (2-D, rerun) | CMI (PCA summary) | PCA / 2-D |",
    )
    lines.append("|---|---|---|---|---|")
    for name in payload["target_labels"]:
        old = payload["existing_audit_2d_summary_cmi"][name]
        rerun = payload["rerun_2d_summary_cmi"][name]
        pca = payload["pca_summary_cmi"][name]
        ratio = (pca / old) if (old and np.isfinite(old) and old > 0) else float("nan")
        lines.append(
            f"| {name} | {_fmt(old)} | {_fmt(rerun)} | {_fmt(pca)} | {_fmt(ratio, 3)} |",
        )
    lines.append("")
    lines.append(
        f"Release-gate CMI floor: ≥ {0.02} nats. Interpretation per label is "
        "folded into the per-label report cards and `SUMMARY.md`.",
    )
    lines.append("")

    new_text = (text.rstrip() + "\n\n" + "\n".join(lines)) if text else "\n".join(lines)
    path.write_text(new_text)


if __name__ == "__main__":
    main()
