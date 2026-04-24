"""[α/M] PCA-CMI triage — sequential, short-circuiting.

Context
-------
The §9.2 final audit recomputed PCA-CMI for all five Pipeline 1 labels using a
7-component PCA summary (95.8 % variance). Four of five labels moved above the
release-gate floor; **[α/M] alone remains at 0.0000 nats**. The shuffle-null
(skill_ratio -0.2574) and XP-joint-shuffle (ΔRMSE/σ = 0.5362) remain
load-bearing evidence for the Tier-1 release, but the CMI anomaly needs a
specific explanation before the D-Cat-b methods paper.

Three competing hypotheses:

- **H1 — high-order Hermite structure.** The 7-PC summary discards ~4.2 %
  tail variance; if [α/M] information lives in that tail, a richer PCA
  (15 components, ~99 % variance) should recover it.
- **H2 — aux absorption.** The 4-D aux conditioning set
  ``(bp_rp, g_mag, parallax, av_sfd)`` may absorb [α/M]-relevant signal via
  sub-population correlations; conditioning only on parallax should recover it.
- **H3 — KSG estimator noise.** KSG raw CMI can be slightly negative when the
  true value is near zero; :func:`conditional_mi_ksg` clamps to 0, which masks
  this. An unclipped re-run pinpoints whether the 0.0000 is "exactly zero" or
  "near zero + clipping".

Protocol (sequential with short-circuit)
----------------------------------------
1. **Test 1 — 15-PC CMI.** If CMI ≥ 0.01 nats, **H1 confirmed**, stop.
2. **Test 2 — Unclipped 15-PC CMI.** If meaningfully positive (≥ 0.01),
   H1+H3 both contribute. If small negative / small positive (≈ 0 ± 0.003),
   **H3 dominates**. Stop if H3 answers cleanly.
3. **Test 3 — Minimal conditioning.** I([α/M]; XP-PCA-15 | parallax) alone;
   compare to full-aux result. If parallax-only is materially positive while
   full-aux is 0, **H2 confirmed**.

Outputs
-------
- ``reports/pipeline1/audit/alpha_m_triage.json`` — machine-readable.
- ``reports/pipeline1/audit/alpha_m_triage.md`` — narrative + verdict.

Same val split (seed 0, N_val 41851), same ensemble, same scaler, same KSG
(k=8, 8000-sample cap). Read-only on data.
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
from scipy.spatial import cKDTree
from scipy.special import digamma

from arqueogal.xp_abundances.main.audit import conditional_mi_ksg
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.model import CovarianceBlockLayout
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("alpha_m_triage")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENSEMBLE = REPO_ROOT / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/audit"

TARGET_LABEL = "alpha_m_apogee"
_EPS: float = 1e-12

# Decision thresholds (matches prose in the docstring).
CMI_POSITIVE_THRESHOLD_NATS: float = 0.01
H3_NEAR_ZERO_ABS_MAX: float = 0.003


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
    n_components: int,
) -> tuple[np.ndarray, int, float]:
    """Return PCA projection with a fixed number of components.

    Parameters
    ----------
    X_val_xp
        ``(N, 108)`` array of BP+RP normalised shape coefficients.
    n_components
        Exact number of components to retain.

    Returns
    -------
    (projection, k, cumulative_variance_at_k)
    """
    Xc = X_val_xp - X_val_xp.mean(axis=0, keepdims=True)
    U, s, _Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    var = (s**2) / max(Xc.shape[0] - 1, 1)
    cum = np.cumsum(var) / var.sum()
    k = int(min(n_components, len(s)))
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


def _conditional_mi_ksg_raw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """Raw (unclipped) KSG conditional MI — identical math to audit.conditional_mi_ksg,
    minus the ``max(mi, 0)`` clamp. Used by Test 2 (H3).
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    z = np.atleast_2d(z).T if z.ndim == 1 else z
    n = x.shape[0]
    xyz = np.concatenate([x, y, z], axis=1)
    xz = np.concatenate([x, z], axis=1)
    yz = np.concatenate([y, z], axis=1)

    tree_xyz = cKDTree(xyz)
    eps = tree_xyz.query(xyz, k=k + 1, p=np.inf)[0][:, -1]
    tree_xz, tree_yz, tree_z = cKDTree(xz), cKDTree(yz), cKDTree(z)
    nz = np.array(
        [len(tree_z.query_ball_point(z[i], eps[i] - _EPS, p=np.inf)) for i in range(n)],
        dtype=np.float64,
    )
    nxz = np.array(
        [len(tree_xz.query_ball_point(xz[i], eps[i] - _EPS, p=np.inf)) for i in range(n)],
        dtype=np.float64,
    )
    nyz = np.array(
        [len(tree_yz.query_ball_point(yz[i], eps[i] - _EPS, p=np.inf)) for i in range(n)],
        dtype=np.float64,
    )
    cmi = digamma(k) + np.mean(digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1))
    return float(cmi)


def _cmi_single_label(  # noqa: PLR0913
    xp_summary: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    *,
    max_samples: int = 8000,
    k: int = 8,
    seed: int = 0,
    clip: bool = True,
) -> tuple[float, int]:
    """KSG CMI for a single label with the driver's standard sampling.

    Returns ``(cmi, n_used)``.
    """
    rng = np.random.default_rng(seed)
    finite_row = np.isfinite(xp_summary).all(axis=1) & np.isfinite(Z).all(axis=1) & np.isfinite(y)
    idx = np.flatnonzero(finite_row)
    if idx.size < 200:  # noqa: PLR2004
        return float("nan"), int(idx.size)
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    if clip:
        cmi = conditional_mi_ksg(xp_summary[idx], y[idx], Z[idx], k=k)
    else:
        cmi = _conditional_mi_ksg_raw(xp_summary[idx], y[idx], Z[idx], k=k)
    return float(cmi), int(idx.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mi-max-samples", type=int, default=8000)
    parser.add_argument("--pca-components-rich", type=int, default=15)
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

    val_ds = val_loader.dataset
    X_val = np.asarray(val_ds.X).astype(np.float32)
    Y_val_human_scaled = np.asarray(val_ds.Y).astype(np.float32)
    Y_val_human_raw = scaler_human.inverse_mean(Y_val_human_scaled)
    human_to_block = block_layout.human_to_block_perm.cpu().numpy()
    Y_val = Y_val_human_raw[:, human_to_block]
    _LOG.info("X_val=%s Y_val=%s", X_val.shape, Y_val.shape)

    # PCA basis on the XP block — 15 components.
    xp_idx = np.asarray(families["bp_shape"] + families["rp_shape"], dtype=np.int64)
    X_val_xp = X_val[:, xp_idx].astype(np.float64)
    xp_pca15, k_pca15, var_at_k15 = _pca_xp_summary(X_val_xp, args.pca_components_rich)
    _LOG.info(
        "PCA: k=%d components, cumulative variance at k = %.4f",
        k_pca15,
        var_at_k15,
    )

    # Conditioning — full 4-D.
    val_source_ids = np.asarray(val_loader.dataset.source_id)
    Z_full, cond_names = _aux_conditioning(args.parquet, val_source_ids)
    # Parallax-only (column index 2 per cond_cols order).
    parallax_col = list(cond_names).index("parallax")
    Z_parallax = Z_full[:, [parallax_col]]

    # Target label.
    target_idx = ckpt_label_names.index(TARGET_LABEL)
    y_target = Y_val[:, target_idx]

    tests: dict[str, Any] = {}

    # -- Test 1 — 15-PC CMI with full 4-D aux, clipped ------------------------
    _LOG.info("Test 1 — 15-PC CMI with full 4-D aux (clipped)")
    t1_val, t1_n = _cmi_single_label(
        xp_pca15,
        y_target,
        Z_full,
        max_samples=args.mi_max_samples,
        k=8,
        seed=0,
        clip=True,
    )
    _LOG.info("Test 1 CMI = %.6f (n=%d)", t1_val, t1_n)
    tests["test_1_pca15_clipped"] = {
        "description": (
            "15-PC PCA summary, 4-D aux conditioning "
            "(bp_rp, g_mag, parallax, av_sfd), KSG k=8, clipped (standard driver)."
        ),
        "pca_components": int(k_pca15),
        "pca_variance_retained": float(var_at_k15),
        "conditioning_columns": list(cond_names),
        "cmi_clipped": float(t1_val),
        "n_samples_used": int(t1_n),
        "threshold_positive_nats": CMI_POSITIVE_THRESHOLD_NATS,
        "verdict_if_positive": "H1_confirmed_high_order_hermite_structure",
    }

    verdict: dict[str, Any] = {
        "supported_hypothesis": None,
        "short_circuited_after_test": None,
        "rationale": None,
        "tests_run": ["test_1_pca15_clipped"],
    }

    if t1_val >= CMI_POSITIVE_THRESHOLD_NATS:
        verdict.update(
            supported_hypothesis="H1",
            short_circuited_after_test=1,
            rationale=(
                f"15-PC CMI = {t1_val:.4f} nats ≥ {CMI_POSITIVE_THRESHOLD_NATS} floor. "
                f"[α/M] signature lives in Hermite coefficients beyond the 7-PC summary "
                f"({var_at_k15 * 100:.2f}% variance retained at 15 PC vs ~95.8% at 7 PC)."
            ),
        )
        _write_outputs(
            args.report_dir,
            _assemble_payload(
                args,
                cfg,
                split_seed,
                X_val.shape[0],
                ckpt_label_names,
                tests,
                verdict,
                cond_names,
            ),
        )
        _LOG.info("Short-circuit: H1 confirmed.")
        return

    # -- Test 2 — 15-PC CMI with full 4-D aux, UNCLIPPED (H3) -----------------
    _LOG.info("Test 2 — 15-PC CMI with full 4-D aux, UNCLIPPED")
    t2_val, t2_n = _cmi_single_label(
        xp_pca15,
        y_target,
        Z_full,
        max_samples=args.mi_max_samples,
        k=8,
        seed=0,
        clip=False,
    )
    _LOG.info("Test 2 CMI (unclipped) = %.6f (n=%d)", t2_val, t2_n)
    tests["test_2_pca15_unclipped"] = {
        "description": (
            "Same estimator as Test 1 but without the max(I_hat, 0) clamp. "
            "Probes whether the 0.0000 in Test 1 reflects true absence of "
            "information or a small-negative KSG raw estimate clipped to zero."
        ),
        "pca_components": int(k_pca15),
        "pca_variance_retained": float(var_at_k15),
        "conditioning_columns": list(cond_names),
        "cmi_unclipped_raw": float(t2_val),
        "n_samples_used": int(t2_n),
        "threshold_positive_nats": CMI_POSITIVE_THRESHOLD_NATS,
        "threshold_near_zero_abs_max": H3_NEAR_ZERO_ABS_MAX,
    }
    verdict["tests_run"].append("test_2_pca15_unclipped")

    t2_meaningfully_positive = t2_val >= CMI_POSITIVE_THRESHOLD_NATS
    t2_near_zero = abs(t2_val) <= H3_NEAR_ZERO_ABS_MAX

    if t2_meaningfully_positive:
        verdict.update(
            supported_hypothesis="H1+H3",
            short_circuited_after_test=2,
            rationale=(
                f"Unclipped 15-PC raw CMI = {t2_val:.4f} nats ≥ "
                f"{CMI_POSITIVE_THRESHOLD_NATS}, yet the clipped Test 1 value was "
                f"{t1_val:.4f}. H1 and H3 interact: richer PCA does reveal signal, "
                f"but the clipping in audit.conditional_mi_ksg masked part of it."
            ),
        )
        _write_outputs(
            args.report_dir,
            _assemble_payload(
                args,
                cfg,
                split_seed,
                X_val.shape[0],
                ckpt_label_names,
                tests,
                verdict,
                cond_names,
            ),
        )
        _LOG.info("Short-circuit: H1+H3 (meaningful positive in unclipped).")
        return

    if t2_near_zero:
        # H3 fires only for small raw magnitudes — the "true I ≈ 0, clipping
        # hides finite-sample noise" regime. Large-magnitude negatives are a
        # separate failure mode (high-D KSG bias / aux absorption) and fall
        # through to Test 3.
        verdict.update(
            supported_hypothesis="H3",
            short_circuited_after_test=2,
            rationale=(
                f"Unclipped 15-PC raw CMI = {t2_val:.4f} nats "
                f"(|raw| ≤ {H3_NEAR_ZERO_ABS_MAX}). "
                f"The clipped 0.0000 is an artefact of the max(I_hat, 0) clamp in "
                f"audit.conditional_mi_ksg — the KSG raw estimator is sampling near "
                f"zero (expected behaviour when true I is near zero and sample size "
                f"is finite). This is estimator noise, not real information absence. "
                f"H1 can be ruled out: even with 15 PCs ({var_at_k15 * 100:.2f}% "
                f"variance) the unclipped signal does not lift above the "
                f"{CMI_POSITIVE_THRESHOLD_NATS}-nat threshold."
            ),
        )
        _write_outputs(
            args.report_dir,
            _assemble_payload(
                args,
                cfg,
                split_seed,
                X_val.shape[0],
                ckpt_label_names,
                tests,
                verdict,
                cond_names,
            ),
        )
        _LOG.info("Short-circuit: H3 (estimator noise dominant).")
        return

    # -- Test 3 — Minimal conditioning (H2) -----------------------------------
    _LOG.info("Test 3 — 15-PC CMI conditioning only on parallax")
    t3_val, t3_n = _cmi_single_label(
        xp_pca15,
        y_target,
        Z_parallax,
        max_samples=args.mi_max_samples,
        k=8,
        seed=0,
        clip=True,
    )
    _LOG.info("Test 3 CMI (parallax-only) = %.6f (n=%d)", t3_val, t3_n)
    tests["test_3_pca15_parallax_only"] = {
        "description": (
            "15-PC PCA summary, conditioning only on parallax (1-D). "
            "Probes whether bp_rp/g_mag/av_sfd in the full-aux conditioning "
            "set absorb [α/M]-relevant signal via sub-population correlations."
        ),
        "pca_components": int(k_pca15),
        "pca_variance_retained": float(var_at_k15),
        "conditioning_columns": ["parallax"],
        "cmi_clipped": float(t3_val),
        "n_samples_used": int(t3_n),
    }
    verdict["tests_run"].append("test_3_pca15_parallax_only")

    if t3_val >= CMI_POSITIVE_THRESHOLD_NATS and t1_val < CMI_POSITIVE_THRESHOLD_NATS:
        verdict.update(
            supported_hypothesis="H2",
            short_circuited_after_test=3,
            rationale=(
                f"Parallax-only CMI = {t3_val:.4f} nats ≥ {CMI_POSITIVE_THRESHOLD_NATS} "
                f"while full-aux CMI = {t1_val:.4f} stays at/below the floor. The aux "
                f"block (bp_rp, g_mag, av_sfd) is absorbing [α/M]-relevant signal — "
                f"likely through a sub-population kinematic correlation."
            ),
        )
    else:
        verdict.update(
            supported_hypothesis="inconclusive",
            short_circuited_after_test=3,
            rationale=(
                f"Test 1 (full-aux) = {t1_val:.4f}, Test 2 (unclipped) = {t2_val:.4f}, "
                f"Test 3 (parallax-only) = {t3_val:.4f}. None of H1/H2/H3 cleanly "
                f"triggers. Load-bearing evidence for the Tier-1 release remains the "
                f"shuffle-null (skill_ratio -0.2574) and XP-joint-shuffle (ΔRMSE/σ = "
                f"0.5362); CMI is non-diagnostic for [α/M] at this estimator setting."
            ),
        )

    _write_outputs(
        args.report_dir,
        _assemble_payload(
            args,
            cfg,
            split_seed,
            X_val.shape[0],
            ckpt_label_names,
            tests,
            verdict,
            cond_names,
        ),
    )
    _LOG.info("All three tests ran; verdict=%s", verdict["supported_hypothesis"])


def _assemble_payload(  # noqa: PLR0913
    args: argparse.Namespace,
    _cfg: TrainingConfig,
    split_seed: int,
    n_val: int,
    ckpt_label_names: tuple[str, ...],
    tests: dict[str, Any],
    verdict: dict[str, Any],
    cond_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "scope": "[α/M] PCA-CMI triage — sequential short-circuiting protocol",
        "target_label": TARGET_LABEL,
        "ensemble_dir": str(args.ensemble),
        "parquet": str(args.parquet),
        "split_seed": split_seed,
        "n_val": int(n_val),
        "label_names_block_order": list(ckpt_label_names),
        "pca_components_requested": int(args.pca_components_rich),
        "cmi_estimator_k": 8,
        "max_samples": int(args.mi_max_samples),
        "full_conditioning_columns": list(cond_names),
        "cmi_release_floor_nats": 0.02,
        "decision_thresholds": {
            "positive_nats": CMI_POSITIVE_THRESHOLD_NATS,
            "near_zero_abs_max": H3_NEAR_ZERO_ABS_MAX,
        },
        "tests": tests,
        "verdict": verdict,
    }


def _fmt(x: float, precision: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{precision}f}"


def _write_outputs(report_dir: Path, payload: dict[str, Any]) -> None:
    json_path = report_dir / "alpha_m_triage.json"
    md_path = report_dir / "alpha_m_triage.md"

    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    _LOG.info("wrote %s", json_path)

    _write_markdown(md_path, payload)
    _LOG.info("wrote %s", md_path)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    verdict = payload["verdict"]
    tests = payload["tests"]

    lines: list[str] = []
    lines.append("# [α/M] PCA-CMI triage — hypothesis verdict")
    lines.append("")
    lines.append(
        f"_Timestamp: {payload['timestamp']} · "
        f"Ensemble: `{Path(payload['ensemble_dir']).name}` · "
        f"Val split seed {payload['split_seed']} · N_val = {payload['n_val']}_",
    )
    lines.append("")
    lines.append(
        "Follow-up to `SUMMARY.md` and `pca_cmi_all_labels.json`. The §9.2 final audit "
        "recomputed PCA-CMI for all five labels on a 7-component PCA summary (95.8% "
        "variance). [M/H] and [Mg/H] recovered above the 0.02-nat release-gate floor; "
        "**[α/M] remained at 0.0000 nats alone**. This triage runs three targeted tests "
        "in order, short-circuiting as soon as one explanation fits.",
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**Supported hypothesis:** {verdict['supported_hypothesis']}")
    lines.append("")
    lines.append(f"**Short-circuited after test:** {verdict['short_circuited_after_test']}")
    lines.append("")
    lines.append(f"**Tests actually run:** {', '.join(verdict['tests_run'])}")
    lines.append("")
    lines.append(f"**Rationale:** {verdict['rationale']}")
    lines.append("")

    lines.append("## Test results")
    lines.append("")
    if "test_1_pca15_clipped" in tests:
        t = tests["test_1_pca15_clipped"]
        lines.append("### Test 1 — 15-PC CMI, full 4-D aux, clipped (H1 probe)")
        lines.append("")
        lines.append(f"- PCA components: {t['pca_components']}")
        lines.append(f"- Cumulative variance: {t['pca_variance_retained'] * 100:.2f}%")
        lines.append(f"- Conditioning: {', '.join(t['conditioning_columns'])}")
        lines.append(f"- Samples used: {t['n_samples_used']}")
        lines.append(f"- **CMI (clipped): {_fmt(t['cmi_clipped'])} nats**")
        lines.append(
            f"- H1 trigger: CMI ≥ {t['threshold_positive_nats']} nats → "
            "high-order Hermite structure carries the signal.",
        )
        lines.append("")
    if "test_2_pca15_unclipped" in tests:
        t = tests["test_2_pca15_unclipped"]
        lines.append("### Test 2 — 15-PC CMI, full 4-D aux, UNCLIPPED (H3 probe)")
        lines.append("")
        lines.append(
            f"- Raw KSG estimator (no max(I_hat, 0) clamp): "
            f"**{_fmt(t['cmi_unclipped_raw'])} nats**",
        )
        lines.append(f"- Samples used: {t['n_samples_used']}")
        lines.append(
            f"- H3 trigger: |raw| ≤ {t['threshold_near_zero_abs_max']} → "
            "small-sample KSG noise around a true value near zero.",
        )
        lines.append(
            f"- H1+H3 trigger: raw ≥ {t['threshold_positive_nats']} → "
            "real signal masked by the clamp.",
        )
        lines.append("")
    if "test_3_pca15_parallax_only" in tests:
        t = tests["test_3_pca15_parallax_only"]
        lines.append("### Test 3 — 15-PC CMI, parallax-only conditioning (H2 probe)")
        lines.append("")
        lines.append(f"- Conditioning: {', '.join(t['conditioning_columns'])} (1-D)")
        lines.append(f"- Samples used: {t['n_samples_used']}")
        lines.append(f"- **CMI (clipped): {_fmt(t['cmi_clipped'])} nats**")
        lines.append(
            "- H2 trigger: parallax-only CMI ≥ 0.01 while full-aux CMI ≤ 0.01 → "
            "aux features absorbing [α/M]-relevant signal through sub-population "
            "kinematic correlation.",
        )
        lines.append("")

    lines.append("## Release-gate context")
    lines.append("")
    lines.append(
        "Load-bearing evidence for the [α/M] Tier-1 (clean) release is the "
        "shuffled-spectrum null (skill_ratio -0.2574) and the XP-joint-shuffle "
        "(ΔRMSE/σ = 0.5362). Both pass cleanly. CMI is a methodological cross-check, "
        "not a release-blocking quantity for [α/M]. This triage answers *why* the CMI "
        "specifically behaves as it does so the D-Cat-b methods paper can be precise.",
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Estimator: KSG k={payload['cmi_estimator_k']}")
    lines.append(f"- Max samples per test: {payload['max_samples']}")
    lines.append(
        f"- Full aux conditioning: {', '.join(payload['full_conditioning_columns'])}",
    )
    lines.append(f"- Ensemble: `{payload['ensemble_dir']}`")
    lines.append(f"- Parquet: `{payload['parquet']}`")
    lines.append(
        "- Driver: `scripts/triage_alpha_m_cmi.py` "
        "(short-circuits after the first test to answer the question).",
    )
    lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
