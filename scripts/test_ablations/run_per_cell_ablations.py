"""Ablate per-cell tier-promotion gates and compare against the current stack.

Segregated test harness — does not modify the canonical release pipeline or
the main release artefacts. Reads:
- ``release/test_ablations_2026-04-26/predictions_stream1.parquet`` (full
  inference output on Stream 1 with all flags + per-element σ).
- ``data/processed/pipeline1_features_stream1.parquet`` (truth labels:
  ``teff_apogee``, ``logg_apogee``, ``mh_apogee``, ``alpha_m_apogee``,
  ``mg_h_apogee``).

For each ablation config, recompute tier assignments by toggling specific
gates ON / OFF / replaced with a global alternative, then compute metrics
on the held-out test split (stratified split seed=0, fracs=(.70,.15,.15)).

Metrics per element:
- Tier-1 fraction: % of test stars promoted to Tier 1 for that element.
- Tier-1 RMSE: prediction error vs truth on Tier-1 stars.
- Tier-1 bias: mean (pred - truth) on Tier-1 stars.
- Tier-1 σ-coverage at 1σ: fraction of Tier-1 stars where |pred - truth| ≤ σ.
- Tier-1 σ-coverage at 2σ: same at 2σ.
- Tier-2 RMSE: same on Tier-2 stars (regressor + kNN-rescued).
- Tier-1 + Tier-2 RMSE: combined trustworthy-catalog RMSE.

Output: ``release/test_ablations_2026-04-26/ablations.json`` — one block per
config — and ``release/test_ablations_2026-04-26/REPORT.md`` rendered.

Constraint: post-hoc ablation only; for each gate we recompute the tier from
flags already in the parquet (or recompute σ-inflation from σ + a different
threshold). Gates that would need a re-inference (e.g. retraining the
σ-shrinkage with a single-global α instead of per-cell) are flagged for a
separate test.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.xp_abundances.main.data import stratified_split_ids

OUT_DIR = REPO / "release/test_ablations_2026-04-26"
PREDS = OUT_DIR / "predictions_stream1.parquet"
FEATURES = REPO / "data/processed/pipeline1_features_stream1.parquet"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
ELEMENT_TRUTH = {
    "teff": "teff_apogee",
    "logg": "logg_apogee",
    "mh": "mh_apogee",
    "alpha_m": "alpha_m_apogee",
    "mg_h": "mg_h_apogee",
}
ELEMENT_PRED = {e: f"{e}_pred" for e in ELEMENTS}
ELEMENT_SIGMA = {e: f"{e}_sigma" for e in ELEMENTS}

# Production thresholds at the time of the 2026-04-26 ablation. Frozen here
# so the ablation test is reproducible against the snapshot it was run on.
# After 2026-04-26 the alpha_m threshold was tightened from 0.10 → 0.05;
# see release.py and the ablation REPORT.md for justification. To re-run
# this ablation against the new production setting, change "alpha_m": 0.10
# to "alpha_m": 0.05 below.
SIGMA_INFLATED_THRESHOLDS_PROD: dict[str, float] = {
    "teff": 150.0,
    "logg": 0.30,
    "mh": 0.20,
    "alpha_m": 0.10,
    "mg_h": 0.20,
}

# Current production OOD + caveat sets.
OOD_FLAGS_PROD = ("ood_joint_flag", "latent_support_flag", "ood_aux_mahalanobis_flag")
CAVEAT_FLAGS_PROD = (
    "regime_b_flag",
    "mode_ambiguous_flag",
    "ood_disagreement_flag",
    "aux_missing_any",
    "dist_prior_dominated",
)
AUX_ASSISTED = ("alpha_m", "mg_h")


@dataclass
class AblationConfig:
    """A specific gate configuration to evaluate."""

    name: str
    description: str
    ood_flags: tuple[str, ...] = OOD_FLAGS_PROD
    caveat_flags: tuple[str, ...] = CAVEAT_FLAGS_PROD
    sigma_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(SIGMA_INFLATED_THRESHOLDS_PROD)
    )
    use_kin_ood: bool = True
    # If ``global_sigma_threshold_in_sigma_units`` is set, replace the per-element
    # thresholds with a single threshold scaled to each element's training-σ
    # marginal — i.e. "k×σ_train" for some constant k, applied to all elements.
    global_sigma_threshold_in_sigma_units: float | None = None
    # Per-element caveat overrides. If an element appears here, its tuple of
    # caveat flags REPLACES ``caveat_flags`` for that element only. Used for
    # selective caveats — e.g., "mode_ambiguous on α/M only".
    caveat_flags_per_element: dict[str, tuple[str, ...]] | None = None


def _flags_to_mask(df: pd.DataFrame, flags: tuple[str, ...]) -> np.ndarray:
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    for col in flags:
        if col in df.columns:
            mask |= df[col].fillna(False).to_numpy().astype(bool)
    return mask


def assign_tier(
    df: pd.DataFrame,
    cfg: AblationConfig,
    train_sigma: dict[str, float],
) -> dict[str, np.ndarray]:
    """Recompute per-element release_tier under the given ablation."""
    n = len(df)
    ood = _flags_to_mask(df, cfg.ood_flags)

    kin_ood = np.zeros(n, dtype=bool)
    if cfg.use_kin_ood and "kin_ood_flag" in df.columns:
        kin_ood = df["kin_ood_flag"].fillna(False).to_numpy().astype(bool)

    per_element: dict[str, np.ndarray] = {}
    for elem in ELEMENTS:
        pred = df[ELEMENT_PRED[elem]].to_numpy()
        sigma = df[ELEMENT_SIGMA[elem]].to_numpy()

        # Caveats: per-element override wins over global.
        elem_flags = cfg.caveat_flags
        if cfg.caveat_flags_per_element and elem in cfg.caveat_flags_per_element:
            elem_flags = cfg.caveat_flags_per_element[elem]
        caveat = _flags_to_mask(df, elem_flags)

        # σ-inflation: per-element threshold OR k×σ_train if global is requested.
        if cfg.global_sigma_threshold_in_sigma_units is not None:
            k = cfg.global_sigma_threshold_in_sigma_units
            thr = k * train_sigma[elem]
        else:
            thr = cfg.sigma_thresholds[elem]
        sigma_inflated = np.nan_to_num(sigma, nan=0.0) > thr

        elem_caveat = caveat | sigma_inflated
        if elem in AUX_ASSISTED:
            elem_caveat = elem_caveat | kin_ood

        elem_nan = np.isnan(pred)
        tier = np.ones(n, dtype=np.int8)
        tier[elem_caveat] = 2
        tier[elem_nan | ood] = 3
        per_element[elem] = tier
    return per_element


def metrics_for_config(
    df_full: pd.DataFrame,
    truth: dict[str, np.ndarray],
    test_idx: np.ndarray,
    cfg: AblationConfig,
    train_sigma: dict[str, float],
) -> dict[str, Any]:
    tier_per_elem = assign_tier(df_full, cfg, train_sigma)

    out: dict[str, Any] = {
        "name": cfg.name,
        "description": cfg.description,
        "n_test": int(len(test_idx)),
        "per_element": {},
    }

    for elem in ELEMENTS:
        tier = tier_per_elem[elem][test_idx]
        pred = df_full[ELEMENT_PRED[elem]].to_numpy()[test_idx]
        sigma = df_full[ELEMENT_SIGMA[elem]].to_numpy()[test_idx]
        y = truth[elem][test_idx]
        ok = np.isfinite(pred) & np.isfinite(y) & np.isfinite(sigma)

        block: dict[str, Any] = {}
        for tier_id, tag in [(1, "tier1"), (2, "tier2"), (3, "tier3")]:
            mask = ok & (tier == tier_id)
            n = int(mask.sum())
            block[tag] = {"n": n}
            if n > 0:
                err = pred[mask] - y[mask]
                z = err / np.maximum(sigma[mask], 1e-9)
                block[tag]["rmse"] = float(np.sqrt(np.mean(err**2)))
                block[tag]["bias"] = float(np.mean(err))
                block[tag]["mae"] = float(np.mean(np.abs(err)))
                block[tag]["coverage_1sigma"] = float(np.mean(np.abs(z) <= 1.0))
                block[tag]["coverage_2sigma"] = float(np.mean(np.abs(z) <= 2.0))
                block[tag]["frac_of_test"] = float(n / max(int(ok.sum()), 1))

        # Trustworthy combined (Tier 1 + Tier 2)
        mask = ok & (tier <= 2)
        n = int(mask.sum())
        if n > 0:
            err = pred[mask] - y[mask]
            block["tier12"] = {
                "n": n,
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "frac_of_test": float(n / max(int(ok.sum()), 1)),
            }
        out["per_element"][elem] = block
    return out


def main() -> None:
    print(f"loading predictions: {PREDS}")
    df_pred = pd.read_parquet(PREDS)
    print(f"  {len(df_pred):,} rows × {len(df_pred.columns)} cols")

    print(f"loading features (truth): {FEATURES}")
    feat_cols = ["source_id"] + list(ELEMENT_TRUTH.values())
    feat = pd.read_parquet(FEATURES, columns=feat_cols)
    print(f"  {len(feat):,} rows")
    feat = feat.drop_duplicates(subset="source_id", keep="first")
    df_pred = df_pred.drop_duplicates(subset="source_id", keep="first")
    print(f"  after dedup — preds {len(df_pred):,}, feat {len(feat):,}")

    df = df_pred.merge(feat, on="source_id", how="inner")
    print(f"merged on source_id: {len(df):,} rows")

    # Compute the test-split source IDs from the same stratification used during
    # training. We need the columns the stratifier expects.
    print("rebuilding 70/15/15 stratified split (seed=0)")
    # Stratifier needs: fe_h_apogee, teff_apogee, b_deg (DEFAULT_STRAT_COLS).
    feat_for_split = pd.read_parquet(
        FEATURES, columns=["source_id", "teff_apogee", "fe_h_apogee", "b_deg"]
    )
    split_ids = stratified_split_ids(feat_for_split, seed=0, fracs=(0.70, 0.15, 0.15))
    test_source_ids = set(split_ids["test"])
    test_mask = df["source_id"].isin(test_source_ids).to_numpy()
    test_idx = np.flatnonzero(test_mask)
    print(f"  test split: {len(test_idx):,} stars (of {len(df):,} merged)")

    # Per-element training-σ marginal — needed for the global-σ-threshold ablation.
    train_source_ids = set(split_ids["train"])
    train_mask = df["source_id"].isin(train_source_ids).to_numpy()
    train_idx = np.flatnonzero(train_mask)
    train_sigma = {}
    for elem, truth_col in ELEMENT_TRUTH.items():
        y = df[truth_col].to_numpy()[train_idx]
        ok = np.isfinite(y)
        train_sigma[elem] = float(np.std(y[ok])) if ok.any() else 1.0
    print("per-element training-σ (used for global-σ threshold ablation):")
    for elem, s in train_sigma.items():
        print(f"  {elem}: {s:.4g}")

    # Truth dict.
    truth = {elem: df[ELEMENT_TRUTH[elem]].to_numpy() for elem in ELEMENTS}

    # ---- Ablation configs ----
    configs = [
        AblationConfig(
            name="baseline_prod",
            description="Current production stack — all per-cell gates enabled.",
        ),
        AblationConfig(
            name="no_mode_ambiguous",
            description="Drop mode_ambiguous_flag from caveats.",
            caveat_flags=tuple(c for c in CAVEAT_FLAGS_PROD if c != "mode_ambiguous_flag"),
        ),
        AblationConfig(
            name="no_regime_b",
            description="Drop regime_b_flag from caveats.",
            caveat_flags=tuple(c for c in CAVEAT_FLAGS_PROD if c != "regime_b_flag"),
        ),
        AblationConfig(
            name="no_kin_ood",
            description="Disable kin_ood demotion of aux-assisted elements.",
            use_kin_ood=False,
        ),
        AblationConfig(
            name="no_mahalanobis",
            description="Drop ood_joint_flag (XP-Mahalanobis) from OOD set.",
            ood_flags=tuple(c for c in OOD_FLAGS_PROD if c != "ood_joint_flag"),
        ),
        AblationConfig(
            name="no_latent_support",
            description="Drop latent_support_flag.",
            ood_flags=tuple(c for c in OOD_FLAGS_PROD if c != "latent_support_flag"),
        ),
        AblationConfig(
            name="no_aux_mahalanobis",
            description="Drop ood_aux_mahalanobis_flag.",
            ood_flags=tuple(c for c in OOD_FLAGS_PROD if c != "ood_aux_mahalanobis_flag"),
        ),
        AblationConfig(
            name="no_aux_missing",
            description="Drop aux_missing_any from caveats.",
            caveat_flags=tuple(c for c in CAVEAT_FLAGS_PROD if c != "aux_missing_any"),
        ),
        AblationConfig(
            name="no_dist_prior",
            description="Drop dist_prior_dominated from caveats.",
            caveat_flags=tuple(c for c in CAVEAT_FLAGS_PROD if c != "dist_prior_dominated"),
        ),
        AblationConfig(
            name="no_disagreement",
            description="Drop ood_disagreement_flag from caveats.",
            caveat_flags=tuple(c for c in CAVEAT_FLAGS_PROD if c != "ood_disagreement_flag"),
        ),
        AblationConfig(
            name="sigma_global_0p5x",
            description="Replace per-element σ thresholds with 0.5× σ_train.",
            global_sigma_threshold_in_sigma_units=0.5,
        ),
        AblationConfig(
            name="sigma_global_1x",
            description="Replace per-element σ thresholds with 1.0× σ_train.",
            global_sigma_threshold_in_sigma_units=1.0,
        ),
        AblationConfig(
            name="sigma_global_2x",
            description="Replace per-element σ thresholds with 2.0× σ_train.",
            global_sigma_threshold_in_sigma_units=2.0,
        ),
        AblationConfig(
            name="all_caveats_off",
            description="Drop ALL caveat flags (only OOD demotes).",
            caveat_flags=(),
        ),
        AblationConfig(
            name="all_ood_off",
            description="Drop ALL OOD flags (no Tier-3 demotions from OOD; "
            "only NaN predictions go to Tier 3).",
            ood_flags=(),
        ),
        AblationConfig(
            name="minimal_gates",
            description="Drop ALL caveats AND OOD flags. Only NaN → Tier 3, σ-inflation → Tier 2.",
            ood_flags=(),
            caveat_flags=(),
        ),
        # ---- The three configs that bracket the 2026-04-26 recommendation ----
        AblationConfig(
            name="prod_alpha_tightened",
            description="Production stack with α/M σ-threshold tightened from "
            "0.10 → 0.05 dex (≈0.5×σ_train). Isolates the σ-tighten "
            "effect from the gate-set simplification.",
            sigma_thresholds={
                "teff": 150.0,
                "logg": 0.30,
                "mh": 0.20,
                "alpha_m": 0.05,
                "mg_h": 0.20,
            },
        ),
        AblationConfig(
            name="recommended_no_alpha_tighten",
            description="Recommended gate-set simplification but with α/M "
            "σ-threshold left at 0.10. Isolates the gate-set effect "
            "from the σ-tighten.",
            ood_flags=("ood_joint_flag",),
            caveat_flags=(),
            caveat_flags_per_element={"alpha_m": ("mode_ambiguous_flag",)},
            use_kin_ood=False,
            sigma_thresholds={
                "teff": 150.0,
                "logg": 0.30,
                "mh": 0.20,
                "alpha_m": 0.10,
                "mg_h": 0.20,
            },
        ),
        AblationConfig(
            name="recommended",
            description="Final 2026-04-26 recommendation. Mahalanobis OOD only; "
            "mode-ambiguous demotion on α/M only; kin_ood disabled "
            "(Stream-3 layer concern); all other gates dropped; "
            "α/M σ-threshold tightened to 0.05.",
            ood_flags=("ood_joint_flag",),
            caveat_flags=(),
            caveat_flags_per_element={"alpha_m": ("mode_ambiguous_flag",)},
            use_kin_ood=False,
            sigma_thresholds={
                "teff": 150.0,
                "logg": 0.30,
                "mh": 0.20,
                "alpha_m": 0.05,
                "mg_h": 0.20,
            },
        ),
    ]

    print(f"\nrunning {len(configs)} ablation configs...")
    results: list[dict[str, Any]] = []
    for cfg in configs:
        m = metrics_for_config(df, truth, test_idx, cfg, train_sigma)
        results.append(m)
        # Quick stdout summary of Tier 1 RMSE per element.
        t1_rmse = {
            e: m["per_element"][e].get("tier1", {}).get("rmse", float("nan")) for e in ELEMENTS
        }
        t1_frac = {
            e: m["per_element"][e].get("tier1", {}).get("frac_of_test", 0.0) for e in ELEMENTS
        }
        msg = ", ".join(f"{e}:f={t1_frac[e]:.3f}/RMSE={t1_rmse[e]:.3f}" for e in ELEMENTS)
        print(f"  {cfg.name:24s} {msg}")

    out_path = OUT_DIR / "ablations.json"
    out_path.write_text(
        json.dumps(
            {
                "n_test": int(len(test_idx)),
                "train_sigma": train_sigma,
                "ablations": results,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
