"""Tests for ``scripts/run_pipeline1_inference.py``.

Placement rationale
-------------------
Script tests live under ``tests/scripts/`` (mirroring ``scripts/``) so driver-
level integration tests are separate from library-module unit tests. The
building blocks the driver wires together (``load_ensemble``,
``predict_ensemble``, ``apply_frozen_zscore``, OOD / Regime-B) already have
unit tests under ``tests/xp_abundances/main/`` and ``tests/data/``; these
tests exercise the glue — CLI-free entrypoint ``run_inference``, schema
detection, output column contract, sidecar integrity, atomic-write
semantics, and fingerprint-drift rejection.

Test-infrastructure trade-off
-----------------------------
The production 5-label ensemble ships 5 checkpoints at ~10 MB each with a
139-D feature layout, and fitting the training Mahalanobis OOD bundle
requires reading the ~800 MB Stream-1 parquet. Loading all of that inside
unit tests is both slow and memory-hungry. We instead:

- Build a stand-in 2-member 5-label ensemble on a tiny :class:`FeatureLayout`
  (10-D XP + 2 c0 scalars + 0 residuals + 2 aux = 14-D input), injected via
  the ``layout`` kwarg on :func:`run_inference`.
- Build a tiny reference parquet for the OOD bundle on the same layout
  (~200 rows of gaussian noise in z-score space).
- Emit a minimal provenance JSON with the *real* live basis fingerprint
  from :func:`_build_hermite_basis`, so :func:`verify_basis_fingerprint`
  passes on the happy path and the drift test can flip one character to
  trigger :class:`FrozenStatsMismatchError`.

Production validation of the full-size ensemble against the 1.3M-row Stream 3
input is Thread 3 work, not the unit-test layer.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_pipeline1_inference as driver
from arqueogal.data.frozen_stats import FrozenStatsMismatchError
from arqueogal.data.gaia_xp import _build_hermite_basis
from arqueogal.xp_abundances.main.bimodality import BimodalityGrid
from arqueogal.xp_abundances.main.config import TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelScaler,
    LabelTiers,
)
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import save_checkpoint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIVE_LABELS: tuple[str, ...] = (
    "teff_apogee",
    "logg_apogee",
    "mh_apogee",
    "alpha_m_apogee",
    "mg_h_apogee",
)

TINY_BP_INDICES: tuple[int, ...] = tuple(range(1, 6))  # 5 BP coefs
TINY_RP_INDICES: tuple[int, ...] = tuple(range(1, 6))  # 5 RP coefs
TINY_AUX_COLS: tuple[str, ...] = ("parallax", "g_mag")  # 2 aux
TINY_RESIDUAL_COLS: tuple[str, ...] = ()  # 0 residuals
TINY_INPUT_DIM = 5 + 5 + 2 + 0 + 2  # = 14

# Wider aux layout used for the aux-missingness-flag tests. Covers all three
# flag channels (IR, parallax, extinction) — the minimal tiny layout carries
# only 2 aux cols, which is not enough to exercise the three flag channels.
FLAG_AUX_COLS: tuple[str, ...] = (
    "parallax",
    "parallax_error",
    "j_mag",
    "h_mag",
    "k_mag",
    "w1_mag",
    "w2_mag",
    "av_edenhofer",
    "av_sfd",
    "av_lallement",
)
FLAG_INPUT_DIM = 5 + 5 + 2 + 0 + len(FLAG_AUX_COLS)


def _flag_layout() -> FeatureLayout:
    """Tiny layout augmented with the aux columns needed for the flag tests."""
    return FeatureLayout(
        xp_bp_indices=TINY_BP_INDICES,
        xp_rp_indices=TINY_RP_INDICES,
        xp_scalar_cols=("bp_c0_z", "rp_c0_z"),
        residual_cols=TINY_RESIDUAL_COLS,
        aux_cols=FLAG_AUX_COLS,
    )


def _tiny_layout() -> FeatureLayout:
    return FeatureLayout(
        xp_bp_indices=TINY_BP_INDICES,
        xp_rp_indices=TINY_RP_INDICES,
        xp_scalar_cols=("bp_c0_z", "rp_c0_z"),
        residual_cols=TINY_RESIDUAL_COLS,
        aux_cols=TINY_AUX_COLS,
    )


def _five_label_tiers() -> LabelTiers:
    return LabelTiers.five_label()


def _five_label_block_layout() -> CovarianceBlockLayout:
    return CovarianceBlockLayout(
        block_sizes=(5,),
        n_diagonal_only=0,
        label_order_block=FIVE_LABELS,
        label_order_human=FIVE_LABELS,
    )


def _save_tiny_member(
    ensemble_dir: Path,
    layout: FeatureLayout,
    seed: int,
) -> Path:
    """Write one tiny 5-label checkpoint into ``ensemble_dir/member_seed<seed>/``."""
    tiers = _five_label_tiers()
    block_layout = _five_label_block_layout()
    torch.manual_seed(seed)
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=8,
            trunk_hidden=(16, 8),
            head_hidden=8,
            dropout=0.0,
        )
    )
    log_temp = torch.tensor(0.0)
    cfg = TrainingConfig(
        latent_dim=8,
        trunk_hidden=(16, 8),
        head_hidden=8,
        dropout=0.0,
        use_c0_scalars=True,
    )
    # Build a non-trivial scaler so `save_checkpoint` does not reject it.
    rng = np.random.default_rng(seed)
    Y_fit = np.column_stack(
        [
            rng.normal(4600.0, 250.0, size=256),  # teff
            rng.normal(2.4, 0.5, size=256),  # logg
            rng.normal(-0.25, 0.35, size=256),  # mh
            rng.normal(0.09, 0.10, size=256),  # alpha_m
            rng.normal(-0.15, 0.29, size=256),  # mg_h
        ]
    ).astype(np.float32)
    scaler = LabelScaler.fit(Y_fit, FIVE_LABELS)

    member_dir = ensemble_dir / f"member_seed{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    path = member_dir / f"xp_abundances_main_ensemble_5label_seed{seed}_best.pt"
    save_checkpoint(
        path,
        model=model,
        log_temp=log_temp,
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=scaler,
        seed=seed,
    )
    return path


def _write_frozen_stats_sidecar(
    sidecar_path: Path,
    *,
    fingerprint: str,
    n_ratios: int = 54,
) -> None:
    """Emit a minimal provenance JSON compatible with load_frozen_zscore_stats."""
    coef_block = {
        "bp": {},
        "rp": {},
        "sigma_floor": 1e-20,
        "n_reference_population": 200,
        "reference_population": "test fixture",
    }
    for i in range(1, n_ratios + 1):
        coef_block["bp"][str(i)] = {"mu": 0.0, "sigma": 1.0}
        coef_block["rp"][str(i)] = {"mu": 0.0, "sigma": 1.0}
    payload = {
        "extra": {
            "basis_fingerprint_sha256": fingerprint,
            "c0_zscore_frozen": {
                "bp": {"mu_log10": 0.0, "sigma_log10": 1.0},
                "rp": {"mu_log10": 0.0, "sigma_log10": 1.0},
                "n_reference_population": 200,
                "reference_population": "test fixture",
            },
            "coef_norm_zscore_frozen": coef_block,
        },
    }
    sidecar_path.write_text(json.dumps(payload))


def _synthetic_input_df(
    n: int,
    layout: FeatureLayout,
    *,
    schema: str = "zscored",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "source_id": np.arange(n, dtype=np.int64) + 10**10,
        "b_deg": rng.uniform(-80.0, 80.0, size=n).astype(np.float64),
    }
    for col in layout.bp_coef_cols:
        data[col] = rng.standard_normal(n).astype(np.float32)
    for col in layout.rp_coef_cols:
        data[col] = rng.standard_normal(n).astype(np.float32)
    if schema == "zscored":
        data["bp_c0_z"] = rng.standard_normal(n).astype(np.float32)
        data["rp_c0_z"] = rng.standard_normal(n).astype(np.float32)
    else:
        # Raw schema: the driver z-scores the full 54-wide BP/RP block against
        # the frozen σ table, then subsets to layout indices. Populate the
        # whole 1..54 range so the shape check in apply_frozen_zscore passes.
        data["bp_c0_log"] = rng.standard_normal(n).astype(np.float32)
        data["rp_c0_log"] = rng.standard_normal(n).astype(np.float32)
        for i in range(1, 55):
            bp_col = f"bp_coef_norm_{i}"
            rp_col = f"rp_coef_norm_{i}"
            if bp_col not in data:
                data[bp_col] = rng.standard_normal(n).astype(np.float32)
            if rp_col not in data:
                data[rp_col] = rng.standard_normal(n).astype(np.float32)
    for col in layout.residual_cols:
        data[col] = np.abs(rng.standard_normal(n)).astype(np.float32)
    for col in layout.aux_cols:
        data[col] = rng.standard_normal(n).astype(np.float32)
    # selection_prob passthrough exercised elsewhere — here score from (b, g_mag).
    return pd.DataFrame(data)


def _write_tiny_mode_ambiguous_grid(path: Path) -> Path:
    """Write a tiny synthetic bimodality grid + provenance sidecar for tests.

    The grid spans (3500–6000 K, 0–4 dex log g, −3..+0.5 dex [M/H]) with
    moderate cell sizes, and flags one arbitrary cell as bimodal so the
    query path exercises both flag states. Tests that care about the exact
    flag rate override the grid; the default fixture just needs the
    artefact to exist so the driver can load it.
    """
    teff_edges = np.array([3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0])
    logg_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mh_edges = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5])
    shape = (len(teff_edges) - 1, len(logg_edges) - 1, len(mh_edges) - 1)
    is_bimodal = np.zeros(shape, dtype=bool)
    is_bimodal[2, 2, 3] = True  # one arbitrary flagged cell
    n_per_cell = np.full(shape, 100, dtype=np.int32)
    grid = BimodalityGrid(
        teff_edges=teff_edges,
        logg_edges=logg_edges,
        mh_edges=mh_edges,
        is_bimodal=is_bimodal,
        n_per_cell=n_per_cell,
        min_cell_n=50,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    grid.save(path, provenance={"source": "test fixture"})
    return path


def _install_stub_selection_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit the selection-function artefact — tests don't ship one."""

    def _const_selection_prob(b_deg, g_mag, *, artifact_path=None):
        return np.full(np.asarray(b_deg).shape, 0.85, dtype=np.float64)

    monkeypatch.setattr(driver, "score_selection_prob", _const_selection_prob)


def _seed_fixture_env(
    tmp_path: Path,
    n_rows: int = 24,
) -> dict[str, Path]:
    """Build ensemble_dir, frozen stats sidecar, input parquet, OOD parquet."""
    layout = _tiny_layout()
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir(parents=True)
    for s in (0, 1):
        _save_tiny_member(ensemble_dir, layout, seed=s)

    frozen_stats_path = tmp_path / "stream1.provenance.json"
    fingerprint = _build_hermite_basis()["fingerprint_sha256"]
    _write_frozen_stats_sidecar(frozen_stats_path, fingerprint=fingerprint)

    input_df = _synthetic_input_df(n_rows, layout, seed=1)
    input_path = tmp_path / "input.parquet"
    input_df.to_parquet(input_path, index=False)

    # OOD training parquet — only needs the 108-D block cols, but in the tiny
    # layout that's 10-D; populate accordingly.
    ood_df = pd.DataFrame(
        {
            **{
                c: np.random.default_rng(2).standard_normal(200).astype(np.float32)
                for c in [*layout.bp_coef_cols, *layout.rp_coef_cols]
            },
        }
    )
    ood_path = tmp_path / "ood_train.parquet"
    ood_df.to_parquet(ood_path, index=False)

    grid_path = _write_tiny_mode_ambiguous_grid(tmp_path / "mode_ambiguous_grid.npz")

    output_path = tmp_path / "out.parquet"
    return {
        "ensemble_dir": ensemble_dir,
        "frozen_stats": frozen_stats_path,
        "input": input_path,
        "ood_train": ood_path,
        "mode_ambiguous_grid": grid_path,
        "output": output_path,
        "layout": layout,
        "n_rows": n_rows,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Driver runs on a 24-row synthetic input and writes a valid output."""
    env = _seed_fixture_env(tmp_path, n_rows=24)
    _install_stub_selection_function(monkeypatch)

    prov = driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=8,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    assert env["output"].is_file()
    assert prov["n_input_rows"] == env["n_rows"]
    assert prov["n_output_rows"] == env["n_rows"]


def test_output_schema_columns_and_dtypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every required column exists with the expected dtype."""
    env = _seed_fixture_env(tmp_path, n_rows=16)
    _install_stub_selection_function(monkeypatch)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=8,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    out = pd.read_parquet(env["output"])
    expected_mean_cols = [f"{n}_pred" for n in driver.LABEL_SHORT_NAMES]
    expected_sigma_cols = [f"{n}_sigma" for n in driver.LABEL_SHORT_NAMES]
    expected_epi_cols = [f"{n}_epistemic_var" for n in driver.LABEL_SHORT_NAMES]
    expected_cov_cols = [f"cov_{i}_{j}" for i in range(5) for j in range(i, 5)]
    required = [
        "source_id",
        *expected_mean_cols,
        *expected_sigma_cols,
        *expected_cov_cols,
        *expected_epi_cols,
        "ood_mahalanobis_score",
        "ood_disagreement_flag",
        "ood_joint_flag",
        "regime_b_flag",
        "mode_ambiguous_flag",
        "mode_ambiguous_in_grid",
        "selection_prob",
    ]
    for col in required:
        assert col in out.columns, f"missing output column: {col}"
    assert out["source_id"].dtype == np.int64
    for c in expected_mean_cols + expected_sigma_cols + expected_cov_cols + expected_epi_cols:
        assert out[c].dtype == np.float32, f"{c}: expected float32, got {out[c].dtype}"
    assert out["ood_mahalanobis_score"].dtype == np.float32
    for c in (
        "ood_disagreement_flag",
        "ood_joint_flag",
        "regime_b_flag",
        "mode_ambiguous_flag",
        "mode_ambiguous_in_grid",
    ):
        assert out[c].dtype == bool, f"{c}: expected bool"
    assert out["selection_prob"].dtype == np.float32
    # cov_i_i diagonal must equal sigma_i ** 2 within float32 tolerance.
    for i, name in enumerate(driver.LABEL_SHORT_NAMES):
        diag = out[f"cov_{i}_{i}"].to_numpy()
        sig_sq = out[f"{name}_sigma"].to_numpy() ** 2
        assert np.allclose(diag, sig_sq, rtol=1e-4, atol=1e-4)


def test_row_count_preservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _seed_fixture_env(tmp_path, n_rows=31)
    _install_stub_selection_function(monkeypatch)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=7,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    out = pd.read_parquet(env["output"])
    assert len(out) == 31


def test_provenance_sidecar_validity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _seed_fixture_env(tmp_path, n_rows=12)
    _install_stub_selection_function(monkeypatch)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    sidecar = env["output"].with_suffix(env["output"].suffix + ".provenance.json")
    assert sidecar.is_file()
    prov = json.loads(sidecar.read_text())
    required_keys = {
        "output_file",
        "script",
        "timestamp_utc",
        "git_sha",
        "device",
        "n_input_rows",
        "n_output_rows",
        "input",
        "ensemble",
        "frozen_stats",
        "ood",
        "regime_b",
        "mode_ambiguous",
        "selection_prob",
        "label_tiers",
        "prior_augmented_release_notes",
        "columns",
    }
    missing = required_keys - set(prov.keys())
    assert not missing, f"sidecar missing keys: {missing}"
    # Tier annotations: Option 2.
    assert prov["label_tiers"] == {
        "teff": "T1",
        "logg": "T1-caveat",
        "mh": "T1",
        "alpha_m": "T1",
        "mg_h": "T1",
    }
    # Release notes must contain the exact user-ratified statements.
    assert "2.4x improvement" in prov["prior_augmented_release_notes"]["teff"]
    assert "30% improvement" in prov["prior_augmented_release_notes"]["logg"]
    # SHA-256 recorded for input.
    assert len(prov["input"]["sha256"]) == 64
    # Per-member SHAs recorded.
    assert len(prov["ensemble"]["member_sha256"]) == 2


def test_basis_fingerprint_mismatch_fails_fast_with_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver raises FrozenStatsMismatchError and leaves no partial output."""
    env = _seed_fixture_env(tmp_path, n_rows=12)
    _install_stub_selection_function(monkeypatch)
    # Corrupt the stored fingerprint.
    payload = json.loads(env["frozen_stats"].read_text())
    real = payload["extra"]["basis_fingerprint_sha256"]
    bad = "0" * len(real) if real[0] != "0" else "1" * len(real)
    payload["extra"]["basis_fingerprint_sha256"] = bad
    env["frozen_stats"].write_text(json.dumps(payload))

    with pytest.raises(FrozenStatsMismatchError):
        driver.run_inference(
            ensemble_dir=env["ensemble_dir"],
            input_parquet=env["input"],
            frozen_stats_path=env["frozen_stats"],
            output_parquet=env["output"],
            batch_size=4,
            device=torch.device("cpu"),
            ood_threshold=0.5,
            regime_b_config=None,
            ood_training_parquet=env["ood_train"],
            mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
            layout=env["layout"],
        )
    assert not env["output"].exists()
    sidecar = env["output"].with_suffix(env["output"].suffix + ".provenance.json")
    assert not sidecar.exists()


def test_schema_detection_raw_inputs_apply_frozen_zscore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An input with bp_c0_log/rp_c0_log is detected as raw and z-scored in."""
    env = _seed_fixture_env(tmp_path, n_rows=10)
    _install_stub_selection_function(monkeypatch)
    layout = env["layout"]
    raw_df = _synthetic_input_df(10, layout, schema="raw", seed=7)
    raw_path = tmp_path / "input_raw.parquet"
    raw_df.to_parquet(raw_path, index=False)

    calls: list[str] = []
    orig_apply = driver.apply_frozen_zscore

    def _spy(*args, **kwargs):
        calls.append("called")
        return orig_apply(*args, **kwargs)

    monkeypatch.setattr(driver, "apply_frozen_zscore", _spy)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=raw_path,
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=layout,
    )
    assert calls, "apply_frozen_zscore should have been called for raw input"
    sidecar = env["output"].with_suffix(env["output"].suffix + ".provenance.json")
    prov = json.loads(sidecar.read_text())
    assert prov["input"]["schema_detected"] == "raw"


def test_zscored_inputs_bypass_frozen_zscore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already-z-scored inputs do NOT re-apply frozen z-score."""
    env = _seed_fixture_env(tmp_path, n_rows=10)
    _install_stub_selection_function(monkeypatch)

    calls: list[str] = []
    orig = driver.apply_frozen_zscore

    def _spy(*args, **kwargs):
        calls.append("called")
        return orig(*args, **kwargs)

    monkeypatch.setattr(driver, "apply_frozen_zscore", _spy)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    assert not calls
    sidecar = env["output"].with_suffix(env["output"].suffix + ".provenance.json")
    prov = json.loads(sidecar.read_text())
    assert prov["input"]["schema_detected"] == "zscored"


def test_atomic_write_on_failure_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If writing the parquet raises mid-write, no tmp/leftover file survives."""
    env = _seed_fixture_env(tmp_path, n_rows=10)
    _install_stub_selection_function(monkeypatch)

    def _explode(self, *args, **kwargs):
        raise RuntimeError("simulated interrupt mid-write")

    # Make pandas.DataFrame.to_parquet explode under _atomic_write_parquet.
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _explode)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        driver.run_inference(
            ensemble_dir=env["ensemble_dir"],
            input_parquet=env["input"],
            frozen_stats_path=env["frozen_stats"],
            output_parquet=env["output"],
            batch_size=4,
            device=torch.device("cpu"),
            ood_threshold=0.5,
            regime_b_config=None,
            ood_training_parquet=env["ood_train"],
            mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
            layout=env["layout"],
        )
    assert not env["output"].exists()
    leftovers = [
        p
        for p in env["output"].parent.iterdir()
        if p.name.startswith(env["output"].name + ".") and p.name.endswith(".tmp")
    ]
    assert not leftovers, f"leftover tmp files: {leftovers}"


def test_nonmatching_label_names_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong label set in checkpoint trips the 5-label guard."""
    env = _seed_fixture_env(tmp_path, n_rows=10)
    _install_stub_selection_function(monkeypatch)
    # Build a 3-label checkpoint directly and drop it in.
    ensemble_dir = tmp_path / "ensemble_bad"
    ensemble_dir.mkdir()
    layout = env["layout"]
    bad_tiers = LabelTiers(
        tier1=("teff_apogee", "logg_apogee"),
        tier2=("mh_apogee",),
        tier3=(),
    )
    bad_block_layout = CovarianceBlockLayout(
        block_sizes=(2,),
        n_diagonal_only=1,
        label_order_block=bad_tiers.all_labels,
        label_order_human=bad_tiers.all_labels,
    )
    torch.manual_seed(0)
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=bad_block_layout,
            latent_dim=8,
            trunk_hidden=(16, 8),
            head_hidden=8,
            dropout=0.0,
        )
    )
    cfg = TrainingConfig(
        latent_dim=8,
        trunk_hidden=(16, 8),
        head_hidden=8,
        dropout=0.0,
        use_c0_scalars=True,
    )
    Y_fit = np.random.default_rng(0).standard_normal((64, 3)).astype(np.float32)
    scaler = LabelScaler.fit(Y_fit, bad_tiers.all_labels)
    (ensemble_dir / "member_seed0").mkdir()
    save_checkpoint(
        ensemble_dir / "member_seed0" / "bad_seed0_best.pt",
        model=model,
        log_temp=torch.tensor(0.0),
        cfg=cfg,
        layout=layout,
        tiers=bad_tiers,
        label_scaler=scaler,
        seed=0,
    )
    with pytest.raises(RuntimeError, match="5-label"):
        driver.run_inference(
            ensemble_dir=ensemble_dir,
            input_parquet=env["input"],
            frozen_stats_path=env["frozen_stats"],
            output_parquet=env["output"],
            batch_size=4,
            device=torch.device("cpu"),
            ood_threshold=0.5,
            regime_b_config=None,
            ood_training_parquet=env["ood_train"],
            mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
            layout=layout,
        )


def test_raw_schema_mahalanobis_rate_plausible_on_training_like_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the Phase-3b Mahalanobis-on-raw-schema bug.

    Original failure mode: :func:`_xp_108d_block` read raw ``bp_coef_norm_*``
    columns directly and fed them to the Mahalanobis bundle, whose centroid
    and precision were fit in z-scored space. On Stream-3 raw input this
    produced ``mahalanobis_rate == 1.0``.

    Construction: frozen stats carry non-trivial ``(mu, sigma) = (5.0, 2.0)``
    per BP/RP coefficient. Raw input is drawn from ``N(mu, sigma)``; after
    applying ``apply_frozen_zscore`` it is ``N(0, 1)``. The OOD training
    parquet is also ``N(0, 1)``. Matched distributions ⇒ low flag rate.
    Regression (skipping z-score) ⇒ features N(5, 2) vs z-scored centroid at 0,
    Mahalanobis distances explode, rate → 1.0.
    """
    layout = _tiny_layout()
    # Stand up ensemble + OOD training as usual
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir(parents=True)
    for s in (0, 1):
        _save_tiny_member(ensemble_dir, layout, seed=s)

    # Custom frozen-stats sidecar with non-identity (mu, sigma) per coef.
    # Shift mu far enough that raw (un-z-scored) features sit ~12 σ from the
    # OOD centroid — so the bug path produces Mahalanobis distance ≫ 20 and
    # the fix path produces Mahalanobis ~ sqrt(chi²₁₀) ≈ 3. Wide regression
    # margin both ways.
    fingerprint = _build_hermite_basis()["fingerprint_sha256"]
    coef_mu, coef_sigma = 12.0, 2.0
    coef_block = {
        "bp": {},
        "rp": {},
        "sigma_floor": 1e-20,
        "n_reference_population": 200,
        "reference_population": "regression-fixture-nontrivial",
    }
    for i in range(1, 55):
        coef_block["bp"][str(i)] = {"mu": coef_mu, "sigma": coef_sigma}
        coef_block["rp"][str(i)] = {"mu": coef_mu, "sigma": coef_sigma}
    frozen_stats_path = tmp_path / "stream1.provenance.json"
    frozen_stats_path.write_text(
        json.dumps(
            {
                "extra": {
                    "basis_fingerprint_sha256": fingerprint,
                    "c0_zscore_frozen": {
                        "bp": {"mu_log10": 0.0, "sigma_log10": 1.0},
                        "rp": {"mu_log10": 0.0, "sigma_log10": 1.0},
                        "n_reference_population": 200,
                        "reference_population": "regression-fixture-nontrivial",
                    },
                    "coef_norm_zscore_frozen": coef_block,
                },
            }
        )
    )

    # Raw input drawn from N(coef_mu, coef_sigma) — post-z-score this is N(0, 1)
    n_rows = 80
    rng = np.random.default_rng(13)
    data: dict[str, np.ndarray] = {
        "source_id": np.arange(n_rows, dtype=np.int64) + 10**10,
        "b_deg": rng.uniform(-80.0, 80.0, size=n_rows).astype(np.float64),
        "bp_c0_log": rng.standard_normal(n_rows).astype(np.float32),
        "rp_c0_log": rng.standard_normal(n_rows).astype(np.float32),
    }
    for i in range(1, 55):
        data[f"bp_coef_norm_{i}"] = rng.normal(
            coef_mu,
            coef_sigma,
            size=n_rows,
        ).astype(np.float32)
        data[f"rp_coef_norm_{i}"] = rng.normal(
            coef_mu,
            coef_sigma,
            size=n_rows,
        ).astype(np.float32)
    for col in layout.aux_cols:
        data[col] = rng.standard_normal(n_rows).astype(np.float32)
    input_path = tmp_path / "input_raw.parquet"
    pd.DataFrame(data).to_parquet(input_path, index=False)

    # OOD training: z-scored N(0, 1) — matches input post-z-score. One RNG
    # for all columns so they are statistically independent (naive
    # ``default_rng(seed)`` per column re-seeds and duplicates every column,
    # collapsing the empirical covariance to rank-1 and making the Mahalanobis
    # precision matrix catastrophic).
    ood_rng = np.random.default_rng(2)
    n_ref = 300
    ood_cols = [*layout.bp_coef_cols, *layout.rp_coef_cols]
    ood_df = pd.DataFrame({c: ood_rng.standard_normal(n_ref).astype(np.float32) for c in ood_cols})
    ood_path = tmp_path / "ood_train.parquet"
    ood_df.to_parquet(ood_path, index=False)

    grid_path = _write_tiny_mode_ambiguous_grid(tmp_path / "mode_ambiguous_grid.npz")

    output_path = tmp_path / "out.parquet"
    _install_stub_selection_function(monkeypatch)
    driver.run_inference(
        ensemble_dir=ensemble_dir,
        input_parquet=input_path,
        frozen_stats_path=frozen_stats_path,
        output_parquet=output_path,
        batch_size=16,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=ood_path,
        mode_ambiguous_grid_path=grid_path,
        layout=layout,
    )
    out = pd.read_parquet(output_path)

    # (1) Mahalanobis scores must be finite on a clean raw input
    scores = out["ood_mahalanobis_score"].to_numpy()
    n_finite = int(np.isfinite(scores).sum())
    assert n_finite == n_rows, f"expected all {n_rows} Mahalanobis scores finite, got {n_finite}"

    # (2) The regression bar: Mahalanobis distance median. With the z-score
    # applied, features ~ N(0, 1), match the z-scored OOD centroid, so the
    # distance is distributed as sqrt(chi²₁₀) with median ≈ 3.1. Without the
    # z-score, raw features ~ N(mu, sigma) sit O(sqrt(mu² × 10)) = O(38)
    # distance units from the z-scored centroid; the flag threshold (99th pct
    # of chi²₁₀) is at ~4.8, so the bug path lights up the flag on every row.
    # We check median distance (not flag rate) because (a) it's a quieter
    # signal than flag rate on noisy tiny layouts, (b) joint_flag folds in
    # ensemble disagreement which is near-max on untrained stand-in
    # checkpoints. Threshold 10.0 sits well above chi²₁₀ sqrt-median (~3)
    # and well below the bug's O(38).
    mahal_median = float(np.median(scores))
    assert mahal_median < 10.0, (
        f"Mahalanobis distance median {mahal_median:.2f} too high on matched "
        f"raw input — the z-score is likely not being applied inside "
        f"_xp_108d_block (Phase-3b regression; chi²₁₀ sqrt-median ≈ 3.1)"
    )


def test_schema_detection_requires_one_c0_convention() -> None:
    with pytest.raises(ValueError, match="missing both"):
        driver._detect_input_schema({"source_id"})
    with pytest.raises(ValueError, match="both z-scored and raw"):
        driver._detect_input_schema(
            {"bp_c0_z", "rp_c0_z", "bp_c0_log", "rp_c0_log"},
        )
    assert driver._detect_input_schema({"bp_c0_z", "rp_c0_z"}) == "zscored"
    assert driver._detect_input_schema({"bp_c0_log", "rp_c0_log"}) == "raw"


# ---------------------------------------------------------------------------
# Aux-missingness flag tests + nan_to_num regression
# ---------------------------------------------------------------------------


def _seed_flag_fixture_env(tmp_path: Path, n_rows: int = 8) -> dict[str, Path]:
    """Variant of :func:`_seed_fixture_env` using :func:`_flag_layout`.

    The checkpoint, frozen-stats sidecar, and OOD training parquet are all
    sized to the wider aux-layout so the fixtures exercise the three flag
    channels end-to-end.
    """
    layout = _flag_layout()
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir(parents=True)
    for s in (0, 1):
        _save_tiny_member(ensemble_dir, layout, seed=s)

    frozen_stats_path = tmp_path / "stream1.provenance.json"
    fingerprint = _build_hermite_basis()["fingerprint_sha256"]
    _write_frozen_stats_sidecar(frozen_stats_path, fingerprint=fingerprint)

    input_df = _synthetic_input_df(n_rows, layout, seed=1)
    # Make parallax large enough to not trip the parallax_over_error<5 test
    # unless a row explicitly injects NaN into one of the inputs.
    input_df["parallax"] = np.full(n_rows, 5.0, dtype=np.float32)
    input_df["parallax_error"] = np.full(n_rows, 0.25, dtype=np.float32)  # S/N=20
    # _synthetic_input_df only emits layout.aux_cols; g_mag is not in the
    # flag-layout aux set but the selection-prob scorer needs it. Inject a
    # placeholder column so the scorer path is happy — a monkeypatched
    # stub returns a constant regardless of its value.
    input_df["g_mag"] = np.full(n_rows, 14.0, dtype=np.float32)
    input_path = tmp_path / "input.parquet"
    input_df.to_parquet(input_path, index=False)

    ood_df = pd.DataFrame(
        {
            **{
                c: np.random.default_rng(2).standard_normal(200).astype(np.float32)
                for c in [*layout.bp_coef_cols, *layout.rp_coef_cols]
            },
        }
    )
    ood_path = tmp_path / "ood_train.parquet"
    ood_df.to_parquet(ood_path, index=False)

    grid_path = _write_tiny_mode_ambiguous_grid(tmp_path / "mode_ambiguous_grid.npz")

    output_path = tmp_path / "out.parquet"
    return {
        "ensemble_dir": ensemble_dir,
        "frozen_stats": frozen_stats_path,
        "input_df": input_df,
        "input_path": input_path,
        "ood_train": ood_path,
        "mode_ambiguous_grid": grid_path,
        "output": output_path,
        "layout": layout,
        "n_rows": n_rows,
    }


def _run_with_injected_nans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injections: dict[int, list[str]],
) -> pd.DataFrame:
    """Run the driver against a fixture with NaN injected per row.

    ``injections`` maps row_index -> list of column names to set NaN in
    that row. Returns the output DataFrame.
    """
    env = _seed_flag_fixture_env(tmp_path, n_rows=max(injections.keys()) + 1 + 2)
    _install_stub_selection_function(monkeypatch)
    df = env["input_df"].copy()
    for row_idx, cols in injections.items():
        for col in cols:
            df.loc[row_idx, col] = np.nan
    df.to_parquet(env["input_path"], index=False)

    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input_path"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    return pd.read_parquet(env["output"])


def test_nan_to_num_regression_produces_finite_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One NaN per aux column, one row at a time — all predictions must be finite.

    This is the core regression test for the training-vs-inference
    nan_to_num mismatch. Before the fix, a single NaN in any aux column
    propagated through the trunk to produce NaN μ and Σ for that star.
    """
    # Inject NaN in a different aux col on each of rows 0..9.
    aux_cols_to_probe = FLAG_AUX_COLS  # all 10 aux cols
    injections = {i: [col] for i, col in enumerate(aux_cols_to_probe)}
    out = _run_with_injected_nans(tmp_path, monkeypatch, injections)

    label_cols = [f"{n}_pred" for n in driver.LABEL_SHORT_NAMES]
    sigma_cols = [f"{n}_sigma" for n in driver.LABEL_SHORT_NAMES]
    for col in label_cols + sigma_cols:
        arr = out[col].to_numpy()
        assert np.isfinite(arr).all(), (
            f"{col} has non-finite entries — nan_to_num mismatch regression. "
            f"non-finite mask: {~np.isfinite(arr)}"
        )


def test_ir_missing_flag_truth_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ir_missing_flag is True iff any IR column is NaN for that row."""
    # Row 0: j_mag NaN; row 1: w2_mag NaN; row 2: all clean; row 3: j+h NaN.
    injections = {
        0: ["j_mag"],
        1: ["w2_mag"],
        3: ["j_mag", "h_mag"],
    }
    out = _run_with_injected_nans(tmp_path, monkeypatch, injections)
    flags = out["ir_missing_flag"].to_numpy()
    assert flags[0]
    assert flags[1]
    assert not flags[2]
    assert flags[3]
    # Only IR flag should fire in these rows — parallax and extinction clean.
    assert not out["parallax_missing_flag"].iloc[0]
    assert not out["extinction_missing_flag"].iloc[0]


def test_parallax_missing_flag_truth_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """parallax_missing_flag fires on NaN parallax, NaN error, or low S/N."""
    env = _seed_flag_fixture_env(tmp_path, n_rows=6)
    _install_stub_selection_function(monkeypatch)
    df = env["input_df"].copy()
    # Row 0: parallax NaN (flag should be True).
    df.loc[0, "parallax"] = np.nan
    # Row 1: parallax_error NaN (flag should be True).
    df.loc[1, "parallax_error"] = np.nan
    # Row 2: low S/N — parallax/error = 1 < 5 (flag should be True).
    df.loc[2, "parallax"] = 1.0
    df.loc[2, "parallax_error"] = 1.0
    # Row 3: healthy S/N — parallax/error = 20 > 5 (flag should be False).
    df.loc[3, "parallax"] = 5.0
    df.loc[3, "parallax_error"] = 0.25
    # Row 4: boundary at exactly 5.0 — flag should be False (>= 5 keeps).
    df.loc[4, "parallax"] = 5.0
    df.loc[4, "parallax_error"] = 1.0
    # Row 5: zero-error guard (flag should be True — div-by-zero).
    df.loc[5, "parallax"] = 5.0
    df.loc[5, "parallax_error"] = 0.0
    df.to_parquet(env["input_path"], index=False)

    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input_path"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    out = pd.read_parquet(env["output"])
    flags = out["parallax_missing_flag"].to_numpy()
    assert flags[0]
    assert flags[1]
    assert flags[2]
    assert not flags[3]
    assert not flags[4]
    assert flags[5]


def test_extinction_missing_flag_requires_all_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extinction_missing_flag trips only when ALL THREE A_V maps are NaN."""
    # Row 0: one A_V NaN -> False.
    # Row 1: two A_V NaN -> False.
    # Row 2: all three NaN -> True.
    # Row 3: all clean -> False.
    injections = {
        0: ["av_edenhofer"],
        1: ["av_edenhofer", "av_sfd"],
        2: ["av_edenhofer", "av_sfd", "av_lallement"],
    }
    out = _run_with_injected_nans(tmp_path, monkeypatch, injections)
    flags = out["extinction_missing_flag"].to_numpy()
    assert not flags[0]
    assert not flags[1]
    assert flags[2]
    assert not flags[3]


def test_compound_aux_missing_any_is_logical_or(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aux_missing_any == ir_missing OR parallax_missing OR extinction_missing."""
    # Row 0: IR only; row 1: parallax only; row 2: extinction only;
    # row 3: IR + extinction; row 4: all clean; row 5: IR+parallax+extinction.
    injections = {
        0: ["j_mag"],
        1: ["parallax"],
        2: ["av_edenhofer", "av_sfd", "av_lallement"],
        3: ["k_mag", "av_edenhofer", "av_sfd", "av_lallement"],
        5: ["w1_mag", "parallax", "av_edenhofer", "av_sfd", "av_lallement"],
    }
    out = _run_with_injected_nans(tmp_path, monkeypatch, injections)
    ir = out["ir_missing_flag"].to_numpy()
    plx = out["parallax_missing_flag"].to_numpy()
    ext = out["extinction_missing_flag"].to_numpy()
    compound = out["aux_missing_any"].to_numpy()
    expected = ir | plx | ext
    assert np.array_equal(compound, expected), (
        f"aux_missing_any != OR of three: compound={compound} expected={expected}"
    )
    # Spot-check specific rows:
    assert compound[0] and ir[0] and not plx[0] and not ext[0]
    assert compound[1] and plx[1] and not ir[1] and not ext[1]
    assert compound[2] and ext[2] and not ir[2] and not plx[2]
    assert compound[3] and ir[3] and ext[3]
    assert not compound[4]
    assert compound[5] and ir[5] and plx[5] and ext[5]


def test_ood_joint_flag_independent_of_aux_missingness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting all aux NaN (with clean XP) must NOT trip ood_joint_flag alone.

    Aux-missingness is a DATA-availability signal; the Mahalanobis OOD flag
    is fit on the 108-D XP block only. Setting aux NaN shouldn't light up
    the Mahalanobis half; the ensemble-disagreement half is noise-dependent
    on a random-init tiny ensemble, so we check the Mahalanobis sub-flag
    directly (it's what "OOD on XP distribution" means here).
    """
    env = _seed_flag_fixture_env(tmp_path, n_rows=6)
    _install_stub_selection_function(monkeypatch)
    df = env["input_df"].copy()
    # Row 0: every aux col NaN, XP clean.
    for col in FLAG_AUX_COLS:
        df.loc[0, col] = np.nan
    df.to_parquet(env["input_path"], index=False)

    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input_path"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    out = pd.read_parquet(env["output"])
    # Aux flags all three fire for row 0.
    assert out["ir_missing_flag"].iloc[0]
    assert out["parallax_missing_flag"].iloc[0]
    assert out["extinction_missing_flag"].iloc[0]
    assert out["aux_missing_any"].iloc[0]
    # But the Mahalanobis score on the 108-D XP block is not NaN
    # (XP block is clean) — aux-missingness does not contaminate the
    # XP-distribution OOD metric.
    mahal = out["ood_mahalanobis_score"].iloc[0]
    assert np.isfinite(mahal), f"Mahalanobis score contaminated by aux NaN: {mahal}"
    # And the predictions for this row are finite (nan_to_num worked).
    for name in driver.LABEL_SHORT_NAMES:
        assert np.isfinite(out[f"{name}_pred"].iloc[0]), (
            f"{name}_pred not finite on aux-NaN row — nan_to_num regression"
        )
        assert np.isfinite(out[f"{name}_sigma"].iloc[0])


def test_output_schema_includes_aux_missingness_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four new flag columns are present with bool dtype."""
    env = _seed_flag_fixture_env(tmp_path, n_rows=6)
    _install_stub_selection_function(monkeypatch)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input_path"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    out = pd.read_parquet(env["output"])
    for col in (
        "ir_missing_flag",
        "parallax_missing_flag",
        "extinction_missing_flag",
        "aux_missing_any",
    ):
        assert col in out.columns, f"missing flag column: {col}"
        assert out[col].dtype == bool, f"{col}: expected bool, got {out[col].dtype}"


def test_provenance_records_aux_missingness_definitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sidecar carries the flag definitions, thresholds, rates, and counts."""
    env = _seed_flag_fixture_env(tmp_path, n_rows=6)
    _install_stub_selection_function(monkeypatch)
    df = env["input_df"].copy()
    df.loc[0, "j_mag"] = np.nan  # one IR-missing row for non-zero rate.
    df.to_parquet(env["input_path"], index=False)
    driver.run_inference(
        ensemble_dir=env["ensemble_dir"],
        input_parquet=env["input_path"],
        frozen_stats_path=env["frozen_stats"],
        output_parquet=env["output"],
        batch_size=4,
        device=torch.device("cpu"),
        ood_threshold=0.5,
        regime_b_config=None,
        ood_training_parquet=env["ood_train"],
        mode_ambiguous_grid_path=env["mode_ambiguous_grid"],
        layout=env["layout"],
    )
    sidecar = env["output"].with_suffix(env["output"].suffix + ".provenance.json")
    prov = json.loads(sidecar.read_text())
    aux = prov["aux_missingness"]
    assert set(aux["definitions"].keys()) == {
        "ir_missing_flag",
        "parallax_missing_flag",
        "extinction_missing_flag",
        "aux_missing_any",
    }
    assert aux["definitions"]["ir_missing_flag"]["ir_cols"] == list(driver.IR_COLS)
    assert (
        aux["definitions"]["parallax_missing_flag"]["parallax_over_error_min"]
        == driver.PARALLAX_OVER_ERROR_MIN
    )
    assert aux["definitions"]["extinction_missing_flag"]["extinction_cols"] == list(
        driver.EXTINCTION_COLS,
    )
    # Flag rates non-zero for IR (we injected a NaN) and zero for extinction.
    assert aux["flag_rates"]["ir_missing_flag"] > 0.0
    assert aux["flag_rates"]["extinction_missing_flag"] == 0.0
    # Layout resolution echoes back which configured cols were in aux_cols.
    assert aux["layout_resolution"]["parallax_in_layout"] is True


# Silence unused-import complaints about `os`, `shutil`, `signal` — they exist
# for future interrupt-based atomic-write variants the reviewer may add.
_ = (os, shutil, signal)
