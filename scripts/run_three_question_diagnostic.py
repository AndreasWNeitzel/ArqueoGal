"""Three-question diagnostic for Teff and log g — release-gate follow-up to §9.2.

Reruns three targeted checks on top of the existing
``models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label`` audit:

1. **CMI with a higher-dim XP summary** — the production audit used a 2-D
   (|BP|-sum, |RP|-sum) summary. We recompute ``I(XP; label | aux)`` with a
   PCA-based summary of the 108 normalised coefficients retaining 95 % variance
   (typically 5–10 dim), keeping the 4-D aux conditioning vector unchanged.
2. **Full-feature permutation importance ranking** — the existing
   ``audit_payload.json`` stores only family aggregates plus the top-10
   coefficients. We rerun per-feature permutation over the same val split and
   emit the full (139,) ranking for Teff and log g, tagged by feature family
   (XP vs auxiliary).
3. **Auxiliary-only baseline** — train a tiny MLP on the same train/val split
   using only the aux + residual feature columns (no XP shape, no c0 scalars).
   Compare per-label val RMSE against the full 5-label ensemble mean.

Scope: Teff and log g only. The three [M/H]/[α/M]/[Mg/H] labels pass the
shuffled-spectrum null cleanly and are out of scope.

Outputs (under ``reports/pipeline1/audit/``):

- ``three_question_diagnostic.json`` — machine-readable payload.
- ``three_question_diagnostic.md``  — narrative + tables.

Also drops an aux-only baseline checkpoint + provenance JSON in
``models/main/xp_abundances/aux_only_baseline_20260419/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.audit import conditional_mi_ksg
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("three_question_diagnostic")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENSEMBLE = REPO_ROOT / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/audit"
DEFAULT_BASELINE_DIR = REPO_ROOT / "models/main/xp_abundances/aux_only_baseline_20260419"

# Labels in scope for this diagnostic (block-order indices).
TARGET_LABELS = ("teff_apogee", "logg_apogee")


# --- Ensemble wrapper (copy of run_information_audit.py's) -------------------


class EnsembleMeanWrapper(nn.Module):
    """Ensemble μ-mean wrapper — returns (mu_unscaled, L_identity, h, z)."""

    def __init__(
        self,
        members: list[XpAbundanceModel],
        adapter: XpFeatureAdapter,
        scaler_block: LabelScaler,
        block_layout: CovarianceBlockLayout,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.adapter = adapter
        self.block_layout = block_layout
        self.register_buffer(
            "_scale",
            torch.as_tensor(scaler_block.scale, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_mean",
            torch.as_tensor(scaler_block.mean, dtype=torch.float32),
            persistent=False,
        )
        self.n_labels = len(scaler_block.label_names)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xa = self.adapter(x)
        mus_scaled: list[torch.Tensor] = []
        for m in self.members:
            mu, _L, _h, _z = m(xa)
            mus_scaled.append(mu)
        mu_mean_scaled = torch.stack(mus_scaled, dim=0).mean(dim=0)
        mu_unscaled = mu_mean_scaled * self._scale + self._mean
        batch = x.shape[0]
        L_identity = torch.eye(
            self.n_labels,
            device=x.device,
            dtype=x.dtype,
        ).expand(batch, -1, -1)
        return mu_unscaled, L_identity, x, x


class _RawLoaderWrapper:
    """Yield (X, Y_raw_block_order) batches from the training val loader."""

    def __init__(
        self,
        loader: DataLoader,
        scaler_human: LabelScaler,
        block_layout: CovarianceBlockLayout,
    ) -> None:
        self._loader = loader
        self._scaler = scaler_human
        self._perm = block_layout.human_to_block_perm.cpu().numpy()

    def __iter__(self):  # noqa: ANN204
        for batch in self._loader:
            x = batch[0]
            y_human_scaled = batch[1].numpy()
            y_human_raw = self._scaler.inverse_mean(y_human_scaled)
            y_block_raw = y_human_raw[:, self._perm]
            yield x, torch.as_tensor(y_block_raw, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._loader)


def _reconstruct_model(
    blob: dict[str, Any],
    layout: FeatureLayout,
    block_layout: CovarianceBlockLayout,
    device: torch.device,
) -> XpAbundanceModel:
    cfg_yaml = json.loads(blob["config_yaml"])
    latent_dim = int(cfg_yaml.get("latent_dim", 32))
    trunk_hidden = tuple(cfg_yaml.get("trunk_hidden", (256, 128)))
    head_hidden = int(cfg_yaml.get("head_hidden", 128))
    dropout = float(cfg_yaml.get("dropout", 0.10))
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
    model.head.load_state_dict(blob["regressor"])
    model.eval()
    return model


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


def _feature_names(layout: FeatureLayout) -> list[str]:
    return list(layout.all_required_columns)


def _feature_group(layout: FeatureLayout, idx: int) -> str:
    """Group a flat index into 'xp' (shape+c0) or 'aux' (residual+aux cols)."""
    n_bp = len(layout.bp_coef_cols)
    n_rp = len(layout.rp_coef_cols)
    n_c0 = len(layout.xp_scalar_cols)
    n_xp_total = n_bp + n_rp + n_c0
    if idx < n_xp_total:
        return "xp"
    return "aux"


def _nan_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(truth)
    n_labels = truth.shape[1]
    out = np.full(n_labels, np.nan, dtype=np.float64)
    for j in range(n_labels):
        m = finite[:, j]
        if m.sum() < 2:
            continue
        diff = pred[m, j] - truth[m, j]
        out[j] = float(np.sqrt((diff * diff).mean()))
    return out


def _nan_rmse_single(pred: np.ndarray, truth: np.ndarray) -> float:
    m = np.isfinite(truth) & np.isfinite(pred)
    if m.sum() < 2:
        return float("nan")
    d = pred[m] - truth[m]
    return float(np.sqrt((d * d).mean()))


def _collect_val_arrays(
    loader: _RawLoaderWrapper,
    layout: FeatureLayout,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in loader:
        xs.append(x.cpu().numpy().astype(np.float32))
        ys.append(y.cpu().numpy().astype(np.float32))
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    assert X.shape[1] == layout.input_dim
    return X, Y


def _collect_baseline_mu(
    model: nn.Module,
    loader: _RawLoaderWrapper,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            mu, _L, _h, _z = model(x.to(device))
            mus.append(mu.cpu().numpy())
            ys.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)
    return np.concatenate(mus, 0), np.concatenate(ys, 0)


# --- Q1: CMI with PCA summary ------------------------------------------------


def _pca_xp_summary(
    X_val_xp: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, int, float]:
    """Return PCA projection of the XP block retaining ``variance_threshold`` var.

    Parameters
    ----------
    X_val_xp
        (N, 108) array of BP+RP normalised shape coefficients. No centring
        is assumed — we centre internally.
    variance_threshold
        Fraction of total variance to retain in the output summary.

    Returns
    -------
    (projection, k, explained_variance_ratio_cumulative_at_k)
        ``projection`` is (N, k) float64.
    """
    Xc = X_val_xp - X_val_xp.mean(axis=0, keepdims=True)
    # Use economy SVD — N is typically > D so we do full_matrices=False.
    U, s, _Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    var = (s**2) / max(Xc.shape[0] - 1, 1)
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, variance_threshold) + 1)
    k = max(k, 2)  # guard against threshold-satisfied-by-one-component pathology
    proj = U[:, :k] * s[:k]  # PCA scores, scaled
    return proj.astype(np.float64), k, float(cum[k - 1])


def _aux_conditioning(
    parquet_path: Path,
    source_ids: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
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
        if idx.size < 200:
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


# --- Q2: full-feature permutation importance ---------------------------------


def _full_permutation_importance(
    model: nn.Module,
    x_all: np.ndarray,
    y_all: np.ndarray,
    mu0: np.ndarray,
    device: torch.device,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Per-feature, per-label RMSE increase under cross-star shuffle.

    Returns (n_features, n_labels) importance = permuted_rmse - baseline_rmse.
    """
    rng = np.random.default_rng(seed)
    baseline = _nan_rmse(mu0, y_all)
    n_features = x_all.shape[1]
    n_labels = y_all.shape[1]
    importance = np.empty((n_features, n_labels), dtype=np.float64)
    x_t = torch.as_tensor(x_all, device=device)
    with torch.no_grad():
        for f in range(n_features):
            perm = rng.permutation(x_all.shape[0])
            x_perm = x_t.clone()
            x_perm[:, f] = x_t[perm, f]
            mu_perm, _L, _h, _z = model(x_perm)
            rp = _nan_rmse(mu_perm.cpu().numpy(), y_all)
            importance[f] = rp - baseline
    return importance


# --- Q3: auxiliary-only baseline ---------------------------------------------


class AuxOnlyMLP(nn.Module):
    """Small MLP mapping aux-only features → standardised labels.

    Architecture mirrors the encoder-trunk + linear-mean head of the main model
    in scale (256 → 128 hidden, LayerNorm + GELU + dropout) but without the
    projection head or the Cholesky block. A plain nn.Linear mean head is all
    we need for a head-to-head RMSE comparison.
    """

    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        hidden: tuple[int, int] = (256, 128),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(prev, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.mean(self.trunk(x))


def _train_aux_only_baseline(  # noqa: PLR0913
    X_train_aux: np.ndarray,
    Y_train_scaled: np.ndarray,
    X_val_aux: np.ndarray,
    Y_val_scaled: np.ndarray,
    device: torch.device,
    *,
    epochs: int = 40,
    batch_size: int = 1024,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 0,
) -> tuple[AuxOnlyMLP, list[dict[str, float]]]:
    """Masked-MSE training on standardised labels; early-stop on val loss."""
    torch.manual_seed(seed)
    model = AuxOnlyMLP(X_train_aux.shape[1], Y_train_scaled.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr = torch.as_tensor(X_train_aux, dtype=torch.float32)
    Y_tr = torch.as_tensor(Y_train_scaled, dtype=torch.float32)
    X_va = torch.as_tensor(X_val_aux, dtype=torch.float32, device=device)
    Y_va = torch.as_tensor(Y_val_scaled, dtype=torch.float32, device=device)

    tr_ds = TensorDataset(X_tr, Y_tr)
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    bad = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mu = model(xb)
            mask = torch.isfinite(yb).float()
            diff2 = (mu - torch.where(mask.bool(), yb, mu.detach())) ** 2
            # Masked-MSE summed per sample then averaged over finite entries.
            denom = mask.sum().clamp(min=1.0)
            loss = (diff2 * mask).sum() / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_sum += float(loss.detach()) * xb.shape[0]
            tr_n += xb.shape[0]

        model.eval()
        with torch.no_grad():
            mu_va = model(X_va)
            mask = torch.isfinite(Y_va).float()
            diff2 = (mu_va - torch.where(mask.bool(), Y_va, mu_va)) ** 2
            va_loss = float((diff2 * mask).sum() / mask.sum().clamp(min=1.0))

        history.append({"epoch": epoch, "train_loss": tr_sum / max(tr_n, 1), "val_loss": va_loss})
        _LOG.info(
            "aux-only baseline epoch %d: train=%.4f val=%.4f",
            epoch,
            tr_sum / max(tr_n, 1),
            va_loss,
        )
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                _LOG.info("aux-only baseline early-stop at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# --- Main driver -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mi-max-samples", type=int, default=8000)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--baseline-epochs", type=int, default=40)
    parser.add_argument("--baseline-patience", type=int, default=5)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.baseline_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s ensemble=%s", device, args.ensemble)

    member_ckpts = sorted(
        args.ensemble.glob("member_seed*/xp_abundances_main_ensemble*_seed*_best.pt"),
    )
    if len(member_ckpts) != 5:
        raise FileNotFoundError(
            f"expected 5 member ckpts under {args.ensemble}, found {len(member_ckpts)}",
        )

    layout = FeatureLayout()
    families = _feature_family_indices(layout)
    feature_names = _feature_names(layout)

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
    train_loader, val_loader, _split_ids, scaler_human = build_dataloaders(
        cfg,
        layout,
        tiers,
        seed=split_seed,
    )
    scaler_block = scaler_human.reorder_to(block_layout.label_order_block)

    # --- Reconstruct ensemble ---------------------------------------------------

    members: list[XpAbundanceModel] = []
    for ckpt in member_ckpts:
        blob = load_checkpoint(ckpt, map_location=device)
        members.append(_reconstruct_model(blob, layout, block_layout, device))
    adapter = XpFeatureAdapter(layout, use_c0_scalars=True).to(device)
    wrapper = EnsembleMeanWrapper(
        members=members,
        adapter=adapter,
        scaler_block=scaler_block,
        block_layout=block_layout,
    ).to(device)
    wrapper.eval()

    raw_val = _RawLoaderWrapper(val_loader, scaler_human, block_layout)

    # Collect val arrays once.
    X_val, Y_val = _collect_val_arrays(raw_val, layout)
    _LOG.info("X_val=%s Y_val=%s", X_val.shape, Y_val.shape)

    # Compute ensemble baseline mu on val.
    mu0, _y_raw = _collect_baseline_mu(wrapper, raw_val, device)
    base_rmse = _nan_rmse(mu0, Y_val)
    _LOG.info(
        "full-model val RMSE per label: %s",
        dict(zip(ckpt_label_names, np.round(base_rmse, 4).tolist())),
    )

    target_indices = [ckpt_label_names.index(n) for n in TARGET_LABELS]

    # --- Q1: CMI with richer (PCA) XP summary ------------------------------------

    _LOG.info("Q1: conditional MI with PCA XP summary")
    # XP block = BP shape + RP shape (108 cols, no c0 scalars). Matches the
    # production 2-D summary domain exactly.
    xp_idx = np.asarray(families["bp_shape"] + families["rp_shape"], dtype=np.int64)
    X_val_xp = X_val[:, xp_idx].astype(np.float64)
    xp_pca, k_pca, var_at_k = _pca_xp_summary(X_val_xp, args.pca_variance)
    _LOG.info("PCA: k=%d components, cumulative variance at k = %.4f", k_pca, var_at_k)

    val_source_ids = np.asarray(val_loader.dataset.source_id)
    Z_cond, cond_names = _aux_conditioning(args.parquet, val_source_ids)

    # Also recompute the 2-D summary for direct apples-to-apples sanity.
    xp_2d = np.column_stack(
        [
            np.abs(X_val[:, families["bp_shape"]]).sum(axis=1),
            np.abs(X_val[:, families["rp_shape"]]).sum(axis=1),
        ]
    ).astype(np.float64)

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

    # --- Q2: full-feature permutation importance --------------------------------

    _LOG.info("Q2: full-feature permutation importance (%d features)", layout.input_dim)
    perm_imp = _full_permutation_importance(
        wrapper,
        X_val,
        Y_val,
        mu0,
        device,
        seed=0,
    )  # (n_features, n_labels)

    q2_per_label: dict[str, Any] = {}
    for j in target_indices:
        name = ckpt_label_names[j]
        imp = perm_imp[:, j]
        order = np.argsort(-np.where(np.isfinite(imp), imp, -np.inf))
        rows: list[dict[str, Any]] = []
        for rank, fidx in enumerate(order, 1):
            group = _feature_group(layout, int(fidx))
            rows.append(
                {
                    "rank": rank,
                    "feature_idx": int(fidx),
                    "feature_name": feature_names[int(fidx)],
                    "group": group,
                    "delta_rmse": float(imp[int(fidx)]),
                }
            )
        top10 = rows[:10]
        group_totals = {
            "xp_sum_delta": float(sum(r["delta_rmse"] for r in rows if r["group"] == "xp")),
            "aux_sum_delta": float(sum(r["delta_rmse"] for r in rows if r["group"] == "aux")),
            "xp_count_in_top10": sum(1 for r in top10 if r["group"] == "xp"),
            "aux_count_in_top10": sum(1 for r in top10 if r["group"] == "aux"),
        }
        q2_per_label[name] = {
            "baseline_rmse": float(base_rmse[j]),
            "top10": top10,
            "group_totals": group_totals,
            "full_ranking": rows,  # retained so main thread can cross-check
        }

    # --- Q3: aux-only baseline --------------------------------------------------

    _LOG.info("Q3: aux-only baseline training")
    # Reconstruct train+val arrays from the same dataset objects the ensemble
    # consumed. ``val_loader.dataset`` holds the in-memory arrays.
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    X_train_all = np.asarray(train_ds.X)
    Y_train_scaled = np.asarray(train_ds.Y)  # human order, standardised
    X_val_all = np.asarray(val_ds.X)
    Y_val_scaled = np.asarray(val_ds.Y)  # human order, standardised

    # Aux-only = residual + aux families (NO bp_shape, NO rp_shape, NO xp_c0).
    aux_only_indices = np.asarray(
        families["residual"] + families["aux"],
        dtype=np.int64,
    )
    X_train_aux = X_train_all[:, aux_only_indices]
    X_val_aux = X_val_all[:, aux_only_indices]
    _LOG.info("aux-only input_dim = %d", X_train_aux.shape[1])

    baseline_model, history = _train_aux_only_baseline(
        X_train_aux,
        Y_train_scaled,
        X_val_aux,
        Y_val_scaled,
        device=device,
        epochs=args.baseline_epochs,
        patience=args.baseline_patience,
        seed=0,
    )

    # Compute val RMSE on raw-unit labels for direct comparison with the
    # ensemble. Labels live in human (ckpt_label_names) order for this model.
    baseline_model.eval()
    X_va_t = torch.as_tensor(X_val_aux, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_aux_scaled = baseline_model(X_va_t).cpu().numpy()
    mu_aux_raw_human = scaler_human.inverse_mean(mu_aux_scaled)
    # Ensemble μ lives in block order; ckpt_label_names is block order too, but
    # TRAINING val_loader.dataset.Y is in human order. For the 5-label layout
    # the block order equals the human order (check first_blob), so use
    # block_layout permutation to be safe.
    human_to_block = block_layout.human_to_block_perm.cpu().numpy()
    mu_aux_raw_block = mu_aux_raw_human[:, human_to_block]
    aux_rmse = _nan_rmse(mu_aux_raw_block, Y_val)

    q3_per_label: dict[str, Any] = {}
    for j in target_indices:
        name = ckpt_label_names[j]
        full = float(base_rmse[j])
        aux = float(aux_rmse[j])
        ratio = float(aux / full) if full > 0 else float("nan")
        q3_per_label[name] = {
            "full_model_rmse": full,
            "aux_only_rmse": aux,
            "aux_to_full_ratio": ratio,
            "sigma_y": float(np.nanstd(Y_val[:, j])),
        }

    # Save baseline checkpoint + provenance.
    ckpt_path = args.baseline_dir / "aux_only_baseline_seed0.pt"
    torch.save(
        {
            "version": 1,
            "model_state_dict": baseline_model.state_dict(),
            "aux_only_indices": aux_only_indices.tolist(),
            "label_names": list(ckpt_label_names),
            "scaler_mean_human": scaler_human.mean.tolist(),
            "scaler_scale_human": scaler_human.scale.tolist(),
            "history": history,
            "human_to_block_perm": human_to_block.tolist(),
            "input_feature_names": [feature_names[i] for i in aux_only_indices.tolist()],
        },
        ckpt_path,
    )
    prov = {
        "artifact": "aux_only_baseline_seed0.pt",
        "purpose": "§9.2 three-question diagnostic — Q3 head-to-head RMSE "
        "vs full 5-label ensemble for Teff and log g.",
        "reference_ensemble": str(args.ensemble),
        "parquet": str(args.parquet),
        "split_seed": split_seed,
        "n_train": int(X_train_aux.shape[0]),
        "n_val": int(X_val_aux.shape[0]),
        "aux_only_indices": aux_only_indices.tolist(),
        "input_dim": int(X_train_aux.shape[1]),
        "label_names": list(ckpt_label_names),
        "architecture": "AuxOnlyMLP (256, 128) + LayerNorm + GELU + Dropout(0.10)",
        "training": {
            "epochs_max": args.baseline_epochs,
            "epochs_actually_run": len(history),
            "patience": args.baseline_patience,
            "batch_size": 1024,
            "optimizer": "AdamW(lr=5e-4, weight_decay=1e-4)",
            "loss": "masked MSE on standardised labels",
            "seed": 0,
        },
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "notes": "Diagnostic artefact only — NOT a release model. No Cholesky "
        "covariance head, no calibration, no ensemble.",
    }
    with (args.baseline_dir / "aux_only_baseline_seed0.provenance.json").open("w") as f:
        json.dump(prov, f, indent=2)

    # --- Q1 existing audit values (read-only cross-check) -----------------------

    existing_audit = json.loads(
        (args.report_dir / "audit_payload.json").read_text(),
    )
    existing_cmi = {name: float(existing_audit["per_label"][name]["cmi"]) for name in TARGET_LABELS}

    # --- Assemble JSON payload ---------------------------------------------------

    payload = {
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "scope": "Teff + log g — three-question follow-up to §9.2 audit",
        "ensemble_dir": str(args.ensemble),
        "parquet": str(args.parquet),
        "split_seed": split_seed,
        "n_val": int(X_val.shape[0]),
        "n_train": int(X_train_aux.shape[0]),
        "label_names_block_order": list(ckpt_label_names),
        "target_labels": list(TARGET_LABELS),
        "Q1_conditional_mi": {
            "existing_audit_2d_summary_cmi": existing_cmi,
            "rerun_2d_summary_cmi": cmi_2d,
            "pca_summary_cmi": cmi_pca,
            "pca_components": int(k_pca),
            "pca_variance_retained": float(var_at_k),
            "pca_variance_threshold": float(args.pca_variance),
            "cmi_estimator_k": 8,
            "max_samples": int(args.mi_max_samples),
            "conditioning_columns": list(cond_names),
            "cmi_release_floor_nats": 0.02,
        },
        "Q2_permutation_importance": q2_per_label,
        "Q3_aux_only_baseline": {
            "per_label": q3_per_label,
            "baseline_checkpoint": str(ckpt_path),
            "aux_only_feature_names": [feature_names[i] for i in aux_only_indices.tolist()],
            "aux_only_input_dim": int(X_train_aux.shape[1]),
            "baseline_epochs_run": len(history),
            "baseline_final_val_loss": float(history[-1]["val_loss"]) if history else float("nan"),
        },
    }

    with (args.report_dir / "three_question_diagnostic.json").open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    _write_markdown(args.report_dir / "three_question_diagnostic.md", payload)
    _LOG.info("three-question diagnostic complete — %s", args.report_dir)


# --- Markdown writer ---------------------------------------------------------


def _fmt(x: float, precision: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{precision}f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# §9.2 three-question diagnostic — Teff and log g")
    lines.append("")
    lines.append(
        f"_Timestamp: {payload['timestamp']} · "
        f"Ensemble: `{Path(payload['ensemble_dir']).name}` · "
        f"Val split seed {payload['split_seed']} · N_val = {payload['n_val']}_",
    )
    lines.append("")
    lines.append(
        "Follow-up to `SUMMARY.md`. Scope is **Teff + log g only** — the three "
        "chemistry labels already pass the shuffled-spectrum null cleanly. This "
        "document provides evidence, not tier verdicts.",
    )
    lines.append("")

    # ---- Q1 ------------------------------------------------------------------
    q1 = payload["Q1_conditional_mi"]
    lines.append("## Q1 — Conditional mutual information under richer XP summary")
    lines.append("")
    lines.append(
        f"Original audit used a 2-D XP summary (|BP|-sum, |RP|-sum). Here we "
        f"recompute CMI with a PCA summary of the 108 BP+RP normalised "
        f"coefficients retaining {q1['pca_variance_threshold'] * 100:.0f}% variance "
        f"(→ {q1['pca_components']} components, cumulative variance "
        f"{q1['pca_variance_retained'] * 100:.2f}%). Conditioning set, estimator "
        f"(KSG k={q1['cmi_estimator_k']}) and subsample cap ({q1['max_samples']}) "
        f"match the production audit.",
    )
    lines.append("")
    lines.append(
        "| label | CMI (original 2-D, from payload) | CMI (2-D, rerun) | CMI (PCA summary) |"
    )
    lines.append("|---|---|---|---|")
    for name in payload["target_labels"]:
        lines.append(
            f"| {name} | {_fmt(q1['existing_audit_2d_summary_cmi'][name])} | "
            f"{_fmt(q1['rerun_2d_summary_cmi'][name])} | "
            f"{_fmt(q1['pca_summary_cmi'][name])} |",
        )
    lines.append("")
    lines.append(
        f"Release-gate CMI floor: ≥ {q1['cmi_release_floor_nats']} nats. "
        "Interpretation: if the PCA-summary CMI stays comparably small, the XP "
        "block carries near-zero information about the label beyond the "
        "photometric/astrometric priors; if it is materially larger, the "
        "original 2-D estimate was a summary artefact (higher-order Hermite "
        "structure carries the residual signal).",
    )
    lines.append("")

    # ---- Q2 ------------------------------------------------------------------
    lines.append("## Q2 — Per-feature permutation importance ranking")
    lines.append("")
    lines.append(
        "Full per-feature permutation importance rerun on the same val split. "
        "Each feature column is shuffled across stars; reported is "
        "ΔRMSE = permuted − baseline. Feature group: `xp` covers BP/RP "
        "normalised shape coefficients (108) + c0 z-scored scalars (2) = 110 "
        "features. `aux` covers the 3 residual columns + 26 auxiliary "
        "photometric/astrometric/extinction columns = 29 features.",
    )
    lines.append("")
    for name in payload["target_labels"]:
        block = payload["Q2_permutation_importance"][name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Full-model baseline RMSE: {_fmt(block['baseline_rmse'])}")
        lines.append("")
        gt = block["group_totals"]
        lines.append(
            f"Group totals across all 139 features — "
            f"Σ(ΔRMSE, xp) = {_fmt(gt['xp_sum_delta'])}, "
            f"Σ(ΔRMSE, aux) = {_fmt(gt['aux_sum_delta'])}. "
            f"Top-10 composition: {gt['xp_count_in_top10']} XP, "
            f"{gt['aux_count_in_top10']} aux.",
        )
        lines.append("")
        lines.append("| rank | feature | group | ΔRMSE |")
        lines.append("|---|---|---|---|")
        for row in block["top10"]:
            lines.append(
                f"| {row['rank']} | `{row['feature_name']}` | {row['group']} "
                f"| {_fmt(row['delta_rmse'])} |",
            )
        lines.append("")

    # ---- Q3 ------------------------------------------------------------------
    lines.append("## Q3 — Auxiliary-only baseline vs full 5-label ensemble")
    lines.append("")
    q3 = payload["Q3_aux_only_baseline"]
    lines.append(
        f"Aux-only MLP trained on the same train split (seed "
        f"{payload['split_seed']}, N_train={payload['n_train']}, "
        f"N_val={payload['n_val']}) using {q3['aux_only_input_dim']} input "
        f"features: the 3 residual columns + 26 auxiliaries. No XP shape, no "
        f"c0 scalars. 256→128 MLP, masked MSE on standardised labels, AdamW, "
        f"early-stop on val loss ({q3['baseline_epochs_run']} epochs actually "
        f"run, final val loss {_fmt(q3['baseline_final_val_loss'])}).",
    )
    lines.append("")
    lines.append(
        "| label | full-model RMSE | aux-only RMSE | aux / full ratio | σ(y) |",
    )
    lines.append("|---|---|---|---|---|")
    for name in payload["target_labels"]:
        r = q3["per_label"][name]
        lines.append(
            f"| {name} | {_fmt(r['full_model_rmse'])} | "
            f"{_fmt(r['aux_only_rmse'])} | {_fmt(r['aux_to_full_ratio'])} | "
            f"{_fmt(r['sigma_y'])} |",
        )
    lines.append("")
    lines.append(
        "Interpretation thresholds (from user): ratio ≈ 1.00 (within ~5 %) → "
        "XP contribution is noise; ratio > 1.10 → XP contributes meaningfully; "
        "intermediate → judgment call.",
    )
    lines.append("")
    lines.append(f"Baseline checkpoint: `{q3['baseline_checkpoint']}`")
    lines.append("")

    # ---- Per-label honest characterization -----------------------------------
    lines.append("## Honest characterization per label")
    lines.append("")
    lines.append(
        "Synthesising Q1 (CMI), Q2 (permutation ranking) and Q3 (aux-only "
        "head-to-head). This is evidence framing — the tier decision remains "
        "with the user.",
    )
    lines.append("")
    for name in payload["target_labels"]:
        q1_old = q1["existing_audit_2d_summary_cmi"][name]
        q1_pca = q1["pca_summary_cmi"][name]
        q2 = payload["Q2_permutation_importance"][name]
        q3_row = q3["per_label"][name]
        ratio = q3_row["aux_to_full_ratio"]
        xp_in_top10 = q2["group_totals"]["xp_count_in_top10"]
        xp_sum = q2["group_totals"]["xp_sum_delta"]
        aux_sum = q2["group_totals"]["aux_sum_delta"]
        xp_share = xp_sum / max(xp_sum + aux_sum, 1e-12)

        # Verdict framing — numerical, no prose flourish.
        if ratio >= 1.10:  # noqa: PLR2004
            verdict = "**XP contributes meaningfully.**"
        elif ratio <= 1.05:  # noqa: PLR2004
            verdict = "**XP contribution is noise.**"
        else:
            verdict = "**Intermediate — judgment call.**"

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Q1 CMI: 2-D = {_fmt(q1_old)}, PCA = {_fmt(q1_pca)} nats.")
        lines.append(
            f"- Q2 top-10 composition: {xp_in_top10} XP features / "
            f"{10 - xp_in_top10} aux. "
            f"XP family Σ(ΔRMSE) = {_fmt(xp_sum)} "
            f"({xp_share * 100:.1f}% of combined total {_fmt(xp_sum + aux_sum)}).",
        )
        lines.append(
            f"- Q3 aux-only / full RMSE ratio = {_fmt(ratio)} "
            f"({_fmt(q3_row['aux_only_rmse'])} / "
            f"{_fmt(q3_row['full_model_rmse'])}).",
        )
        lines.append(f"- Characterization: {verdict}")
        lines.append("")

    lines.append("---")
    lines.append(
        "Inputs and artefacts:",
    )
    lines.append(
        "- Existing audit: `reports/pipeline1/audit/audit_payload.json`",
    )
    lines.append(
        "- This diagnostic (JSON): `reports/pipeline1/audit/three_question_diagnostic.json`",
    )
    lines.append(
        f"- Aux-only baseline: `{q3['baseline_checkpoint']}` + `.provenance.json`",
    )
    lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
