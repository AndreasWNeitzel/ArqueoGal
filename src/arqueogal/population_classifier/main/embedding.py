"""Parametric UMAP embedding — research_brief §10.1.

Pipeline 2 replaces classical UMAP with a **neural-network parametric UMAP**
(Sainburg, McInnes & Gentner 2021, arXiv:2009.12981). Three things the
parametric variant buys us over classical UMAP that matter for D-Cat-d:

1. **Out-of-sample transform** — classical UMAP can only embed the training
   set. Parametric UMAP is a learned ``f_θ: R^D → R^d`` that runs on any new
   star at inference cost = one forward pass.
2. **MC-ensemble friendly** — drawing ``N_MC = 50`` perturbed feature
   realisations and re-embedding each costs 50 forward passes per star, not
   50 full UMAP refits (would be a ~2-day job on RTX 3060).
3. **Differentiable** — downstream gradient-based diagnostics and ensembling
   can flow through the embedding.

The loss is UMAP's standard cross-entropy between the high-D fuzzy
simplicial set (computed once by ``umap-learn``) and the low-D fuzzy graph
implied by the current embedding. We use negative sampling — sampling pairs
uniformly at random as implied negatives — exactly as in McInnes+2018 and
Sainburg+2021.

The module is intentionally thin: we delegate the fuzzy-simplicial-set
construction to ``umap.umap_.fuzzy_simplicial_set`` (CPU only; the graph
build is fast) and do the neural-net training in PyTorch so it stays on
GPU when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

_MIN_DIST_FLOOR: float = 1e-4  # same as low-D distance floor used by umap-learn


@dataclass(frozen=True, slots=True)
class ParametricUMAPConfig:
    """Knobs exposed to callers.

    ``n_neighbors``, ``min_dist``, and ``n_components`` have the same
    semantics as classical UMAP. ``hidden_dims`` controls the encoder trunk;
    keep it modest since we run on CPU in tests and RTX 3060 for real runs.
    """

    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.10
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    n_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    negative_sample_rate: int = 5
    seed: int = 0


class ParametricUMAPEncoder(nn.Module):
    """MLP encoder ``f_θ: R^D → R^d``.

    Identical shape to the trunk used by Pipeline-1 (LayerNorm + GELU + MLP),
    so implementation review patterns transfer. The final layer is a plain
    ``Linear`` — no activation — since the UMAP loss expects real-valued
    embedding coordinates.
    """

    def __init__(
        self, input_dim: int, n_components: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_components))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _umap_loss(
    z_i: torch.Tensor, z_j: torch.Tensor, z_neg: torch.Tensor,
    *, a: float, b: float,
) -> torch.Tensor:
    """UMAP attractive + repulsive cross-entropy loss.

    ``z_i``, ``z_j``: ``(E, d)`` embedding of the two ends of each positive
    edge. ``z_neg``: ``(E, M, d)`` embedding of ``M`` negative partners per
    edge. Returns a scalar mean-loss.
    """
    d_pos = (z_i - z_j).pow(2).sum(dim=-1)
    q_pos = 1.0 / (1.0 + a * d_pos.clamp_min(_MIN_DIST_FLOOR).pow(b))
    attractive = -torch.log(q_pos.clamp_min(_MIN_DIST_FLOOR))

    d_neg = (z_i.unsqueeze(1) - z_neg).pow(2).sum(dim=-1)
    q_neg = 1.0 / (1.0 + a * d_neg.clamp_min(_MIN_DIST_FLOOR).pow(b))
    repulsive = -torch.log((1.0 - q_neg).clamp_min(_MIN_DIST_FLOOR)).mean(dim=1)

    return (attractive + repulsive).mean()


@dataclass
class ParametricUMAP:
    """Parametric UMAP trainer + inferencer.

    Typical lifecycle::

        pu = ParametricUMAP(ParametricUMAPConfig())
        pu.fit(X)                  # trains encoder on full or sub-sample
        Z = pu.transform(X_new)    # out-of-sample → (N_new, n_components)
        pu.save("encoder.pt")
        pu2 = ParametricUMAP.load("encoder.pt")

    ``graph_`` (sparse COO) is the UMAP fuzzy simplicial set from the fit.
    ``a``/``b`` are the low-D distance kernel parameters derived from
    ``config.min_dist``.
    """

    config: ParametricUMAPConfig
    encoder: ParametricUMAPEncoder | None = None
    input_dim: int | None = None
    a: float | None = None
    b: float | None = None
    history: list[float] = field(default_factory=list)

    # --- training ----------------------------------------------------------

    def _build_graph(
        self, X: np.ndarray, *, random_state: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (heads, tails, weights) sampled from the fuzzy graph.

        Each positive edge appears ``⌈weight · n_epochs⌉`` times, matching
        umap-learn's ``make_epochs_per_sample`` — approximate: we just use
        the raw weights as sampling probabilities, which is simpler and
        works for test-scale runs.
        """
        graph, _sigmas, _rhos = fuzzy_simplicial_set(
            X=X, n_neighbors=self.config.n_neighbors,
            random_state=random_state, metric="euclidean",
        )
        graph = graph.tocoo()
        heads = np.asarray(graph.row, dtype=np.int64)
        tails = np.asarray(graph.col, dtype=np.int64)
        weights = np.asarray(graph.data, dtype=np.float32)
        return heads, tails, weights

    def fit(self, X: np.ndarray, *, device: torch.device | None = None) -> ParametricUMAP:
        """Train the encoder on ``X`` (``(N, D)`` float32)."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rng_np = np.random.RandomState(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.input_dim = X.shape[1]
        self.a, self.b = find_ab_params(spread=1.0, min_dist=self.config.min_dist)
        self.encoder = ParametricUMAPEncoder(
            input_dim=self.input_dim,
            n_components=self.config.n_components,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(device)

        heads, tails, weights = self._build_graph(X, random_state=rng_np)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        n = X.shape[0]

        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)
        n_edges = len(heads)
        n_per_epoch = max(n_edges, self.config.batch_size)
        for _ in range(self.config.n_epochs):
            idx_all = torch.as_tensor(
                rng_np.choice(n_edges, size=n_per_epoch, p=(weights / weights.sum())),
                device=device,
            )
            epoch_loss = 0.0
            n_batches = max(n_per_epoch // self.config.batch_size, 1)
            for b_idx in range(n_batches):
                batch = idx_all[b_idx * self.config.batch_size:
                                (b_idx + 1) * self.config.batch_size]
                if batch.numel() == 0:
                    continue
                i_idx = torch.as_tensor(heads, device=device)[batch]
                j_idx = torch.as_tensor(tails, device=device)[batch]
                neg_idx = torch.randint(
                    0, n,
                    (batch.numel(), self.config.negative_sample_rate),
                    device=device,
                )
                z_i = self.encoder(X_t[i_idx])
                z_j = self.encoder(X_t[j_idx])
                z_neg = self.encoder(X_t[neg_idx.reshape(-1)]).reshape(
                    batch.numel(), self.config.negative_sample_rate, -1,
                )
                loss = _umap_loss(z_i, z_j, z_neg, a=self.a, b=self.b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
            self.history.append(epoch_loss / max(n_batches, 1))
            _ = w  # retained for future edge-weighted variants
        return self

    # --- inference ---------------------------------------------------------

    def transform(
        self, X: np.ndarray, *, device: torch.device | None = None,
    ) -> np.ndarray:
        """Run the trained encoder on new data. No graph build — just forward."""
        if self.encoder is None:
            raise RuntimeError("ParametricUMAP is not fitted")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} cols, encoder trained on {self.input_dim}",
            )
        device = device or next(self.encoder.parameters()).device
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.as_tensor(X, dtype=torch.float32, device=device))
        return z.cpu().numpy()

    def fit_transform(
        self, X: np.ndarray, *, device: torch.device | None = None,
    ) -> np.ndarray:
        self.fit(X, device=device)
        return self.transform(X, device=device)

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist encoder + config + graph kernel params to ``path``."""
        if self.encoder is None:
            raise RuntimeError("ParametricUMAP is not fitted")
        blob: dict[str, Any] = {
            "config": self.config,
            "input_dim": self.input_dim,
            "a": self.a,
            "b": self.b,
            "state_dict": self.encoder.state_dict(),
            "history": self.history,
        }
        torch.save(blob, path)

    @classmethod
    def load(
        cls, path: str | Path, *, device: torch.device | None = None,
    ) -> ParametricUMAP:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(path, map_location=device, weights_only=False)
        cfg: ParametricUMAPConfig = blob["config"]
        enc = ParametricUMAPEncoder(
            input_dim=blob["input_dim"],
            n_components=cfg.n_components,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(device)
        enc.load_state_dict(blob["state_dict"])
        enc.eval()
        instance = cls(
            config=cfg, encoder=enc, input_dim=blob["input_dim"],
            a=float(blob["a"]), b=float(blob["b"]),
            history=list(blob.get("history", [])),
        )
        return instance


__all__ = [
    "ParametricUMAP",
    "ParametricUMAPConfig",
    "ParametricUMAPEncoder",
]
