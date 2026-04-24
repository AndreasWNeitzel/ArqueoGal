"""Halfway-checkpoint UMAP harness for Pipeline 1 contrastive pretraining.

Run A's halfway gate (#132). After contrastive pretraining converges, we embed
a held-out slice of stars through the pretrained encoder trunk, run UMAP on the
latent ``h`` (not the L2-normalised projection ``z``), and save three scatter
plots coloured by Tier-1 labels — ``teff_apogee`` / ``mh_apogee`` / ``logg_apogee``.

The purpose is the visual gate research_brief §9.1 requires: Tier-1 gradients
must appear smooth and non-degenerate on a 2-D projection of the trunk.
If any of the three plots shows shattered or discontinuous structure in a
label that the sanity battery already confirmed lives in the feature matrix,
halt Run A before supervised fine-tune.

This module is deliberately stateless — :func:`compute_halfway_embedding`
takes a pretrained :class:`XpAbundanceModel`, returns the 2-D embedding, and
:func:`save_halfway_plots` writes the three-panel figure. Callers (the driver
script) assemble the held-out batch and persist any machine-consumable
summary JSON themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.model import XpAbundanceModel


@dataclass
class HalfwayEmbedding:
    """UMAP embedding of the pretrained trunk on a held-out slice.

    ``embedding`` is (N, 2); ``labels`` holds the three Tier-1 columns used
    for colouring. ``n_stars`` and ``n_finite`` give the finite-label counts
    reported in the audit JSON.
    """

    embedding: np.ndarray
    labels: dict[str, np.ndarray]  # {"teff_apogee": ..., "mh_apogee": ..., "logg_apogee": ...}
    n_stars: int
    n_finite: dict[str, int] = field(default_factory=dict)


def compute_halfway_embedding(  # noqa: PLR0913 — the UMAP knobs stay explicit
    model: XpAbundanceModel,
    adapter: XpFeatureAdapter,
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    device: torch.device,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    umap_seed: int = 0,
    batch_size: int = 2048,
) -> HalfwayEmbedding:
    """Embed ``X`` through the trunk and reduce ``h`` to 2-D with UMAP.

    ``labels`` must contain the three Tier-1 columns by name. UMAP is run on
    the pre-projection hidden state ``h``, not ``z`` — ``z`` is L2-normalised
    for SupCon's cosine similarity and squashes magnitude structure the
    downstream regressor uses. We keep the richer ``h`` view for the gate.
    """
    import umap  # local import — heavy dependency

    model.eval()
    adapter.eval()
    h_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_batch = torch.as_tensor(
                X[start : start + batch_size], dtype=torch.float32, device=device
            )
            x_adapted = adapter(x_batch)
            h, _z = model.encoder(x_adapted)
            h_chunks.append(h.cpu().numpy())
    H = np.concatenate(h_chunks, axis=0)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=umap_seed,
    )
    emb = reducer.fit_transform(H).astype(np.float32)

    n_finite = {k: int(np.isfinite(v).sum()) for k, v in labels.items()}
    return HalfwayEmbedding(
        embedding=emb,
        labels=labels,
        n_stars=len(X),
        n_finite=n_finite,
    )


def save_halfway_plots(
    he: HalfwayEmbedding,
    out_dir: Path,
    *,
    prefix: str = "halfway",
) -> list[Path]:
    """Write three scatter plots — one per Tier-1 label — to ``out_dir``.

    Each PNG is named ``{prefix}_umap_{column}.png``. Returns the list of
    written paths for the driver to record in the audit JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cmap_for = {
        "teff_apogee": "plasma",
        "mh_apogee": "viridis",
        "logg_apogee": "cividis",
        "alpha_m_apogee": "coolwarm",
        "mg_h_apogee": "coolwarm",
    }
    label_texts = {
        "teff_apogee": r"$T_\mathrm{eff}$ / K (APOGEE)",
        "mh_apogee": "[M/H] (APOGEE)",
        "logg_apogee": r"$\log g$ (APOGEE)",
        "alpha_m_apogee": r"[$\alpha$/M] (APOGEE)",
        "mg_h_apogee": "[Mg/H] (APOGEE)",
    }
    for col, values in he.labels.items():
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        finite = np.isfinite(values)
        sc = ax.scatter(
            he.embedding[finite, 0],
            he.embedding[finite, 1],
            c=values[finite],
            cmap=cmap_for.get(col, "viridis"),
            s=3,
            alpha=0.7,
            linewidths=0,
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(
            f"Halfway trunk UMAP — coloured by {col} ({finite.sum():,}/{he.n_stars:,} stars)",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label_texts.get(col, col))
        fig.tight_layout()
        path = out_dir / f"{prefix}_umap_{col}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


__all__ = [
    "HalfwayEmbedding",
    "compute_halfway_embedding",
    "save_halfway_plots",
]
