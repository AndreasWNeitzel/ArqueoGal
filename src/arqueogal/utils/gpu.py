"""Device selection + cuML→CPU fallback — utils/DESIGN.md.

Two responsibilities:

1. **Device selection.** ``get_device()`` returns a Torch device with
   graceful CPU fallback — critical for CPU-only HPC nodes.
2. **Algorithm fallback.** ``get_umap_class()`` / ``get_hdbscan_class()``
   return the GPU-accelerated cuML variant when available and the CPU
   umap-learn / hdbscan class otherwise, both exposing the sklearn-ish
   ``.fit(X)`` / ``.fit_predict(X)`` API.
"""

from __future__ import annotations

from typing import Any


def get_device(prefer: str = "auto") -> Any:
    """Return a :class:`torch.device`.

    Parameters
    ----------
    prefer : {"auto", "cuda", "cpu"}
        ``"auto"`` — CUDA if available, else CPU. ``"cuda"`` forces CUDA
        (raises if unavailable). ``"cpu"`` forces CPU.
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if prefer != "auto":
        raise ValueError(f"prefer must be 'auto'|'cuda'|'cpu', got {prefer!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_vram(required_mb: float, device: Any = None) -> bool:
    """Return True if ``device`` has ≥ ``required_mb`` MiB of free VRAM.

    Returns True unconditionally on CPU (infinite "VRAM"). For CUDA devices
    the check queries ``torch.cuda.mem_get_info`` (free, total) — allocated
    memory within the current process is counted as free from the OS
    perspective, but we use the driver's free-memory number which is the
    physically available headroom.
    """
    import torch

    if device is None:
        device = get_device()
    if device.type != "cuda":
        return True
    free_bytes, _total = torch.cuda.mem_get_info(device)
    return (free_bytes / (1024 * 1024)) >= required_mb


def get_umap_class() -> type:
    """cuML UMAP if available, else umap-learn UMAP."""
    try:
        from cuml.manifold import UMAP as _CuUMAP  # noqa: N811
    except ImportError:
        pass
    else:
        return _CuUMAP
    from umap import UMAP as _CpuUMAP  # noqa: N811

    return _CpuUMAP


def get_hdbscan_class() -> type:
    """cuML HDBSCAN if available, else hdbscan.HDBSCAN."""
    try:
        from cuml.cluster import HDBSCAN as _CuHDB  # noqa: N811
    except ImportError:
        pass
    else:
        return _CuHDB
    from hdbscan import HDBSCAN as _CpuHDB  # noqa: N811

    return _CpuHDB


__all__ = [
    "check_vram",
    "get_device",
    "get_hdbscan_class",
    "get_umap_class",
]
