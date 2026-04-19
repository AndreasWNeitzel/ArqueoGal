"""Global seed + determinism controls — utils/DESIGN.md.

A single ``set_global_seed`` entrypoint seeds:

- Python's :mod:`random`,
- NumPy's legacy global state (``np.random.seed``),
- PyTorch CPU + (if available) CUDA,
- and returns a fresh :class:`numpy.random.Generator` the caller can thread
  through their own code. New NumPy code should prefer this generator over
  the legacy global RNG.

``set_full_determinism`` additionally flips ``torch.backends.cudnn.deterministic``
and disables benchmark. That incurs a significant training slowdown and is
opt-in — per DESIGN, use only for debugging repro bugs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed every global RNG we use and return a new NumPy generator.

    Parameters
    ----------
    seed
        Non-negative integer seed. Applied to :mod:`random`, NumPy's legacy
        global state, PyTorch (CPU + CUDA if available), and Python's
        ``PYTHONHASHSEED``.

    Returns
    -------
    numpy.random.Generator
        A fresh ``default_rng(seed)`` — prefer threading this through
        instead of relying on ``np.random.*`` module-level calls.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def set_full_determinism(seed: int) -> np.random.Generator:
    """Seed everything **and** force cuDNN determinism (slow — debug only)."""
    rng = set_global_seed(seed)
    try:
        import torch
    except ImportError:
        return rng
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch ≥ 1.8 exposes this; guard for older installs.
    if hasattr(torch, "use_deterministic_algorithms"):
        # warn_only=True: avoid raising on kernels that don't have deterministic
        # implementations (e.g. scatter_add on CUDA) — caller knows they're in
        # debug mode.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return rng


__all__ = [
    "set_full_determinism",
    "set_global_seed",
]
