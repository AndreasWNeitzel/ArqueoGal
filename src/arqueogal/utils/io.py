"""Parquet + checkpoint I/O helpers — utils/DESIGN.md.

Parquet is the standard for ArqueoGal feature / catalog artefacts (see
``data_acquisition.md`` §5.2). Checkpoints carry a ``version`` tag and
handle ``_orig_mod.`` prefix stripping that ``torch.compile`` injects.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "ArqueoGalCheckpointError",
    "load_checkpoint",
    "load_parquet",
    "save_checkpoint",
    "save_parquet",
    "streaming_parquet_reader",
]

# Current checkpoint format version. Bump on breaking changes. Missing
# version ⇒ treated as legacy (version 0) with a warning.
CHECKPOINT_VERSION = 1


class ArqueoGalCheckpointError(RuntimeError):
    """Raised when a checkpoint file fails validation."""


def load_parquet(path: str | Path, columns: list[str] | None = None) -> Any:
    """Read a Parquet file as pandas DataFrame.

    We intentionally default to pandas (not cudf) for compatibility with
    CPU-only HPC nodes. Callers with large frames on a CUDA box can wrap
    with ``cudf.from_pandas`` themselves — keeping the fallback explicit
    avoids hidden GPU OOMs.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    table = pq.read_table(path, columns=columns)
    return table.to_pandas()


def save_parquet(
    df: Any,
    path: str | Path,
    compression: str = "snappy",
) -> Path:
    """Write ``df`` to ``path`` atomically (temp file → rename).

    Atomic write (temp file → rename) prevents half-written Parquet files
    on crash; see ``docs/data_acquisition.md`` §14.4.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # cudf DataFrames expose the same .to_parquet API as pandas.
    df.to_parquet(tmp, compression=compression, index=False)
    tmp.replace(path)
    return path


def streaming_parquet_reader(
    path: str | Path,
    batch_size: int,
) -> Iterator[Any]:
    """Yield pandas DataFrames of up to ``batch_size`` rows at a time."""
    import pyarrow.parquet as pq

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def save_checkpoint(path: str | Path, **state: Any) -> Path:
    """Save a training / model checkpoint.

    ``state`` keys are written as-is. A ``version`` key is always injected
    (overwriting any caller-supplied value, intentionally — we own the
    format). Write is atomic.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_out = dict(state)
    state_out["version"] = CHECKPOINT_VERSION
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_out, tmp)
    tmp.replace(path)
    return path


def _strip_orig_mod_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip ``_orig_mod.`` prefix that ``torch.compile`` injects.

    Only strips when **every** key carries the prefix — avoids silently
    corrupting user-named weights.
    """
    prefix = "_orig_mod."
    if not state_dict:
        return state_dict
    if not all(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def load_checkpoint(
    path: str | Path,
    device: Any = None,
    *,
    weights_only: bool = True,
    strict_version: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint with version check + ``_orig_mod.`` prefix fix.

    Parameters
    ----------
    weights_only
        Forwarded to :func:`torch.load`. Default ``True`` per PyTorch 2.6+
        guidance; set to ``False`` only for legacy checkpoints that store
        arbitrary Python objects.
    strict_version
        If True, raise on version mismatch. Otherwise warn.
    """
    import warnings

    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    map_location = device if device is not None else "cpu"
    try:
        state = torch.load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except Exception as exc:  # noqa: BLE001 — fall-back path for legacy pickles
        if weights_only:
            # Retry without weights_only — legacy checkpoint may contain a
            # pickled config/namespace that fails weights_only safety check.
            warnings.warn(
                f"load_checkpoint: weights_only=True failed ({exc}); retrying "
                "with weights_only=False. This is unsafe for untrusted files.",
                stacklevel=2,
            )
            state = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        else:
            raise

    if not isinstance(state, dict):
        raise ArqueoGalCheckpointError(
            f"checkpoint {path} is not a dict (got {type(state).__name__})",
        )

    version = state.get("version", 0)
    if version != CHECKPOINT_VERSION:
        msg = f"checkpoint {path} version {version} != current {CHECKPOINT_VERSION}"
        if strict_version:
            raise ArqueoGalCheckpointError(msg)
        warnings.warn(msg, stacklevel=2)

    # Strip _orig_mod. prefix anywhere it appears under a "state_dict"-style
    # key — but don't recurse wildly; cover the usual cases.
    for key in ("state_dict", "model_state_dict", "model"):
        sub = state.get(key)
        if isinstance(sub, dict):
            with contextlib.suppress(TypeError):
                state[key] = _strip_orig_mod_prefix(sub)

    return state
