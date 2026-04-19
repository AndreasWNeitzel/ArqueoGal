"""Shared utilities — see ``utils/DESIGN.md``.

Re-exports the lightweight, import-cheap helpers (``reproducibility``,
``gpu``, ``io``, ``config``, ``coordinates``) directly. Plotting
(matplotlib) and coordinates' astropy chain are deliberately **not**
imported at module top-level — they're accessed via
``from arqueogal.utils import plotting`` on demand.
"""

from arqueogal.utils.config import (
    ConfigValidationError,
    load_config,
    to_yaml,
)
from arqueogal.utils.gpu import (
    check_vram,
    get_device,
    get_hdbscan_class,
    get_umap_class,
)
from arqueogal.utils.io import (
    CHECKPOINT_VERSION,
    ArqueoGalCheckpointError,
    load_checkpoint,
    load_parquet,
    save_checkpoint,
    save_parquet,
    streaming_parquet_reader,
)
from arqueogal.utils.reproducibility import (
    set_full_determinism,
    set_global_seed,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "ArqueoGalCheckpointError",
    "ConfigValidationError",
    "check_vram",
    "get_device",
    "get_hdbscan_class",
    "get_umap_class",
    "load_checkpoint",
    "load_config",
    "load_parquet",
    "save_checkpoint",
    "save_parquet",
    "set_full_determinism",
    "set_global_seed",
    "streaming_parquet_reader",
    "to_yaml",
]
