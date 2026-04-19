# Shared Utilities — Design Document

## Files

```
src/arqueogal/utils/
├── __init__.py
├── config.py          ← YAML config loader with validation
├── coordinates.py     ← Galactic coordinate/velocity transformations
├── plotting.py        ← Publication-quality plot infrastructure (A&A format)
├── io.py              ← Parquet I/O, checkpoint save/load, streaming helpers
├── reproducibility.py ← Global seed setting, deterministic mode
└── gpu.py             ← Device selection, VRAM monitoring, cuML/sklearn fallback
```

## config.py

Nested dataclasses loaded from YAML (pattern from TESS_ML), with improvements:

- **Validation layer**: Validate all numeric ranges on load (e.g., 0 < lr < 1,
  0 < val_frac < 1, epochs > 0, batch_size > 0). Raise ValueError with descriptive
  messages for invalid configs.
- **Unknown key detection**: Warn (not silently ignore) if YAML contains keys not
  in the dataclass schema.
- **Type checking**: Verify that values match expected types.
- **Path resolution**: Resolve relative paths against config file location.
- **Serialization**: `to_yaml()` for checkpoint embedding.

## coordinates.py

Galactic coordinate and velocity transformations using astropy + galpy:

- `equatorial_to_galactic(ra, dec, parallax, pmra, pmdec, rv)` → (l, b, d, U, V, W)
- `galactic_velocities_to_cylindrical(U, V, W, R, phi)` → (v_R, v_phi, v_z)
- `compute_orbital_params(R, z, v_R, v_phi, v_z, potential)` → (ecc, Z_max, R_peri,
  R_apo, L_z, E_tot)
- Use astropy.units throughout. Strip only at output boundaries.
- Bayesian distance from parallax (prior from Galactic model, following Rodrigues+2014).

## plotting.py

Publication-quality infrastructure inherited from TESS_ML plot_config.py:

- A&A figure sizes: single column 8.8 cm, double column 18.3 cm
- LaTeX rendering with fallback if LaTeX not installed (detect and warn, don't crash)
- Consistent rcParams applied on import
- Color scheme: colorblind-safe palettes available as option
- Helper functions:
  - `hexbin_with_colorbar(ax, x, y, ...)` — for large N (never raw scatter)
  - `density_2d(ax, x, y, bins, ...)` — 2D histogram with proper normalization
  - `residual_panel(ax, y_true, y_pred, label, ...)` — standardized residual plot
  - `coverage_curve(ax, sigma, error, ...)` — P(|ε| < nσ) vs nσ
  - `save_figure(fig, path, formats=['png', 'pdf'])` — save in multiple formats

## io.py

- `load_parquet(path, columns=None)` → DataFrame (pandas or cudf depending on size)
- `save_parquet(df, path, compression='snappy')`
- `streaming_parquet_reader(path, batch_size)` → generator of DataFrames
- `save_checkpoint(path, **kwargs)` — with version tag, weights_only=True support
- `load_checkpoint(path, device)` — with _orig_mod. prefix stripping, version check

## reproducibility.py

```python
def set_global_seed(seed: int) -> None:
    """Set seed for torch, numpy, python random, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full determinism also requires:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # But this significantly slows training. Use only for debugging.
```

## gpu.py

```python
def get_device() -> torch.device:
    """Select device with graceful CPU fallback."""

def check_vram(required_mb: float) -> bool:
    """Check if sufficient VRAM is available."""

def get_umap_class():
    """Return cuml.manifold.UMAP if available, else umap.UMAP."""

def get_hdbscan_class():
    """Return cuml.cluster.HDBSCAN if available, else hdbscan.HDBSCAN."""
```
