"""Extinction features — §8 of data_acquisition.md.

Two independent Av estimates feed Pipeline 1:

1. **Line-of-sight 3D-map composition** (§8.2/§8.4) — Edenhofer+2024 for
   d < 1.25 kpc, Lallement+2022 for 1.25–3 kpc, SFD beyond. Composition
   routing + unit conversion are implemented here via pluggable query
   callables so the logic is fully offline-testable. The actual map
   downloads are an opt-in step triggered by :func:`get_default_queries`
   (which imports ``dustmaps`` and expects the maps to have been fetched
   once via each submodule's ``fetch()`` helper).

2. **Neighborhood-median GSP-Phot Av** (§8.3) — purely numerical helper that
   runs on per-star Gaia ``ag_gspphot`` values and their 3D positions. Zero
   additional disk footprint. Implemented here.

The two are then injected as separate features so the ML learns when to
deviate from the prior — individual per-star Av from any single 3D dust
map is noisy; the neighborhood-median + line-of-sight pair is the robust
feature set (see ``docs/research_brief.md`` §5).

``dustmaps`` and ``astropy`` are imported lazily inside the helpers that
need them — the core composition and neighborhood code stay pure NumPy /
SciPy / pandas so they can be exercised in CI without any map fixtures.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

NEAR_BOUNDARY_KPC: Final[float] = 1.25
"""§8.2: stars within this heliocentric distance are served by Edenhofer+2024."""

FAR_BOUNDARY_KPC: Final[float] = 3.0
"""§8.2: between :data:`NEAR_BOUNDARY_KPC` and this distance, Lallement+2022
is used. Beyond this, the SFD 2D map (integrated to infinity) is substituted."""

SFD_TO_AV_COEFF: Final[float] = 2.742
"""§8.2: A_V / E(B−V) from Schlafly & Finkbeiner 2011 for the SFD map on
the Landolt V band. Keep as a module-level constant so tests can assert it
without reading the code."""

DEFAULT_NEIGHBORHOOD_RADIUS_PC: Final[float] = 75.0
"""§8.3: 50–100 pc radius for the neighborhood median. Default 75 pc
balances locality (high-extinction cloud resolution) against neighbour
count (shot noise in the median at < 10 neighbours)."""

MIN_NEIGHBORS_FOR_MEDIAN: Final[int] = 5
"""Below this, the median is too noisy to trust — reported as NaN so the
downstream ML sees an explicit missing value instead of a garbage number."""


# -----------------------------------------------------------------------------
# §8.3 neighborhood-median Av
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeighborhoodAvFeatures:
    """Per-star §8.3 features. Length equals the input row count.

    All arrays are ``float32`` to match the XP / photometry dtypes used by
    Pipeline 1. ``n_neighbors`` is ``int32``.
    """

    av_nbhd_median: np.ndarray
    av_nbhd_std: np.ndarray
    n_neighbors: np.ndarray

    def to_frame(self, source_ids: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """Return a DataFrame with the three columns, optionally keyed by source_id."""
        cols = {
            "av_nbhd_median": self.av_nbhd_median,
            "av_nbhd_std": self.av_nbhd_std,
            "n_neighbors": self.n_neighbors,
        }
        if source_ids is not None:
            return pd.DataFrame({"source_id": np.asarray(source_ids), **cols})
        return pd.DataFrame(cols)


def galactic_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray, distance_pc: np.ndarray) -> np.ndarray:
    """Convert (RA, Dec, d) → heliocentric Cartesian (x, y, z) in pc.

    Uses the direct spherical-to-Cartesian transform on *equatorial* angles
    — §8.3 neighbourhoods are defined by 3D proximity, and the choice of
    heliocentric vs. galactic axes does not change the neighbour set, only
    the axis labels. Equatorial avoids an astropy coordinate-frame round-
    trip (20× faster for 1.5 M stars).
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    d = np.asarray(distance_pc, dtype=np.float64)
    cos_dec = np.cos(dec)
    x = d * cos_dec * np.cos(ra)
    y = d * cos_dec * np.sin(ra)
    z = d * np.sin(dec)
    return np.column_stack([x, y, z])


def neighborhood_av_features(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_pc: np.ndarray,
    ag_gspphot: np.ndarray,
    *,
    radius_pc: float = DEFAULT_NEIGHBORHOOD_RADIUS_PC,
    min_neighbors: int = MIN_NEIGHBORS_FOR_MEDIAN,
    include_self: bool = False,
) -> NeighborhoodAvFeatures:
    """Compute §8.3 neighborhood-median A_G features.

    For each star, finds all stars within a 3D ball of ``radius_pc`` pc in
    heliocentric Cartesian coordinates and reports the median and standard
    deviation of their ``ag_gspphot`` values (excluding NaNs).

    NaN inputs
    ----------
    - Stars with NaN ``distance_pc`` are silently excluded from the tree —
      they contribute no neighbours and receive NaN outputs.
    - Stars with NaN ``ag_gspphot`` are kept in the tree (they still have
      positions) but excluded from the median / std computation over their
      neighbours. A star whose own A_G is NaN still gets a neighbourhood
      statistic.

    Parameters
    ----------
    ra_deg, dec_deg, distance_pc
        Per-star equatorial coordinates and distance.
    ag_gspphot
        Per-star Gaia GSP-Phot ``ag_gspphot`` (magnitudes).
    radius_pc
        3D ball radius. §8.3 recommends 50–100 pc.
    min_neighbors
        Below this count, the median / std are reported as NaN. A star's
        own entry is not counted (see ``include_self``).
    include_self
        If ``True``, a star's own A_G is included in its own neighbour set.
        Default is ``False`` (§8.3 intent: the median should represent the
        ambient field, not be biased toward the star itself).

    Returns
    -------
    NeighborhoodAvFeatures
        Three arrays of length N.
    """
    ra_arr = np.asarray(ra_deg, dtype=np.float64)
    dec_arr = np.asarray(dec_deg, dtype=np.float64)
    d_arr = np.asarray(distance_pc, dtype=np.float64)
    av_arr = np.asarray(ag_gspphot, dtype=np.float64)

    _validate_lengths(ra_arr, dec_arr, d_arr, av_arr)
    if radius_pc <= 0:
        raise ValueError(f"radius_pc must be positive, got {radius_pc}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")

    n = ra_arr.shape[0]
    xyz = galactic_to_xyz(ra_arr, dec_arr, d_arr)

    has_pos = np.isfinite(xyz).all(axis=1)
    n_pos = int(has_pos.sum())
    logger.info(
        "neighborhood_av: %d/%d stars have finite 3D positions; building tree on those",
        n_pos,
        n,
    )
    if n_pos == 0:
        return _empty_features(n)

    tree_points = xyz[has_pos]
    tree_av = av_arr[has_pos]
    tree = cKDTree(tree_points)

    # Remap positional indices in the tree back to the original rows so we
    # can, if requested, drop a star's own contribution.
    pos_to_global = np.flatnonzero(has_pos)
    global_to_pos = np.full(n, -1, dtype=np.int64)
    global_to_pos[pos_to_global] = np.arange(n_pos)

    # cKDTree.query_ball_point rejects NaN — query only the finite rows.
    finite_neighbor_lists = tree.query_ball_point(tree_points, r=radius_pc)

    median = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    n_neigh = np.zeros(n, dtype=np.int32)

    for pos_idx, global_idx in enumerate(pos_to_global):
        neigh = finite_neighbor_lists[pos_idx]
        if not neigh:
            continue
        neigh_arr = np.asarray(neigh, dtype=np.int64)
        if not include_self:
            neigh_arr = neigh_arr[neigh_arr != pos_idx]
            if neigh_arr.size == 0:
                continue
        av_slice = tree_av[neigh_arr]
        finite = np.isfinite(av_slice)
        finite_vals = av_slice[finite]
        k = finite_vals.size
        n_neigh[global_idx] = k
        if k >= min_neighbors:
            median[global_idx] = np.median(finite_vals)
            std[global_idx] = np.std(finite_vals, ddof=0)

    return NeighborhoodAvFeatures(
        av_nbhd_median=median.astype(np.float32),
        av_nbhd_std=std.astype(np.float32),
        n_neighbors=n_neigh,
    )


# -----------------------------------------------------------------------------
# §8.2 line-of-sight 3D-map composition
# -----------------------------------------------------------------------------


class DustQuery(Protocol):
    """Minimal structural type for a line-of-sight dust query.

    Matches the ``__call__(coords) -> np.ndarray`` contract we wrap around
    each of the three backends. The real ``dustmaps`` classes expose a
    ``.query(coords)`` method; :func:`get_default_queries` converts those
    into plain callables.
    """

    def __call__(self, coords) -> np.ndarray:  # noqa: ANN001 — astropy SkyCoord
        ...


@dataclass(frozen=True, slots=True)
class ComposedAv:
    """Per-star §8.2 composition outputs.

    ``av`` is the composed A_V in magnitudes. ``source`` flags which map
    was used per star: ``0`` = Edenhofer+2024 (near), ``1`` = Lallement+2022
    (mid), ``2`` = SFD (far), ``-1`` = no valid value (NaN distance or map
    returned NaN for every backend).
    """

    av: np.ndarray
    source: np.ndarray

    def to_frame(self, source_ids: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        cols = {"av_los": self.av, "av_los_source": self.source}
        if source_ids is not None:
            return pd.DataFrame({"source_id": np.asarray(source_ids), **cols})
        return pd.DataFrame(cols)


def compose_av(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_pc: np.ndarray,
    *,
    near_query: DustQuery,
    mid_query: DustQuery,
    far_query: DustQuery,
    near_boundary_kpc: float = NEAR_BOUNDARY_KPC,
    far_boundary_kpc: float = FAR_BOUNDARY_KPC,
    sfd_to_av: float = SFD_TO_AV_COEFF,
) -> ComposedAv:
    """§8.2 line-of-sight A_V composition.

    Routes each star to the appropriate 3D-dust-map backend based on its
    heliocentric distance:

    - ``distance < near_boundary_kpc`` → ``near_query`` (Edenhofer+2024,
      returns A_V in mag directly).
    - ``near_boundary_kpc ≤ distance < far_boundary_kpc`` → ``mid_query``
      (Lallement+2022, returns A_V in mag directly).
    - ``distance ≥ far_boundary_kpc`` → ``far_query`` (SFD, returns
      E(B−V) — multiplied here by ``sfd_to_av`` to obtain A_V).

    Queries are invoked as ``near_query(coords)``, where ``coords`` is the
    subset ``SkyCoord`` (with ``distance=...`` attached for 3D maps) for the
    stars falling in that distance bin. This decouples the composition
    logic from ``dustmaps`` — tests pass plain callables returning ndarrays.

    NaN distances produce ``av=nan`` and ``source=-1``.

    Parameters
    ----------
    ra_deg, dec_deg, distance_pc
        Per-star equatorial coordinates and heliocentric distance (pc).
    near_query, mid_query, far_query
        Callables honouring :class:`DustQuery`. See
        :func:`get_default_queries` for the production wiring.
    near_boundary_kpc, far_boundary_kpc
        Distance-bin boundaries. Defaults match §8.2.
    sfd_to_av
        Multiplicative conversion from SFD E(B−V) to A_V.

    Returns
    -------
    ComposedAv
    """
    _validate_lengths(ra_deg, dec_deg, distance_pc)
    if near_boundary_kpc <= 0 or far_boundary_kpc <= near_boundary_kpc:
        raise ValueError(
            f"need 0 < near_boundary_kpc < far_boundary_kpc; got "
            f"{near_boundary_kpc=}, {far_boundary_kpc=}"
        )

    n = ra_deg.shape[0]
    av = np.full(n, np.nan, dtype=np.float64)
    source = np.full(n, -1, dtype=np.int8)

    dist_kpc = np.asarray(distance_pc, dtype=np.float64) / 1000.0
    has_pos = np.isfinite(dist_kpc) & (dist_kpc > 0)

    near_mask = has_pos & (dist_kpc < near_boundary_kpc)
    mid_mask = has_pos & (dist_kpc >= near_boundary_kpc) & (dist_kpc < far_boundary_kpc)
    far_mask = has_pos & (dist_kpc >= far_boundary_kpc)

    logger.info(
        "compose_av: %d near (<%.2f kpc), %d mid (%.2f–%.2f kpc), %d far (≥%.2f kpc), "
        "%d excluded (bad distance)",
        int(near_mask.sum()),
        near_boundary_kpc,
        int(mid_mask.sum()),
        near_boundary_kpc,
        far_boundary_kpc,
        int(far_mask.sum()),
        far_boundary_kpc,
        int((~has_pos).sum()),
    )

    for mask, query, src_code, transform in (
        (near_mask, near_query, 0, lambda x: x),
        (mid_mask, mid_query, 1, lambda x: x),
        (far_mask, far_query, 2, lambda x: x * sfd_to_av),
    ):
        if not mask.any():
            continue
        coords = _build_dust_coords(ra_deg[mask], dec_deg[mask], distance_pc[mask])
        result = np.asarray(transform(np.asarray(query(coords), dtype=np.float64)))
        av[mask] = result
        source[mask] = src_code
        # If the backend returned NaN for any star, mark the source as -1 so the
        # downstream ML can distinguish "map said nothing" from "distance bin
        # not covered". Bin-mask indices remain as assigned for stars with a
        # finite value; the -1 overwrite only touches NaN results.
        nan_result = ~np.isfinite(result)
        if nan_result.any():
            source_indices = np.flatnonzero(mask)[nan_result]
            source[source_indices] = -1

    return ComposedAv(av=av.astype(np.float32), source=source)


def get_default_queries() -> tuple[Callable, Callable, Callable]:
    """Build the production (Edenhofer, Lallement, SFD) query callables.

    Imports ``dustmaps`` lazily and wraps each backend's ``.query(coords)``
    method. Raises :class:`ImportError` with an actionable hint if a
    backend module is missing (e.g. ``dustmaps.lallement2022`` is not in
    every wheel — users must supply their own wrapper in that case).

    This function IS NOT called automatically by :func:`compose_av` — the
    caller must wire it in explicitly, keeping the import side-effects
    (map file access, tens of MB of cached data loaded into RAM) opt-in.
    """
    from dustmaps.edenhofer2023 import Edenhofer2023Query
    from dustmaps.sfd import SFDQuery

    try:
        from dustmaps.lallement2022 import Lallement2022Query
    except ImportError as exc:
        raise ImportError(
            "dustmaps.lallement2022 is not available in this dustmaps build. "
            "Install a version that ships it, or supply a custom mid-distance "
            "query callable to compose_av(..., mid_query=your_query)."
        ) from exc

    eden = Edenhofer2023Query()
    lall = Lallement2022Query()
    sfd = SFDQuery()
    return (lambda c: eden.query(c), lambda c: lall.query(c), lambda c: sfd.query(c))


# -----------------------------------------------------------------------------
# §8.5 Lallement+2022 line-of-sight integration (direct CDS fetch, no dustmaps)
# -----------------------------------------------------------------------------
#
# The cube is a 3D extinction-density map at 550 nm (units A0/pc) in heliocentric
# galactic Cartesian coordinates. Axes: X → Galactic centre, Y → rotation (l=90°),
# Z → NGP. Voxel step = 10 pc. Sun at voxel (300.5, 300.5, 40.5) — cube size
# (X=601, Y=601, Z=81) → ±3 kpc × ±3 kpc × ±0.4 kpc. Shipped gzipped via CDS at
# ftp://cdsarc.u-strasbg.fr/cats/J/A+A/661/A147/cube_ext.fits.gz.
#
# We DO NOT depend on dustmaps.lallement2022 — per docs/data_acquisition.md
# §8.5, upgrading dustmaps risks RAPIDS-pinned dep churn. This helper reads the
# FITS cube directly with astropy.io.fits and interpolates with
# scipy.ndimage.map_coordinates.

LALLEMENT2022_VOXEL_STEP_PC: Final[float] = 10.0
"""Lallement+2022 cube voxel step in parsecs (FITS header STEP = 10)."""

LALLEMENT2022_SUN_VOXEL: Final[tuple[float, float, float]] = (300.5, 300.5, 40.5)
"""Sun position in (X, Y, Z) voxel coordinates (0-indexed, float; from header
SUN_POSX / SUN_POSY / SUN_POSZ)."""


def load_lallement2022_cube(cube_path):  # noqa: ANN001, ANN202 — pathlib.Path + ndarray
    """Load the Lallement+2022 extinction-density cube from disk.

    Accepts either ``cube_ext.fits`` or ``cube_ext.fits.gz`` — ``astropy.io.fits``
    transparently decompresses on the fly. Returns the cube data as a
    ``float32`` ndarray in FITS axis order ``(Z, Y, X)``, matching the header
    layout (NAXIS3=81, NAXIS2=601, NAXIS1=601).
    """
    from astropy.io import fits

    with fits.open(cube_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    return data


def lallement2022_query(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_pc: np.ndarray,
    *,
    cube: np.ndarray,
    voxel_step_pc: float = LALLEMENT2022_VOXEL_STEP_PC,
    sun_voxel: tuple[float, float, float] = LALLEMENT2022_SUN_VOXEL,
    sample_step_pc: float = 10.0,
) -> np.ndarray:
    """Integrate Lallement+2022 extinction density along Sun → star LOS.

    Computes A_V (mag) as the line integral of the cube's extinction density
    (nominally A0(550 nm)/parsec) from the Sun to each star. Stars whose LOS
    leaves the cube bounds at any sample — or whose distance is NaN / ≤ 0 —
    get NaN; the downstream ML sees explicit missingness rather than a
    silently-clipped value.

    Parameters
    ----------
    ra_deg, dec_deg, distance_pc
        ICRS coordinates and heliocentric distance (pc). Length-N arrays.
    cube
        Pre-loaded density cube from :func:`load_lallement2022_cube`. Shape
        (NZ, NY, NX) in FITS axis order. Lazy load is the caller's job so
        batched calls don't re-read the file.
    voxel_step_pc
        Voxel size in pc (default 10.0 from header).
    sun_voxel
        Sun position in (X, Y, Z) voxel coordinates (0-indexed float).
    sample_step_pc
        LOS sampling step for trapezoidal integration. Default matches the
        voxel step so every voxel crossing contributes once; smaller values
        add cost without adding information because the cube is only
        resolved at 25 pc (RESOL header keyword).

    Returns
    -------
    av : ndarray, float32
        Integrated A_V in magnitudes per star. NaN where the LOS exits the
        cube or the distance is not usable.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from scipy.ndimage import map_coordinates

    _validate_lengths(ra_deg, dec_deg, distance_pc)
    if voxel_step_pc <= 0 or sample_step_pc <= 0:
        raise ValueError(
            f"steps must be positive; got voxel={voxel_step_pc}, sample={sample_step_pc}"
        )

    nz, ny, nx = cube.shape
    sun_i, sun_j, sun_k = sun_voxel

    d = np.asarray(distance_pc, dtype=np.float64)
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    n = d.shape[0]

    # Use astropy to get galactic (l, b) — exact frame transform, small cost
    # at N ~ 10^6.
    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    l_rad = gal.l.to_value(u.rad)
    b_rad = gal.b.to_value(u.rad)
    cos_b = np.cos(b_rad)
    uhat_x = cos_b * np.cos(l_rad)
    uhat_y = cos_b * np.sin(l_rad)
    uhat_z = np.sin(b_rad)

    valid = np.isfinite(d) & (d > 0)
    out = np.full(n, np.nan, dtype=np.float64)
    if not valid.any():
        return out.astype(np.float32)

    # Per-star integration. We could vectorise a common-N-sample grid, but
    # distances span a factor of >>10 and the cube is only ±3 kpc anyway —
    # iterating and skipping out-of-bounds stars up front is both simpler
    # and faster (array of ~100 points per star, ~1.5 M stars → < 30 s).
    idx_valid = np.flatnonzero(valid)
    # FITS axis order for map_coordinates: (Z, Y, X) → numpy indices [k, j, i].
    for s_idx in idx_valid:
        d_s = d[s_idx]
        n_samples = max(2, int(np.ceil(d_s / sample_step_pc)) + 1)
        s = np.linspace(0.0, d_s, n_samples)
        x_pc = s * uhat_x[s_idx]
        y_pc = s * uhat_y[s_idx]
        z_pc = s * uhat_z[s_idx]
        i_vox = x_pc / voxel_step_pc + sun_i
        j_vox = y_pc / voxel_step_pc + sun_j
        k_vox = z_pc / voxel_step_pc + sun_k
        # Bail on LOS exits cube — NaN to the ML so it picks up the signal.
        if (
            i_vox.min() < 0
            or i_vox.max() > nx - 1
            or j_vox.min() < 0
            or j_vox.max() > ny - 1
            or k_vox.min() < 0
            or k_vox.max() > nz - 1
        ):
            continue
        # map_coordinates expects a (ndim, N) array in FITS numpy order [k, j, i].
        coords = np.vstack([k_vox, j_vox, i_vox])
        dens = map_coordinates(cube, coords, order=1, mode="constant", cval=np.nan)
        if not np.isfinite(dens).all():
            continue
        out[s_idx] = float(np.trapezoid(dens, s))

    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# internals
# -----------------------------------------------------------------------------


def _build_dust_coords(ra_deg: np.ndarray, dec_deg: np.ndarray, distance_pc: np.ndarray):  # noqa: ANN202 — astropy SkyCoord
    """Build a SkyCoord carrying a 3D distance — the shape dustmaps accepts."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(
        ra=np.asarray(ra_deg) * u.deg,
        dec=np.asarray(dec_deg) * u.deg,
        distance=np.asarray(distance_pc) * u.pc,
        frame="icrs",
    )


def _validate_lengths(*arrs: np.ndarray) -> None:
    if not arrs:
        return
    lengths = {a.shape[0] for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f"input arrays have mismatched lengths: {sorted(lengths)}")


def _empty_features(n: int) -> NeighborhoodAvFeatures:
    return NeighborhoodAvFeatures(
        av_nbhd_median=np.full(n, np.nan, dtype=np.float32),
        av_nbhd_std=np.full(n, np.nan, dtype=np.float32),
        n_neighbors=np.zeros(n, dtype=np.int32),
    )


__all__ = [
    "DEFAULT_NEIGHBORHOOD_RADIUS_PC",
    "FAR_BOUNDARY_KPC",
    "LALLEMENT2022_SUN_VOXEL",
    "LALLEMENT2022_VOXEL_STEP_PC",
    "MIN_NEIGHBORS_FOR_MEDIAN",
    "NEAR_BOUNDARY_KPC",
    "SFD_TO_AV_COEFF",
    "ComposedAv",
    "DustQuery",
    "NeighborhoodAvFeatures",
    "compose_av",
    "galactic_to_xyz",
    "get_default_queries",
    "lallement2022_query",
    "load_lallement2022_cube",
    "neighborhood_av_features",
]
