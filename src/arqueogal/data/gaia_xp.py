"""Gaia DR3 XP continuous-mean-spectrum extraction and preprocessing.

Pipeline 1's raw input. The preprocessing sequence is fixed by
``data_acquisition.md`` §6.4 and **must be applied in order**:

1. Ye+2024 NN flux-correction (required — see ``docs/data_acquisition.md`` §6.4).
2. Normalise ``coeff[1:]`` by ``coeff[0]`` (per BP, per RP).
3. ``coeff[0] → log10(coeff[0])``; then z-score across the whole dataset.
4. Propagate errors exactly (no linearisation approximations).
5. Downcast to float32, drop raw coefficients from the analysis-ready file.

This module splits the work into four public entry points so each step is
testable in isolation and the expensive z-score (dataset-level statistic) can
be deferred until all batches are concatenated:

- :func:`fetch_xp_coefficients` — §6.3 batched pyvo query.
- :func:`apply_ye2024_correction` — §6.4 step 1 (Ye+2024 NN flux correction).
- :func:`normalise_xp` — §6.4 steps 2, 4, 5 (no z-score).
- :func:`zscore_c0` — §6.4 step 3.
- :func:`xp_sanity_check` — §6.5.

The §6.3 query drops ``coefficient_correlations`` (too heavy for the 5 GB
budget). Record the drop in provenance via :func:`arqueogal.data.provenance`.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.tap import (
    BATCH_PLACEHOLDER,
    DEFAULT_ASYNC_TIMEOUT_SEC,
    batched_fetch_df,
)

logger = logging.getLogger(__name__)

XP_TABLE: Final[str] = "gaiadr3.xp_continuous_mean_spectrum"
XP_COEFF_LEN: Final[int] = 55
XP_BATCH_SIZE: Final[int] = 5_000
"""§6.3: XP arrays are ~3 KB/row; 5 000 per batch keeps response under AIP
async limits. Larger batches time out."""

_XP_SELECT: Final[str] = """\
       x.source_id,
       x.bp_coefficients, x.bp_coefficient_errors,
       x.rp_coefficients, x.rp_coefficient_errors,
       x.bp_standard_deviation, x.rp_standard_deviation,
       x.bp_n_measurements, x.rp_n_measurements,
       x.bp_n_relevant_bases, x.rp_n_relevant_bases"""

XP_QUERY_ADQL: Final[str] = f"""\
SELECT
{_XP_SELECT}
FROM {XP_TABLE} AS x
WHERE x.source_id IN ({BATCH_PLACEHOLDER})
"""

XP_QUERY_ADQL_UPLOAD: Final[str] = f"""\
SELECT
{_XP_SELECT}
FROM {XP_TABLE} AS x
JOIN tap_upload.ids AS u ON x.source_id = u.source_id
"""
"""TAP UPLOAD variant of :data:`XP_QUERY_ADQL` — mandatory on AIP above
~5 k IDs per batch (inline IN hits the 100 KB gateway ceiling). Use with
:func:`arqueogal.data.tap.batched_upload_fetch_df`."""

_COEFF_COLS: Final[tuple[str, ...]] = (
    "bp_coefficients",
    "bp_coefficient_errors",
    "rp_coefficients",
    "rp_coefficient_errors",
)

_LN10: Final[float] = math.log(10.0)


# -----------------------------------------------------------------------------
# §6.3 fetch
# -----------------------------------------------------------------------------


def fetch_xp_coefficients(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    batch_size: int = XP_BATCH_SIZE,
    mode: Literal["async", "sync", "auto"] = "async",
    checkpoint_dir: Path | str | None = None,
    adql: str = XP_QUERY_ADQL,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Fetch §6.3 XP coefficients for ``source_ids``, concatenated.

    Same contract as :func:`arqueogal.data.gaia_enrich.enrich_source_ids`
    (batched, resumable via per-batch Parquet checkpoints) but with the XP
    query template and the §6.3 default batch size of 5 000.

    Returns
    -------
    pd.DataFrame
        Columns: ``source_id``, four array columns (:data:`_COEFF_COLS`),
        plus ``bp_standard_deviation``, ``rp_standard_deviation``,
        ``bp_n_measurements``, ``rp_n_measurements``,
        ``bp_n_relevant_bases``, ``rp_n_relevant_bases``. Array columns are
        stored as Python lists / object dtype at this stage; downcast happens
        in :func:`normalise_xp`.
    """
    return batched_fetch_df(
        service,
        source_ids,
        adql,
        batch_size=batch_size,
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="xp_batch",
        timeout_sec=timeout_sec,
    )


# -----------------------------------------------------------------------------
# §6.4 step 1 — Ye+2024 flux correction
# -----------------------------------------------------------------------------
#
# Ye et al. 2025 (A&A 695 A75; peer-reviewed of arXiv:2411.19105) publish a
# NN that reduces relative XP flux systematics from 3.2–3.7% to 1.2–2.4%.
# Public release: doi:10.5281/zenodo.14028588 (concept; current version
# zenodo.14712749), archive GaiaXP-correction_V0.zip. We vendor the trained
# weights + scaler under ``data/external/ye2024/GaiaXP-correction_V0/model/``
# (SHA-256 pinned in the provenance sidecar) and port the inference path
# from their ``xp_correction.py`` to our offline batched pipeline.
#
# Key design differences vs the Ye reference implementation:
#   1. Their script fetches XP + photometry per source_id via astroquery.gaia.
#      We do not — astroquery.gaia is forbidden (see docs/data_acquisition.md
#      §14.3). Instead we pass our locally-fetched raw XP coefficients directly
#      into ``gaiaxpy.calibrate`` / ``gaiaxpy.generate`` as a DataFrame
#      (offline mode).
#   2. We batch over 5 000 stars at a time to cap peak memory < 2 GB.
#   3. We return both the dereddened corrected flux (what Pipeline 1 wants)
#      and a per-star flag so downstream code can distinguish real stars from
#      the ones that failed synthetic-photometry construction.

YE2024_SAMPLING_NM: Final[np.ndarray] = np.geomspace(360.0, 990.0, 330)
"""§6.4 / Ye 2025 grid — 330 geometric-spaced wavelengths 360 → 990 nm.
This is the grid the NN was trained on. Overriding is only defensible if
you're porting the NN to a different sampling."""

YE2024_N_OUTPUT: Final[int] = 330

YE2024_INPUT_FEATURES: Final[tuple[str, ...]] = (
    "color_skymap_1",
    "color_skymap_2",
    "color_skymap_3",
    "u",
    "v",
    "g",
    "i",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "bp_g",
    "rp_g",
    "EBV",
)
"""14 NN inputs. Order matches Ye's ``read_nn_model.py`` param list — do
not permute; the scaler vectors are aligned against this ordering."""

YE2024_N_INPUT: Final[int] = 14

YE2024_LAYERS: Final[tuple[int, ...]] = (128, 512, 1024, 1024, 1024, 1024, 1024)
"""Hidden-layer sizes h1..h7 from Ye+2024's NonLinearRegressionModel."""

YE2024_SFD_TO_AV: Final[float] = 2.742
"""Schlafly & Finkbeiner 2011 conversion from SFD E(B-V) to A_V at Rv=3.1."""

YE2024_FLAG_OK: Final[int] = 0
YE2024_FLAG_NO_SYNTH_PHOT: Final[int] = 1
YE2024_FLAG_CALIBRATE_FAIL: Final[int] = 2


def apply_ye2024_correction(  # noqa: PLR0913, PLR0915 — tuning knobs + batch loop
    xp_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    *,
    model_dir: Path | str | None = None,
    batch_size: int = 5_000,
    sampling_nm: np.ndarray = YE2024_SAMPLING_NM,
    device: str | None = None,
    deredden: bool = True,
) -> pd.DataFrame:
    """Apply Ye+2024 NN flux-correction to raw XP coefficients.

    Ports the inference path from Ye+2024 ``xp_correction.py`` (Zenodo
    concept DOI ``10.5281/zenodo.14028588``; current version record
    ``14712749``). The neural network operates on *sampled* spectra — we
    run ``gaiaxpy.calibrate`` offline on the raw coefficients, apply
    CCM89 + SFD dereddening, and predict the flux correction from 14
    colour + photometry + E(B-V) inputs. Output is the *dereddened
    corrected flux*, ready to feed Pipeline 1.

    Parameters
    ----------
    xp_df
        Raw XP rows, one per ``source_id``, with the schema returned by
        :func:`fetch_xp_coefficients` (``bp_coefficients``,
        ``rp_coefficients`` as length-55 array columns, plus the standard
        deviations and bases counts).
    coords_df
        Minimum columns ``(source_id, ra, dec)`` — ICRS degrees at
        Ep=2016.0. Joined on ``source_id`` internally. Any star in
        ``xp_df`` missing from ``coords_df`` is dropped with a warning.
    model_dir
        Directory containing ``nn_model_pattern.pth``, ``scaler_mean.txt``,
        ``scaler_scale.txt``. Defaults to the vendored copy under
        ``data/external/ye2024/GaiaXP-correction_V0/model``.
    batch_size
        Stars per inference batch (default 5000). Larger batches push
        peak memory past 4 GB due to the ``calibrate`` output array.
    sampling_nm
        Wavelength grid in nm. Defaults to :data:`YE2024_SAMPLING_NM` —
        the exact grid Ye's NN was trained on.
    device
        ``"cuda"`` / ``"cpu"``. ``None`` picks CUDA when available.
    deredden
        ``True`` → return dereddened corrected flux (the Pipeline 1 input).
        ``False`` → re-redden the corrected spectrum for observation-frame
        diagnostics.

    Returns
    -------
    pd.DataFrame
        One row per input ``source_id`` that joined to ``coords_df``,
        in the same order as ``xp_df.source_id``. Columns:

        - ``source_id`` (int64)
        - ``corrected_flux`` (float32 array, length ``len(sampling_nm)``,
          NaN for stars with ``ye2024_flag > 0``)
        - ``a_v_sfd`` (float32) — SFD A_V used for dereddening
        - ``ye2024_flag`` (int8) — :data:`YE2024_FLAG_OK` (0) on success,
          :data:`YE2024_FLAG_NO_SYNTH_PHOT` (1) when synthetic photometry
          had NaN inputs, :data:`YE2024_FLAG_CALIBRATE_FAIL` (2) when
          ``gaiaxpy.calibrate`` failed on that row.

        The wavelength grid is not carried per-row — it is constant and
        available as :data:`YE2024_SAMPLING_NM`. Store it alongside the
        output in the provenance sidecar.

    Notes
    -----
    The NN operates on *sampled* spectra, not Hermite coefficients.
    Downstream Pipeline-1 code must either (a) feed the corrected
    sampled flux directly into the model, or (b) re-project onto the
    Hermite basis before applying §6.4 steps 2-5 (normalise by c_0,
    log+zscore c_0). This choice is left to
    ``src/arqueogal/xp_abundances/`` — do not silently pick one here.
    """
    import torch
    from gaiaxpy import PhotometricSystem, calibrate, generate

    _patch_gaiaxpy_cast_output()

    if "source_id" not in xp_df.columns:
        raise KeyError("xp_df must include a 'source_id' column")
    for col in ("source_id", "ra", "dec"):
        if col not in coords_df.columns:
            raise KeyError(f"coords_df must include a {col!r} column")
    if len(sampling_nm) != YE2024_N_OUTPUT:
        raise ValueError(
            f"sampling_nm must have {YE2024_N_OUTPUT} points (NN output width); "
            f"got {len(sampling_nm)}"
        )

    resolved_model_dir = _resolve_ye2024_model_dir(model_dir)
    picked_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "apply_ye2024_correction: model=%s, device=%s, batch=%d, n=%d",
        resolved_model_dir,
        picked_device,
        batch_size,
        len(xp_df),
    )

    nn_model, scaler_mean, scaler_scale = _load_ye2024_model(resolved_model_dir, picked_device)

    # Align xp_df to coords_df on source_id. We preserve xp_df's ordering.
    coords_trim = coords_df[["source_id", "ra", "dec"]].drop_duplicates("source_id")
    joined = xp_df.merge(coords_trim, on="source_id", how="left")
    missing_coords = int(joined["ra"].isna().sum())
    if missing_coords:
        logger.warning(
            "%d/%d stars have no (ra, dec) in coords_df — dropping",
            missing_coords,
            len(joined),
        )
        joined = joined.dropna(subset=["ra", "dec"]).reset_index(drop=True)

    n = len(joined)
    if n == 0:
        return pd.DataFrame(
            {
                "source_id": np.empty(0, dtype=np.int64),
                "corrected_flux": pd.Series([], dtype=object),
                "a_v_sfd": np.empty(0, dtype=np.float32),
                "ye2024_flag": np.empty(0, dtype=np.int8),
            }
        )

    # Pre-compute SFD A_V once — fast, vectorised.
    a_v_sfd = _sfd_av(joined["ra"].to_numpy(), joined["dec"].to_numpy())

    out_flux = np.full((n, YE2024_N_OUTPUT), np.nan, dtype=np.float32)
    out_flag = np.full(n, YE2024_FLAG_OK, dtype=np.int8)

    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        batch = _with_gaiaxpy_required_cols(joined.iloc[lo:hi])
        logger.info("Ye+2024 batch %d..%d / %d", lo, hi, n)
        try:
            calib, _ = calibrate(
                batch,
                sampling=sampling_nm,
                truncation=True,
                save_file=False,
            )
        except Exception as exc:  # noqa: BLE001 — GaiaXPy may raise various types
            logger.warning(
                "gaiaxpy.calibrate failed on batch %d..%d (%s); marking flag=2",
                lo,
                hi,
                exc,
            )
            out_flag[lo:hi] = YE2024_FLAG_CALIBRATE_FAIL
            continue

        flux = np.vstack(calib["flux"].to_numpy()).astype(np.float64)
        # Per-star dereddening via CCM89.
        av_batch = a_v_sfd[lo:hi]
        flux_dered = _ccm89_deredden(flux, sampling_nm, av_batch)

        syn_sky = generate(
            batch,
            photometric_system=PhotometricSystem.Sky_Mapper,
            error_correction=True,
            save_file=False,
        )
        syn_gaia = generate(
            batch,
            photometric_system=PhotometricSystem.Gaia_DR3_Vega,
            error_correction=True,
            save_file=False,
        )
        # gaiaxpy may drop rows where synthesis fails; re-align to batch on
        # source_id so syn_sky / syn_gaia always have len(batch) rows, with
        # NaN where gaiaxpy couldn't synthesize. _build_ye2024_features then
        # propagates the NaN into `features`, which the `valid` mask catches.
        syn_sky = _align_to_batch(batch, syn_sky)
        syn_gaia = _align_to_batch(batch, syn_gaia)
        features = _build_ye2024_features(syn_sky, syn_gaia, av_batch)
        # features: (N_batch, 14), with NaN where any passband was missing.
        valid = np.isfinite(features).all(axis=1)
        invalid = ~valid
        if invalid.any():
            out_flag[lo:hi][invalid] = YE2024_FLAG_NO_SYNTH_PHOT

        if valid.any():
            x = (features[valid] - scaler_mean) / scaler_scale
            x_t = torch.from_numpy(x.astype(np.float32)).to(picked_device)
            with torch.inference_mode():
                nn_out = nn_model(x_t).cpu().numpy()  # (k, 330)
            # Ye's correction recipe: flux_correct = flux - NN * mean(flux).
            mean_flux = np.nanmean(flux_dered[valid], axis=1, keepdims=True)
            corrected = flux_dered[valid] - nn_out * mean_flux
            if not deredden:
                corrected = _ccm89_redden(corrected, sampling_nm, av_batch[valid])
            out_flux[lo:hi][valid] = corrected.astype(np.float32)

    n_ok = int((out_flag == YE2024_FLAG_OK).sum())
    n_nophot = int((out_flag == YE2024_FLAG_NO_SYNTH_PHOT).sum())
    n_failcal = int((out_flag == YE2024_FLAG_CALIBRATE_FAIL).sum())
    logger.info(
        "Ye+2024 done: %d OK, %d no-synth-phot, %d calibrate-failed",
        n_ok,
        n_nophot,
        n_failcal,
    )

    return pd.DataFrame(
        {
            "source_id": joined["source_id"].to_numpy().astype(np.int64),
            "corrected_flux": [row for row in out_flux],
            "a_v_sfd": a_v_sfd.astype(np.float32),
            "ye2024_flag": out_flag,
        }
    )


def _resolve_ye2024_model_dir(model_dir: Path | str | None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    # Repo layout: src/arqueogal/data/gaia_xp.py → parents[3] is repo root.
    repo = Path(__file__).resolve().parents[3]
    return repo / "data" / "external" / "ye2024" / "GaiaXP-correction_V0" / "model"


def _load_ye2024_model(model_dir: Path, device: str):  # noqa: ANN202 — torch.nn.Module
    """Instantiate Ye+2024's NN and load weights + scaler."""
    import torch
    import torch.nn as nn

    mean = np.loadtxt(model_dir / "scaler_mean.txt").astype(np.float32)
    scale = np.loadtxt(model_dir / "scaler_scale.txt").astype(np.float32)
    if mean.shape != (YE2024_N_INPUT,) or scale.shape != (YE2024_N_INPUT,):
        raise ValueError(
            f"scaler shape mismatch: expected ({YE2024_N_INPUT},), "
            f"got mean={mean.shape}, scale={scale.shape}"
        )

    class _Ye2024Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h1, h2, h3, h4, h5, h6, h7 = YE2024_LAYERS
            self.linear1 = nn.Linear(YE2024_N_INPUT, h1)
            self.linear2 = nn.Linear(h1, h2)
            self.linear3 = nn.Linear(h2, h3)
            self.linear4 = nn.Linear(h3, h4)
            self.linear5 = nn.Linear(h4, h5)
            self.linear6 = nn.Linear(h5, h6)
            self.linear7 = nn.Linear(h6, h7)
            self.linear8 = nn.Linear(h7, YE2024_N_OUTPUT)

        def forward(self, x):  # noqa: ANN001, ANN201 — torch.Tensor
            relu = nn.functional.relu
            x = relu(self.linear1(x))
            x = relu(self.linear2(x))
            x = relu(self.linear3(x))
            x = relu(self.linear4(x))
            x = relu(self.linear5(x))
            x = relu(self.linear6(x))
            x = relu(self.linear7(x))
            return self.linear8(x)

    net = _Ye2024Net()
    state = torch.load(model_dir / "nn_model_pattern.pth", map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net, mean, scale


def _sfd_av(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Query SFD E(B-V) and convert to A_V via :data:`YE2024_SFD_TO_AV`."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from dustmaps.sfd import SFDQuery

    coords = SkyCoord(
        ra=np.asarray(ra_deg) * u.deg,
        dec=np.asarray(dec_deg) * u.deg,
        frame="icrs",
    )
    ebv = np.asarray(SFDQuery()(coords), dtype=np.float64)
    return (ebv * YE2024_SFD_TO_AV).astype(np.float32)


_GAIAXPY_N_CORR: Final[int] = XP_COEFF_LEN * (XP_COEFF_LEN - 1) // 2
"""Length of the lower-triangular correlation vector GaiaXPy expects when
``bp_n_parameters == rp_n_parameters == :data:`XP_COEFF_LEN`\u200a``. 55 × 54 / 2 = 1485."""


def _with_gaiaxpy_required_cols(batch: pd.DataFrame) -> pd.DataFrame:
    """Attach the columns ``gaiaxpy.{calibrate,generate}`` require but §6.3
    intentionally did not fetch.

    ``bp_n_parameters`` / ``rp_n_parameters`` — set to :data:`XP_COEFF_LEN`
    (55) because our §6.3 ADQL selects the fixed-length coefficient arrays;
    Gaia DR3 XP continuous records are themselves 55-length.

    ``bp_coefficient_correlations`` / ``rp_coefficient_correlations`` —
    populated with zero vectors of length :data:`_GAIAXPY_N_CORR`. §6.1
    drops covariances per the 5 GB budget (each band is ~12 KB/star; 5.4 GB
    total across the union). GaiaXPy only consumes correlations to
    propagate flux errors; Ye+2024 only reads the mean flux from
    ``calibrate`` and the mean magnitudes from ``generate``, so zeroing the
    correlations affects the returned ``flux_error`` / ``flux_error_*``
    columns (unused here) but not the quantities Ye's NN ingests.
    Provenance records this substitution — see ``ingest_xp.py`` corrections
    list.
    """
    out = batch.copy()
    n = len(out)
    zero_corr = np.zeros(_GAIAXPY_N_CORR, dtype=np.float32)
    if "bp_n_parameters" not in out.columns:
        out["bp_n_parameters"] = np.int16(XP_COEFF_LEN)
    if "rp_n_parameters" not in out.columns:
        out["rp_n_parameters"] = np.int16(XP_COEFF_LEN)
    if "bp_coefficient_correlations" not in out.columns:
        out["bp_coefficient_correlations"] = [zero_corr for _ in range(n)]
    if "rp_coefficient_correlations" not in out.columns:
        out["rp_coefficient_correlations"] = [zero_corr for _ in range(n)]
    return out


def _align_to_batch(batch: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    """Left-merge gaiaxpy photometry output back to batch on ``source_id``.

    gaiaxpy's ``generate`` may drop rows where synthesis fails (or leave NA
    source_ids). A left merge against batch restores row count and ordering;
    missing photometry columns become NaN, which
    :func:`_build_ye2024_features` then propagates into the feature matrix so
    the ``valid`` mask downstream catches them as
    :data:`YE2024_FLAG_NO_SYNTH_PHOT`.
    """
    batch_sids = batch["source_id"].to_numpy()
    syn_sids_raw = syn["source_id"]
    if syn_sids_raw.dtype.name == "Int64":
        mask = syn_sids_raw.notna()
        sub = syn.loc[mask].copy()
        sub["source_id"] = sub["source_id"].astype("int64")
    else:
        sub = syn.copy()
    sub = sub.drop_duplicates("source_id", keep="first")
    keep_cols = [c for c in sub.columns if c != "source_id"]
    left = pd.DataFrame({"source_id": batch_sids})
    out = left.merge(sub, on="source_id", how="left")
    # Guarantee row count matches batch exactly.
    assert len(out) == len(batch), f"_align_to_batch row mismatch: {len(out)} != {len(batch)}"
    # Keep only the photometry columns so downstream .to_numpy() calls see
    # np.nan (object dtype rather than pandas nullable) — matches the pre-
    # patch behavior _build_ye2024_features was written against.
    for col in keep_cols:
        if out[col].dtype.kind == "O":
            continue
        out[col] = out[col].astype("float64")
    return out


_GAIAXPY_CAST_PATCHED: bool = False


def _patch_gaiaxpy_cast_output() -> None:
    """Tolerate NA in gaiaxpy's internal ``cast_output`` source_id/solution_id cast.

    When a batch contains even a single star for which gaiaxpy cannot
    synthesize photometry, its internal ``photometry_df`` has NA in the
    ``source_id``/``solution_id`` column. Vanilla
    :func:`gaiaxpy.core.generic_functions.cast_output` then crashes with
    ``ValueError: cannot convert NA to integer`` on ``.astype('int64')``,
    aborting the whole batch. This patch swaps in a tolerant version that
    falls back to pandas' nullable ``Int64`` when NA is present — yielding
    a DataFrame whose non-source_id photometry columns carry NaN for the
    failing rows, which :func:`_build_ye2024_features` then catches via
    :func:`np.isfinite` so the batch survives with only the bad rows
    flagged :data:`YE2024_FLAG_NO_SYNTH_PHOT`.
    """
    global _GAIAXPY_CAST_PATCHED
    if _GAIAXPY_CAST_PATCHED:
        return
    from gaiaxpy.core import generic_functions as _gf

    _cast_int_columns = ("source_id", "solution_id")

    def _tolerant_cast_output(output):  # noqa: ANN001
        df = output if isinstance(output, pd.DataFrame) else output.data
        for column in df.columns:
            if column in _cast_int_columns:
                try:
                    df[column] = df[column].astype("int64")
                except (ValueError, TypeError):
                    df[column] = df[column].astype("Int64")
        return df

    _gf.cast_output = _tolerant_cast_output
    # Downstream modules import by name, so rebind at each call site too.
    for _mod_path in (
        "gaiaxpy.calibrator.calibrator",
        "gaiaxpy.converter.converter",
        "gaiaxpy.generator.generator",
        "gaiaxpy.error_correction.error_correction",
        "gaiaxpy.colour_equation.xp_filter_system_colour_equation",
        "gaiaxpy.input_reader.file_reader",
    ):
        import importlib

        try:
            _mod = importlib.import_module(_mod_path)
            if hasattr(_mod, "cast_output"):
                _mod.cast_output = _tolerant_cast_output
        except ImportError:
            continue
    _GAIAXPY_CAST_PATCHED = True


def _ccm89_deredden(flux: np.ndarray, sampling_nm: np.ndarray, a_v: np.ndarray) -> np.ndarray:
    """Per-star CCM89 dereddening: `flux * 10^(0.4 * A_lambda)`.

    ``flux`` has shape (N, M); ``sampling_nm`` shape (M,); ``a_v`` shape (N,).
    """
    import extinction

    wave_ang = np.asarray(sampling_nm) * 10.0
    out = np.empty_like(flux)
    for i, av in enumerate(a_v):
        if not np.isfinite(av):
            out[i] = flux[i]
            continue
        ext = extinction.ccm89(wave=wave_ang, a_v=float(av), r_v=3.1, unit="aa")
        out[i] = extinction.remove(extinction=ext, flux=flux[i], inplace=False)
    return out


def _ccm89_redden(flux: np.ndarray, sampling_nm: np.ndarray, a_v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_ccm89_deredden` — apply extinction back."""
    import extinction

    wave_ang = np.asarray(sampling_nm) * 10.0
    out = np.empty_like(flux)
    for i, av in enumerate(a_v):
        if not np.isfinite(av):
            out[i] = flux[i]
            continue
        ext = extinction.ccm89(wave=wave_ang, a_v=float(av), r_v=3.1, unit="aa")
        out[i] = extinction.apply(extinction=ext, flux=flux[i], inplace=False)
    return out


def _build_ye2024_features(
    syn_sky: pd.DataFrame, syn_gaia: pd.DataFrame, a_v: np.ndarray
) -> np.ndarray:
    """Construct the (N, 14) Ye+2024 input feature matrix from synthetic photometry.

    Column ordering matches :data:`YE2024_INPUT_FEATURES` and Ye's
    ``read_nn_model.py`` ``para`` list.  EBV is recovered from A_V via the
    same constant Ye uses (``A_V / 2.742``) rather than re-querying SFD.
    """
    u_mag = syn_sky["SkyMapper_mag_u"].to_numpy()
    v_mag = syn_sky["SkyMapper_mag_v"].to_numpy()
    g_mag = syn_sky["SkyMapper_mag_g"].to_numpy()
    i_mag = syn_sky["SkyMapper_mag_i"].to_numpy()
    G = syn_gaia["GaiaDr3Vega_mag_G"].to_numpy()
    BP = syn_gaia["GaiaDr3Vega_mag_BP"].to_numpy()
    RP = syn_gaia["GaiaDr3Vega_mag_RP"].to_numpy()
    ebv = np.asarray(a_v, dtype=np.float64) / YE2024_SFD_TO_AV

    feats = np.stack(
        [
            g_mag - i_mag,  # color_skymap_1 = g - i
            v_mag - g_mag - 0.9 * (g_mag - i_mag),  # color_skymap_2
            u_mag - v_mag - 0.9 * (g_mag - i_mag),  # color_skymap_3
            u_mag,
            v_mag,
            g_mag,
            i_mag,
            G,
            BP,
            RP,
            BP - RP,
            BP - G,
            RP - G,
            ebv,
        ],
        axis=1,
    ).astype(np.float64)
    return feats


# -----------------------------------------------------------------------------
# §6.4 steps 2, 4, 5 — per-star normalisation and error propagation
# -----------------------------------------------------------------------------


def normalise_xp(df: pd.DataFrame) -> pd.DataFrame:
    """Per-star XP normalisation and error propagation (§6.4 steps 2, 4, 5).

    For each star and each band ∈ {BP, RP}:

    - Validates that the coefficient array has length :data:`XP_COEFF_LEN`.
    - Validates that ``c_0 > 0`` and no coefficient is NaN (§6.5).
    - Produces a 55-element ``{band}_coeffs_norm`` array:

      - ``[0] = log10(c_0)`` (un-z-scored — feed to :func:`zscore_c0` later).
      - ``[i] = c_i / c_0`` for ``i = 1..54``.

    - Produces a 55-element ``{band}_coeff_errs_norm`` array:

      - ``[0] = σ_0 / (c_0 · ln 10)`` — error on ``log10(c_0)``.
      - ``[i] = sqrt( (σ_i/c_0)² + (c_i·σ_0/c_0²)² )`` — error on ratio.

    - Downcasts all output arrays to float32 (§6.2: loss < 1e-7, well below
      the ~1e-4 relative precision required for abundance ML).

    The raw coefficient columns (:data:`_COEFF_COLS`) are dropped from the
    returned DataFrame — retain the input if you need them for diagnostics.
    """
    required = {"source_id", *_COEFF_COLS}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"normalise_xp requires columns {sorted(missing)}")

    bp_coeffs_norm, bp_errs_norm = _normalise_band(
        df["bp_coefficients"], df["bp_coefficient_errors"], band="bp"
    )
    rp_coeffs_norm, rp_errs_norm = _normalise_band(
        df["rp_coefficients"], df["rp_coefficient_errors"], band="rp"
    )

    out = df.drop(columns=list(_COEFF_COLS)).copy()
    out["bp_coeffs_norm"] = list(bp_coeffs_norm)
    out["bp_coeff_errs_norm"] = list(bp_errs_norm)
    out["rp_coeffs_norm"] = list(rp_coeffs_norm)
    out["rp_coeff_errs_norm"] = list(rp_errs_norm)
    # Convenience scalar columns — the pre-z-score log10(c0) per band.
    out["bp_c0_log"] = bp_coeffs_norm[:, 0].astype(np.float32)
    out["rp_c0_log"] = rp_coeffs_norm[:, 0].astype(np.float32)
    return out


def _normalise_band(
    coeffs_col: pd.Series, errs_col: pd.Series, *, band: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stack, validate, normalise, propagate errors. Returns (coeffs, errs)."""
    coeffs = _stack_arrays(coeffs_col, label=f"{band}_coefficients")
    errs = _stack_arrays(errs_col, label=f"{band}_coefficient_errors")

    if coeffs.shape != errs.shape:
        raise ValueError(f"{band}: coeffs shape {coeffs.shape} != errs shape {errs.shape}")

    # §6.5 checks (positive c0, no NaN).
    c0 = coeffs[:, 0]
    nan_rows = np.isnan(coeffs).any(axis=1) | np.isnan(errs).any(axis=1)
    if nan_rows.any():
        idx = np.flatnonzero(nan_rows)
        raise ValueError(
            f"{band}: {nan_rows.sum()} rows contain NaN in coefficients / errors "
            f"(first offending index {idx[0]}); reject upstream per §6.5."
        )
    nonpos = c0 <= 0
    if nonpos.any():
        idx = np.flatnonzero(nonpos)
        raise ValueError(
            f"{band}: {nonpos.sum()} rows have c_0 <= 0 (first at row {idx[0]}); "
            f"reject upstream per §6.5."
        )

    # §6.4 step 2 — normalise ratios.
    ratios = coeffs.copy()
    ratios[:, 1:] = coeffs[:, 1:] / c0[:, np.newaxis]
    # §6.4 step 3 (the per-star part) — log10 at index 0. Z-score is deferred.
    ratios[:, 0] = np.log10(c0)

    # §6.4 step 4 — exact error propagation.
    ratios_errs = np.empty_like(coeffs)
    sigma_0 = errs[:, 0]
    # Error on log10(c_0) = σ_0 / (c_0 · ln 10).
    ratios_errs[:, 0] = sigma_0 / (c0 * _LN10)
    # Error on c_i / c_0 — exact.
    term1 = errs[:, 1:] / c0[:, np.newaxis]
    term2 = coeffs[:, 1:] * sigma_0[:, np.newaxis] / (c0[:, np.newaxis] ** 2)
    ratios_errs[:, 1:] = np.sqrt(term1**2 + term2**2)

    return ratios.astype(np.float32), ratios_errs.astype(np.float32)


def _stack_arrays(col: pd.Series, *, label: str) -> np.ndarray:
    """Turn an object-dtype column of 55-element arrays into a (N, 55) ndarray."""
    stacked = np.stack([np.asarray(x, dtype=np.float64) for x in col])
    if stacked.ndim != 2 or stacked.shape[1] != XP_COEFF_LEN:
        raise ValueError(f"{label}: expected (N, {XP_COEFF_LEN}) ndarray, got {stacked.shape}")
    return stacked


# -----------------------------------------------------------------------------
# §6.4 step 3 — dataset-level z-score of the zeroth coefficient
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class XpC0Stats:
    """Z-score normaliser state for ``{bp,rp}_c0_log`` — carry with the model."""

    bp_c0_log_mean: float
    bp_c0_log_std: float
    rp_c0_log_mean: float
    rp_c0_log_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bp_c0_log_mean": self.bp_c0_log_mean,
            "bp_c0_log_std": self.bp_c0_log_std,
            "rp_c0_log_mean": self.rp_c0_log_mean,
            "rp_c0_log_std": self.rp_c0_log_std,
        }


def zscore_c0(df: pd.DataFrame, stats: XpC0Stats | None = None) -> tuple[pd.DataFrame, XpC0Stats]:
    """Z-score ``bp_c0_log`` / ``rp_c0_log`` → ``bp_c0_z`` / ``rp_c0_z``.

    On training ingestion, call with ``stats=None`` to fit-and-apply (returns
    the fitted statistics so they can be persisted alongside the model). At
    inference, pass the saved :class:`XpC0Stats` to apply the same transform
    — this is critical: re-fitting the z-score on inference data shifts the
    input distribution.
    """
    for col in ("bp_c0_log", "rp_c0_log"):
        if col not in df.columns:
            raise KeyError(f"zscore_c0 requires {col!r} — run normalise_xp first")

    if stats is None:
        stats = XpC0Stats(
            bp_c0_log_mean=float(df["bp_c0_log"].mean()),
            bp_c0_log_std=float(df["bp_c0_log"].std(ddof=0)),
            rp_c0_log_mean=float(df["rp_c0_log"].mean()),
            rp_c0_log_std=float(df["rp_c0_log"].std(ddof=0)),
        )

    if stats.bp_c0_log_std == 0 or stats.rp_c0_log_std == 0:
        raise ValueError(f"z-score std is zero; stats={stats}")

    out = df.copy()
    out["bp_c0_z"] = (
        (df["bp_c0_log"].astype(np.float64) - stats.bp_c0_log_mean) / stats.bp_c0_log_std
    ).astype(np.float32)
    out["rp_c0_z"] = (
        (df["rp_c0_log"].astype(np.float64) - stats.rp_c0_log_mean) / stats.rp_c0_log_std
    ).astype(np.float32)
    return out, stats


# -----------------------------------------------------------------------------
# §6.5 sanity checks (stand-alone — the per-row checks are inside normalise_xp)
# -----------------------------------------------------------------------------


def xp_sanity_check(df: pd.DataFrame) -> dict[str, int]:
    """Run §6.5 sanity checks on a post-fetch (pre-normalise) DataFrame.

    Returns a dict with counts of offending rows per check. Raises
    ``ValueError`` if any check has a non-zero count.
    """
    counts: dict[str, int] = {}
    for col in _COEFF_COLS:
        if col not in df.columns:
            raise KeyError(f"xp_sanity_check requires column {col!r}")

    for col in ("bp_coefficients", "rp_coefficients"):
        stacked = _stack_arrays(df[col], label=col)
        counts[f"{col}_nan_rows"] = int(np.isnan(stacked).any(axis=1).sum())
        counts[f"{col}_nonpos_c0"] = int((stacked[:, 0] <= 0).sum())

    if "has_xp_continuous" in df.columns:
        missing = (~df["has_xp_continuous"].astype(bool)).sum()
        counts["has_xp_continuous_false"] = int(missing)

    bad = {k: v for k, v in counts.items() if v > 0}
    if bad:
        raise ValueError(f"§6.5 XP sanity checks failed: {bad}")

    return counts


# ---------------------------------------------------------------------------
# §6.4 step 2 — Hermite re-projection of Ye+2024 sampled flux onto coefficients
# ---------------------------------------------------------------------------
#
# Per research_brief.md §7.2 (2026-04-18 decision): Pipeline 1 operates on
# 55 BP + 55 RP coefficients, not on the Ye 330-element sampled flux. The
# re-projection is a linear projection onto a basis that is:
#
#  (1) orthonormal w.r.t. the trapezoidal inner product on YE2024_SAMPLING_NM
#      (physical-integral weighting, NOT counting measure — c_0 preserves its
#      integrated-flux-proxy interpretation);
#  (2) derived from per-band Hermite functions φ_n(u) = H_n(u) e^{-u²/2} /
#      sqrt(2^n n! sqrt(π)) on a hard-coded pseudo-wavelength u = (λ−λ₀)/Δλ;
#  (3) built once on a fixed reference grid at import time and cached. The
#      R factor of QR(√W · Φ_raw) is the canonical basis definition; its
#      SHA-256 fingerprint is written to every provenance sidecar that emits
#      coefficients. Basis drift (any change to grid, centres, scales, or
#      band ranges) must bump HERMITE_REPROJECTION_VERSION and re-materialise
#      downstream artefacts.
#
# Errors (§6.4 step 5): Ye+2024 emits a point prediction, so per-coefficient
# errors are not propagated here. The chosen proxy is the per-star
# reprojection_residual_rms feature. See research_brief.md §7.2 and the
# option-(a) decision in data_acquisition.md §6.4 step 5.

HERMITE_REPROJECTION_VERSION: Final[str] = "v1.0"
"""Bump when any of the constants below change."""

HERMITE_BP_RANGE_NM: Final[tuple[float, float]] = (360.0, 660.0)
HERMITE_BP_CENTER_NM: Final[float] = 510.0
HERMITE_BP_SCALE_NM: Final[float] = 30.0
"""BP: u ∈ [-5.0, +5.0] over [360, 660] nm."""

HERMITE_RP_RANGE_NM: Final[tuple[float, float]] = (660.0, 990.0)
HERMITE_RP_CENTER_NM: Final[float] = 825.0
HERMITE_RP_SCALE_NM: Final[float] = 33.0
"""RP: u ∈ [-5.0, +5.0] over [660, 990] nm."""

HERMITE_N_BASIS: Final[int] = 55

XP_FIT_FLAG_OK: Final[int] = 0
XP_FIT_FLAG_RESIDUAL_HIGH: Final[int] = 1
"""`xp_fit_flag` values. Threshold comes from the smoke-test p99; set at
materialisation time and persisted in the output provenance."""


def _hermite_functions_stable(n_max: int, u: np.ndarray) -> np.ndarray:
    """Evaluate φ_0 … φ_{n_max} at ``u`` via the stable 3-term recurrence.

    The physicist Hermite functions
    ``φ_n(u) = H_n(u) e^{-u²/2} / √(2^n n! √π)``
    satisfy
    ``φ_{n+1}(u) = u √(2/(n+1)) φ_n(u) − √(n/(n+1)) φ_{n-1}(u)``
    with ``φ_0 = e^{-u²/2} / π^(1/4)``. The recurrence keeps every ``φ_n`` in
    the normal numerical range, unlike direct ``H_n``·exp which evaluates
    catastrophically-cancelling products of 10^40-scale values at ``n ≳ 30``.

    Returns
    -------
    np.ndarray
        Shape ``(u.size, n_max + 1)`` float64 — rows are grid points, columns
        are ``n = 0, 1, …, n_max``.
    """
    phi = np.empty((n_max + 1, u.size), dtype=np.float64)
    phi[0] = np.exp(-0.5 * u * u) / (np.pi**0.25)
    if n_max >= 1:
        phi[1] = u * math.sqrt(2.0) * phi[0]
    for n in range(1, n_max):
        phi[n + 1] = u * math.sqrt(2.0 / (n + 1)) * phi[n] - math.sqrt(n / (n + 1)) * phi[n - 1]
    return phi.T


def _trapezoidal_weights(grid_nm: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights on an arbitrarily-spaced 1-D grid."""
    w = np.empty_like(grid_nm, dtype=np.float64)
    w[0] = 0.5 * (grid_nm[1] - grid_nm[0])
    w[-1] = 0.5 * (grid_nm[-1] - grid_nm[-2])
    w[1:-1] = 0.5 * (grid_nm[2:] - grid_nm[:-2])
    return w


@lru_cache(maxsize=1)
def _build_hermite_basis() -> dict:
    """Construct per-band orthonormal Hermite bases on YE2024_SAMPLING_NM.

    Returns
    -------
    dict with keys ``bp``, ``rp`` (each a per-band dict with ``mask``, ``lam``,
    ``w``, ``phi_raw``, ``R``, ``psi``), plus ``fingerprint_sha256``.

    ``psi`` is the orthonormal basis evaluated at the band's grid points:
    ``psi.T @ diag(w) @ psi == I``. Coefficients are ``c = psi.T @ (w * f)``.
    Reconstruction is ``f_hat = psi @ c``.
    """
    lam_full = YE2024_SAMPLING_NM
    w_full = _trapezoidal_weights(lam_full)

    bands = {
        "bp": (HERMITE_BP_RANGE_NM, HERMITE_BP_CENTER_NM, HERMITE_BP_SCALE_NM),
        "rp": (HERMITE_RP_RANGE_NM, HERMITE_RP_CENTER_NM, HERMITE_RP_SCALE_NM),
    }
    out: dict = {}
    for band, ((lo, hi), center, scale) in bands.items():
        # Hard split at 660 nm: BP is [lo, hi) (right-open), RP is [lo, hi].
        if band == "bp":
            mask = (lam_full >= lo) & (lam_full < hi)
        else:
            mask = (lam_full >= lo) & (lam_full <= hi)
        lam = lam_full[mask].astype(np.float64)
        w = w_full[mask].astype(np.float64)
        u = (lam - center) / scale
        phi_raw = _hermite_functions_stable(HERMITE_N_BASIS - 1, u)
        # Orthonormal w.r.t. ⟨f, g⟩_W = Σ w_i f_i g_i: QR of √W · Φ gives a Q
        # with Qᵀ Q = I, hence Ψ := Q / √W satisfies Ψᵀ W Ψ = I. Using Q
        # directly (instead of Ψ = Φ R⁻¹) avoids amplifying roundoff through
        # an ill-conditioned R at high n.
        sqrt_w = np.sqrt(w)
        a = phi_raw * sqrt_w[:, None]
        q, r = np.linalg.qr(a, mode="reduced")
        # QR is sign-ambiguous up to column sign of Q / row sign of R. Enforce
        # the positive-diagonal convention on R — this fixes the Ψ columns
        # to have ⟨ψ_n, φ_n⟩_W > 0, so c_0 = ⟨ψ_0, f⟩_W has the sign of the
        # integrated flux (positive for normal stars).
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        q = q * signs[None, :]
        r = r * signs[:, None]
        psi = q / sqrt_w[:, None]
        out[band] = {
            "mask": mask,
            "lam": lam,
            "w": w,
            "phi_raw": phi_raw,
            "R": r,
            "psi": psi,
        }

    # Canonical fingerprint — hashes version + constants + grid + both Ψ.
    # Ψ (not R) is the quantity actually used for coefficient extraction, so
    # hashing it makes the fingerprint bit-exact to the basis the pipeline
    # uses. Any change to the grid or constants perturbs Ψ and thus the hash.
    h = hashlib.sha256()
    h.update(HERMITE_REPROJECTION_VERSION.encode())
    h.update(
        np.array(
            [
                HERMITE_BP_CENTER_NM,
                HERMITE_BP_SCALE_NM,
                HERMITE_RP_CENTER_NM,
                HERMITE_RP_SCALE_NM,
                *HERMITE_BP_RANGE_NM,
                *HERMITE_RP_RANGE_NM,
                HERMITE_N_BASIS,
            ],
            dtype=np.float64,
        ).tobytes()
    )
    h.update(lam_full.astype(np.float64).tobytes())
    h.update(out["bp"]["psi"].astype(np.float64).tobytes())
    h.update(out["rp"]["psi"].astype(np.float64).tobytes())
    out["fingerprint_sha256"] = h.hexdigest()
    return out


def reproject_ye_to_hermite(flux_sampled: np.ndarray) -> dict:
    """Project Ye+2024 sampled flux onto orthonormal Hermite coefficients.

    Parameters
    ----------
    flux_sampled
        Array of shape ``(N, 330)``; each row is a Ye-corrected sampled flux
        on :data:`YE2024_SAMPLING_NM`.

    Returns
    -------
    dict
        - ``bp_coeffs`` (N, 55) float32
        - ``rp_coeffs`` (N, 55) float32
        - ``reprojection_residual_rms_bp`` (N,) float32
        - ``reprojection_residual_rms_rp`` (N,) float32
        - ``reprojection_residual_rms`` (N,) float32 — combined-band RMS
        - ``basis_version`` str
        - ``basis_fingerprint_sha256`` str

    Notes
    -----
    No coefficient-level errors are returned: Ye+2024 does not expose a
    predictive distribution. Use ``reprojection_residual_rms`` as the
    per-star fit-quality proxy (option (a) in data_acquisition.md §6.4).
    """
    if flux_sampled.ndim != 2 or flux_sampled.shape[1] != YE2024_N_OUTPUT:
        raise ValueError(
            f"flux_sampled must have shape (N, {YE2024_N_OUTPUT}), got {flux_sampled.shape}"
        )
    basis = _build_hermite_basis()
    flux = np.asarray(flux_sampled, dtype=np.float64)

    results: dict = {
        "basis_version": HERMITE_REPROJECTION_VERSION,
        "basis_fingerprint_sha256": basis["fingerprint_sha256"],
    }
    rms_bands: list[np.ndarray] = []
    for band in ("bp", "rp"):
        b = basis[band]
        f_band = flux[:, b["mask"]]  # (N, M_band)
        # c = Ψᵀ W f  → shape (N, 55)
        coeffs = f_band @ (b["psi"] * b["w"][:, None])
        f_hat = coeffs @ b["psi"].T
        resid = f_band - f_hat
        rms = np.sqrt(np.mean(resid * resid, axis=1))
        results[f"{band}_coeffs"] = coeffs.astype(np.float32)
        results[f"reprojection_residual_rms_{band}"] = rms.astype(np.float32)
        rms_bands.append(resid)
    resid_all = np.concatenate(rms_bands, axis=1)
    results["reprojection_residual_rms"] = np.sqrt(np.mean(resid_all * resid_all, axis=1)).astype(
        np.float32
    )
    return results


__all__ = [
    "HERMITE_BP_CENTER_NM",
    "HERMITE_BP_RANGE_NM",
    "HERMITE_BP_SCALE_NM",
    "HERMITE_N_BASIS",
    "HERMITE_REPROJECTION_VERSION",
    "HERMITE_RP_CENTER_NM",
    "HERMITE_RP_RANGE_NM",
    "HERMITE_RP_SCALE_NM",
    "XP_BATCH_SIZE",
    "XP_COEFF_LEN",
    "XP_FIT_FLAG_OK",
    "XP_FIT_FLAG_RESIDUAL_HIGH",
    "XP_QUERY_ADQL",
    "XP_QUERY_ADQL_UPLOAD",
    "XP_TABLE",
    "YE2024_FLAG_CALIBRATE_FAIL",
    "YE2024_FLAG_NO_SYNTH_PHOT",
    "YE2024_FLAG_OK",
    "YE2024_INPUT_FEATURES",
    "YE2024_LAYERS",
    "YE2024_N_INPUT",
    "YE2024_N_OUTPUT",
    "YE2024_SAMPLING_NM",
    "YE2024_SFD_TO_AV",
    "XpC0Stats",
    "apply_ye2024_correction",
    "fetch_xp_coefficients",
    "normalise_xp",
    "reproject_ye_to_hermite",
    "xp_sanity_check",
    "zscore_c0",
]
