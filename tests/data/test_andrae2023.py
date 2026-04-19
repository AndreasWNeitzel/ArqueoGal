"""Offline tests for arqueogal.data.andrae2023 — §5.2 FITS loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import pandas as pd

from arqueogal.data.andrae2023 import (
    ANDRAE2023_DEFAULT_HDU,
    ANDRAE2023_ZENODO_RECORD,
    KEPT_COLUMNS,
    VIZIER_KEPT_COLUMNS,
    load_andrae2023,
    load_andrae2023_parquet,
)


def _write_fits(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write ``columns`` as HDU 1 of a new FITS, primary HDU empty."""
    table = Table(columns)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data=table)])
    hdul.writeto(path, overwrite=True)


def test_zenodo_record_matches_section_5_2() -> None:
    assert ANDRAE2023_ZENODO_RECORD == "7945154"


def test_default_hdu_is_1() -> None:
    assert ANDRAE2023_DEFAULT_HDU == 1


def test_kept_columns_covers_stratification_axes() -> None:
    for col in (
        "source_id", "teff_xgboost", "logg_xgboost", "mh_xgboost", "phot_g_mean_mag",
    ):
        assert col in KEPT_COLUMNS


def test_load_returns_dataframe_with_default_columns(tmp_path: Path) -> None:
    path = tmp_path / "andrae.fits"
    n = 6
    _write_fits(
        path,
        {
            "source_id": np.arange(1, n + 1, dtype=np.int64),
            "teff_xgboost": np.linspace(4500.0, 5100.0, n, dtype=np.float32),
            "logg_xgboost": np.linspace(1.5, 3.0, n, dtype=np.float32),
            "mh_xgboost": np.linspace(-1.5, 0.2, n, dtype=np.float32),
            "phot_g_mean_mag": np.linspace(8.0, 15.0, n, dtype=np.float32),
        },
    )

    df = load_andrae2023(path)
    assert len(df) == n
    assert list(df.columns) == list(KEPT_COLUMNS)
    assert df["source_id"].dtype == np.int64


def test_load_accepts_subset_of_columns(tmp_path: Path) -> None:
    path = tmp_path / "andrae.fits"
    _write_fits(
        path,
        {
            "source_id": np.arange(3, dtype=np.int64),
            "teff_xgboost": np.ones(3, dtype=np.float32) * 4800.0,
            "logg_xgboost": np.ones(3, dtype=np.float32) * 2.5,
            "mh_xgboost": np.zeros(3, dtype=np.float32),
            "phot_g_mean_mag": np.ones(3, dtype=np.float32) * 12.0,
        },
    )

    df = load_andrae2023(path, columns=("source_id", "teff_xgboost"))
    assert list(df.columns) == ["source_id", "teff_xgboost"]
    assert len(df) == 3


def test_load_raises_on_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "andrae.fits"
    _write_fits(
        path,
        {
            "source_id": np.arange(3, dtype=np.int64),
            "teff_xgboost": np.ones(3, dtype=np.float32) * 4800.0,
            # logg_xgboost intentionally missing
            "mh_xgboost": np.zeros(3, dtype=np.float32),
            "phot_g_mean_mag": np.ones(3, dtype=np.float32) * 12.0,
        },
    )
    with pytest.raises(KeyError, match="logg_xgboost"):
        load_andrae2023(path)


def test_vizier_kept_columns_contains_stratification_axes() -> None:
    for col in ("source_id", "teff", "logg", "fe_h", "g_mag"):
        assert col in VIZIER_KEPT_COLUMNS


def test_load_andrae2023_parquet_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "andrae.parquet"
    n = 4
    df_in = pd.DataFrame(
        {c: (np.arange(n, dtype=np.int64) if c == "source_id"
             else np.full(n, 0.5, dtype=np.float32))
         for c in VIZIER_KEPT_COLUMNS}
    )
    df_in.to_parquet(path, index=False)

    df = load_andrae2023_parquet(path)
    assert list(df.columns) == list(VIZIER_KEPT_COLUMNS)
    assert len(df) == n
    assert df["source_id"].dtype == np.int64


def test_load_andrae2023_parquet_subset(tmp_path: Path) -> None:
    path = tmp_path / "andrae.parquet"
    df_in = pd.DataFrame(
        {c: (np.arange(3, dtype=np.int64) if c == "source_id"
             else np.full(3, 0.5, dtype=np.float32))
         for c in VIZIER_KEPT_COLUMNS}
    )
    df_in.to_parquet(path, index=False)

    df = load_andrae2023_parquet(path, columns=("source_id", "teff", "logg"))
    assert list(df.columns) == ["source_id", "teff", "logg"]


def test_load_case_insensitive_column_lookup(tmp_path: Path) -> None:
    """FITS column names are case-insensitive; loader must honour that."""
    path = tmp_path / "andrae.fits"
    _write_fits(
        path,
        {
            "SOURCE_ID": np.arange(2, dtype=np.int64),
            "TEFF_XGBOOST": np.ones(2, dtype=np.float32) * 4800.0,
            "LOGG_XGBOOST": np.ones(2, dtype=np.float32) * 2.5,
            "MH_XGBOOST": np.zeros(2, dtype=np.float32),
            "PHOT_G_MEAN_MAG": np.ones(2, dtype=np.float32) * 12.0,
        },
    )
    df = load_andrae2023(path)
    assert len(df) == 2
    # Column names follow the *requested* casing (lower-case defaults).
    assert "source_id" in df.columns
    assert "phot_g_mean_mag" in df.columns
