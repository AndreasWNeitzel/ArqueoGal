"""Tests for utils.plotting — lightweight smoke tests.

These do not compare pixel output. They verify that the helpers run
without error under a non-interactive backend and that ``save_figure``
produces files in the requested formats.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from arqueogal.utils.plotting import (  # noqa: E402
    AA_DOUBLE_COLUMN_IN,
    AA_SINGLE_COLUMN_IN,
    WONG_PALETTE,
    coverage_curve,
    density_2d,
    hexbin_with_colorbar,
    residual_panel,
    save_figure,
    set_aa_style,
)


def test_aa_widths_positive() -> None:
    assert 0 < AA_SINGLE_COLUMN_IN < AA_DOUBLE_COLUMN_IN


def test_wong_palette_has_eight_colours() -> None:
    assert len(WONG_PALETTE) == 8


def test_set_aa_style_applies_rc_params() -> None:
    set_aa_style(usetex=False, colorblind=True, font_size=9.0)
    assert matplotlib.rcParams["font.size"] == 9.0
    assert matplotlib.rcParams["text.usetex"] is False


def test_hexbin_with_colorbar_runs() -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    x = rng.standard_normal(500)
    y = rng.standard_normal(500)
    hb = hexbin_with_colorbar(ax, x, y)
    assert hb is not None
    plt.close(fig)


def test_density_2d_runs() -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    im = density_2d(ax, rng.standard_normal(200), rng.standard_normal(200))
    assert im is not None
    plt.close(fig)


def test_residual_panel_runs() -> None:
    fig, ax = plt.subplots()
    y = np.linspace(0, 1, 50)
    yp = y + 0.01 * np.sin(10 * y)
    residual_panel(ax, y, yp, label="[Fe/H]", sigma=np.full_like(y, 0.02))
    plt.close(fig)


def test_coverage_curve_matches_gaussian_for_gaussian_errors() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    sigma = np.ones(n)
    errors = rng.standard_normal(n)  # true 1σ
    fig, ax = plt.subplots()
    coverage_curve(ax, sigma, errors)
    # If we pulled out the line we plotted, empirical at 1σ should be ~0.68.
    lines = ax.get_lines()
    assert len(lines) >= 2
    # first line is empirical.
    xs, ys = lines[0].get_xdata(), lines[0].get_ydata()
    at_one = np.interp(1.0, xs, ys)
    assert abs(at_one - 0.6827) < 0.05
    plt.close(fig)


def test_save_figure_writes_multiple_formats(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "fig", formats=("png", "pdf"))
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
    plt.close(fig)
