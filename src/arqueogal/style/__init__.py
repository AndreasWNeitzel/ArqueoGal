"""Single source of truth for ArqueoGal v1.4 talk-figure aesthetics.

Every figure script does ``from arqueogal.style import *``.
"""

from arqueogal.style.figure import (
    annotate_corner,
    colorbar,
    fit_aspect_axes,
    hexbin_density,
    median_per_cell,
    reference_line,
    save,
)
from arqueogal.style.labels import LABELS
from arqueogal.style.mpl import OKABE_CYCLE, apply_style
from arqueogal.style.palette import (
    ACCENT_PRIMARY,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    CHROME,
    OKABE_ITO,
    STREAM,
    TIER,
)

__all__ = [
    "ACCENT_PRIMARY",
    "ACCENT_SUCCESS",
    "ACCENT_WARNING",
    "CHROME",
    "LABELS",
    "OKABE_CYCLE",
    "OKABE_ITO",
    "STREAM",
    "TIER",
    "annotate_corner",
    "apply_style",
    "colorbar",
    "fit_aspect_axes",
    "hexbin_density",
    "median_per_cell",
    "reference_line",
    "save",
]
