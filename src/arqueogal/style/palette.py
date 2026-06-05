"""ArqueoGal v1.4 colour constants. Frozen hex codes, do not edit."""

from __future__ import annotations

# Categorical (Okabe-Ito, CVD-safe, in deterministic cycle order).
OKABE_ITO: dict[str, str] = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "red_purple": "#CC79A7",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "yellow":     "#F0E442",
    "black":      "#000000",
}

# Tier mapping (ABSOLUTE; identical across every figure).
TIER: dict[str, str] = {
    "T1": "#009E73",   # green; science-grade
    "T2": "#D55E00",   # vermillion; rare-but-real chemistry / output-OOD
    "T3": "#CC79A7",   # red-purple; do-not-label / input-OOD
}

# Stream mapping (ABSOLUTE).
STREAM: dict[str, str] = {
    "S1": "#0072B2",   # APOGEE x XP training cohort
    "S2": "#D55E00",   # TESS asteroseismic giants, cross-check
    "S3": "#009E73",   # Andrae+23 RGB inference cohort
}

# Slide chrome.
CHROME: dict[str, str] = {
    "title":         "#1A1D2E",
    "body":          "#2B2D42",
    "muted":         "#5C6378",
    "grid":          "#D0D3DC",
    "divider":       "#E5E7EB",
    "surface":       "#FFFFFF",
    "surface_alt":   "#F8F9FB",
    "surface_dark":  "#0E1117",
    "near_black":    "#1A1D2E",
}

ACCENT_PRIMARY = "#0072B2"
ACCENT_WARNING = "#D55E00"
ACCENT_SUCCESS = "#009E73"
