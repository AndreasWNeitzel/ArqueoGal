# ArqueoGal v1.4 talk style guide (single source of truth)

The constants in this file are mirrored in
``src/arqueogal/style/`` (palette.py, mpl.py, labels.py, figure.py).
Every figure script does ``from arqueogal.style import *``.
Anything inline-redefined is a defect.

## Palette (frozen hex)

```python
OKABE_ITO = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "red_purple": "#CC79A7",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "yellow":     "#F0E442",   # always with edgecolor on white
    "black":      "#000000",
}

# Tier mapping (ABSOLUTE; identical across every figure)
TIER = {"T1": "#009E73", "T2": "#D55E00", "T3": "#CC79A7"}

# Stream mapping (ABSOLUTE)
STREAM = {"S1": "#0072B2", "S2": "#D55E00", "S3": "#009E73"}

# Slide chrome
CHROME = {
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
```

## Colormaps

| Use case                           | Default     |
|-------------------------------------|-------------|
| Density / scalar map / counts       | viridis     |
| Density on dark backgrounds         | magma       |
| Grayscale-survival, CVD-strict      | cividis     |
| Diverging around 0 (residuals, alpha) | BrBG_r    |
| Diverging on instrument systematics | PuOr        |

Banned: jet, rainbow, hsv, gist_rainbow, seismic, nipy_spectral.
Any deviation needs a `# style: deliberate, see docs/decisions/00XX.md`
comment.

Reference / annotation colours:
- 1:1 reference line: `#000000`, lw=0.8, ls="--".
- APOGEE floor: `#000000`, lw=1.0, ls="--", alpha=0.7.
- p99 cuts: matched to the tier they cut (T2 vermillion, T3 red-purple).
- Sun marker: filled white star, edgecolor `#000000`, edgewidth 0.8,
  size 10.

## Typography

Font stack: Inter (preferred), DejaVu Sans fallback, Arial.
Math: dejavusans + mathtext, NOT cm.

| Element                       | Size | Weight | Color    |
|-------------------------------|------|--------|----------|
| Slide title                   | 28   | 600    | #1A1D2E  |
| Slide subtitle                | 14   | 400    | #5C6378  |
| Figure suptitle               | NONE | NONE   | NONE     |
| Panel title (in-figure)       | 13   | 600    | #2B2D42  |
| Axis label                    | 12   | 400    | #2B2D42  |
| Tick label                    | 11   | 400    | #2B2D42  |
| Legend                        | 10   | 400    | #2B2D42  |
| Numeric annotation in panel   | 10   | 400    | #2B2D42  |
| Figure caption                | 10   | italic | #5C6378  |
| Slide footer / page number    | 9    | 400    | #5C6378  |
| Big stat numeric              | 60   | 700    | accent   |

Two weights only: 400 and 600. Italic reserved for captions and
slide-bottom takeaway bands. Sentence case for all titles, panel
titles, axis labels.

## Layout grid

Slide: 16:9, 13.33 x 7.5 in. Margins L/R 0.6 in, T/B 0.4 in.

| Zone        | y range          | Use                            |
|-------------|------------------|--------------------------------|
| Title       | 0.40 to 0.95     | Slide title, 28pt              |
| Subtitle    | 0.95 to 1.45     | Slide subtitle, 14pt           |
| Content     | 1.55 to 6.95     | Figures, cards                 |
| Caption     | 6.40 to 6.85     | Optional bottom takeaway       |
| Footer      | 7.10 to 7.40     | Project mark + page number     |

Figure aspect-ratio templates:
- Wide banner (sample-flow): figsize=(12.0, 2.5)
- Wide hero, single panel: figsize=(10.0, 5.5)
- 2-up compare: figsize=(11.0, 4.5), gridspec 1x2
- 1x3 grid: figsize=(11.0, 4.0)
- 1x4 grid: figsize=(11.0, 4.0)
- 2x2 grid: figsize=(11.0, 6.0)
- 2x3 grid: figsize=(11.0, 5.5)
- Square hero (chemistry plane): figsize=(6.0, 6.0)
- Mollweide stack (2 sky maps): figsize=(11.0, 5.0)

## Math typography

Every axis label routes through `arqueogal.style.labels.LABELS`. Never
inline a string. Examples: `LABELS["Teff"]`, `LABELS["alpha_M"]`,
`LABELS["Rgal"]`.

In `python-pptx` text frames (slide titles / subtitles), keep unicode:
`T_eff`, `log g`, `[α/M]`, `nu_max`. Do NOT mix LaTeX (`$...$`) into
pptx text frames; they render as literal dollar signs.

## Save protocol

```python
def save(fig, name, *, out_dir=None):
    out = (out_dir or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", dpi=300, bbox_inches="tight",
                pad_inches=0.10, transparent=True)
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight",
                pad_inches=0.10, transparent=True)
    plt.close(fig)
```

The deck embeds PNG. PDF is the archival vector copy.

## Banned aesthetics

- No figure suptitle. The slide carries the title.
- No drop shadows. python-pptx default shadows are removed via XML
  manipulation on every shape.
- No accent lines under titles.
- No full-width coloured bars on content slides. Title slide allows a
  single vertical accent rule on the left, 4-pt wide.
- No cream / beige / off-white backgrounds. Surface is `#FFFFFF`
  or `#0E1117`.
- No emoji.
- No em-dashes anywhere in output.
- No centered body text. Titles centered or left-aligned (left
  preferred). Subtitle and body left-aligned.
- No decorative icons in scientific panels.
- No banned colormaps.
- No legend frames (`frameon=False`).
- No grid on density plots. Grid only on line plots, bar charts,
  histograms.

## rcParams (canonical)

See `src/arqueogal/style/mpl.py:apply_style`.

## Per-figure QA checklist

Before save:
- [ ] No suptitle.
- [ ] Every text element uses Inter (or fallback chain).
- [ ] Math labels via `LABELS[...]`.
- [ ] cmap is on the allowed list.
- [ ] vmin / vmax explicitly set (not auto).
- [ ] Reference lines correctly coloured and styled.
- [ ] Annotations not overlapping data or other annotations.
- [ ] Legend frameless.
- [ ] Figure size matches the §3 spec.
- [ ] Saved as both PDF and PNG with the canonical name.
- [ ] Version string in any embedded label is `v1.4`.
