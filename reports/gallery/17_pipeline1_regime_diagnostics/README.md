# Stage 17 — Pipeline 1 regime diagnostics

Per-regime `release_tier` diagnostics for the two Stream-3 arms.

## Figures

- `regime_diagnostics_volume.png` — volume-limited arm (d ≤ 2.5 kpc).
- `regime_diagnostics_uniform.png` — uniform-in-G arm.
- `tier_summary.json` — tier counts per arm.

Each figure is a 3×3 panel:

| row | what |
|---|---|
| 1 | Tier 1 / 2 / 3 stacked fraction vs G-mag, A_V (nbhd-median), distance |
| 2 | Tier-3 fraction on Galactic Mollweide, HR diagram, chemistry plane |
| 3 | σ by label × tier, Tier-3 flag mix, summary bar |

## What the release tiers mean

Defined in `src/arqueogal/xp_abundances/main/release.py` and assigned to
each Stream-3 prediction parquet by `scripts/assign_release_tier.py`.

- **Tier 1 — per-star science**: all structural gates pass.
- **Tier 2 — statistical / ensemble only**: passes the OOD gates but
  carries a caveat (`regime_b_flag`, `mode_ambiguous_flag`,
  `ood_disagreement_flag`, or `aux_missing_any`).
- **Tier 3 — withheld**: OOD-flagged (`ood_joint_flag` or
  `latent_support_flag`) or NaN in any `*_pred` column.

Hard-kill (Tier 3) trumps caveat (Tier 2). Missing flag columns are
treated as `False` — demotions must be explicit in the input.

## How to read row 1

The stacked bars show the fraction of each tier per bin; the dark
step-line (right-axis, log) is the raw count per bin. Green collapses
where the model works; red grows where it struggles.

## How to read row 2

Hexbin Tier-3 fraction, `mincnt` set so that sparsely-sampled bins do not
dominate the colormap. Yellow = ~100% Tier 3, black = ~0%.

## How to read row 3

- σ-by-tier: box-plot of the calibrated σ per label, stratified by tier.
  Tier 3 σ is systematically broader; Tier 1 σ shows the core σ a
  downstream consumer should expect on clean stars.
- Flag mix: the share of Tier-3 rows each gate flags (non-exclusive — a
  row may trip multiple gates). Useful for diagnosing which gate is
  doing the work on a given arm.
- Summary bar: absolute tier fractions with counts.

## Rebuild

```bash
PYTHONPATH=src python scripts/assign_release_tier.py
PYTHONPATH=src python scripts/gallery/plot_17_pipeline1_regime_diagnostics.py
```
