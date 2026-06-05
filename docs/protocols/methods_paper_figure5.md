# Protocol: methods-paper Figure 5 — extinction-recipe ablation

**Status:** Harness implemented (2026-04-29). Code: `scripts/ablation_extinction_recipe.py` + `tests/scripts/test_ablation_extinction_recipe.py`. Production run pending the v2 joint-loss ensemble checkpoint and two prediction parquets — one trained with the Av-as-feature-only recipe, one with the hybrid-D recipe documented in `docs/protocols/extinction_correction.md` — evaluated on real Stream-1 holdout data.

## 1. Why this protocol

The 2026-04-29 literature review settled on hybrid-D dereddening (Yuan+2013 broadband ratios + CCM89 R_V=3.1, A_V kept as residual feature). The methods paper §5 requires a quantitative figure that proves the recipe choice on real Stream-1 holdout data. The two-reviewer disagreement at recommendation time (literature-grounder said "Av as feature is the consensus", domain-reviewer said "the consensus is hybrid-D, the lit-grounder misread the abstracts") makes this ablation load-bearing for an A&A referee response.

## 2. Three diagnostic panels

For each of the five production labels (Teff, log g, [M/H], [α/M], [Mg/H]):

1. **Residual vs A_V** (`residual_vs_av_<element>.pdf`). Per-A_V-bin median + 16-84 % envelope of `residual = pred − truth`, both recipes overlaid. The methods-paper claim is that hybrid-D's median sits flat on zero across A_V while baseline shows a positive slope. The fitted slopes go into `slopes.csv` + `slopes.json`.

2. **Residual vs Galactic latitude** (`residual_vs_quadrant_<element>.pdf`). Per-|b|-bin median residual under each recipe. A model leaking extinction shows latitude-correlated residual; hybrid-D should erase that.

3. **Intrinsic-colour correlation** (`intrinsic_colour_vs_alpha_m.pdf`). Two-panel scatter of dereddened (BP − RP)_0 vs predicted [α/M]. Slope-only fit; should be near zero under hybrid-D and visibly tilted under baseline.

## 3. Quantitative gate

The `--out` directory carries `summary.json` with verdict ∈ {`hybrid-D wins`, `inconclusive`}. The gate is encoded in `scripts.ablation_extinction_recipe.AblationConfig.slope_improvement_required = 0.30`: hybrid-D wins iff every element's `|slope_hybrid|` is ≤ (1 − 0.30) · `|slope_baseline|`. The 30 % floor is the methods-paper-defensible threshold (matches Hattori+2024's published filter rationale; arXiv:2404.01269).

## 4. Production-run recipe

```bash
PYTHONPATH=src python scripts/ablation_extinction_recipe.py \
    --baseline release/extinction_ablation_baseline.parquet \
    --hybrid   release/extinction_ablation_hybrid.parquet \
    --truth    release/stream1_holdout_truth.parquet \
    --out      reports/ablations/extinction_recipe/
```

The two prediction parquets must be **indexed-aligned 1:1** to the truth parquet on `source_id`. The truth parquet must carry per-row `av_los`, `b_deg`, `bp_rp_dered`, and `<element>_truth` columns for the five elements. Producing those parquets means: (a) train two ensembles (5 seeds each) under each recipe with `arqueogal.xp_abundances.main.training`, (b) run them on the Stream-1 holdout, (c) dump the predictions side-by-side with the truth labels.

## 5. Expected production results

The production run on Stream-1 holdout data will produce quantitative slopes per element, relative improvement across recipes, and a final verdict (hybrid-D wins iff all elements show ≥ 30% improvement). Real residuals are noisier than synthetic, so absolute slopes will be larger; relative improvement is the load-bearing metric for the ablation story.

## 6. Common-misread guard

If a future reviewer claims hybrid-D loses on real data, check first:

1. Was the hybrid-D ensemble trained with the dereddened broadbands actually in `DEFAULT_AUX_COLS`? If the model was given the raw IR columns plus av_los, it is "hybrid-D-on-paper but Av-as-feature-in-practice". The frozen-stats fingerprint is the load-bearing check.
2. Did the truth parquet carry `bp_rp_dered` derived from the same A_V column the model was trained on? Mismatched A_V columns will make the intrinsic-colour panel unreliable.
3. Were both ensembles trained for the same number of epochs at the same learning-rate schedule? An under-trained hybrid-D model will not yet have absorbed the new feature meaning.

## 7. References

- `docs/protocols/extinction_correction.md` (the recipe under test).
- Yuan, Liu & Xiang 2013, MNRAS 430, 2188 (broadband ratios).
- Schlafly+2016, ApJ 821, 78 (R_V variance bound).
- Hattori 2024, arXiv:2404.01269 (E(B−V) < 0.1 filter rationale → 30 % gate).
- `scripts/ablation_extinction_recipe.py` (runtime).
