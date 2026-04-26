# ADR 0014. Contrastive α/M-blindness catastrophe and β-NLL σ-inflation

**Status:** accepted
**Date:** 2026-04-21
**Supersedes:** relevant parts of ADR 0011 (β=0.5 retained as default).

> **Note (2026-04-22):** the "Pipeline-2 σ-gate" and "Pipeline 2 feature matrix and
> classifier also rerun" references below describe the original in-repo
> population-classification module, which has since been moved to the
> **Starfold** repository. The v1.0/v1.1 checkpoints' non-production status
> still holds; any Starfold-side rerun is now a downstream concern.

## Context

After the v1.1 tag work (inverse-frequency [M/H] weighting, Pipeline-2 σ-gate)
we inspected Stream-3 chemical predictions and the contrastive halfway-UMAPs
together and found two catastrophes:

1. **Smeared chemical continuum.** The contrastive encoder does not separate
   low-α and high-α disk stars into disjoint structures in the halfway-UMAPs.
   Instead there is a single smeared α continuum. TESS_ML's prototype encoder
   on the same data did separate them cleanly.

2. **Prototype attractor cluster.** A collection of Stream-3 stars receives
   supervised-head predictions at ``[α/M] ≈ +0.11`` with very small dispersion
   and ``[M/H] ≈ -1.0 ± 0.2``. Their APOGEE-truth values (for the
   validation-set subsample) span an order of magnitude wider. The head is
   not just biased, it's collapsed into a prototype that ignores the input.

## Diagnosis

Two compounding bugs were responsible. Each bug, on its own, the DESIGN
documents *warned about*; neither was visible in a single-stage unit test
because their symptoms only show up in the **composition** of the two stages.

### Bug A, contrastive encoder trained α/M-blind

`scripts/run_contrastive_pretrain.py` built its `LossWeights` with
``supcon_label_n_first=3``. SupCon's Gaussian-kernel pair weight used only
the first three labels ``(Teff, logg, [M/H])``. Rationale at the time: "Tier-1
atmospherics only; leave chemistry for supervised fine-tune so we don't
double-count." What actually happened:

- For two stars at the same ``(Teff, logg, [M/H])`` but different ``[α/M]``,
  the SupCon kernel gave them weight → 1 (treated as the same class).
- The encoder was trained to map such pairs to nearby projections.
- A low-α disk RGB and a high-α halo RGB at the same atmospherics became
  **indistinguishable** to the encoder.

This matches observation (1) exactly.

### Bug B, β-NLL at β=0.5 absorbs μ-bias into σ on a quasi-frozen encoder

`scripts/run_ensemble.py` used ``LossWeights(supcon=0, beta_nll=1, beta=0.5)``
with ``encoder_lr_ratio=0.1``. Seitzer's β-NLL variance-weighting
``(Π diag Σ)^(β/n)`` (per DESIGN, multivariate generalisation of
``σ²^β``) detaches the weight from gradient flow. At β=0.5:

- For data-sparse Kiel cells (warm upper-RGB, metal-poor), the supervised
  head cannot learn the correct μ because the encoder is already α/M-blind
  (Bug A). The residual is large and systematic.
- β-NLL's down-weighting concentrates the head on *fitting σ* such that the
  weighted loss is small, this is exactly the pathology Seitzer's paper
  discusses: β=0.5 is a compromise that *partially* absorbs μ-bias into σ
  instead of exposing it as mean error.
- With the encoder quasi-frozen (encoder_lr_ratio=0.1), the head has no
  escape route, it cannot ask the encoder for a better feature. It
  collapses onto a single prototype that minimises the detached-σ-weighted
  NLL: the ``[α/M]=+0.11, [M/H]=-1`` attractor.

This matches observation (2) exactly, and also matches methods-paper
Finding #3 ("β=0.5 absorbs per-cell μ bias into inflated σ; β=0 exposes
the bias as explicit mean error"), which was identified earlier as a
*methodology result*, not a production-blocker. It became a
production-blocker once Bug A degraded the encoder enough that the head had
no good μ available for many cells.

## Decisions

### D1, contrastive uses all 5 production labels, not just Tier-1

`run_contrastive_pretrain.py` sets ``supcon_label_n_first=None`` and passes
``LabelTiers.five_label()``. The SupCon kernel now weights on
``{Teff, logg, [M/H], [α/M], [Mg/H]}``, matching the supervised head's
label space exactly. Stars with different chemistry at the same atmospherics
are no longer treated as maximally positive.

### D2. SupCon is NaN-safe

`supcon_soft_positive` now masks any pair where either label row has a NaN
in any dim → weight 0. Required because per-element abundances have 1-5%
NaN rates (V ~5.3%, Mg/Fe ~1.6%); without the mask, ``d2 = (ya - yk)²``
NaN-propagates and the whole loss is NaN from epoch 0. First observed
2026-04-21 when switching from ``n_first=3`` to all labels.

### D3, ensemble switches to β=0 with a small SupCon auxiliary

`run_ensemble.py` sets ``LossWeights(supcon=0.1, beta_nll=1.0, beta=0.0)``
and ``grad_norm_abort_threshold=500.0`` (β=0 canary). Rationale:

- **β=0.** Pure Gaussian NLL. Per-cell μ bias surfaces as explicit mean
  error in the loss, not buried in σ. This forces the head to actually
  fit the mean, which (with a non-α/M-blind encoder from D1) it can now do.
- **supcon=0.1.** Small contrastive auxiliary keeps the latent geometry
  from collapsing as the head re-learns with β=0. Zero SupCon would let
  the latent drift.
- **grad_norm_abort_threshold=500.** Pure Gaussian NLL can explode on
  high-σ samples; canary aborts training with a diagnostic instead of
  letting NaN grads propagate silently.

ADR 0011's "β=0.5 retained as default" remains valid *as a methodology
baseline*, we still want to compare calibration under both regimes when
writing up Finding #3. But β=0 is the production retrain choice for
Pipeline 1 v2.

## Consequences

- Encoder checkpoint `models/main/xp_abundances/20260421_38a993e_1371f1a/`
  and ensemble checkpoints tagged `pipeline1-v1-2026-04-19` /
  `pipeline1-v1.1` are retained for audit and for the methods paper's
  Finding #3 comparison, but are **no longer production**.
- Pipeline 1 v2 inference on Stream 3 must rerun end-to-end. Predictions
  from v1.0 and v1.1 are not to be used for science downstream of this
  date. Pipeline 2 feature matrix and classifier also rerun.
- `docs/plan/06_methods_paper.md` Finding #3 is now also an empirical
  production result, not only a methodology finding, update when writing
  the paper outline.
- No change to Tier-promotion protocol (research_brief §3.3), tiers
  are label-space promotion, not training-recipe decisions.

## Verification (acceptance criteria)

1. Contrastive halfway-UMAP colored by [α/M] shows visible α-separation
   (not a single smeared continuum).
2. Stream-3 predictions show no cluster of stars at
   ``[α/M]≈+0.11, [M/H]≈-1`` with <0.03 dex internal dispersion.
3. Per-cell μ bias on validation partition is **not** masked by
   σ-inflation: reliability diagram coverage stays within ±5% of nominal
   after β=0 retrain.
4. Sanity battery and calibration harness pass without the per-[M/H]-bin
   gate firing at −1.0 metal-poor bin.
