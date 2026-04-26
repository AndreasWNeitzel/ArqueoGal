# 5 Negative results

The methods paper deliberately publishes the experiments that did not
work: the contrastive-α/M-blind catastrophe, the 2-D KSG conditional MI
estimator that was deprecated as biased, and the GP-smoothed calibration
that was tested and rejected for production. We document them because
each carries a methodological lesson, because referees will eventually
ask, and because a full audit trail strengthens the catalog's
credibility (hostile_referee_committee.md cross-cutting threats §3).

## 5.1 The contrastive-α/M-blind catastrophe (ADR-0014)

**What was tried.** A SimCLR-style supervised contrastive pretraining
phase for the encoder, prior to the supervised regression head training,
using a soft-positive-Gaussian-kernel weighting on the per-star label
distance (Khosla et al. 2020, arXiv:2004.11362). The hypothesis: the
contrastive objective would learn a more semantically structured XP
embedding that would generalise better to the per-element heads than
joint training from scratch.

**What happened.** The contrastive pretraining converged and produced
visually appealing latent-space clusters when projected to 2-D. But the
downstream per-element heads systematically produced biased [α/M]
predictions, with the bias correlating with the star's distance from
the disc-population mean in the contrastive embedding. The effect was
largest in the metal-poor regime ([Fe/H] < −0.7) and produced
catastrophic disagreement with APOGEE-derived [α/M] for stars in
known halo populations. The contrastive embedding had collapsed the
α-rich-vs-α-poor distinction into a kinematically-coloured manifold
that the head could not reliably project back.

**Resolution.** The contrastive-pretraining phase was removed from
production training. The supervised-only baseline (joint training of
encoder and per-element heads from scratch) was restored as
production. The per-element [α/M] head no longer systematically biases
metal-poor halo stars; the residual prior-dominance is now an explicit
release-side caveat (§3.5) rather than a hidden architectural failure.

**Lesson.** Self-supervised representations, even with label
supervision via soft positives, can collapse population-prior
correlations into the embedding in ways that the downstream regression
head cannot disentangle. For low-resolution stellar spectroscopy,
joint supervised training preserves the option to keep population
priors visible (and surfaceable) in the released uncertainty
quantification.

## 5.2 The 2-D KSG CMI deprecation (ADR-0009)

**What was tried.** Conditional mutual information CMI(XP ; label | aux)
was originally estimated by summarising the 108-D XP block via two
scalar statistics (the sum of |BP coefficients| and the sum of |RP
coefficients|) and feeding them to a Kraskov-Stögbauer-Grassberger
k-nearest-neighbour MI estimator (Kraskov et al. 2004, Phys. Rev. E 69,
066138). The rationale: dimensionality reduction makes the KSG
estimator tractable in the data-poor regime.

**What happened.** Audit of the resulting CMI values against
ground-truth-known information channels (notably Teff, where the XP
broadband colour is known to encode strong information) showed a ~5×
inflation: the 2-D KSG estimator reported Teff CMI values
inconsistent with both the empirical residual reduction from XP and
the theoretical Fisher-information bound. Cross-checking with a
PCA-based summary at 7 components (≥ 95 % variance retention)
restored consistency.

**Resolution.** The 2-D summary is deprecated. The PCA-summary KSG CMI
with 7 components is the production estimator. Earlier audit reports
that used the 2-D summary are flagged with `legacy_2d` provenance
metadata; users reading those reports should treat the CMI numbers
as historical, not load-bearing.

**Lesson.** KSG MI estimators on coarse summaries of high-dimensional
inputs can introduce systematic biases that are hard to detect
without a calibrated check. PCA-based summaries with ≥ 95 % variance
preservation provide a more transparent dimensionality reduction.

## 5.3 The GP-smoothed calibration rejection (ADR-0003)

**What was tried.** A Gaussian-process-smoothed alternative to the
empirical-Bayes shrinkage calibration (§3.3), with the GP modelling
per-cell σ as a smooth function of (Teff, logg, [M/H]). The
hypothesis: smoother per-cell σ surfaces would reduce binning
artefacts and improve calibration in undersampled regime cells.

**What happened.** The GP-smoothed calibration was implemented as
`gp_smoothed_per_cell_per_label_scale` in `uncertainty.py` and
benchmarked against the production τ = 50 empirical-Bayes shrinkage on
the validation set. Across the five released elements, ECE differences
were within ±0.5 percentage points and per-cell coverage within ±2
percentage points of the empirical-Bayes baseline. The GP introduced
117 outgoing edges in the calibration graph (the largest single
function in the project's uncertainty module) and required tuning of
GP hyperparameters that did not generalise across the regime grid.

**Resolution.** The empirical-Bayes shrinkage is the production
calibration. The GP-smoothed code is retained as a methodology-
comparison toggle (`run_calibration.py --apply-gp-smoothing`) but is
not invoked in any release pipeline. A future code-cleanup pass may
delete the GP path entirely (META_META §10 user decision 9).

**Lesson.** When a more flexible model fails to outperform a simpler
shrinkage on calibrated metrics, the simpler model is preferred for
production. The negative result is publishable in its own right
because it falsifies a plausible hypothesis (smoother surfaces help)
and saves future effort on a path that does not pay off.

## 5.4 What we did not do, and why not

A τ-hyperparameter sweep across the empirical-Bayes shrinkage strength
(τ ∈ {10, 20, 30, 50, 100, 150}; bayesian_uq.md MAJOR-1) was not
performed before this release. The protocol document
`docs/protocols/tau_sweep.md` scopes the work; it remains future
work. The production τ = 50 is empirically validated against
per-cell coverage but not against a full ECE-vs-τ curve; the
sensitivity could shift the calibration recommendation by up to ~1
percentage point of ECE in either direction.

A held-out open-cluster benchmark (test 3 of the §3.3 promotion
protocol; protocol scoped in `docs/protocols/open_cluster_benchmark.md`)
was not run before this release. Standard benchmarks (M67, NGC 6791,
NGC 2420, NGC 6819, NGC 2477, NGC 5822, plus 4 others identified in
the protocol document) require excluding cluster members from training
and re-running the full pipeline; the 2 weeks of compute were not
committed before the v1 tag.
