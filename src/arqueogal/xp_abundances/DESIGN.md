# `arqueogal.xp_abundances` — Design (pipeline-level)

## Purpose

Pipeline 1: semi-supervised multi-task regression from Gaia DR3 XP Hermite coefficients to
APOGEE DR19-calibrated chemical abundances on red giants, with heteroscedastic covariant
uncertainty that is *calibrated* (reliability + coverage) rather than merely quoted.

Delivers to **D-Cat-b** (Month 6 / Aug 2026) and feeds Pipeline 2 features.

**Primary reference**: `docs/research_brief.md` §§2–9. This file is a stub; all science,
label tiering, preprocessing order, validation protocol, and information-content audit live
there.

## Scope

- **Magnitude regime**: G ≤ 17.65 (XP-native). We do NOT push to G > 17. Focus is depth of
  treatment — calibration, extension to additional elements via the §3.3 statistical
  promotion protocol, honest separation of spectrum-driven from prior-driven predictions.
- **Stars**: red giants, Teff ∈ [4000, 5500] K, log g ∈ [1.0, 3.5].
- **Labels — Tier 1** (per-star reliable): Teff, log g, [Fe/H], Av, d.
- **Labels — Tier 2** (per-star with inflated σ, population-level recommended): [α/M],
  [C/Fe], [N/Fe], [C/N] for RGB in validity domain, [Mg/Fe] if separable.
- **Labels — Tier 3**: not released per-star.

## Main vs experimental

`main/` is frozen during deliverable sprints; only bug fixes and validation changes allowed
between milestones. Feature work lives under `experimental/`. Cross-imports are rejected.
See `main/DESIGN.md` and `experimental/DESIGN.md` for each arm.

## Hard rules (specific to this pipeline)

- **Preprocessing order is fixed** (data_acquisition.md §6.4): (1) Ye+2024 NN flux-
  correction, (2) normalise coefficients 1–54 by coefficient 0, (3) log10 + z-score
  coefficient 0, (4) propagate errors under division. Do not reorder. Do not silently
  skip Ye+2024.
- **Mészáros+2025 [X/M]/Teff corrections** applied to DR19 labels *before* `flag_bad` cut.
- **Retrain from scratch on DR19** — no DR17-trained weights reused.
- **Uncertainty calibration is a release gate**. Reliability diagrams (per Teff×log g×[Fe/H]
  cell), 68/95/99% coverage tests on hold-out, post-hoc temperature scaling or isotonic
  regression if miscalibrated. Conformal intervals released as alternative product.
- **Information-content audit** (research_brief §9.2) required for every released label.
  Labels failing shuffled-spectrum null or conditional-MI tests do not release per-star.
- **No new element promoted to Tier 2** without the full six-test §3.3 protocol.
- **No position features** by default; if included, release as a clearly flagged ablation
  variant.

## References

Andrae+2023 (MNRAS 521 3527); Zhang+2023 (MNRAS 524 1855); Li+2024 AspGap (ApJ 974 42);
Guiglion+2024 (A&A 682 A9); Anders+2024 SHBoost (A&A 691 A127); Fallows & Sanders 2024
(MNRAS 531 2126); Ye+2024 (arXiv:2411.19105); Buck & Schwarz 2024 (arXiv:2410.16081);
Mészáros+2025 (arXiv:2506.07845); Kendall & Gal 2017; Guo+2017; Angelopoulos & Bates 2023.
