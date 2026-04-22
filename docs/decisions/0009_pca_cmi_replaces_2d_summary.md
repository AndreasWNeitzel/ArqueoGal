# ADR-0009: Deprecate 2-D KSG CMI summary; use PCA summary

**Date**: 2026-04-19 · **Status**: Accepted, protocol update pending implementation

## Context

§9.2 audit test 5 originally computed conditional mutual information I(XP; y | aux)
using a 2-D XP summary (|BP|-sum, |RP|-sum) as the high-dim conditioning variable. The
KSG (Kraskov-Stögbauer-Grassberger) estimator is known to be biased on low-D summaries
of high-D signals because:

- Too-low-D summaries lose information the estimator treats as noise → underestimation.
- Boundary effects at the summary's support → overestimation in some regimes.
- Near-threshold values are susceptible to estimator variance.

Cross-label CMI comparison at 2026-04-19 showed the 2-D summary was unreliable:

| Label | 2-D CMI | PCA CMI (7 comp, 95.8% var) | Ratio |
|---|---|---|---|
| Teff | 0.1352 | 0.0296 | 4.6× over |
| log g | 0.0401 | 0.0311 | 1.3× over |
| [M/H] | 0.0088 | 0.0357 | 4.1× under |
| [α/M] | 0.0000 | 0.0000 | — |
| [Mg/H] | 0.0000 | 0.0357 | — |

Three of five labels' 2-D CMI is badly biased relative to the PCA summary. The
apparent "Teff has high CMI" was a 4.6× overestimate.

## Decision

**Deprecate 2-D CMI in audit.py test 5.** Primary methodology: PCA summary with ≥95%
variance explained (7 components for ~95.8%). 2-D summary retained as a supplementary
cross-check only. Update `research_brief.md §9.2` to specify PCA methodology.

## Rationale

- Evidence is unambiguous: 2-D → PCA changes the CMI estimate by factors of 4–5×
  in both directions for three of five labels. A method with that much variance is
  not reliable.
- PCA summary captures 95%+ of XP variance — the KSG estimator operates on a
  near-complete representation.
- Cost is trivial (PCA is cheap; adding 7 components to the conditional estimator
  is unchanged in structure).

## Alternatives rejected

- **Full-dim KSG on 110-D XP** — KSG scales poorly in dimensionality; variance
  dominates by ~20 D.
- **Deep-MI estimators (MINE, InfoNCE)** — more flexible but introduce estimator
  complexity and hyperparameter tuning not needed for the audit scale.
- **Keep 2-D as primary, flag as approximate** — user-facing methodology should
  use the best available estimate, not a known-biased one.

## Consequences

- `audit.py` test 5 implementation change: primary summary is PCA with ≥95% var;
  2-D summary output as supplementary diagnostic.
- Methods-paper note: the 2-D vs PCA comparison is itself methodology content
  (when to trust CMI estimates in XP-spectroscopy-style audits).
- Stream 3-era audits and future label promotions use PCA summary as primary.

## Methodology note

This was surfaced during the cross-label CMI consistency pass. The methodology paper
should explicitly document the bias rather than silently use the corrected method —
it is publishable content.

## Needs clarification

- Has the code change actually landed in `audit.py`? The ratification happened in
  conversation, but no PR/commit has been documented. Confirm before claiming §9.2
  uses PCA as primary.
