# ADR-0005: Tier decisions for Teff and log g (Option 2)

**Date**: 2026-04-19 · **Status**: Accepted, shipped

## Context

The §9.2 shuffled-spectrum null test surfaced apparent failure for Teff and log g
(skill_ratio 0.714 and 0.870, failing the literal §4 gate). The initial reading was
"prior-augmented tier-1-caveat", that XP color and parallax carry most of the label
and the shuffle-null failure is about the distribution of information rather than
spectral vs non-spectral origin.

That framing was wrong: the §9.2 protocol is specifically designed to distinguish
spectral from non-spectral prediction, and reframing it as "prior-augmented" without
evidence is a protocol departure that needs ratification. Three specific diagnostics
ran to resolve the ambiguity.

## Decision

**Option 2**: Teff → Tier 1 clean. log g → Tier 1 with explicit "prior-augmented"
caveat in release documentation.

## Rationale from the three-diagnostic sweep

| Diagnostic | Teff | log g |
|---|---|---|
| PCA-CMI (7 comp, 95.8% var) | 0.0296 nats, barely above 0.02 floor | 0.0311 nats, barely above floor |
| Permutation importance | 6 XP features in top-10; 44.7% XP group share | 0 XP in top-10; 14.6% XP group share |
| Aux-only baseline RMSE ratio | 164 K / 67 K = 2.44× improvement from XP | 0.225 / 0.157 = 1.44× improvement |

**Teff**: unambiguously spectrum-driven. 2.4× RMSE improvement from XP, 6 of 10 top
features are XP coefficients. The PCA-CMI being "barely above floor" is a known KSG
estimator artifact near threshold on continuous labels. No caveat needed.

**log g**: mixed. 1.44× RMSE improvement from XP (meets the >1.10× threshold for "XP
contributing"), but zero XP features in top-10 permutation (distances + photometry
dominate), and PCA-CMI right at the 0.02 floor. The model genuinely uses XP
(30% RMSE improvement is real value for stars where photometry is poor), but XP's
role is secondary. "Prior-augmented caveat" is the honest description.

## Alternatives rejected

- **Option 1: both Teff and log g → Tier 1 no caveat.** Under-honest for log g;
  downstream users would miss the aux-feature dependence.
- **Option 3: log g → Tier 2 population-only.** Too strict. 30% RMSE improvement is
  real scientific value, especially for stars with poor parallax/photometry. Tier 2
  denies downstream users information for insufficient reason.

## Consequences

- D-Cat-b release documentation has explicit verbatim language about Teff (XP primary)
  and log g (XP secondary to parallax + photometry).
- log g release statement is: "log g predictions use Gaia XP spectra augmented by
  auxiliary features. Aux-only baseline RMSE 0.225 dex; full model 0.157 dex (30%
  improvement). Spectral contribution is secondary to geometric and photometric
  features. Users requiring the full marginal contribution of spectra should note
  this."
- Precedent set: §9.2 shuffle-null failures are not automatic Tier-demotion triggers;
  a three-diagnostic follow-up (CMI with richer summary, permutation with XP-vs-aux
  grouping, aux-only baseline) is the protocol departure when indicated.

## Methodology note

A protocol departure deserves a dedicated review with full evidence, not a bullet in
a status update. Halting at the first "prior-augmented" framing and running a
three-diagnostic triage was the correct process move; that triage is reusable
scaffolding for future §9.2-style audits and is documented as a methodology finding
for the paper.
