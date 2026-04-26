# Protocol: Fisher / Cramer-Rao bounds for per-element abundance recovery

**Status:** Scoped. Computation pending Stream 3 / methods-paper drafting.
**Authoring trigger:** External peer review CRITICAL #1; META_META §8 P1-1.
**Owner:** Andreas Neitzel + advisor input on forward-modelling tooling.

## 1 Why this protocol

The methods paper claim is that Pipeline 1 recovers per-element abundances
"at the level of APOGEE DR19 fidelity from Gaia BP/RP" within the regimes
where this is information-theoretically supported. That sentence is
defensible only when we publish the Cramer-Rao lower bound (CRLB) on
per-element σ at representative SNRs and compare to the catalog's
released σ. The XP-information-content review and the external peer
reviewer both flag the absence of this comparison as the #1 blocker
before submission. This protocol scopes the work.

## 2 Theoretical framework

The Fisher-information matrix for per-element abundance recovery from a
forward-modelled BP/RP coefficient vector is, schematically:

  I[θ_i, θ_j] = E[ ∂log p(F|θ) / ∂θ_i × ∂log p(F|θ) / ∂θ_j ]

where F is the BP/RP coefficient block (108-D after the project's
preprocessing chain), θ = (Teff, logg, [M/H], [α/M], [Mg/H], ...) is the
parameter vector, and p(F|θ) is the per-coefficient noise model. The
Cramer-Rao bound for any unbiased estimator of θ_i is:

  σ²_θ_i ≥ [I⁻¹]_{i,i}

Per-element CRLB depends on:

- The synthesis grid used to forward-model F(θ), typically MARCS plus
  Synspec or ASS for the H-band-equivalent + optical reasoning at the
  XP-projected resolution.
- The per-coefficient noise model, De Angeli et al. 2023 quantifies
  per-coefficient uncertainty as a function of G-magnitude.
- The SNR at the target G-magnitude, Pipeline 1 caps at G < 17;
  representative bins at G = 12, 14, 16, 17.

We follow the Ting and Weinberg 2022 framework (arXiv:2102.04007) for
spectroscopic Fisher information, with the noise model adapted to the
BP/RP coefficient space.

## 3 Computation plan

### Step A: Synthetic XP coefficient grid

Generate a synthetic spectral grid covering the production regime: Teff
∈ [3500, 6500] K (50 K steps), logg ∈ [0.5, 4.5] (0.25 dex steps).
[M/H] ∈ [−2.0, +0.5] (0.1 dex steps). Use MARCS (Gustafsson et al.
2008, A&A 486, 951) for cool stars and ATLAS9 (Castelli and Kurucz
2003, IAU 210, A20) for warm stars; document the grid boundary
(typically Teff = 5500 K).

Project synthetic spectra through Gaia gaiaxpy 2.x to recover the BP/RP
coefficient representation, then apply the project's fixed pre-
processing chain (Ye+2025 → coef-1..54 normalisation by c0 → log+z-score
c0 → frozen per-coefficient z-score). Output: per-grid-point 108-D
coefficient vector.

### Step B: Numerical derivatives

For each grid point, compute the per-coefficient derivative ∂F/∂θ_i
numerically via finite differences. Step sizes: ΔTeff = 50 K, Δlogg =
0.10 dex, Δ[X/M] = 0.05 dex.

### Step C: Noise covariance

Per-coefficient noise σ at G = (12, 14, 16, 17) is read from De Angeli
et al. 2023 (A&A 674, A10) and reformulated post-Ye+2025 correction.
The diagonal noise covariance Σ_F is sufficient for the leading-order
Fisher analysis; off-diagonal correlations are second-order.

### Step D: Fisher matrix and CRLB

Per (G-magnitude, regime cell) pair, assemble the Fisher matrix:

  I[θ_i, θ_j] = (∂F/∂θ_i)^T · Σ_F⁻¹ · (∂F/∂θ_j)

Invert to get the Cramer-Rao bound on σ_θ_i. Report at the regime-cell
mean per element per G-bin.

### Step E: Comparison to released σ

Plot CRLB vs Pipeline 1 released σ per element per G-bin in a 5-panel
diagnostic figure (Methods Figure 5 per the figure manifest). Identify
labels and magnitudes where Pipeline 1 is at-the-bound (consistent with
optimal extraction), above the bound (information loss to be
investigated), or below the bound (prior dominance, see §3.5 of the
methods paper).

## 4 Expected outcomes

Order-of-magnitude expectations (xp_information_content.md):

- **Teff:** CRLB ~ 50–100 K at G = 14; Pipeline 1 reports ~ 80 K.
  Expected at the bound.
- **[Fe/H]:** CRLB ~ 0.10 dex; Pipeline 1 ~ 0.10 dex. At the bound.
- **logg:** CRLB ~ 0.25 dex from XP alone; Pipeline 1 ~ 0.15 dex.
  Below the bound, the parallax-based prior is doing the heavy
  lifting (consistent with §3.5).
- **[α/M]:** CRLB ~ 0.20 dex from XP alone; Pipeline 1 ~ 0.07 dex.
  Far below the bound, confirms aux-prior dominance.
- **[Mg/H]:** CRLB ~ 0.20 dex; Pipeline 1 ~ 0.10 dex. Substantially
  below the bound, also aux-prior dominated; Mg b in BP is weak at
  XP resolution.

These are predictions; the actual Fisher analysis falsifies or
confirms them.

## 5 Effort estimate

3–5 days of focused work:
- Day 1: synthetic grid generation (using existing MARCS+Synspec
  pipeline if available, otherwise build a minimal grid).
- Day 2: numerical derivatives + noise model.
- Day 3: Fisher matrix assembly + CRLB computation per grid point.
- Day 4: diagnostic figure + per-element comparison.
- Day 5: methods-paper §3.5.5 prose update with quantified CRLBs.

## 6 Acceptance criterion

Per-element CRLB is published in a methods-paper Figure 5 or appendix
table. Each label and G-bin combination identifies whether Pipeline 1
is at-the-bound, above, or below. The reframing of [α/M] and [Mg/H]
as "aux-assisted" (§3.5) is then quantitatively defensible: their
released σ is far below the spectroscopic Fisher bound, confirming
the aux-prior contribution.

## 7 References

- Ting and Weinberg 2022, ApJ 927, 209 (arXiv:2102.04007).
- De Angeli et al. 2023, A&A 674, A2.
- Gustafsson et al. 2008, A&A 486, 951 (MARCS).
- Castelli and Kurucz 2003, IAU 210, A20 (ATLAS9).
- Witten et al. 2022, MNRAS 516, 3254 (BP/RP information content).
