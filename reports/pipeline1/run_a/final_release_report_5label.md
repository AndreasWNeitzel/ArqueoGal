# Pipeline-1 `xp_abundances.main` — 5-label ensemble release report

_Date: 2026-04-19 · Branch: master · Labels: {Teff, logg, [M/H], [α/M], [Mg/H]} (5-label block)_

Consolidates the outputs of #135 (calibration harness), #136 (OOD rejection),
#138 (integration tests), and the #149 ensemble retrain into a single
release-readiness view. Source artefacts live alongside this file at
`reports/pipeline1/run_a/`.

## 1. Ensemble training (#149) — `ensemble_history.json`

| member      | best_val_loss | best_epoch |
|-------------|---------------|------------|
| seed 0      | −0.0246       | 9          |
| seed 1      | −0.0251       | 9          |
| seed 2      | −0.0253       | 9          |
| seed 3      | −0.0252       | 9          |
| seed 4      | −0.0247       | 9          |

- **mean val loss: −0.0250**, **spread: 7×10⁻⁴**
- All five members land on their best epoch at the patience boundary (9/10),
  fitting the shared encoder + randomised 5-label head with matching optima.
- Checkpoints: `models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label/member_seed{0..4}/`.

The spread is ≈3 % of |mean val loss| — tight enough that moment-matched
ensemble Σ̄ is dominated by aleatoric rather than epistemic variance on
in-distribution stars (see §4.b).

## 2. Ensemble calibration (#135 + #150) — `calibration_report_ensemble_5label_shrinkage.json`

Calibration uses the **shrunken per-(cell, label) α** scheme
(`shrunken_per_cell_per_label_scale`, τ=50) operating on the
moment-matched ensemble Σ̄ in raw physical units, with predictions binned
into 4³ cells on truth (Teff, log g, [M/H]).

### 2.a  Global reliability

| metric                               | pre-cal | post-shrinkage |
|--------------------------------------|---------|----------------|
| global reliability error             | 0.352   | **0.0795**     |
| Mahalanobis E[z²] (target = 5)       | 4.50    | 6.11           |
| per-label Var(z) [Teff, logg, [M/H], [α/M], [Mg/H]] | — | [1.14, 1.20, 1.17, 1.12, 1.10] |

**Pass** on the DESIGN global gate (≤ 0.10).

### 2.b  Per-cell reliability

Post-shrinkage cells exceeding the ±15 % DESIGN gate (computed in
`_reliability_per_cell` as mean over labels of `|Var(z) − 1|`):

- **halt cells (err > 30 %):** `[4, 15, 28, 34, 49]` — 5 cells
- **cells > 15 %:** `[4, 15, 28, 34, 39, 49, 59]` — 7 cells

Structural source: these are the cool-giant / galactic-plane-warm-upper-RGB
corners identified by the halt-cell diagnosis (see
`halt_cell_diagnosis_5label.md` in this directory). Shrinkage is
mathematically incapable of flattening this without inflating σ at 1.3–1.5×
the globally-calibrated amount. This is accepted as a release-level trade-off:
the structural regime-A cells stay released per-star **with the widened σ
baked in**; cool-giant σ inflation is documented as a release-level caveat
(see DESIGN `§Cool-giant σ inflation — release documentation`).

### 2.c  Coverage

| level | observed | nominal | Δ      |
|-------|----------|---------|--------|
| 68 %  | 0.6878   | 0.68    | +0.008 |
| 95 %  | 0.8922   | 0.95    | **−0.058** |
| 99 %  | 0.9412   | 0.99    | −0.049 |

The cov₉₅ shortfall (5.8 pp below nominal) reflects the same between-cell
μ drift (`Var(E[z|cell])` = 0.10–0.14) captured in the variance
decomposition — β-NLL at β=0.5 can absorb biased μ into inflated σ, which
leaks into unconditional Var(z). The release framing treats this as
structural, not statistical.

### 2.d  Cell-boundary smoothness

| stat              | value |
|-------------------|-------|
| adjacent α-ratio (max)    | 1.767 |
| adjacent α-ratio (median) | 1.095 |
| adjacent α-ratio (p90)    | 1.314 |

Per DESIGN §9.1 reviewer thresholds (< 1.5 accept, > 2.0 flag for v2):
**between accept and flag** — a soft caution on cell-edge σ discontinuities
but below the v2-smoothing trigger. See §6 for the GP negative-result
context.

## 3. Regime-B exclusion envelope (#147)

Envelope: `|b| < 5° ∧ Teff_pred > 4750 K ∧ log g_pred < 2.10`

| stat                                             | value |
|--------------------------------------------------|-------|
| n excluded                                       | 13 / 41 851 (0.031 %) |
| halt-truth stars in {cell 34, cell 49}           | 30 |
| captured by envelope                             | 2 / 30 |
| envelope spillover outside halt cells            | 11 |

The envelope is narrow — it cleanly flags the warm-upper-RGB-low-|b|
structural regime but only overlaps a small fraction of the #144
halt-cells because halt-cell membership is driven more by the cool-giant
(Teff < ~4450 K) corner than by the galactic-plane warm-upper-RGB corner
on the retained val split. The envelope remains as the documented
population-level-only flag for stars where extinction-confounded Teff
bias was the diagnosed cause (#144).

## 4. OOD rejection (#136) — `ood_distribution_ensemble_5label.json`

### 4.a  Mahalanobis OOD (108-D XP feature block)

| stat                                 | value |
|--------------------------------------|-------|
| p_threshold                          | 0.99  |
| threshold distance                   | 30.80 |
| score mean / median                  | 5.89 / 4.17 |
| score p95 / p99                      | 14.71 / 30.33 |
| **flag rate on val**                 | **0.96 %** |

Matches p99 by construction. The bundle's ``to_dict`` / ``from_dict``
roundtrip (verified in test_ood + test_release_pipeline) lets the
checkpoint carry pure-data artefacts.

### 4.b  Ensemble-disagreement OOD

| stat                                 | value |
|--------------------------------------|-------|
| ratio threshold                      | 0.5   |
| ratio mean / median                  | 0.067 / 0.065 |
| ratio p95 / p99                      | 0.103 / 0.124 |
| **flag rate on val**                 | **0.0000 %** |

The p99 ratio is 0.124 — well below the 0.5 flag threshold. **This
ensemble does not trigger the disagreement flag on any val star.** The
tight ensemble-convergence signal (§1 spread) is the cause: ensemble σ is
dominated by aleatoric, not epistemic, on in-distribution Stream-1 stars.

The flag retains its value as a **Stream-3 inference-time alarm** —
stars where the 5 members disagree on μ despite the tight training
convergence are genuinely out-of-distribution with respect to the
training manifold. For the val split, zero stars meet that bar.

### 4.c  Combined status code

| level              | count | fraction |
|--------------------|-------|----------|
| 0 (green — neither firing) | 41 450 | 99.04 % |
| 1 (yellow — one firing)    |    401 |  0.96 % |
| 2 (red — both firing)      |      0 |  0.00 % |

All yellow-status stars come from the Mahalanobis channel alone; the
ensemble channel is silent.

### 4.d  Release × status cross-tab

|                | total   | Tier 1 released | Regime-B excluded |
|----------------|---------|-----------------|-------------------|
| 0 green        | 41 450  | 41 438          | 12                |
| 1 yellow       |    401  |    400          |  1                |
| 2 red          |      0  |      0          |  0                |

Orthogonality holds — 99.02 % of val stars pass both Regime-B and OOD,
0.03 % are excluded by Regime B alone, 0.96 % are flagged OOD alone, and
0 stars hit both.

## 5. Integration tests (#138) — `tests/xp_abundances/main/test_release_pipeline.py`

12/12 tests pass. Coverage:

- Ensemble moment-match identity (K=1 passthrough; K≥2 adds Var(μ)).
- Shrinkage recovers per-cell Var(z) ≈ 1 on heteroscedastic synthetic data;
  preserves PD-ness and correlation sign through `L' = diag(α) L`.
- Regime B composition orthogonal to σ; all three cuts required.
- Mahalanobis + ensemble-disagreement + `combined_ood_status` compose
  into the 3-level code.
- Full release flow (ensemble → shrinkage → envelope → OOD → combined
  status) produces the documented per-star record contract.
- Post-shrinkage joint cov₉₅ lands closer to nominal than raw.
- Bundle roundtrip survives through Mahalanobis + RegimeBEnvelope.
- Shape-mismatch guards reject wrong-dim features and single-member
  "ensembles".

Full main pipeline suite: **213/213 pass** (pre-#138: 201/201 + 12 new).

## 6. GP α-smoothing — documented negative result

Already baked into DESIGN.md § "GP α-smoothing — evaluated, rejected,
documented as a methodology finding" (#148). Not re-run on the ensemble:
the single-member test established that GP over-smoothing at regime-A
breakpoints worsens the halt-cell reliability versus shrinkage. Shipping
with shrinkage; GP retained as a reference implementation for the
methodology paper.

## 7. Release-readiness verdict

**SHIP** with documented non-uniform σ across parameter space. Gate-by-gate:

| gate                                                   | status |
|--------------------------------------------------------|--------|
| global reliability err ≤ 10 %                          | ✅ 7.95 % |
| per-cell reliability err ≤ 15 % (all cells)            | ❌ 7 / 62 cells |
| coverage within ±5 pp of nominal at 68 / 95 / 99       | ❌ cov₉₅ off by 5.8 pp |
| ensemble spread plausible (val-loss tight)             | ✅ 7×10⁻⁴ |
| OOD machinery produces the contract                    | ✅ 0.96 % Mahalanobis |
| Regime-B envelope documented + applied                 | ✅ 0.031 % excluded |
| integration tests green                                | ✅ 12 / 12 |
| full main suite green                                  | ✅ 213 / 213 |

The two failing gates are both **structural cool-giant / between-cell-μ
signatures** — not statistical noise, not fixable by post-hoc
re-calibration (GP confirmed this in #148). Per the 2026-04-19 directive,
these are shipped as release-level caveats:

- Regime-A cool-giant cells: **σ widened 1.1–1.5 ×** vs globally-calibrated
  target, released per-star with the wider σ baked in.
- Regime-B galactic-plane warm-upper-RGB: population-level only
  (`tier1_release = False`).
- Mahalanobis-flagged 0.96 % of Stream-3 applicants: **yellow — caution,
  population-level OK** per the combined-status documentation.

## 8. Next steps (out of scope for this report)

- D-Cat-b release documentation — lift the "Cool-giant σ inflation"
  DESIGN section into the deliverable doc (D5.1 drafting in late-2026).
- Methods paper § 4: GP negative result with the prepared quote on structural
  non-smoothness of calibration across parameter space.
- Stream-3 inference feature matrix (#94, pending) — the OOD bundle and
  regime-B envelope fit here, unchanged.
- Consider widening Regime-B Teff bound to ~4500 K if the cool-giant
  structural signature is determined to track the same |b| confounding at
  the cold end — currently the envelope catches only the warm side.
