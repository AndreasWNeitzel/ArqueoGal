# §9.2 information-content audit — Pipeline 1 v1 (5-label) summary

_Timestamp: 2026-04-19 15:51 UTC · Ensemble: `20260419_nogit_a0e10aa_ensemble_5label` · Val split seed 0 · N_val = 41851_
_Final tier decisions ratified 2026-04-19 (Option 2) after three-diagnostic follow-up._

**Overall go/no-go: GO — D-Cat-b (Aug 2026) release packaging is cleared.**

## Final tier decisions (Option 2, ratified 2026-04-19)

| label | tier | evidence summary |
|---|---|---|
| teff_apogee | **Tier 1 (clean)** | XP primary; aux-only RMSE 164 K vs full 67 K (2.4× improvement); 6 XP in top-10 permutation; distance features single strongest individual driver. |
| logg_apogee | **Tier 1 with prior-augmented caveat** | XP secondary; aux-only RMSE 0.225 dex vs full 0.157 dex (30% improvement); 0 XP in top-10 permutation; PCA-CMI 0.031 nats > 0.02 floor. |
| mh_apogee | **Tier 1 (clean)** | Null skill_ratio −0.039; XP joint ΔRMSE/σ = 0.691; shuffle null passes cleanly. |
| alpha_m_apogee | **Tier 1 (clean)** | Null skill_ratio −0.257; XP joint ΔRMSE/σ = 0.536; shuffle null passes cleanly. |
| mg_h_apogee | **Tier 1 (clean)** | Null skill_ratio +0.062; XP joint ΔRMSE/σ = 0.602; shuffle null passes cleanly. |

## Release-statement verbatim text

Downstream documentation for D-Cat-b must use the following language verbatim.

**Teff (Tier 1, no caveat):**

> Teff predictions use Gaia XP spectra as the primary information source, augmented by parallax and magnitudes. Aux-only baseline achieves RMSE 164 K; the full model achieves 67 K (2.4× improvement). XP coefficients account for 6 of the 10 top-ranked features in permutation importance analysis.

**log g (Tier 1 with explicit prior-augmented caveat):**

> log g predictions use Gaia XP spectra augmented by auxiliary features (parallax, magnitudes, extinction). An aux-only baseline MLP achieves RMSE 0.225 dex on the validation set; adding XP spectral information improves this to 0.157 dex (30% improvement). The spectral contribution is secondary to geometric and photometric features. Users requiring the full marginal contribution of spectra to log g predictions should note this and consider their use case accordingly.

## Process (how the tier decisions were reached)

The §9.2 shuffled-spectrum null test surfaced a potential concern for Teff and log g (null skill_ratio 0.714 and 0.870, failing the literal §4 gate). A three-diagnostic follow-up (PCA-CMI with ≥95% variance summary, per-feature permutation importance with XP-vs-auxiliary grouping, and an auxiliary-only baseline MLP comparison) was executed to adjudicate. Results directed Option 2: Teff carries XP as primary signal; log g carries XP as secondary signal. Tier assignments reflect this evidence.

The three info-rich labels ([M/H], [α/M], [Mg/H]) pass the literal §4 shuffled-spectrum null cleanly and do not need follow-up. For methodology consistency across the full label set, PCA-CMI was also recomputed for these three labels on the same val split and PCA basis used for Teff/log g:

| label | CMI (2-D, original audit) | CMI (PCA summary, 7 comp, 95.8% var) | PCA / 2-D | vs 0.02 floor |
|---|---|---|---|---|
| teff_apogee | 0.1352 | 0.0296 | 0.22× | just above |
| logg_apogee | 0.0401 | 0.0311 | 0.78× | just above |
| mh_apogee | 0.0088 | **0.0357** | 4.06× | clearly above |
| alpha_m_apogee | 0.0000 | **0.0000** | — | **STILL zero** |
| mg_h_apogee | 0.0000 | **0.0357** | — | clearly above |

The [M/H] and [Mg/H] recoveries (2-D collapsed → PCA well above floor) confirm that the 2-D summary underestimated their CMI, as predicted. **[α/M] is the outlier**: PCA-CMI remains 0.0000 even at 7 components. A three-test sequential triage (`reports/pipeline1/audit/alpha_m_triage.md`; driver `scripts/triage_alpha_m_cmi.py`) was executed to choose between three candidate explanations. Result: **H2 confirmed** — aux absorption. With a richer 15-component PCA summary (98.87% variance) and the full 4-D aux conditioning set, CMI remains 0.0000 nats (ruling out H1). The unclipped raw KSG estimate is −0.0880, well outside the small-sample-noise regime (ruling out H3 as the dominant cause). With conditioning on **parallax alone**, the same 15-PC summary produces CMI = **0.1125 nats** — a factor of ~56 above the 0.02-nat release-gate floor. The aux block (bp_rp, g_mag, av_sfd) is therefore absorbing [α/M]-relevant signal, almost certainly through a sub-population correlation (α-rich stars are kinematically hot and live in reddened, low-latitude regions; aux features co-vary with α-enhancement across the disc). This is not a release blocker: [α/M]'s shuffled-spectrum null (skill_ratio −0.2574) and XP-joint-shuffle (ΔRMSE/σ = 0.5362) are the load-bearing evidence and both pass cleanly. The CMI result is the expected behaviour for an information-rich label correlated with the conditioning basis, not a signature of weak spectral content. Cited in the D-Cat-b methods paper.

Cited artefacts:
- Three-diagnostic report: `reports/pipeline1/audit/three_question_diagnostic.md` + `.json`.
- PCA-CMI across all labels: `reports/pipeline1/audit/pca_cmi_all_labels.json` (produced by `scripts/run_pca_cmi_all_labels.py`).
- Aux-only baseline checkpoint: `models/main/xp_abundances/aux_only_baseline_20260419/aux_only_baseline_seed0.pt` (+ `.provenance.json`).

## Methodology note — 2-D CMI deprecated, PCA summary is primary

**Three of five labels showed substantial bias in the 2-D summary**:

- **Teff: 0.1352 → 0.0296 nats** (4.6× upward inflation under the 2-D summary).
- **[M/H]: 0.0088 → 0.0357 nats** (4× underestimation; 2-D collapsed below the release-gate floor).
- **[Mg/H]: 0.0000 → 0.0357 nats** (pathological collapse under 2-D; PCA clearly above floor).

log g was mildly biased (0.0401 → 0.0311 nats, 1.3× inflation); [α/M] requires the triage treatment described above.

**As of the Pipeline 1 v1 release, the 2-D XP summary (`|BP|-sum`, `|RP|-sum`) is deprecated from the §9.2 audit protocol.** PCA-summary KSG CMI (≥ 95% variance retained; default 7 components for Gaia XP) is now the primary Test-5 estimator. Audit-protocol version bumped to **v1.1**. The 2-D estimator is retained behind a `legacy_2d` flag in `src/arqueogal/xp_abundances/main/audit.py` for reproducibility of historical report cards only and emits a `DeprecationWarning` on use. See `docs/research_brief.md §9.2.1`, the `src/arqueogal/xp_abundances/main/audit.py` module docstring, and the `src/arqueogal/xp_abundances/DESIGN.md` "Audit protocol" changelog entry for the codified protocol.

## Cross-label evidence (original §9.2 tests)

| label | shuffle-null verdict | tier | RMSE | σ(y) | real skill | null/real skill | CMI 2-D nats | XP joint ΔRMSE/σ |
|---|---|---|---|---|---|---|---|---|
| teff_apogee | prior-augmented → Tier 1 (Option 2) | T1 |  67.0956 | 267.5588 |   0.7492 |   0.7142 |   0.1352 |   0.2141 |
| logg_apogee | prior-augmented → Tier 1-caveat | T1-caveat | 0.1571 |   0.5159 |   0.6955 |   0.8698 |   0.0401 |   0.0906 |
| mh_apogee | information-rich | T1 |   0.1154 |   0.3443 |   0.6648 |  -0.0393 |   0.0088 |   0.6910 |
| alpha_m_apogee | information-rich | T1 |   0.0547 |   0.0954 |   0.4264 |  -0.2574 |   0.0000 |   0.5362 |
| mg_h_apogee | information-rich | T1 |   0.1041 |   0.2902 |   0.6415 |   0.0619 |   0.0000 |   0.6017 |

## Per-label report cards

- [`teff_apogee_report_card.md`](./teff_apogee_report_card.md)
- [`logg_apogee_report_card.md`](./logg_apogee_report_card.md)
- [`mh_apogee_report_card.md`](./mh_apogee_report_card.md)
- [`alpha_m_apogee_report_card.md`](./alpha_m_apogee_report_card.md)
- [`mg_h_apogee_report_card.md`](./mg_h_apogee_report_card.md)

## Tests executed

1. LOOCO (per-XP-coefficient zero-out, aggregated to family).
2. Permutation importance (per-feature shuffle; RMSE increase).
4. Shuffled-spectrum null (within (Teff, log g) cell permutation of the 110 XP Hermite + 2 c0 columns).
5. Conditional MI (Kraskov KSG) I(XP-summary; y | aux prior) — 2-D summary in the original audit; PCA summary (7 components, 95.8% variance) in the three-diagnostic follow-up and the cross-label consistency pass.

## Tests deferred

- 3. SHAP — awaiting `shap` in the pinned RAPIDS 25.10 env.
- 6. Decorrelated sub-sample — stub per DESIGN §9.2.
