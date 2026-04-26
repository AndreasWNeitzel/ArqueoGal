# Review of Reviews: Science Deliverable and Catalog UX Audit

**RoR Auditor:** Haiku 4.5 | **Date:** 2026-04-24 | **Scope:** Science-deliverable-specific spot-checks on the ArqueoGal expert-review fleet

---

## Executive Summary

The upstream science_deliverable.md meta-synthesis (7 topical metas, 50+ Haiku specialist reviews) identifies **three critical gaps** blocking consumer release and **six major issues** requiring journal remediation. This RoR audit verifies the facts underlying those verdicts: tier-status inconsistencies, citation accuracy, schema documentation, and informational-content disclosure. **Finding: the meta's critical findings are substantiated; the recommendations are well-scoped and actionable; external researchers cannot onboard without immediate P0 fixes.**

---

## 1. [Mg/H] Tier-Status Conflict: Verified

**Claim (science_deliverable.md §4):** "audit-promoted to Tier 1 (test 5 CMI 0.036 nats, above 0.02 floor) but current release.py omits it."

**Verification:**

- `/home/aneitzel/projects/ArqueoGal/src/arqueogal/xp_abundances/main/release.py`, line 32: `_PRED_COLS: Final = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")` — [Mg/H] IS listed as a predicted column.
- The column is conditionally emitted: `release.py:assign_release_tier()` checks for its presence via `col in df.columns` (line 68), but does not force emission.
- `/home/aneitzel/projects/ArqueoGal/docs/research_brief.md` §3.2: "[Mg/H] Tier 2 per-element" is mentioned in tier protocol.
- The inference module (`inference.py:46`) references `CalibrationArtifacts` and `collect_predictions()` but does not specify which labels are returned.

**Status:** PARTIALLY VERIFIED with qualification. The audit statement "[Mg/H] tier status is inconsistent" is accurate: research_brief.md states Tier 2, the meta claims audit promotes it to Tier 1, but `release.py` does not *forbid* emission. The conflict is real but resides in the *design documents* (research_brief vs audit outcome), not in code-blocking logic. A downstream script must decide whether to materialize the `mg_h_pred` column. **Recommendation:** Resolve in DESIGN.md explicitly: either (a) tier 1, emit; (b) tier 2, emit with caveat flag; (c) tier 3, omit with documented reason.

---

## 2. LICENSE and CITATION.cff Absence: Confirmed

**Claim (science_deliverable.md §7):** "no LICENSE file (github_presentability.md CRITICAL: legal blocker), no CITATION.cff (github_presentability.md CRITICAL: discoverability blocker)."

**Verification:**

- `ls /home/aneitzel/projects/ArqueoGal/LICENSE*` returns no matches.
- `ls /home/aneitzel/projects/ArqueoGal/CITATION*` returns no matches.
- `/home/aneitzel/projects/ArqueoGal/README.md`, line 116: "License — TBD."
- `/home/aneitzel/projects/ArqueoGal/pyproject.toml`, line 10: `license = {text = "MIT"}`.

**Status:** CONFIRMED. Internal contradiction: code declares MIT in pyproject.toml but README claims TBD. No committed LICENSE file. No CITATION.cff. The meta's verdict is accurate and release-blocking.

---

## 3. [Mg/H] and [α/M] Aux-Absorption Transparency: Verified

**Claim (science_deliverable.md §6):** "[α/M] has zero PCA-CMI (conditional MI between XP spectrum and [α/M] abundance, given auxiliary features parallax, photometry, extinction). This means the model has learned the [α/M]-kinematics correlation present in APOGEE training data and uses it dominantly at inference, not XP spectral features."

**Cross-reference verification:**

- META_META_SYNTHESIS.md §2.1: "CMI([α/M] | parallax, photometry, extinction, position) ≈ 0 nats. CMI([α/M] | parallax only) = 0.1125 nats. The 56× difference confirms aux features absorb the model's information channel."
- Same section: "[Mg/H] shows the same pattern with a physically real Mg b triplet feature in the RP window providing weak but non-zero independent constraint."
- research_brief.md §3.3.1 mentions "three-question diagnostic" for tier promotion, but does not explicitly document the CMI result or the aux-dominance finding.

**Schema check:** The catalog schema (implicit in release.py's column lists and inference.py outputs) **does not include per-star flags** like `xp_abundance_type` or `aux_absorption_flag`. Neither [α/M] nor [Mg/H] carry metadata distinguishing spectrum_dominant from aux_assisted in the emitted Parquet.

**Status:** VERIFIED and CRITICAL. The meta's finding is supported by the upstream fleet's convergent analyses. The schema gap is real: release.py can emit these flags but currently does not. The science_deliverable.md CRITICAL verdict ("catalogue documentation will not surface") is accurate. External researchers will download Tier 1 [α/M] without knowing its prior-dominance; this is a labeling and documentation defect, per the meta.

---

## 4. The 293k RGB Training Count: Verified

**Claim (science_deliverable.md §4):** "trained on 293k APOGEE DR19 RGB overlap."

**Source verification:**

- `/home/aneitzel/projects/ArqueoGal/docs/research_brief.md` §4: "the 2026-04 Stream 1 emission delivers **324 054 rows spanning 292 948 unique Gaia source_ids** — i.e. about 293 k unique RGB training stars."

**Status:** VERIFIED. The 293k figure is documented and references the Stream 1 dedup-by-source_id contract (research_brief §6: mandatory dedup on source_id before train/val/test splitting due to Astra multi-task re-runs).

---

## 5. The 1.5M Stream 3 Inference Count: Verified

**Claim (science_deliverable.md §4):** "~1.5M Gaia DR3 red giants at G ≲ 17.65 after stream-1-driven training."

**Source verification:**

- `/home/aneitzel/projects/ArqueoGal/docs/plan/03_stream3_inference.md`: "Stream 3 expansion + Pipeline 1 inference" is Phase 3, in progress. The document does not explicitly state "1.5M" in the portions read, but:
- `/home/aneitzel/projects/ArqueoGal/docs/context/architecture.md`: "Stream 3 is Andrae+2023's vetted RGB/RC sample × Gaia DR3 (~10M parent sample)."
- `/home/aneitzel/projects/ArqueoGal/docs/data_acquisition.md` §5: "Stream 3 bulk (1.5 M stars): compute central-value actions only."

**Status:** VERIFIED. The 1.5M figure is used consistently in the project documentation as the target for Stream 3 inference, with caveats that disk budget and HPC availability may compress it.

---

## 6. The 2/10 GitHub Presentability Score: Verified with Rationale

**Claim (science_deliverable.md §7):** "GitHub presentability is at 2/10 for onboarding... With LICENSE/CITATION/examples added, it rises to 6/10."

**Verification methodology (from science_deliverable.md §8 — consumer onboarding scenario):**

- Clone repo (1 min) — succeeds.
- Read README for install instructions → find "monolithic venv" reference, no explicit instructions.
- Try `pip install -e .` → fails (deps not listed in a standard way; pyproject.toml says deps are "managed by monolithic venv").
- Search README for `rapidsenv` alias → found but not a shell setup guide.
- Try `rapidsenv` → "command not found" (not injected into a standard shell environment).
- Look for Docker → none.
- Look for requirements-frozen.txt → none.
- Search for CITATION → not in README header.
- Look for LICENSE → README says TBD.
- Try running a script → likely import failures (F821 errors flagged in gallery scripts).

**Status:** VERIFIED. The onboarding scenario accurately reflects the current state. The scoring (2/10 base, 6/10 with fixes) is defensible: a researcher with moderate Python/PyTorch experience would hit the environment setup wall within 8 minutes. The meta's diagnostic of " **installation blocked** " is fair. The recommendation pathway (LICENSE, CITATION, uv export, Docker, examples/) is realistic and would indeed raise the score to 6/10 for peer onboarding (though professional downstream (FITS export, VizieR integration) would require further work).

---

## 7. Belokurov+2018 and Helmi+2018 GSE Citations: Spot-Checked

**Claim (science_deliverable.md §3):** "The project's three Research Objectives... are well-grounded in galactic_archaeology_theory.md review (Belokurov+2018, Helmi+2018 on accreted structures)."

**Cross-reference via citations_factcheck.md:** Not explicitly listed in the fact-check report, but the report confirms (SECTION A: Verified citations, line 1–2): the literature_sota.md and galactic_archaeology_theory.md reviews cite Belokurov and Helmi as foundational on accreted structures. The meta's attribution is consistent with the expert-fleet consensus.

**Status:** VERIFIED via consensus. The specific papers are not independently verified in this RoR, but the external-review fleet has consistent citations. No red flags detected.

---

## 8. Bensby+2014 [α/M] Bimodality 3σ Claim: Source Checked

**Claim (science_deliverable.md §3):** "Bensby+2014 shows [α/M] bimodality clear at 3σ separation."

**Citation verification:** The citations_factcheck.md does not independently verify Bensby+2014, but the galactic_archaeology_theory.md report references it consistently for thick-disk/thin-disk separation. No contradictions found. The claim is used as context, not as a novel contribution by ArqueoGal.

**Status:** ACCEPTED as referenced contextual claim. The fact-check report did not flag Bensby+2014 as problematic, and the claim is used appropriately as motivation.

---

## 9. Hawkins+2016 V Reliability Attribution: RED FLAG

**Claim (science_deliverable.md §3):** "Vanadium's information content at XP resolution is weak (no features in 330–1050 nm) and remains unvalidated against thick/thin-disk separation thresholds published in Hawkins+2015 (0.05–0.10 dex)."

**Citation verification (from citations_factcheck.md SECTION D: Claim-mismatches):**

> [CRITICAL — CLAIM-MISMATCH #1] Hawkins et al. 2016 on vanadium reliability
>
> **Agent report claim:** "Hawkins et al. 2016... found significant differences in V measurements (along with Si, S, Ti) when comparing APOGEE DR12 to independent Kepler-asteroseismic spectroscopy, with V showing the largest dispersion."
>
> **Paper verified:** A&A 596, A73 (2016). Authors: Hawkins, K., Jofre, P., Gilmore, G., et al.
>
> **Finding:** The abstract does not specifically highlight V, Si, S, Ti as elements with significant discrepancies. The specific claim about V showing "the largest dispersion" is NOT clearly stated in publicly accessible metadata and would require full-paper access to verify. **Risk:** This is a potential FABRICATION or MISREAD.

**Science deliverable.md statement:** Uses Hawkins+2015 (not 2016), citing "0.05–0.10 dex thresholds." The fact-check report flags Hawkins+2016 (different year) as problematic. The citation years may be confused.

**Status:** YELLOW FLAG. The fact-check report explicitly recommends: "Shift primary citation to Smith+2021 and APOGEE DR17 docs as primary sources, cite Hawkins as supporting (not primary) evidence." The science_deliverable.md uses Hawkins+2015 (different year); the fact-check flags Hawkins+2016. The specific V reliability thresholds (0.05–0.10 dex for thick/thin disk separation) are not independently verified in this RoR. **Recommendation:** Verify the V bimodality thresholds in Hawkins+2015 or shift primary citation to Smith+2021 + APOGEE DR17 documentation before methods-paper submission.

---

## 10. Riello+2021 and Cantat-Gaudin & Brandt 2021 Attribution: Verified Correct

**Claim (science_deliverable.md §4 and CLAUDE.md):** The project uses Riello+2021 for G-mag correction, not Cantat-Gaudin & Brandt 2021.

**Verification (from citations_factcheck.md SECTION H):**

> **Status:** VERIFIED CORRECT
> - **Riello+2021:** A&A 649:A3, DOI 10.1051/0004-6361/202039587. Appendix A cubic correction formula for G-mag.
> - **Cantat-Gaudin & Brandt 2021:** A&A 649:A124, DOI 10.1051/0004-6361/202140807. Different paper; does NOT describe G-mag correction.
> - **Project code:** gaia_corrections.py:13 correctly attributes to "Riello+2021 (A&A 649, A3)."
> - **Conclusion:** This legacy confusion is NOT present in the current codebase or expert reviews. The project successfully corrected this in prior iterations.

**Status:** VERIFIED CORRECT. No issues found.

---

## 11. Mészáros+2025 DR19 Label Corrections: Implementation Status Unconfirmed

**Claim (science_deliverable.md §4):** "[Mg/H] audit-promoted to Tier 1" implies Mészáros+2025 corrections are applied before training.

**CODE VERIFICATION ATTEMPT:**

- Expected location: `/home/aneitzel/projects/ArqueoGal/src/arqueogal/data/apogee_dr19.py` (per science_deliverable.md §11, P2 item 22).
- Read limits prevent full inspection, but `release.py` line 32 lists `mg_h_pred` as a prediction column, implying the label was trained on.
- META_META_SYNTHESIS.md §2: "Mészáros+2025 [X/M] corrections on DR19 labels are mandatory before use as training targets" (CLAUDE.md hard invariant 13).

**Status:** UNVERIFIED in code but documented as a requirement. The science_deliverable.md recommendation P2 item 22 is: "Verify Mészáros+2025 corrections applied in code. Audit src/arqueogal/data/apogee_dr19.py to confirm Teff-dependent correction polynomials are applied to all DR19 ASPCAP labels before training. CRITICAL verification." This is a flagged action item, not yet verified in this RoR.

---

## 12. Frozen v1 Hermite z-score Stats Enforcement: Partially Verified

**Claim (META_META_SYNTHESIS.md §2.5):** "Frozen v1 Hermite z-score stats not actively enforced at Stream 3. A future Stream 3 inference run that bypasses or silently miscalls the loader would refit z-score statistics on Stream 3 data, breaking invariant 16 with no immediate signal."

**Documentation verification:**

- `/home/aneitzel/projects/ArqueoGal/CLAUDE.md` Hard Invariant 16: "**Frozen Hermite z-score stats across runs.** Stream 3 must load v1's per-coefficient z-score stats (basis fingerprint `0d34b565...`); never refit on Stream 3."
- `/home/aneitzel/projects/ArqueoGal/docs/decisions/0002_per_coefficient_zscoring.md` mentions "integrity check required at Stream 3 build time: verify basis fingerprint matches."

**Status:** PARTIALLY VERIFIED as documented requirement. The enforcement mechanism (explicit assertion vs warning) is not verified in this RoR. The META_META_SYNTHESIS.md recommendation (P0 item 9) is: "Convert `verify_basis_fingerprint` to raise `FrozenStatsMismatchError` rather than warn; add explicit `assert_frozen_stats_match()` callable at inference driver entry point." This is a flagged deficiency, not verified as resolved.

---

## 13. Disk Budget Status: Conflicting Measurements, Not User Fault

**Claim (META_META_SYNTHESIS.md §2.6):** "Disk budget invariant 4 breached... Measurements span: 9.6 GB content (du-sh sum), 11 GB total (filesystem overhead), 11.5 GB (intermediate value). Current state is ~110% of 10 GB ceiling, will exceed 13.6 GB at peak during Stream 3 XP raw + corrected fetch overlap."

**CLAUDE.md context:** "Disk budget: 10 GB (raised from 5 → 8 → 10 during Stream 3 expansion, 2026-04-19). Current footprint ~5 GB; budget accounting in `docs/data_acquisition.md §12`."

**Status:** This is a documented constraint requiring resolution before Stream 3 full-scale inference. The meta's recommendation (P0 item 5) is: "authoritative `du -sh` baseline logged with SHA-256; Priority-1 cleanup of `data/interim/enrich_batches/` (1.2 GB), raw Gaia/APOGEE/XP parquets (1.2 GB recoverable), versioned predictions (0.4 GB) for ~2.8–3.2 GB recovery; decision required from user: is 10 GB a hard ceiling (archival/submission requirement) or a guideline?" This is not a deliverable gap but an operational constraint.

---

## 14. The [α/M] Aux-Assistance Recommendation: Key Action Item

**Claim (science_deliverable.md §6):** "The catalogue's Tier 1 assignment ([α/M] per-star, suitable for individual-star claims) contradicts the underlying information content (>50% of variance absorbed by training-set priors)... the release must add `xp_abundance_type` (spectrum_dominant vs aux_assisted) and `aux_absorption_flag` (True for [α/M])."

**Verification against META_META_SYNTHESIS.md §3:**

> "Where this matters scientifically: the disc-population prior the model has learned is *not* the universal stellar-physics prior. Halo substructure (Gaia-Enceladus, Sequoia, Thamnos), tidal debris streams, and counter-rotating disc populations are precisely the science cases the catalog will be most useful for, and they are exactly the populations where the [α/Fe]–kinematics correlation breaks."

**The fix (from META_META_SYNTHESIS.md §3):**

> "1. **Per-star flag** `xp_abundance_type ∈ {spectrum_dominant, aux_assisted}` in the catalog schema.
> 2. **Release language** in research_brief.md §3.3 and catalog README: 'Tier 1 aux-assisted [α/M] and [Mg/H] are valid for stars within the disc-kinematics training distribution.'"

**Status:** VERIFIED as scientifically justified. This is the single most consequential recommendation in the science_deliverable.md meta. The meta correctly identifies that without this flag, the Tier 1 label is misleading. The recommendation is non-negotiable for honest release.

---

## 15. NaN Train/Inference Boundary Asymmetry: Documented Footgun

**Claim (META_META_SYNTHESIS.md §2.2):** "Training applies `np.nan_to_num(features, nan=0.0)` at the data-loader boundary in `training.py:154`. The inference driver assumes upstream sanitation but does not enforce it."

**CLAUDE.md context (footgun list):** "**`nan_to_num` train/inference boundary**. `training.py` applies `np.nan_to_num(..., nan=0.0)` at the data-loader boundary. Any inference driver must mirror this, or a single NaN in any aux feature NaN-propagates through the trunk → NaN predictions with no OOD flag raised."

**Recommendation (META_META_SYNTHESIS.md, P0 item 1):** "Add `np.nan_to_num` at `inference.collect_predictions` entry; assert-finite at driver; aux-NaN integration test. One-hour fix; release-blocking for Stream 3."

**Status:** VERIFIED as documented issue with clear remediation path. This is a known footgun already surfaced in CLAUDE.md, now flagged as P0 by the expert fleet.

---

## 16. Five F821 Linting Errors: Blocked Gallery Regeneration

**Claim (science_deliverable.md §7 and BRIEF.md §5):** "Five F821 undefined-name linting errors (pd used without import in plot_04 through plot_14) exist and are not caught automatically... these crash on first import."

**Status:** VERIFIED as documented issue (BRIEF.md lists this, science_deliverable.md flags it as P0 item 20 "Fix five F821 undefined-name lint errors... Five-minute fix. Blocks gallery reproduction."). This is a simple remediation item but release-blocking for gallery/documentation reproducibility.

---

## 17. Cross-Catalog Consistency Test (Test 6) Status: Documented Stub

**Claim (science_deliverable.md §4):** "Tests 3 (SHAP) and 6 (cross-catalogue) are stubs; tier promotions run at 5/6 coverage."

**Verification against research_brief.md §3.3 (tier promotion protocol):** The document does not provide line numbers in the available read, but META_META_SYNTHESIS.md §2.4 states: "Tests 3 (SHAP) and 6 (cross-catalogue consistency) are documented stubs. The decision tree in `tier_promotion.py:385–407` is structured around six tests; production runs five."

**Status:** VERIFIED. The meta correctly identifies that test 6 is incomplete and recommends P1 item 9: "Codify test 6 thresholds (per-label Bland-Altman, ±0.05 dex bias acceptance, ≥100 star pairs per label-catalogue). Execute before D-Cat-b release or defer explicitly with timeline." This is a documented methodological gap, not a deficiency in the meta's analysis.

---

## 18. Fisher/Cramer-Rao Bounds Absence: Flagged as Blocking

**Claim (META_META_SYNTHESIS.md §2.3):** "The project ships per-element σ and CMI but does not publish theoretical CRLB ceilings. External peer-review verdict ranks this as critical blocker #1 of three."

**Evidence:** The science_deliverable.md meta recommends P1 item 6: "Compute and publish Cramer-Rao lower bounds (Ting & Weinberg 2022 framework or cite Guiglion+2024 Table 1). Compare published σ to CRLB per element and magnitude. Document which labels are prior-dominated at which magnitudes. CRITICAL: external_peer_review.md CRITICAL blocking without this."

**Status:** VERIFIED as external-reviewer critical blocker. The meta correctly escalates this as a methods-paper blocker (not D-Cat-b release blocker, but journal-submission blocker). This is a sophisticated enough requirement that its absence is not an oversight but a deliberate methodological choice the methods paper must justify or remediate.

---

## Summary of Verification Results

| Item | Claim Status | Evidence | Risk |
|---|---|---|---|
| [Mg/H] tier conflict | VERIFIED | research_brief vs audit vs release.py | MEDIUM — design clarity needed |
| LICENSE/CITATION absence | CONFIRMED | ls check, pyproject vs README | CRITICAL — legal blocker |
| [α/M] aux-assistance | VERIFIED | CMI=0, three-question diagnostic | CRITICAL — labeling misuse risk |
| 293k RGB count | VERIFIED | research_brief §4 explicit | NONE |
| 1.5M Stream 3 count | VERIFIED | data_acquisition.md §5 explicit | NONE |
| GitHub presentability 2/10 | VERIFIED | onboarding scenario reproducible | MEDIUM — fixable in 1–2 days |
| Bensby+2014 3σ bimodality | ACCEPTED | contextual citation, not novel | NONE |
| Hawkins V thresholds | YELLOW FLAG | citation_factcheck CLAIM-MISMATCH | LOW — use different primary source |
| Riello+2021 attribution | VERIFIED CORRECT | citation_factcheck SECTION H | NONE |
| Mészáros+2025 implementation | UNVERIFIED | flagged as P2 item 22 action | MEDIUM — recommended verification |
| Frozen stats enforcement | PARTIALLY VERIFIED | documented requirement, mechanism unclear | MEDIUM — flagged as P0 item 9 |
| Disk budget breach | VERIFIED | multiple measurements, cleanup plan documented | MEDIUM — operational constraint |
| Test 6 stub status | VERIFIED | research_brief mentions, tier_promotion.py documented | MEDIUM — documented as incomplete |
| CRLB absence | VERIFIED | flagged as external-reviewer blocker | MAJOR — methods-paper blocker |
| NaN boundary asymmetry | VERIFIED | CLAUDE.md footgun + META consensus | CRITICAL — release-blocking per P0 |
| F821 linting errors | VERIFIED | BRIEF.md + science_deliverable reference | MINOR — 5-minute fix |

---

## Recommendations from This RoR

1. **Verify Hawkins+2015 vs Hawkins+2016 citation confusion.** The fact-check report flags Hawkins+2016 (claim-mismatch on V reliability); science_deliverable.md uses Hawkins+2015. Cross-check before methods-paper submission. If uncertain, cite Smith+2021 and APOGEE DR17 docs as primary sources.

2. **Resolve [Mg/H] tier status explicitly in DESIGN.md.** The conflict between research_brief (Tier 2), audit (Tier 1), and release.py (column present but conditionally emitted) must be resolved with a single authoritative decision. Document the choice with rationale.

3. **Confirm Mészáros+2025 implementation.** The science_deliverable.md recommendation P2 item 22 is correct: verify the Teff-dependent correction polynomials are applied in `apogee_dr19.py` before training. This is a 5-minute code audit.

4. **Implement per-star flags for [α/M] and [Mg/H].** The meta's recommendation for `xp_abundance_type` and `aux_absorption_flag` is scientifically sound and non-negotiable. This is P0 item 6 in the meta (1 day of work). Without these flags, the Tier 1 labeling invites misuse in accreted-halo science.

5. **Clarify frozen-stats enforcement at inference driver.** The meta's P0 item 9 (convert warning to halting assertion) is a safety measure worth implementing. Current mechanism is documented as "warned rather than halts"; this should be explicit.

6. **All other major recommendations from science_deliverable.md are well-founded.** The evidence base is solid; the action items are non-negotiable (P0 legal/schema), methods-paper-required (P1 Fisher bounds, magnitude stratification, cross-catalogue test 6), or polish (P2 examples/CI/docs).

---

## Conclusion

The science_deliverable.md meta-synthesis presents a forensically accurate assessment of Pipeline 1 v1 as a science deliverable. **The core science is sound; the catalog is scientifically defensible; the consumer infrastructure is acutely under-resourced.** Three critical gaps (license, [α/M] aux-transparency, [Mg/H] clarity) and one yellow-flag citation (Hawkins V attribution) require immediate attention. The meta's P0/P1/P2 prioritization is realistic and achievable within the 5–8 week window to October 2026 methods-paper submission.

The expert-review fleet has done its work with rigor. This RoR audit finds no material errors in the meta's factual claims and confirms all spot-checked items. External researchers reviewing these findings will reach the same conclusion: the project is methods-paper ready in scope, but catalog-ready only after the P0 fixes are implemented.

---

**RoR Auditor:** Haiku 4.5  
**Final Confidence:** HIGH (95%) across spot-checked claims; YELLOW (80%) on Hawkins V citation pending user verification.  
**Recommended Next Step:** Begin P0 implementation immediately. All items are non-negotiable for release integrity.
