# Phase 02, §9.2 information-content audit

**Status: Partial. 5 of 6 tests done; tier decisions ratified for all 5 labels. Test 6
(cross-catalogue consistency) blocked on Stream 3 inference.**

## Goal

Produce a per-label report card showing evidence for/against per-star release of each
of the 5 Pipeline 1 labels, and convert the evidence into final tier assignments per
`research_brief.md §3.3`.

## Deliverables

1. `reports/pipeline1/audit/<label>_report_card.md`, one per label (5 files). Shipped.
2. `reports/pipeline1/audit/SUMMARY.md`, consolidated tier assignments with evidence,
   verbatim release-statement text for D-Cat-b. Shipped.
3. `reports/pipeline1/audit/three_question_diagnostic.{md,json}`, the CMI / permutation /
   aux-only triage that resolved the Teff + log g shuffle-null failure. Shipped.
4. `reports/pipeline1/audit/alpha_m_triage.{md,json}`, the three-hypothesis sweep that
   resolved [α/M]'s zero PCA-CMI (H2 parallax-only CMI = 0.1125 nats wins; aux features
   absorb [α/M] variance). Shipped.
5. `reports/pipeline1/audit/pca_cmi_all_labels.json`. PCA-CMI across all 5 labels
   confirming 2-D summary is biased (ADR-0009). Shipped.
6. Test 6 (cross-catalogue consistency) report card. **pending Stream 3 inference**.

## Tier decisions (Option 2, ratified 2026-04-19)

| Label | Tier | Evidence summary |
|---|---|---|
| Teff | **T1 clean** | Aux-only RMSE 164 K vs full 67 K (2.4× improvement); 6 XP features in top-10 permutation |
| log g | **T1 with prior-augmented caveat** | Aux-only 0.225 → 0.157 dex (30% improvement); 0 XP features in top-10; PCA-CMI 0.031 nats just above 0.02 floor |
| [M/H] | **T1 clean** | Null skill_ratio −0.039; XP joint ΔRMSE/σ = 0.691 |
| [α/M] | **T1 clean** | Null skill_ratio −0.257; XP joint ΔRMSE/σ = 0.536; zero PCA-CMI explained by aux absorption (H2), not missing signal |
| [Mg/H] | **T1 clean** | Null skill_ratio +0.062; XP joint ΔRMSE/σ = 0.602 |

## Tests executed

1. LOOCO (per-XP-coefficient zero-out, family-aggregated).
2. Permutation importance (per-feature shuffle, RMSE increase).
3. SHAP. **DEFERRED** (external lib not in pinset).
4. Shuffled-spectrum null (within (Teff, log g) cell permutation).
5. Conditional MI (Kraskov KSG) with PCA summary (7 components, 95.8% var). 2-D summary
   deprecated. ADR-0009.
6. Cross-catalogue consistency. **PENDING** Stream 3 inference overlap with
   AspGap / Guiglion+2024 / SHBoost.

## Acceptance criteria

- All 5 labels have report cards, met.
- Tier decisions for all 5 labels are ratified with documented evidence, met.
- Release-statement text is verbatim-pinned in `SUMMARY.md`, met.
- Test 6 closes the audit. **pending Phase 03 completion**.

## Methodology notes for methods paper

- The 2-D KSG CMI summary is biased upward or downward relative to PCA summary
  (ADR-0009). Publishable as a methodology note.
- Teff + log g failed the literal §9.2 shuffle-null gate. The three-diagnostic triage
  (PCA-CMI, permutation with XP-vs-aux grouping, aux-only baseline) resolved the
  apparent failure as label-specific rather than model-wide. This is publishable as
  "when to run auxiliary diagnostics on §9.2 failures".
- [α/M] had zero PCA-CMI even at 15 components. H2 (parallax-only CMI = 0.1125 nats)
  is the winning explanation: aux features absorb [α/M]'s variance more than other
  chemistry labels. Noted for methods paper; doesn't affect release.

## Needs clarification

- Test 6 execution plan. What does the comparison actually look like? Bland-Altman on
  the overlap with each of AspGap, Guiglion+2024, SHBoost? Combined? Per-label or
  global? Acceptance thresholds not written down.
- The deprecation of 2-D CMI in `audit.py` test 5, ratified in discussion (see
  `docs/decisions/0009_pca_cmi_replaces_2d_summary.md`) but no PR has been
  documented. Confirm whether the code change landed or is still pending.
