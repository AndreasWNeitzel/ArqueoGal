# ArqueoGal — Data Overview

**Scope:** a visual tour of the data ArqueoGal ingests, what gets produced at each stage, and how Stream 1 / Stream 2 / Stream 3 feed the two ML pipelines.
**Companion documents:** [`research_brief.md`](research_brief.md) (scientific rationale, methodology), [`data_acquisition.md`](data_acquisition.md) (TAP queries, cross-matching, preprocessing recipes).
**How to regenerate these figures:** `python scripts/plot_data_overview.py` — writes PNG + PDF into `reports/figures/data_overview/`. Re-run whenever the feature matrices change.

This document is the canonical, one-page entry point for anyone (collaborator, reviewer, future-self) who wants to know *"what data does this project actually have?"* before diving into the brief or the acquisition spec.

---

## 1. Data flow — streams → pipelines → deliverables

![Data flow](../reports/figures/data_overview/panel_01_data_flow.png)

Three acquisition streams feed two machine-learning pipelines, which produce three deliverables on the FCT timeline.

- **Stream 1** (APOGEE DR19 × Gaia DR3 on RGB+RC giants, ~324 k rows) is the **training set** for Pipeline 1. Every row carries calibrated APOGEE DR19 ASPCAP labels ($T_\mathrm{eff}$, $\log g$, [M/H], [Fe/H], [α/M], per-element [X/M]) plus Gaia DR3 XP coefficients, after Mészáros+2025 $T_\mathrm{eff}$-trend corrections on APOGEE DR19 and Lindegren+2021 zpt + Riello+2021 G-mag corrections on Gaia.
- **Stream 2** (TESS Hon+2021 $\nu_\mathrm{max}$ × Gaia DR3, ~158 k rows) is **pre-staged** for Task 4 asteroseismic ages (led externally by Campante/Miglio). Neither Pipeline 1 nor Pipeline 2 consumes it yet — it is here so the ingestion, cross-match, and provenance scaffolding are ready when age labels arrive.
- **Stream 3** (Andrae+2023 vetted RGB+RC × Gaia DR3, ~168 k rows sub-selected from 10.5 M) is the **inference set** — where Pipeline 1 predicts abundances in the XP-only regime, feeding Pipeline 2's chrono-chemo-kinematic feature vector.
- **External** priors (Edenhofer+2024, Lallement+2022, SFD, Bailer-Jones+2021 distances, GSP-Phot neighborhood-median extinction) are composed into every stream's feature row according to `data_acquisition.md` §8.5.

**Training data flows Stream 1 → Pipeline 1.** **Inference data flows Stream 3 → Pipeline 1 → Pipeline 2.** Pipeline 2 produces D-Cat-d (soft membership probabilities); Pipeline 1 outputs feed D-Cat-b (chemical abundances). D5.1 is the open-source release of Pipeline 2.

---

## 2. Sky footprint (Mollweide, galactic coordinates)

![Sky footprint](../reports/figures/data_overview/panel_02_sky_mollweide.png)

Four Mollweide panels in galactic $(\ell, b)$, centred on $\ell = 0$. **Astronomical convention:** longitude increases to the *left* (the observer looks out at the celestial sphere, not down at a terrestrial map).

- **Stream 1** concentrates on the APOGEE-2 footprint: the Galactic plane, the bulge, and the APOGEE-N/APOGEE-S tiling pattern are visible. Off-plane coverage is sparse — this is the fundamental reason Pipeline 1 is needed in the first place (XP can cover the full sky where APOGEE DR19 cannot).
- **Stream 2** tracks the TESS continuous-viewing zones and the ecliptic-poles geometry — wall-to-wall in the polar caps, sparse near the ecliptic.
- **Stream 3 (selected)** is the stratified sub-sample cut from Andrae+2023 — near-uniform coverage after the $(T_\mathrm{eff}, \log g, [\mathrm{M/H}], G)$ stratification, with the Galactic-plane density still dominating because that is where the RGB+RC giants live.
- **Andrae+2023 parent sample** (10.5 M) shows the raw all-sky density of their vetted RGB+RC catalogue before our selection; compare with Stream 3 selected to see where we have deliberately thinned the plane.

---

## 3. Kiel diagram — Stream 1 training set

![Kiel diagram](../reports/figures/data_overview/panel_03_kiel.png)

Stream 1 in the $(T_\mathrm{eff}, \log g)$ plane, coloured by [M/H]. The red dashed rectangle marks the RGB+RC selection window used by both pipelines ($T_\mathrm{eff} \in [4000, 5500]$ K, $\log g \in [1.0, 3.5]$). This panel is the fastest sanity check that (a) the APOGEE DR19 × Gaia cross-match landed on giants rather than dwarfs, (b) the metallicity gradient along the RGB has the right orientation, and (c) the red-clump bunching at $\log g \approx 2.4$ is clearly separated from the RGB bump — which matters for gating [C/N]-based age calibration later (the [C/N]-mass relation does **not** apply to clump stars; see `research_brief.md` §4).

---

## 4. Chemical feature space — Tinsley–Wallerstein

![Tinsley-Wallerstein](../reports/figures/data_overview/panel_04_tinsley_wallerstein.png)

[α/M] vs [Fe/H] hexbin number density for Stream 1 (colorbar: stars per hex bin). **The bimodal high-α / low-α sequence is the single most important structural signal Pipeline 2 must recover.** If our unsupervised clustering on the full 10–11D chrono-chemo-kinematic vector fails to separate these two sequences as distinct components, the method is broken — this is the simplest ground-truth check against known Milky Way disc structure. The visible knee near $[\mathrm{Fe/H}] \approx -0.5$ and the clean low-α plateau near $[\alpha/\mathrm{M}] \approx 0$ are the expected APOGEE DR19 morphology.

---

## 5. Magnitude distribution per stream

![G-mag histogram](../reports/figures/data_overview/panel_05_magnitude_hist.png)

G-band histograms per stream with the Gaia XP native release cutoff (G ≈ 17.65) marked in red. Pipeline 1's scope is the XP-native regime — we do **not** push beyond G = 17. Reading the plot:

- Stream 1 peaks around $G \approx 12\text{–}14$ (APOGEE DR19 is SNR-limited for spectroscopy, not magnitude-limited by XP) and has a long tail towards $G \approx 16$.
- Stream 2 is the brightest (TESS is 4–13 mag in its primary mission) — this is why it will dominate Task 4 asteroseismology but is a minor contributor to the combined catalogue's star count.
- Stream 3 is deliberately distributed across the faint end up to the cutoff — where Pipeline 1's *extension* value over APOGEE DR19 is greatest.

---

## 6. Label availability matrix — Tier 1 / Tier 2 / Tier 3

![Label matrix](../reports/figures/data_overview/panel_06_label_matrix.png)

Per-column finite-value rate on `flag_bad == 0` Stream 1 rows, colour-coded by release tier (`research_brief.md` §3.2):

- **Tier 1 (blue) — per-star reliable:** $T_\mathrm{eff}$, $\log g$, [M/H]. Always released per star; near-100% coverage in Stream 1.
- **Tier 2 (orange) — population-level recommended:** [Fe/H], [α/M], [Mg/Fe], [Al/Fe]. ≥ ~95% coverage. Released per star but with a population-level health check required before citing any given value.
- **Tier 3 (gray) — not released per-star:** Si, Ca, Ti, Mn, Ni, Na, Cr, K, V, S, Ce, C, N, O. Coverage is patchier and element-dependent. An element is promoted Tier 3 → Tier 2 only after passing the full six-test protocol in `research_brief.md` §3.3.

---

## 7. Extinction prior composition

![Extinction priors](../reports/figures/data_overview/panel_07_extinction_priors.png)

Pipeline 1 does not trust any single 3D dust map. Each star carries up to five extinction signals — left panel shows per-source coverage, right panel shows the value distribution:

- `av_edenhofer` — Edenhofer+2024, distance $< 1.25$ kpc, highest resolution near the Sun.
- `av_lallement` — Lallement+2022, $1.25\text{–}3$ kpc, fills the intermediate-distance gap.
- `av_sfd` — SFD integrated asymptotic extinction, for stars beyond the 3D-map reach.
- `av_nbhd_median` — GSP-Phot $A_G$ median over the 32 nearest 3D-Cartesian neighbours. Zero additional disk cost; the ML learns when to deviate from the prior.
- `ag_gspphot` — Gaia DR3 GSP-Phot direct $A_G$ estimate, used as a prior-plus-disagreement feature.

Coverage is deliberately staggered across stream geometry; the neighbourhood-median (red) is a belt-and-braces prior present for every star that has any GSP-Phot neighbour, which is ~all of them.

---

## 8. Stream 1 row-count waterfall

![Row-count waterfall](../reports/figures/data_overview/panel_08_rowcount_waterfall.png)

How ~965 k APOGEE DR19 ASPCAP rows collapse to the ~324 k Stream 1 training rows:

1. `flag_bad == 0` ASPCAP flag — removes catastrophic fits.
2. `SNR_APOGEE > 70` — the minimum for trustworthy abundances (`data_acquisition.md` §3.4).
3. Gaia DR3 XP available — Stream 1 demands an XP coefficient vector for every training star.
4. RGB+RC cut — $T_\mathrm{eff} \in [4000, 5500]$ K, $\log g \in [1.0, 3.5]$, matches the selection window in panel 3.
5. Dedup on `source_id` — APOGEE DR19 multi-visit and re-reductions can produce duplicates.

The final row count is authoritative: it comes straight from `pipeline1_features_stream1.parquet`. The intermediate counts are approximate; see `data_acquisition.md` §3.4 for the exact filter order and the provenance sidecar shipped next to the Parquet for the row-count audit.

---

## Regeneration

```bash
python scripts/plot_data_overview.py                 # full run (~1 min)
python scripts/plot_data_overview.py --skip 2 8      # skip expensive panels
python scripts/plot_data_overview.py --stream1-sample 50000   # faster Stream 1 sky plot
```

Outputs land in `reports/figures/data_overview/panel_{01..08}_*.{png,pdf}`. Both formats are written by default — PNG for docs embedding, PDF for publication-ready inserts.
