# `arqueogal.population_classifier` — Design (pipeline-level)

## Purpose

Pipeline 2: fully unsupervised chrono-chemo-kinematic population classification via
**Parametric UMAP + HDBSCAN with soft memberships**, DBCV-optimised hyperparameters,
MC ensemble over Pipeline-1 calibrated feature uncertainties.

Delivers **D5.1** (open-source GitHub release, Month 10 / Dec 2026) and **D-Cat-d**
(membership probabilities, Month 12 / Feb 2027). Builds on Neitzel+2025 (A&A 695, A243).

**Primary reference**: `docs/research_brief.md` §10 (main), §11 (experimental). This
file is a stub.

## Feature space

10–11 dimensional (expanded from Neitzel+2025's 5D):

```
{age, [Fe/H], [Mg/Fe], [Al/Fe] where available, [C/N] for RGB (evol-stage gated),
 J_R, J_z, L_z, ecc, r_peri/r_apo, E}
```

Rationale: actions (L_z, E) separate GSE / Sequoia / Thamnos far better than LSR-referred
(V_φ, √(U²+W²)). [Mg/Fe] alone beats lumped [α/Fe] on SN-II/Ia timescale resolution.
[Al/Fe] is the single cleanest in-situ vs accreted discriminator (Belokurov & Kravtsov 2022).

Neitzel+2025's 5D `(age, [Fe/H], [α/Fe], V_φ, √(U²+W²))` retained as backwards-compatibility
baseline for reproduction runs.

## Main vs experimental

`main/` is UMAP+HDBSCAN per Neitzel+2025, frozen during deliverable sprints. Experimental
methods (Mahalanobis-UMAP, Aligned UMAP, TDA, deep clustering, diffusion maps, DP-GMM,
hierarchical on embedding) live under `experimental/` and are promoted only after matching
main on DBCV + bootstrap stability + FIRE-2 informedness + held-out-feature consistency.
Cross-imports rejected.

## Validation boundary

- **Subtask 5.1 (hare-and-hounds, FIRE-2 Ananke)** — method ceiling under idealised
  conditions. Metrics: ARI, AMI, Youden J, MCC against simulation ground truth. Prefer
  Remus/Romulus (bimodal per Barry+2026); fall back to m12f for backward compatibility.
- **Subtask 5.2 (real D-Cat)** — no ground truth. Six-diagnostic stack:
  bootstrap-ARI stability (N=500, threshold 0.75), DBCV, permutation-causal feature
  attribution, null-model comparison (multivariate Gaussian + Gaussian copula),
  held-out-feature consistency, literature cross-reference (Dodd+2023, Myeong+2019,
  Horta+2021, Ceccarelli+2024).
- FIRE-2 results do NOT transfer to real-data performance claims. Subtask 5.2 stands
  alone on its own diagnostic stack.

## Hard rules (specific to this pipeline)

- **DBCV maximisation** for hyperparameter selection. **No visual selection.** **No
  persistence-maximisation alone** (favours trivial dense clusters; see research_brief
  §10.7).
- **Soft memberships** via `hdbscan.all_points_membership_vectors()` — required for
  D-Cat-d. GLOSH outlier scores released alongside.
- **MC ensemble N=50** over per-star feature posteriors (including calibrated Pipeline-1 σ).
  Output: mean + std of soft-membership distribution per star per cluster.
- **Parametric UMAP** as main (out-of-sample, differentiable, MC-friendly). cuML UMAP as
  comparison baseline.
- **Bootstrap stability mandatory**. Clusters with median pairwise ARI < 0.5 are rejected
  as artefacts; ARI > 0.75 are stable.
- **Boundary-star flag** when max(soft-membership) < 0.7 or std over MC > 0.15.
- **Evolutionary-stage gating** on [C/N] (RGB only, [Fe/H] > −0.8, Teff 4200–5100 K).
  Red-clump stars do not receive [C/N] features.
- **Collaborator HPC sweep** (Optuna on FIRE-2, persistence-scored) is NOT a substitute for
  our own DBCV grid. Integrate as a prior only if they add DBCV + ground-truth metrics
  (research_brief §10.7). Do not wait for their output.
- **Compute budget**: full MC × DBCV grid × bootstrap on 10⁵ stars ≈ 2–3 days on RTX 3060.
  Plan accordingly.

## References

McInnes+2018 UMAP; Sainburg+2021 Parametric UMAP (arXiv:2009.12981); Campello+2013 HDBSCAN;
Moulavi+2014 DBCV; Hennig 2007 cluster stability; Neitzel+2025 (A&A 695 A243); Myeong+2019;
Koppelman+2019; Belokurov+2018; Dodd+2023; Horta+2021; Belokurov & Kravtsov 2022;
Ceccarelli+2024; Sanderson+2020 / Nguyen+2024 Ananke; Barry+2026; Parul+2025.
