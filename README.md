# ArqueoGal

**Chrono-chemo-kinematic map of the Milky Way disc from Gaia DR3, APOGEE DR19 and TESS asteroseismology.**

Project 2024.15303.PEX (DOI: [10.54499/2024.15303.PEX](https://doi.org/10.54499/2024.15303.PEX)), FCT-funded exploratory research. PI: Tiago Campante (CAUP/IA, Porto). Period: February 2026 – August 2027.

This repository is Andreas Neitzel's (Co-I, PhD student) personal development workspace for two ML pipelines and three deliverables within ArqueoGal, it is not the team-wide repository. Code matures here and is integrated upstream for each deliverable.

---

## What this workspace produces

| Deliverable | Due | Role of this repo |
|---|---|---|
| **D-Cat-b** (supporting contribution) | Aug 2026 (Month 6) | XP-based chemical abundance catalogue for stars without APOGEE DR19 spectroscopy. **Produced here** by Pipeline 1 |
| **D5.1** | Dec 2026 (Month 10) | Open-source ML tool for stellar-population classification. **Produced by Starfold** (separate repo); this repo provides the upstream abundance predictions |
| **D-Cat-d** | Feb 2027 (Month 12) | Stellar-population membership probabilities for the all-sky ArqueoGal catalogue. Produced by running Starfold on this repo's Pipeline 1 predictions |

The scope of this repository is **Pipeline 1**: Gaia DR3 XP coefficients → APOGEE-DR19-calibrated abundances:

- **`xp_abundances`:** semi-supervised multi-task regression with contrastive pretraining + joint-loss fine-tuning. Outputs per-star {[M/H], [α/M], [Fe/H], [Mg/H], Teff, log g} predictions with block-Cholesky covariant uncertainties, plus OOD / support / release-tier quality flags.

Population classification (formerly Pipeline 2 inside this repo) was spun out to **Starfold** on 2026-04-22; the code now lives at <https://github.com/AndreasWNeitzel/Starfold>. See [`docs/plan/04_pipeline2_main.md`](docs/plan/04_pipeline2_main.md) for the integration contract.

---

## Start here

![Data flow](reports/figures/data_overview/panel_01_data_flow.png)

**For a visual tour of the data we ingest and produce:** [`docs/data_overview.md`](docs/data_overview.md), eight panels covering data-flow, sky footprint, Kiel diagram, `[α/M]` vs `[Fe/H]` chemistry, magnitude reach, label availability tiers, extinction priors, and row-count waterfall.

**For scientific rationale and methodology:** [`docs/research_brief.md`](docs/research_brief.md), literature review, tier-promotion protocol, information-content audit, six-diagnostic validation stack for real-data clustering, and the scope of what each pipeline claims.

**For TAP queries, cross-matching, and preprocessing recipes:** [`docs/data_acquisition.md`](docs/data_acquisition.md), the 5 GB-budget data plan, Gaia DR3 corrections, Ye+2024 XP preprocessing order, dust-map composition, provenance sidecar schema.

---

## Repository layout

```
ArqueoGal/
├── docs/            ← project references (research_brief, data_acquisition, data_overview)
├── configs/         ← YAML configs (main/ vs experimental/)
├── src/arqueogal/
│   ├── data/                    ← ingestion, cross-matching, feature engineering
│   ├── xp_abundances/           ← Pipeline 1 (main/ + experimental/)
│   └── utils/                   ← shared coordinates, plotting, I/O, GPU helpers
├── scripts/         ← CLI entry points (ingestion, corrections, feature emission, plotting)
├── notebooks/       ← exploration and visualisation
├── tests/           ← mirrors src/ with separate trees for main/ and experimental/
├── reports/         ← figures (reports/figures/data_overview/ is the canonical overview)
├── data/            ← raw, interim, processed, external (gitignored; 5 GB budget)
└── models/          ← serialised checkpoints (gitignored)
```

Main-pipeline code is frozen during deliverable sprints; feature work goes to `experimental/` and is promoted only after beating main by a documented margin. See `docs/research_brief.md` §11 for the promotion bar.

---

## Environment

- WSL2 Ubuntu, Python 3.12, RAPIDS 25.10 (cudf / cuml / cugraph), PyTorch 2.10 + CUDA 13.
- Activate with `rapidsenv` (shell alias). Do not create new venvs.
- Before any `pip install`, verify it will not bump RAPIDS-pinned versions (cudf, cuml, numpy, pandas, pyarrow).

---

## Credentials. AIP TAP token

Most data sources (GAVO, VizieR, SDSS DR19, MAST) are public and need no auth. The one exception is the **AIP TAP service** (`gaia.aip.de`), which hosts the authenticated Gaia DR3 mirror used by Stream 1, Stream 3, the XP-coefficient fetch (`gaiadr3.xp_continuous_mean_spectrum`), and StarHorse2 distances. Bearer-token auth is the preferred path; YAML user/password is supported for legacy accounts.

**Set up the token (preferred):**

```bash
# one-shot: export in your shell, or add to ~/.bashrc / ~/.zshrc
export GAIA_AIP_TOKEN="<your AIP personal access token>"
```

Generate the token from your AIP Daiquiri profile at <https://gaia.aip.de/accounts/profile/>. The token is sent as an `Authorization: Token <token>` header, never commit it, never log it.

**Alternative. YAML file:**

```yaml
# ~/.arqueogal/credentials.yaml    (chmod 600)
aip:
  user: "your_aip_login"
  password: "your_aip_password"

# optional. ESA Gaia Archive (public access works without)
esa:
  user: "your_esa_login"
  password: "your_esa_password"
```

YAML wins when both YAML and `GAIA_AIP_TOKEN` are present. Custom path via `ARQUEOGAL_CREDENTIALS_PATH`. Full schema in `src/arqueogal/data/credentials.py`.

**Verify:**

```bash
python -c "from arqueogal.data.tap import aip_service; aip_service().search('SELECT TOP 1 source_id FROM gaiadr3.gaia_source')"
```

Without a working token, every Gaia-backed ingestion script (`scripts/fetch_gaia_xp.py`, `scripts/ingest_stream2.py`, `scripts/apply_gaia_corrections.py`'s upstream fetches, etc.) will fail at the first TAP call.

---

## Quickstart: read a published prediction parquet

The catalogue parquets emitted by `release.py` carry per-element predictions, calibrated covariant uncertainties, and the v5 release-tier columns. To open one and filter to the trustworthy slice (Tier 1 + Tier 2):

```python
import pandas as pd

df = pd.read_parquet("release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet")
trustworthy = df[df["release_tier"] <= 2]

# Per-element columns: <elem>_hybrid_pred / <elem>_hybrid_sigma / <elem>_hybrid_source
# (regressor or knn-median substitution under v5 σ-inflation rules).
print(trustworthy[
    ["source_id", "teff_hybrid_pred", "logg_hybrid_pred",
     "mh_hybrid_pred", "alpha_m_hybrid_pred", "release_tier"]
].head())
```

The companion `*.release_tier.json` sidecar records the catalogue schema version, tier counts, and which OOD/caveat flags were active at annotation time. See [`docs/CATALOG_SCHEMA.md`](docs/CATALOG_SCHEMA.md) for the full column reference and [`docs/decisions/0015_v5_release_tier_simplification.md`](docs/decisions/0015_v5_release_tier_simplification.md) for the v5 rationale.

---

## Quickstart: run the release-tier annotation on a fresh inference parquet

If you have produced a fresh prediction parquet via `scripts/run_pipeline1_inference.py`, attach the v5 release-tier columns with:

```python
from pathlib import Path
from arqueogal.xp_abundances.main.release import annotate_parquet

summary = annotate_parquet(Path("path/to/predictions.parquet"))
print(summary)  # {"n_rows": ..., "counts": {1: ..., 2: ..., 3: ...}}
```

`annotate_parquet` is idempotent: it can be re-run after a schema bump and will overwrite the per-element tier columns and sidecar in place. The function emits structured logs at INFO level on entry and exit; configure logging in your driver to capture them.

---

## References

- **Pipeline 1 groundwork:** Ye+2024 (arXiv:[2411.19105](https://arxiv.org/abs/2411.19105)), Andrae+2023, Mészáros+2025 (arXiv:[2506.07845](https://arxiv.org/abs/2506.07845)), Lindegren+2021, Riello+2021 A&A 649 A3 Appendix A.
- **Downstream classifier (Starfold):** Neitzel+2025 A&A 695 A243 (arXiv:[2501.16294](https://arxiv.org/abs/2501.16294)). UMAP+HDBSCAN stellar-population methodology, implemented in the separate Starfold repository at <https://github.com/AndreasWNeitzel/Starfold>.
- **Data maps:** Edenhofer+2024, Lallement+2022, Schlegel-Finkbeiner-Davis 1998, Bailer-Jones+2021 (GAVO `gedr3dist.main`), Queiroz+2023 StarHorse2 v2 (AIP `gaiadr3_contrib.aqueiroz2023_*_v2`).

---

## Licence

TBD, the open-source D5.1 release in December 2026 will ship under an OSI-approved licence; the intermediate code in this workspace is internal project work until then.
