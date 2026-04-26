# ArqueoGal — Onboarding Tutorial Audit

**Date:** 2026-04-26  
**Auditor:** Claude Code (Haiku)  
**Scope:** GitHub onboarding and tutorial readiness for new collaborators

---

## Executive Summary

The repository is scientifically mature (v5 schema shipped, Stream 3 inference underway) but lacks three critical tutorials for GitHub users. The README correctly defers to domain-specific docs (research_brief, data_acquisition, HARDWARE_SETUP), but lacks (a) a first-day dev install walkthrough, (b) a science-user "load and filter the catalogue" recipe, and (c) an "extend the pipeline" tutorial for contributors.

---

## Current State: What Exists

### Strengths
- **README.md**: Excellent high-level overview of deliverables, repository role within ArqueoGal, and references to deeper docs.
- **docs/data_acquisition.md**: Complete TAP query, preprocessing, and provenance spec (60+ sections).
- **docs/research_brief.md**: Scientific protocol and tier-promotion logic clearly documented.
- **docs/CATALOG_SCHEMA.md**: Full column reference with NaN semantics for all release columns.
- **docs/HARDWARE_SETUP.md**: WSL2 memory ceiling, filesystem placement, OOM recovery.
- **docs/plan/**: Phase-by-phase status with current blockers and known stubs.
- **docs/decisions/**: 13 ADRs documenting non-obvious technical choices (e.g., ADR-0015: v5 schema simplification).
- **pyproject.toml**: Minimal but correct (no build-from-source required; venv pre-installed).

### Gaps

**1. No "First-Day Dev Install" Tutorial**

An IA postdoc cloning the repo from GitHub sees:
- README → points to HARDWARE_SETUP, CLAUDE.md, and various docs
- No single sequence: "Clone → activate venv → pull credentials → validate install → run a smoke test"

Current user must:
- Infer that `rapidsenv` is a shell alias (how? where is it defined?)
- Know to set `GAIA_AIP_TOKEN` or create `~/.arqueogal/credentials.yaml` (documented in README §Credentials, but isolated)
- Know that the venv is pre-built at `~/.venvs/rapids25.10_python3.12_cuda13` (documented in global CLAUDE.md, not in README)
- Know that `poetry` / `pip install -e .` are NOT the install method (no installation section present)
- Know to check AIP auth before running any ingestion script

**2. No "Consume the Catalog" Tutorial**

A science user wants to: load the released Parquet, filter to Tier 1, plot a Kiel diagram, histogram [α/M].

Current state:
- CATALOG_SCHEMA.md lists all columns and their semantics.
- No worked example: "Load v5 catalogue → filter on release_tier==1 → make 3 plots"
- No mention of where released catalogues live (`data/processed/stream3_*` or cloud storage?)
- No mention of how to interpret release-tier quality flags (Tier 1 is "ready to cite", Tier 2 is "aggregate studies OK", Tier 3 is "methods work only")
- No example of accessing covariance structure (block-Cholesky covariance stored where? how to unpack?)

**3. No "Extend the Pipeline" Tutorial**

A contributor wants to add a new feature column (e.g., new kinematic flag, extinction correction, distance prior).

Current state:
- Each module has a DESIGN.md file (e.g., `src/arqueogal/xp_abundances/main/DESIGN.md`).
- CLAUDE.md invariant: "DESIGN.md co-commit discipline — any change to column shape/name/dtype requires a same-commit DESIGN.md update."
- No tutorial showing: (1) where to add code, (2) how to test locally, (3) how to emit the new column in the right pipeline stage, (4) how to update DESIGN.md.

---

## Minimal Tutorials to Make GitHub-Ready

### A. First-Day Dev Install (Estimated: 8–10 minutes to read, 5 min to execute)

**Title:** "Get Started in 10 Minutes"

**Content:**

1. **Prerequisites check**: Do you have Python 3.12 and CUDA 13? (link to HARDWARE_SETUP.md)
2. **Clone and shell setup** (2 steps):
   ```bash
   git clone https://github.com/AndreasWNeitzel/ArqueoGal.git
   cd ArqueoGal
   ```
3. **Activate the pre-built venv** (1 step):
   ```bash
   source ~/.venvs/rapids25.10_python3.12_cuda13/bin/activate
   # or use the shell alias: rapidsenv
   ```
4. **Set up credentials** (choose one):
   - Bearer token: `export GAIA_AIP_TOKEN="..."`
   - Or YAML: create `~/.arqueogal/credentials.yaml` (example in README)
5. **Validate install** (3 commands):
   ```bash
   python -c "import arqueogal; print('arqueogal OK')"
   python -c "from arqueogal.data.tap import aip_service; aip_service().search('SELECT TOP 1 source_id FROM gaiadr3.gaia_source')"
   python -m pytest tests/ -k "test_smoke" -v
   ```
6. **Understand the layout** (link to repository layout section in README)

**Where to place:** `docs/tutorials/01_first_day.md` (or integrate into README as "Quick Install" subsection)

---

### B. Consume the Catalog (Estimated: 15–20 minutes to read, 10 min to run)

**Title:** "Load and Filter the Catalogue"

**Content:**

1. **What you'll do**: Load the v5 catalogue (released Parquet), filter to Tier 1 (science-ready), and plot a Kiel diagram.
2. **Prerequisites**: First-day install complete, `pandas` or `polars` familiar.
3. **Step 1: Load the catalogue**
   ```python
   import pandas as pd
   cat = pd.read_parquet("data/processed/stream3_v5_release.parquet")
   print(f"Catalogue: {len(cat)} stars")
   print(cat.columns[:10])
   ```
4. **Step 2: Understand release tiers**
   - Tier 1: `release_tier == 1` — use in science publications
   - Tier 2: `release_tier == 2` — use in aggregate studies with caveats
   - Tier 3: `release_tier == 3` — methods/diagnostics only
5. **Step 3: Filter to Tier 1**
   ```python
   t1 = cat[cat.release_tier == 1]
   print(f"Tier 1: {len(t1)} stars ({100*len(t1)/len(cat):.1f}%)")
   ```
6. **Step 4: Plot Kiel diagram** (worked example with matplotlib)
   ```python
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots()
   ax.scatter(t1.teff, t1.logg, s=1, alpha=0.3)
   ax.invert_yaxis()
   ax.set_xlabel(r"$T_{\rm eff}$ (K)")
   ax.set_ylabel(r"$\log g$ (dex)")
   plt.savefig("kiel.pdf")
   ```
7. **Step 5: Explore abundance distribution**
   ```python
   fig, axes = plt.subplots(2, 2, figsize=(8, 6))
   for ax, col in zip(axes.flat, ["mh", "alpha_m", "feh", "mg_h"]):
       ax.hist(t1[col].dropna(), bins=50, edgecolor='k', alpha=0.7)
       ax.set_xlabel(f"[{col.replace('_', '/')}] (dex)")
   plt.tight_layout()
   plt.savefig("abundances.pdf")
   ```
8. **Covariance and uncertainties**: Brief note that per-star covariance is stored as block-Cholesky factors in `prediction_cov_chol_...` columns; link to CATALOG_SCHEMA.md for details.

**Where to place:** `docs/tutorials/02_load_catalogue.md`

---

### C. Extend the Pipeline (Estimated: 25–30 minutes to read, 30 min to implement + test)

**Title:** "Add a New Feature Column"

**Content:**

1. **What you'll do**: Add a new kinematic quality flag (e.g., "orbit_is_stable") that gets emitted in the v5 release catalogue.
2. **Prerequisites**: First-day install, familiarity with one of the pipeline scripts (e.g., `scripts/build_mode_ambiguous_mask.py`).
3. **Step 1: Understand the pipeline stages**
   - Inference (produces predictions + OOD flags) → `src/arqueogal/xp_abundances/main/inference.py`
   - Release annotation (adds tiers, flags, diagnostics) → `src/arqueogal/xp_abundances/main/release.py`
   - Where to add your code? Depends on whether the flag depends on predictions or is purely diagnostic.
4. **Step 2: Anatomy of a new column**
   - Define the column in the module's DESIGN.md (type, valid range, NaN semantics)
   - Implement the logic in the appropriate script (e.g., `scripts/build_new_flag.py`)
   - Load the flag in the release module and emit it as a new release column
5. **Step 3: Implement the feature** (worked example: kinematic OOD flag)
   ```python
   # scripts/build_kin_ood_flag.py (already exists — use as template)
   def compute_kin_ood_flag(predictions_df):
       """Return boolean flag: True if kinematic coordinates are outliers."""
       # Load pre-trained kinematic model or compute Mahalanobis distance
       # Return pd.Series(bool, name='kin_ood_flag')
   ```
6. **Step 4: Integrate into release** (in `src/arqueogal/xp_abundances/main/release.py`)
   ```python
   kin_flags = pd.read_parquet("path/to/kin_ood_flags.parquet")
   release_df["kin_ood_flag"] = release_df["source_id"].map(kin_flags.set_index("source_id")["kin_ood_flag"])
   ```
7. **Step 5: Update DESIGN.md**
   - Record the new column, its type, valid range, and NaN semantics
   - Commit this alongside the code change (mandatory per CLAUDE.md)
8. **Step 6: Test locally** (minimal smoke test)
   ```bash
   python scripts/build_new_flag.py --input data/processed/predictions.parquet --output data/interim/new_flag.parquet
   python -c "import pandas as pd; df = pd.read_parquet('data/interim/new_flag.parquet'); print(f'{len(df)} rows, {df.isna().sum()} NaNs')"
   ```
9. **Step 7: Run the full release pipeline**
   ```bash
   python scripts/assign_release_tier.py --input data/processed/predictions.parquet --new-flag new_flag
   ```
10. **Common pitfalls**:
    - Forgetting to update DESIGN.md in the same commit (rejected at review)
    - NaN handling: what does NaN mean for your column? (e.g., "kinematic data unavailable" vs "kinematic flag undefined")
    - Schema version bump: if adding a new column, increment CATALOG_SCHEMA.md version field

**Where to place:** `docs/tutorials/03_extend_pipeline.md`

---

## Recommendations for Implementation

1. **Placement**: Create `docs/tutorials/` directory with three `.md` files (01, 02, 03 above).
2. **Integration**: Update README.md with a "Tutorials" section linking to all three.
3. **Effort estimate**: 2–3 hours to write (including code testing and screenshot validation).
4. **Review priority**:
   - High: Tutorial A (first-day install) — blocks all future contributors.
   - Medium: Tutorial B (consume catalogue) — required for science users, methods-paper authors.
   - Low: Tutorial C (extend pipeline) — required only for co-developers (Starfold integration team, future PhD students).

---

## Related Documentation That Could Be Linked

- `docs/context/conventions.md` (code style, testing expectations)
- `docs/decisions/ADR-0015.md` (v5 schema simplification rationale)
- `docs/plan/00_overview.md` (current phase and blockers)
- `tests/` directory structure (shows existing smoke tests and patterns)

---

## Conclusion

The repository is scientifically and technically mature but reads as **Andreas's personal workspace** rather than a **community-ready GitHub repository**. Three concise, worked tutorials would transform it into a resource that can accommodate new IA postdocs, external collaborators, and downstream users of the D-Cat-b catalogue without repeated Slack questions.
