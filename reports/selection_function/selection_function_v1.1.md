# Compound Selection Function — v1.1

**Change from v1:** per-star selection probability is now a **compound** of Ye+2024 `NO_SYNTH_PHOT` retention (the v1 artefact, unchanged) and IR (2MASS/AllWISE) photometric completeness (new in v1.1). v1 remains in place at `reports/selection_function/selection_function_v1.{md,parquet,provenance.json}` for historical reference.

**Compound definition:**

```
p_compound = p_ye_retained · p_ir_complete · p_parallax · p_extinction
```

where `p_parallax` and `p_extinction` are 0/1 gates in v1.1 — they take a data-availability flag (True/False) and map to 1.0/0.0. Smooth per-star parallax- and extinction-availability probabilities are earmarked for v1.2.

## Scorer contract

```python
from arqueogal.data.selection_function import (
    score_selection_prob,             # v1, unchanged
    score_ir_completeness,            # new in v1.1
    score_compound_selection_prob,    # new in v1.1
)

# Unchanged backwards-compatible API:
p_ye = score_selection_prob(b_deg, g_mag)

# New in v1.1:
p_ir = score_ir_completeness(b_deg, g_mag, teff, logg)
bundle = score_compound_selection_prob(
    b_deg, g_mag, teff, logg,
    parallax_over_error=pox, av_missing=False,
)
# → {'p_ye_retained', 'p_ir_complete', 'p_compound', 'components': {...}}
```

## Artefact schema (`selection_function_v1.1.parquet`)

One row per (|b|, G) cell — 25 rows. This is the **|b|×G marginal** view, usable directly by consumers whose pipelines do not carry Teff / log g at the scoring stage. The full 4-D IR-completeness grid lives in its sibling artefact `reports/selection_function/ir_completeness_v1.parquet`.

| column | dtype | notes |
|---|---|---|
| `b_lo`, `b_hi` | float64 | |b| bin edges, deg |
| `g_lo`, `g_hi` | float64 | G bin edges, mag |
| `p_ye_retained` | float64 | from v1 (`1 − P(NO_SYNTH_PHOT)`), Laplace-free; floor 0.01 |
| `p_ir_complete` | float64 | new; `(n_c + 1) / (n_t + 2)`; Laplace-smoothed; floor 0.01, ceil 1.0 |
| `n_total_bg`, `n_complete_bg` | int64 | Stream-1 counts in cell |
| `p_compound_bg` | float64 | `p_ye · p_ir`, clamped to [0.0001, 1.0] |

## Provenance

- Input Stream-1 Parquet SHA-256: `4a98caa75f7ce940be8d9d5f9f8593cd3313eb6b97351871993b8110ea629cc1`.
- v1 Ye-retention Parquet SHA-256: `da0c83d47967d5e29cbcf1cbf0c251f14d7296a6d58e43b4511b4af252c95aae`.
- Git SHA and build timestamp in `selection_function_v1.1.provenance.json`.

## References

- Full IR-completeness methodology: `reports/selection_function/ir_completeness_v1.md`.
- v1 Ye-retention methodology: `reports/selection_function/selection_function_v1.md`.
- Pipeline context: `docs/data_acquisition.md` §6.4 and §6.6.

