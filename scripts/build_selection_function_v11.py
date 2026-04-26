"""Build the ArqueoGal compound selection-function v1.1 artefacts.

This builder extends v1 (Ye+2024 ``NO_SYNTH_PHOT`` retention on ``(|b|, G)``)
with a second component: **P(IR-complete | |b|, G, Teff, log g)** computed on
the Stream 1 training set (``data/processed/pipeline1_features_stream1.parquet``).

Rationale
---------
The IR-dependency diagnostic confirmed the five 2MASS/AllWISE magnitudes are
load-bearing on all five Pipeline-1 labels. Stars with missing IR counterparts
fall into the "IR=0 rare-pattern" regime at inference (training used zero-
imputation on ~0.4 % of rows without 2MASS/AllWISE); their per-star
predictions are scientifically different from the IR-complete majority. For
scientifically-consistent Stream 3 release, D-Cat-b consumers need **both**
per-star probabilities (Ye-retained AND IR-complete), multiplicatively
composed into ``p_compound``.

Outputs
-------
- ``reports/selection_function/ir_completeness_v1.{parquet,md,provenance.json}``
  — the full 5×5×3×2 grid plus the |b|×G marginal fallback.
- ``reports/selection_function/selection_function_v1.1.{parquet,md,provenance.json}``
  — joins Ye retention (from v1) with IR completeness into one artefact
  carrying both scalars (plus Laplace-smoothed probabilities).

Leaves v1 in place for historical reference.

Deterministic; no random state. Read-only on the input; atomic writes on
outputs.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_selection_function_v11")

REPO: Final[Path] = Path(__file__).resolve().parents[1]
INPUT_PARQUET: Final[Path] = REPO / "data" / "processed" / "pipeline1_features_stream1.parquet"
YE_V1_PARQUET: Final[Path] = (
    REPO / "reports" / "selection_function" / "selection_function_v1.parquet"
)
IR_GRID_PARQUET: Final[Path] = (
    REPO / "reports" / "selection_function" / "ir_completeness_v1.parquet"
)
COMPOUND_PARQUET: Final[Path] = (
    REPO / "reports" / "selection_function" / "selection_function_v1.1.parquet"
)

# Grid edges. |b| and G match v1 exactly for compositional ease.
B_EDGES: Final[np.ndarray] = np.array([0.0, 5.0, 10.0, 20.0, 45.0, 90.0], dtype=np.float64)
G_EDGES: Final[np.ndarray] = np.array([2.0, 11.0, 12.5, 14.0, 15.5, 17.65], dtype=np.float64)
# Teff edges: cool giants / mid / warm giants — split roughly ~30/50/20 on Stream 1.
T_EDGES: Final[np.ndarray] = np.array([3000.0, 4400.0, 4900.0, 6500.0], dtype=np.float64)
# log g edges: luminous giants vs lower-RGB/RC — split ~55/45 on Stream 1.
L_EDGES: Final[np.ndarray] = np.array([0.0, 2.5, 5.0], dtype=np.float64)

IR_COLS: Final[tuple[str, ...]] = ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag")

SPARSE_CELL_THRESHOLD_4D: Final[int] = 100  # cells below → marginal fallback
LAPLACE_NUM: Final[float] = 1.0
LAPLACE_DEN: Final[float] = 2.0
PROB_FLOOR: Final[float] = 0.01
PROB_CEIL: Final[float] = 1.0
METHOD_VERSION: Final[str] = "v1-grid-5x5x3x2-laplace"


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    logger.info("wrote %s (%d rows)", path, len(df))


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote %s (%d bytes)", path, len(text))


def _compute_ir_complete_mask(df: pd.DataFrame) -> np.ndarray:
    """IR-complete = all 5 (J, H, K, W1, W2) finite AND non-zero.

    The zero-sentinel is the ``nan_to_num``-substituted "no-counterpart"
    pattern used downstream at inference; on the training feature matrix
    missing counterparts show up as ``NaN`` directly, but we enforce both
    conditions so the definition transfers cleanly to the inference domain.
    """
    mask = np.ones(len(df), dtype=bool)
    for c in IR_COLS:
        v = df[c].to_numpy()
        mask &= np.isfinite(v) & (v != 0.0)
    return mask


def _bin_indices(
    abs_b: np.ndarray, g: np.ndarray, t: np.ndarray, lg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 4-D cell indices for every row. NaN Teff/log g → -1."""
    # |b|, G: clamp to edges — all rows fall inside the grid by construction.
    ab = np.clip(abs_b, B_EDGES[0], np.nextafter(B_EDGES[-1], -np.inf))
    gg = np.clip(g, G_EDGES[0], np.nextafter(G_EDGES[-1], -np.inf))
    ib = np.searchsorted(B_EDGES, ab, side="right") - 1
    ig = np.searchsorted(G_EDGES, gg, side="right") - 1
    ib = np.clip(ib, 0, len(B_EDGES) - 2)
    ig = np.clip(ig, 0, len(G_EDGES) - 2)

    # Teff, log g: NaN-aware. Clamp finite values into [edges[0], edges[-1]].
    finite_t = np.isfinite(t)
    finite_l = np.isfinite(lg)
    it = np.full_like(t, -1.0, dtype=np.int64)
    il = np.full_like(lg, -1.0, dtype=np.int64)
    if finite_t.any():
        tc = np.clip(t[finite_t], T_EDGES[0], np.nextafter(T_EDGES[-1], -np.inf))
        idx_t = np.searchsorted(T_EDGES, tc, side="right") - 1
        idx_t = np.clip(idx_t, 0, len(T_EDGES) - 2)
        it[finite_t] = idx_t
    if finite_l.any():
        lc = np.clip(lg[finite_l], L_EDGES[0], np.nextafter(L_EDGES[-1], -np.inf))
        idx_l = np.searchsorted(L_EDGES, lc, side="right") - 1
        idx_l = np.clip(idx_l, 0, len(L_EDGES) - 2)
        il[finite_l] = idx_l
    return ib, ig, it, il


def build_ir_completeness_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (grid_4d, marginal_bg) tidy DataFrames with Laplace-smoothed P_ir_complete.

    - grid_4d: one row per populated (|b|, G, Teff, log g) cell. Contains
      ``n_total``, ``n_complete``, raw ``rate``, Laplace-smoothed
      ``p_ir_complete``, and a ``dense`` flag (True when n_total ≥ threshold).
    - marginal_bg: one row per (|b|, G) cell — the fallback when a 4-D cell
      is sparse or when Teff/log g are unavailable at inference time.
    """
    abs_b = np.abs(df["b_deg"].to_numpy(dtype=np.float64))
    g = df["g_mag"].to_numpy(dtype=np.float64)
    t = df["teff_apogee"].to_numpy(dtype=np.float64)
    lg = df["logg_apogee"].to_numpy(dtype=np.float64)
    ir_ok = _compute_ir_complete_mask(df)

    ib_arr, ig_arr, it_arr, il_arr = _bin_indices(abs_b, g, t, lg)

    n_b = len(B_EDGES) - 1
    n_g = len(G_EDGES) - 1
    n_t = len(T_EDGES) - 1
    n_l = len(L_EDGES) - 1

    # |b|×G marginal (always dense on Stream 1).
    marg_rows: list[dict[str, float | int]] = []
    for ib in range(n_b):
        for ig in range(n_g):
            mask = (ib_arr == ib) & (ig_arr == ig)
            n_tot = int(mask.sum())
            n_cpl = int((mask & ir_ok).sum())
            rate = float(n_cpl) / n_tot if n_tot > 0 else 0.0
            p_sm = (n_cpl + LAPLACE_NUM) / (n_tot + LAPLACE_DEN) if n_tot > 0 else 0.5
            p_sm = float(np.clip(p_sm, PROB_FLOOR, PROB_CEIL))
            marg_rows.append(
                {
                    "b_lo": float(B_EDGES[ib]),
                    "b_hi": float(B_EDGES[ib + 1]),
                    "g_lo": float(G_EDGES[ig]),
                    "g_hi": float(G_EDGES[ig + 1]),
                    "n_total": n_tot,
                    "n_complete": n_cpl,
                    "rate": rate,
                    "p_ir_complete": p_sm,
                }
            )
    marginal_bg = pd.DataFrame(marg_rows)

    # 4-D grid. Skip fully-empty cells (no rows); but write populated ones
    # whether dense or sparse (Laplace smoothing keeps sparse cells sane).
    grid_rows: list[dict[str, float | int | bool]] = []
    for ib in range(n_b):
        for ig in range(n_g):
            for it in range(n_t):
                for il in range(n_l):
                    mask = (ib_arr == ib) & (ig_arr == ig) & (it_arr == it) & (il_arr == il)
                    n_tot = int(mask.sum())
                    if n_tot == 0:
                        continue
                    n_cpl = int((mask & ir_ok).sum())
                    rate = float(n_cpl) / n_tot
                    p_sm = (n_cpl + LAPLACE_NUM) / (n_tot + LAPLACE_DEN)
                    p_sm = float(np.clip(p_sm, PROB_FLOOR, PROB_CEIL))
                    grid_rows.append(
                        {
                            "b_lo": float(B_EDGES[ib]),
                            "b_hi": float(B_EDGES[ib + 1]),
                            "g_lo": float(G_EDGES[ig]),
                            "g_hi": float(G_EDGES[ig + 1]),
                            "teff_lo": float(T_EDGES[it]),
                            "teff_hi": float(T_EDGES[it + 1]),
                            "logg_lo": float(L_EDGES[il]),
                            "logg_hi": float(L_EDGES[il + 1]),
                            "n_total": n_tot,
                            "n_complete": n_cpl,
                            "rate": rate,
                            "p_ir_complete": p_sm,
                            "dense": bool(n_tot >= SPARSE_CELL_THRESHOLD_4D),
                        }
                    )
    grid_4d = pd.DataFrame(grid_rows)
    logger.info(
        "built 4-D grid: %d populated cells (of %d possible), %d dense (n>=%d), %d sparse",
        len(grid_4d),
        n_b * n_g * n_t * n_l,
        int(grid_4d["dense"].sum()),
        SPARSE_CELL_THRESHOLD_4D,
        int((~grid_4d["dense"]).sum()),
    )
    return grid_4d, marginal_bg


def build_compound_grid(ir_grid: pd.DataFrame, ye_v1_grid: pd.DataFrame) -> pd.DataFrame:
    """Join the Ye-v1 (|b|, G) table with the IR |b|×G marginal to produce the
    compound v1.1 artefact. This table lives alongside the 4-D grid so
    downstream tooling (or external release scripts) can consume the
    compositional |b|×G view in a single read.
    """
    key = ["b_lo", "b_hi", "g_lo", "g_hi"]
    ye = ye_v1_grid[key + ["selection_prob"]].rename(columns={"selection_prob": "p_ye_retained"})
    ir_bg = ir_grid[key + ["n_total", "n_complete", "p_ir_complete"]].rename(
        columns={"n_total": "n_total_bg", "n_complete": "n_complete_bg"}
    )
    out = ye.merge(ir_bg, on=key, how="inner", validate="one_to_one")
    out["p_compound_bg"] = np.clip(
        out["p_ye_retained"].to_numpy() * out["p_ir_complete"].to_numpy(),
        PROB_FLOOR * PROB_FLOOR,  # joint floor = product of the two floors
        PROB_CEIL,
    )
    return out


def _format_ir_md(
    df: pd.DataFrame,
    grid_4d: pd.DataFrame,
    marginal_bg: pd.DataFrame,
    compound: pd.DataFrame,
) -> str:
    """Return the full methodology + results report for v1 IR-completeness."""
    n_total = int(len(df))
    ir_ok = _compute_ir_complete_mask(df)
    n_ir = int(ir_ok.sum())
    global_rate = 100.0 * n_ir / n_total

    worst = grid_4d.loc[grid_4d["p_ir_complete"].idxmin()]
    best = grid_4d.loc[grid_4d["p_ir_complete"].idxmax()]

    lines = [
        "# IR-Completeness Selection-Function Component — v1",
        "",
        f"**Input:** `data/processed/pipeline1_features_stream1.parquet` (N = {n_total:,})",
        "**Artefact:** `reports/selection_function/ir_completeness_v1.parquet` "
        "(4-D grid + |b|×G marginal stacked in two tables; this sidecar lists the 4-D).",
        "**Scorer module:** `src/arqueogal/data/selection_function.py` → "
        "`score_ir_completeness(b_deg, g_mag, teff, logg)`, "
        "`score_compound_selection_prob(...)`.",
        "",
        "---",
        "",
        "## 1. Definition",
        "",
        "A row is **IR-complete** iff all of `j_mag`, `h_mag`, `k_mag`, `w1_mag`, `w2_mag` "
        "are finite and non-zero. The zero-sentinel is excluded because downstream "
        "inference uses `nan_to_num(0.0)` on missing IR rows, so at inference time "
        '`mag == 0` is indistinguishable from "no counterpart"; both conditions are '
        "enforced for definitional transferability from training to inference domains.",
        "",
        f"**Stream 1 global P(IR-complete) = {global_rate:.4f} %** "
        f"(N_complete = {n_ir:,} / N_total = {n_total:,}).",
        "",
        "The IR-dependency diagnostic referenced a ~99.9 % training-domain "
        f"completeness heuristic; the empirical value is {global_rate:.2f} %, in the same "
        "ballpark but slightly lower, driven by faint-in-plane stars.",
        "",
        "## 2. Binning",
        "",
        "Four-dimensional grid, matching the v1 Ye-retention |b|×G axes for compositional ease:",
        "",
        f"- `|b|` (deg): {B_EDGES.tolist()}  → {len(B_EDGES) - 1} bins",
        f"- `G`   (mag): {G_EDGES.tolist()}  → {len(G_EDGES) - 1} bins",
        f"- `Teff` (K):  {T_EDGES.tolist()}  → {len(T_EDGES) - 1} bins (cool / mid / warm giants)",
        f"- `log g`:     {L_EDGES.tolist()}  → {len(L_EDGES) - 1} bins "
        "(luminous giants / lower-RGB+RC)",
        "",
        f"Total possible cells: {(len(B_EDGES) - 1) * (len(G_EDGES) - 1) * (len(T_EDGES) - 1) * (len(L_EDGES) - 1)} "
        f"(5×5×3×2). Populated cells: **{len(grid_4d)}**. "
        f"Dense cells (n ≥ {SPARSE_CELL_THRESHOLD_4D}): "
        f"**{int(grid_4d['dense'].sum())}**. "
        f"Sparse cells (below threshold, scorer backs off to |b|×G marginal at "
        f"runtime): **{int((~grid_4d['dense']).sum())}**.",
        "",
        f"Per-cell Laplace smoothing: "
        f"`p_ir_complete = (n_complete + {LAPLACE_NUM:g}) / (n_total + {LAPLACE_DEN:g})`, "
        f"then clipped to `[{PROB_FLOOR:g}, {PROB_CEIL:g}]`. The Laplace correction "
        "prevents any cell from scoring exactly 0 or 1; the floor/ceil structural.",
        "",
        "## 3. |b|×G marginal (N_complete / N_total per cell)",
        "",
        "The 25-cell |b|×G marginal is the operational fallback when Teff/log g are "
        "unavailable at inference (e.g., before Pipeline 1 predicts them, or for "
        "sparse 4-D cells). It is also the compositional partner of the Ye v1 "
        "retention table.",
        "",
        "| `|b|` (deg) | `G` (mag) | N_total | N_complete | rate | p_ir_complete (Laplace) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in marginal_bg.iterrows():
        lines.append(
            f"| [{r['b_lo']:>4.0f}, {r['b_hi']:>4.0f}) | "
            f"[{r['g_lo']:>5.2f}, {r['g_hi']:>5.2f}) | "
            f"{int(r['n_total']):>6d} | {int(r['n_complete']):>6d} | "
            f"{r['rate']:.4f} | {r['p_ir_complete']:.4f} |"
        )

    lines += [
        "",
        "## 4. Structure beyond |b|×G?",
        "",
        "Across the **4-D grid**, the spread in `p_ir_complete` at constant (|b|, G) — "
        "i.e., the residual variation unlocked by Teff, log g stratification — is "
        "summarised by the worst/best cells:",
        "",
        f"- **Worst** (lowest p_ir_complete): "
        f"|b| ∈ [{worst['b_lo']:.0f}, {worst['b_hi']:.0f}), "
        f"G ∈ [{worst['g_lo']:.2f}, {worst['g_hi']:.2f}), "
        f"Teff ∈ [{worst['teff_lo']:.0f}, {worst['teff_hi']:.0f}), "
        f"log g ∈ [{worst['logg_lo']:.2f}, {worst['logg_hi']:.2f}] "
        f"→ **p = {worst['p_ir_complete']:.4f}** "
        f"(N = {int(worst['n_total'])}, complete = {int(worst['n_complete'])}, "
        f"{'dense' if worst['dense'] else 'sparse — backs off to marginal'}).",
        f"- **Best** (highest p_ir_complete): "
        f"|b| ∈ [{best['b_lo']:.0f}, {best['b_hi']:.0f}), "
        f"G ∈ [{best['g_lo']:.2f}, {best['g_hi']:.2f}), "
        f"Teff ∈ [{best['teff_lo']:.0f}, {best['teff_hi']:.0f}), "
        f"log g ∈ [{best['logg_lo']:.2f}, {best['logg_hi']:.2f}] "
        f"→ **p = {best['p_ir_complete']:.4f}** "
        f"(N = {int(best['n_total'])}).",
        "",
        f"Spread (max − min) across populated dense cells: "
        f"**{grid_4d.loc[grid_4d['dense'], 'p_ir_complete'].max() - grid_4d.loc[grid_4d['dense'], 'p_ir_complete'].min():.4f}** "
        "in probability units. This is load-bearing for the plane-faint corner and "
        "a near-no-op elsewhere — broadly the same structure the Ye-retention "
        "component exhibits.",
        "",
        "## 5. Compound v1.1 |b|×G table (`p_compound = p_ye · p_ir`)",
        "",
        "Joined view of the v1 Ye retention and the v1 IR-completeness marginal. "
        "Used directly by `score_compound_selection_prob` when Teff/log g are "
        "unavailable, and carried in the v1.1 Parquet artefact for release.",
        "",
        "| `|b|` (deg) | `G` (mag) | p_ye_retained | p_ir_complete | p_compound_bg |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, r in compound.iterrows():
        lines.append(
            f"| [{r['b_lo']:>4.0f}, {r['b_hi']:>4.0f}) | "
            f"[{r['g_lo']:>5.2f}, {r['g_hi']:>5.2f}) | "
            f"{r['p_ye_retained']:.4f} | {r['p_ir_complete']:.4f} | "
            f"{r['p_compound_bg']:.4f} |"
        )

    lines += [
        "",
        "## 6. Scoring protocol",
        "",
        "Given `(b_deg, g_mag, teff, logg)`, the scorer takes `|b|`, looks up the 4-D "
        "cell, and returns the smoothed `p_ir_complete`. Fallbacks:",
        "",
        f"- If the 4-D cell is sparse (n < {SPARSE_CELL_THRESHOLD_4D}), return the "
        "|b|×G marginal value.",
        "- If `teff` or `logg` is NaN / missing at call time, return the |b|×G "
        "marginal value directly (no 4-D lookup attempted).",
        "- Out-of-range `|b|`, `G`, `teff`, `logg` are clamped to the nearest edge.",
        "",
        "## 7. Compound probability contract",
        "",
        "```",
        "p_compound = p_ye_retained · p_ir_complete · p_parallax · p_extinction",
        "```",
        "",
        "`p_parallax` and `p_extinction` are simple 0/1 gates in v1.1 "
        "(True → 1.0, False → 0.0). They are placeholders for later work; the "
        "user's requirement is the Ye × IR composition. All four are clamped to "
        f"`[{PROB_FLOOR * PROB_FLOOR:g}, {PROB_CEIL:g}]` — the joint floor is the "
        "product of the two component floors, so inverse weights at the plane-faint "
        "corner remain finite.",
        "",
        "## 8. Build + provenance",
        "",
        "- **Builder:** `scripts/build_selection_function_v11.py`. Deterministic.",
        "- **Inputs:** `data/processed/pipeline1_features_stream1.parquet` "
        "(read-only; SHA-256 in sidecar); v1 Ye retention grid at "
        "`reports/selection_function/selection_function_v1.parquet` (read-only).",
        "- **Atomic writes** on all three outputs (Parquet + MD + provenance JSON).",
        "- **Reproduction:** `PYTHONPATH=src python scripts/build_selection_function_v11.py`.",
        "",
        "## 9. Known limitations",
        "",
        "1. **Parallax + extinction components are 0/1 gates.** v1.2 should replace "
        "them with smooth per-star availability or uncertainty-weighted probabilities.",
        "2. **Stream 1 basis.** The IR-completeness table is computed on the Stream 1 "
        "(APOGEE × Gaia XP) joint selection. Stream 3 at the XP-native faint end may "
        "sample slightly different (|b|, G, Teff, log g) joint distributions; a cross-"
        "check at first Stream 3 ingestion is scheduled (same protocol as v1).",
        "3. **Piecewise-constant inside each 4-D cell.** A star one bin-width inside "
        "a cell gets the same score as one right at the edge. Acceptable at this "
        "stratification depth — smoothing is v1.3 work if the spread ever grows.",
        "4. **Teff bin granularity is coarse (3 bins).** If future data ingestion "
        "produces material 4-D structure inside any of the three Teff bins, refine.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _format_compound_md(
    compound: pd.DataFrame,
    input_sha: str,
    ye_sha: str,
) -> str:
    """Short v1.1 narrative. The full IR methodology lives in its own MD."""
    lines = [
        "# Compound Selection Function — v1.1",
        "",
        "**Change from v1:** per-star selection probability is now a **compound** of "
        "Ye+2024 `NO_SYNTH_PHOT` retention (the v1 artefact, unchanged) and IR "
        "(2MASS/AllWISE) photometric completeness (new in v1.1). v1 remains in place "
        "at `reports/selection_function/selection_function_v1.{md,parquet,provenance.json}` "
        "for historical reference.",
        "",
        "**Compound definition:**",
        "",
        "```",
        "p_compound = p_ye_retained · p_ir_complete · p_parallax · p_extinction",
        "```",
        "",
        "where `p_parallax` and `p_extinction` are 0/1 gates in v1.1 — they take a "
        "data-availability flag (True/False) and map to 1.0/0.0. Smooth per-star "
        "parallax- and extinction-availability probabilities are earmarked for v1.2.",
        "",
        "## Scorer contract",
        "",
        "```python",
        "from arqueogal.data.selection_function import (",
        "    score_selection_prob,             # v1, unchanged",
        "    score_ir_completeness,            # new in v1.1",
        "    score_compound_selection_prob,    # new in v1.1",
        ")",
        "",
        "# Unchanged backwards-compatible API:",
        "p_ye = score_selection_prob(b_deg, g_mag)",
        "",
        "# New in v1.1:",
        "p_ir = score_ir_completeness(b_deg, g_mag, teff, logg)",
        "bundle = score_compound_selection_prob(",
        "    b_deg, g_mag, teff, logg,",
        "    parallax_over_error=pox, av_missing=False,",
        ")",
        "# → {'p_ye_retained', 'p_ir_complete', 'p_compound', 'components': {...}}",
        "```",
        "",
        "## Artefact schema (`selection_function_v1.1.parquet`)",
        "",
        "One row per (|b|, G) cell — 25 rows. This is the **|b|×G marginal** view, "
        "usable directly by consumers whose pipelines do not carry Teff / log g at "
        "the scoring stage. The full 4-D IR-completeness grid lives in its sibling "
        "artefact `reports/selection_function/ir_completeness_v1.parquet`.",
        "",
        "| column | dtype | notes |",
        "|---|---|---|",
        "| `b_lo`, `b_hi` | float64 | |b| bin edges, deg |",
        "| `g_lo`, `g_hi` | float64 | G bin edges, mag |",
        "| `p_ye_retained` | float64 | from v1 (`1 − P(NO_SYNTH_PHOT)`), Laplace-free; floor 0.01 |",
        "| `p_ir_complete` | float64 | new; `(n_c + 1) / (n_t + 2)`; Laplace-smoothed; floor 0.01, ceil 1.0 |",
        "| `n_total_bg`, `n_complete_bg` | int64 | Stream-1 counts in cell |",
        "| `p_compound_bg` | float64 | `p_ye · p_ir`, clamped to [0.0001, 1.0] |",
        "",
        "## Provenance",
        "",
        "- Input Stream-1 Parquet SHA-256: `" + input_sha + "`.",
        "- v1 Ye-retention Parquet SHA-256: `" + ye_sha + "`.",
        "- Git SHA and build timestamp in `selection_function_v1.1.provenance.json`.",
        "",
        "## References",
        "",
        "- Full IR-completeness methodology: `reports/selection_function/ir_completeness_v1.md`.",
        "- v1 Ye-retention methodology: `reports/selection_function/selection_function_v1.md`.",
        "- Pipeline context: `docs/data_acquisition.md` §6.4 and §6.6.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise SystemExit(f"input parquet not found: {INPUT_PARQUET}")
    if not YE_V1_PARQUET.exists():
        raise SystemExit(
            f"v1 Ye-retention artefact not found: {YE_V1_PARQUET}. "
            "Run scripts/build_selection_function_v1.py first."
        )

    logger.info("loading %s", INPUT_PARQUET)
    needed = list(IR_COLS) + ["b_deg", "g_mag", "teff_apogee", "logg_apogee", "ye2024_flag"]
    df = pd.read_parquet(INPUT_PARQUET, columns=needed)
    logger.info("loaded %d rows", len(df))

    ir_ok = _compute_ir_complete_mask(df)
    logger.info(
        "global IR-complete: %d / %d (%.4f %%)",
        int(ir_ok.sum()),
        len(df),
        100.0 * ir_ok.mean(),
    )

    grid_4d, marginal_bg = build_ir_completeness_grid(df)

    # Emit the IR artefact as a single tidy Parquet: 4-D grid rows have
    # teff_lo/teff_hi/logg_lo/logg_hi and 'grid' = '4d'; marginal rows set
    # those to NaN and 'grid' = 'bg'. Downstream code filters by 'grid'.
    ir_4d = grid_4d.copy()
    ir_4d["grid"] = "4d"
    ir_bg = marginal_bg.copy()
    for col in ("teff_lo", "teff_hi", "logg_lo", "logg_hi"):
        ir_bg[col] = np.nan
    ir_bg["rate"] = ir_bg["rate"].astype(float)
    ir_bg["dense"] = True  # |b|×G marginal is always the backstop
    ir_bg["grid"] = "bg"
    ir_all = pd.concat([ir_4d, ir_bg], ignore_index=True)
    _atomic_write_parquet(ir_all, IR_GRID_PARQUET)

    # Compound |b|×G artefact — joins Ye v1 and IR marginal for convenience.
    ye_v1 = pd.read_parquet(YE_V1_PARQUET)
    compound = build_compound_grid(ir_all[ir_all["grid"] == "bg"], ye_v1)
    _atomic_write_parquet(compound, COMPOUND_PARQUET)

    # Report MDs.
    ir_md = _format_ir_md(df, grid_4d, marginal_bg, compound)
    _atomic_write_text(ir_md, IR_GRID_PARQUET.with_suffix(".md"))

    input_sha = _sha256_of(INPUT_PARQUET)
    ye_sha = _sha256_of(YE_V1_PARQUET)
    compound_md = _format_compound_md(compound, input_sha, ye_sha)
    _atomic_write_text(compound_md, COMPOUND_PARQUET.with_suffix(".md"))

    # Provenance — IR artefact.
    n_populated = int(len(grid_4d))
    n_dense = int(grid_4d["dense"].sum())
    n_sparse = int((~grid_4d["dense"]).sum())
    ir_prov = Provenance(
        output_file=str(IR_GRID_PARQUET.relative_to(REPO)),
        script="scripts/build_selection_function_v11.py",
        sources=[
            LocalSource(
                name="Stream 1 Pipeline 1 features",
                path=str(INPUT_PARQUET.relative_to(REPO)),
                sha256=input_sha,
            ),
        ],
        cuts_applied=[],
        corrections=[],
        row_count_before=int(len(df)),
        row_count_after=int(ir_all["n_total"].sum()),
        notes=(
            "IR-completeness grid on (|b|, G, Teff, log g). "
            f"Grid: 5x5x3x2, b={B_EDGES.tolist()}, g={G_EDGES.tolist()}, "
            f"teff={T_EDGES.tolist()}, logg={L_EDGES.tolist()}. "
            f"Populated 4-D cells: {n_populated}; dense: {n_dense}; sparse: {n_sparse}. "
            f"Laplace smoothing (+{LAPLACE_NUM:g}/+{LAPLACE_DEN:g}); "
            f"floor {PROB_FLOOR:g}, ceil {PROB_CEIL:g}. "
            "Sparse-cell fallback: |b|×G marginal (stored in same artefact, grid=='bg')."
        ),
        extra={
            "method_version": METHOD_VERSION,
            "b_edges_deg": B_EDGES.tolist(),
            "g_edges_mag": G_EDGES.tolist(),
            "teff_edges_K": T_EDGES.tolist(),
            "logg_edges_dex": L_EDGES.tolist(),
            "laplace_num": LAPLACE_NUM,
            "laplace_den": LAPLACE_DEN,
            "prob_floor": PROB_FLOOR,
            "prob_ceil": PROB_CEIL,
            "n_populated_4d_cells": n_populated,
            "n_dense_4d_cells": n_dense,
            "n_sparse_4d_cells": n_sparse,
            "sparse_cell_threshold_4d": SPARSE_CELL_THRESHOLD_4D,
            "global_ir_complete_rate": float(ir_ok.mean()),
            "ir_cols": list(IR_COLS),
        },
    )
    write_sidecar(ir_prov)

    # Provenance — compound v1.1 artefact.
    compound_prov = Provenance(
        output_file=str(COMPOUND_PARQUET.relative_to(REPO)),
        script="scripts/build_selection_function_v11.py",
        sources=[
            LocalSource(
                name="Stream 1 Pipeline 1 features",
                path=str(INPUT_PARQUET.relative_to(REPO)),
                sha256=input_sha,
            ),
            LocalSource(
                name="Ye+2024 NO_SYNTH_PHOT retention v1 grid",
                path=str(YE_V1_PARQUET.relative_to(REPO)),
                sha256=ye_sha,
            ),
        ],
        cuts_applied=[],
        corrections=[],
        row_count_before=None,
        row_count_after=int(len(compound)),
        notes=(
            "Compound selection function v1.1: p_compound = p_ye · p_ir · p_parallax "
            "· p_extinction. p_ye from v1 (unchanged). p_ir from IR-completeness "
            "marginal on |b|×G. p_parallax and p_extinction are 0/1 gates. "
            "This Parquet stores the |b|×G marginal view (25 rows); the full 4-D "
            "IR grid lives in ir_completeness_v1.parquet."
        ),
        extra={
            "method_version": "v1.1-compound-bxg-marginal",
            "prob_floor": PROB_FLOOR,
            "prob_ceil": PROB_CEIL,
            "compound_floor": PROB_FLOOR * PROB_FLOOR,
        },
    )
    write_sidecar(compound_prov)

    logger.info("done.")


if __name__ == "__main__":
    main()
