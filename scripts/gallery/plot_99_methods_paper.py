"""Stage 99: methods-paper curated subset.

Selects ~10 flagship figures across stages 01–14 of the Pipeline-1
methodology, re-emits each at publication DPI (300) in both PDF and PNG,
and writes a manifest that links figure → methods-paper section → source
script/function. The population-classification stages (15/16) are
Starfold's concern and are not rendered here; see
``reports/gallery/archive/`` for the historical renders.

Strategy. We monkey-patch `_common.save_fig` so that, while a selected
upstream plot function runs, its `save_fig` call:

  1. Redirects the output path to `reports/gallery/99_methods_paper/`.
  2. Emits both .pdf and .png.
  3. Uses 300 DPI (paper spec).

This keeps the flagship rendering identical to the stage versions — we
do not duplicate the plotting logic. If the stage version is ever
updated, re-running batch 7 inherits the change.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from _common import GALLERY, ROOT, apply_style  # noqa: E402

OUT = GALLERY / "99_methods_paper"
PAPER_DPI = 300


# -----------------------------------------------------------------------------
# Flagship registry: the figures that belong in the methods paper.
#
# Each entry is one call into an existing stage plot_XX module. `figure_name`
# is the stem that both the .pdf and .png will use in OUT/. `section` is the
# anchor in docs/plan/06_methods_paper.md. `caption` is the paper-ready
# caption (authored here, not inherited from stage READMEs).
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Flagship:
    figure_name: str
    source_script: str
    source_function: str
    section: str
    caption: str


FLAGSHIPS: list[Flagship] = [
    Flagship(
        figure_name="01_xp_sed_atlas_by_hrd",
        source_script="plot_01_gaia_xp_raw",
        source_function="sed_atlas_by_hrd",
        section="§2 Data",
        caption=(
            r"Median Gaia DR3 XP reconstructed spectral energy distributions on a "
            r"$T_{\rm eff}$–$\log g$ grid (Streams 1 + 3). The stacked SEDs give a "
            "single-figure picture of the input representation and the HRD coverage "
            "the Pipeline-1 encoder must span."
        ),
    ),
    Flagship(
        figure_name="02_ye_before_after_sed",
        source_script="plot_02_ye_correction",
        source_function="before_after_sed",
        section="§3.1 Preprocessing — Ye+2024 flux correction",
        caption=(
            "Effect of the Ye+2024 neural-network flux correction on a set of "
            "representative reconstructed XP spectra (raw vs. corrected). "
            "Correction magnitude is largest at the blue end and at bright G, "
            "consistent with the flux-zero-point drifts targeted by the method."
        ),
    ),
    Flagship(
        figure_name="03_hermite_zscore_per_coef",
        source_script="plot_03_hermite_reprojection",
        source_function="hermite_zscore_per_coef",
        section="Finding #4 — per-coefficient z-scoring of Hermite coefficients",
        caption=(
            "Per-coefficient distributions of the 110 normalised Hermite "
            r"coefficients ($c_k/c_0$, $k=1\ldots54$ for BP and RP) before (grey) "
            "and after (red) per-coefficient z-scoring. The dominant per-cell mean "
            r"bias on $T_{\rm eff}$ and $[{\rm M}/{\rm H}]$ is absorbed at this "
            "step (−78\\% / −79\\% bias reduction in validation). Published XP "
            "pipelines typically normalise but do not per-coefficient z-score."
        ),
    ),
    Flagship(
        figure_name="04_av_map_stack_stream3",
        source_script="plot_04_extinction",
        source_function="av_map_stack",
        section="§2.3 Extinction stack (Stream 3)",
        caption=(
            r"$A_V$ per Stream-3 star on the sky (Galactic Mollweide) from four "
            "independent sources probing different LOS depths: Edenhofer+2024 (3D, "
            r"truncates at $d=1.25$ kpc; 0\% coverage on Stream-3 RGB by "
            r"construction), Lallement+2022 (3D, $1.25$–$3$ kpc), SFD 1998 (2D "
            r"asymptotic to $\infty$), and GSP-Phot neighbourhood median. The four "
            "maps are not directly comparable — depth differences, not calibration "
            "error, drive the amplitude offsets."
        ),
    ),
    Flagship(
        figure_name="06_omega_total_sky",
        source_script="plot_06_selection_function",
        source_function="omega_total_sky",
        section="§2.4 Selection function",
        caption=(
            r"Stream-3 per-star selection probability $\omega = \omega_{\rm Ye} "
            r"\cdot \omega_{\rm IR} \cdot \omega_\varpi \cdot \omega_{A_V}$ on the "
            "sky (Galactic Mollweide). Used as a per-star weight in downstream "
            "density estimates and as a coverage diagnostic; does not enter "
            "training as a sample weight."
        ),
    ),
    Flagship(
        figure_name="09_feature_correlation_heatmap",
        source_script="plot_09_feature_matrix",
        source_function="feature_correlation_heatmap",
        section="§3.3 Feature matrix",
        caption=(
            "Pearson correlation matrix over the full 153-D Pipeline-1 input (108 "
            r"normalised XP coefs + 2 coef-0 channels + 3 residuals + 40 aux). "
            r"Block structure around BP vs. RP and across the aux block motivates "
            "the PCA-summary estimator used in the §9.2 CMI audit and argues "
            "against raw 2-D KSG CMI (Finding #8)."
        ),
    ),
    Flagship(
        figure_name="12_residual_by_teff_logg",
        source_script="plot_12_pipeline1_validation",
        source_function="residual_by_teff_logg",
        section="Finding #3 — per-cell bias under β-NLL",
        caption=(
            "Per-cell residual (pred − truth) on the Kiel diagram, per label. "
            "Exposes the per-cell mean bias that motivated the pivot from "
            r"$\beta{=}0.5$ to $\beta{=}0$ in the block-Cholesky NLL loss: at "
            r"$\beta{=}0.5$ the bias was absorbed into inflated $\sigma$; at "
            r"$\beta{=}0$ it is visible as an explicit mean error and becomes "
            "the diagnostic target of the per-coefficient z-scoring fix (Finding #4)."
        ),
    ),
    Flagship(
        figure_name="13_aleatoric_vs_epistemic_per_label",
        source_script="plot_13_ensemble_uncertainty",
        source_function="aleatoric_vs_epistemic_per_label",
        section="§4 Uncertainty decomposition",
        caption=(
            r"Aleatoric ($\sigma$) vs. epistemic ($\sigma_{\rm epi}$) uncertainty "
            "per label on validation (log–log). Below the $y{=}x$ line means "
            "aleatoric-dominated (label-noise-limited); above means "
            "epistemic-dominated (model-limited). The 5-label head sits well "
            "below $y{=}x$ for all labels — consistent with the Finding #5 "
            "argument that collapsing $21\\to 5$ labels improves joint-tail "
            "coverage at no per-label marginal cost."
        ),
    ),
    Flagship(
        figure_name="13_regime_b_envelope_footprint",
        source_script="plot_13_ensemble_uncertainty",
        source_function="regime_b_envelope_footprint",
        section="§5 Release scope — Regime B exclusion",
        caption=(
            r"Regime-B envelope ($T_{\rm eff}>4750$ K, $\log g<2.1$, $|b|<5°$) on "
            r"the Stream-1 Kiel diagram (grey = full Stream 1; red = $|b|<5°$ "
            "subsample). Stars falling in this envelope show a systematic $\\sim "
            "1\\sigma$ $T_{\\rm eff}$ over-prediction on validation and are "
            "excluded from the per-star Tier-1 release; the direction-of-bias "
            "puzzle is unresolved and is methods-paper material, not a "
            "release blocker."
        ),
    ),
    Flagship(
        figure_name="14_stream3_pred_chemistry",
        source_script="plot_14_pipeline1_inference",
        source_function="stream3_pred_chemistry",
        section="§6 Stream-3 catalogue preview",
        caption=(
            r"Predicted Stream-3 chemistry plane ($[\alpha/{\rm M}]$ vs. "
            r"$[{\rm M}/{\rm H}]$) under Pipeline-1 v1 (left) and v1.1 "
            r"(right, inverse-frequency $[{\rm M}/{\rm H}]$ weighting in the "
            "supervised loss). v1.1 recovers the metal-poor / $\\alpha$-enhanced "
            "tail that v1 prior-collapsed onto the disc ridge, without altering "
            "the solar-metallicity locus."
        ),
    ),
    # Stages 15–16 (population-classifier σ-gate and HDBSCAN chemical-plane
    # evolution) were removed on 2026-04-22 when population classification
    # moved to the Starfold repository. Historical renders live under
    # reports/gallery/archive/.
]


# -----------------------------------------------------------------------------
# Monkey-patch infrastructure
# -----------------------------------------------------------------------------

_ORIG_SAVE_FIG = _common.save_fig
_EMITTED: dict[str, list[str]] = {}


def _paper_save_fig_factory(figure_name: str):
    """Return a save_fig replacement that writes to OUT/{figure_name}.{pdf,png}.

    The upstream caller passes its own path; we ignore its stem and use our
    `figure_name` instead. We still respect the `tight` kwarg.
    """

    def paper_save_fig(fig, path: Path, *, tight: bool = True) -> None:
        mpl.rcParams["text.usetex"] = False
        OUT.mkdir(parents=True, exist_ok=True)
        if tight:
            fig.tight_layout()
        png = OUT / f"{figure_name}.png"
        pdf = OUT / f"{figure_name}.pdf"
        fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=PAPER_DPI)
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        _EMITTED.setdefault(figure_name, []).extend(
            [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]
        )
        print(f"[paper] wrote {png.relative_to(ROOT)}  +  {pdf.relative_to(ROOT)}")

    return paper_save_fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _prepare_args(mod, spec: Flagship) -> tuple:
    """A couple of stage-01 plot fns take a pre-loaded data dict. Load it here."""
    if spec.source_script == "plot_01_gaia_xp_raw" and spec.source_function in (
        "sed_atlas_by_hrd",
        "example_stars",
    ):
        data = mod._load_xp_subset(n_target=10_000)
        return (data,)
    return ()


def _call_flagship(spec: Flagship) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(spec.source_script)
    except Exception as exc:  # noqa: BLE001
        return False, f"import {spec.source_script}: {type(exc).__name__}: {exc}"
    fn = getattr(mod, spec.source_function, None)
    if fn is None:
        return False, f"{spec.source_script}.{spec.source_function} not found"

    # Patch BOTH the _common module (some plot fns import it directly) AND
    # the target module (which did `from _common import save_fig`).
    patched_save = _paper_save_fig_factory(spec.figure_name)
    orig_common = _common.save_fig
    orig_mod = getattr(mod, "save_fig", None)
    _common.save_fig = patched_save
    if orig_mod is not None:
        setattr(mod, "save_fig", patched_save)
    try:
        args = _prepare_args(mod, spec)
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        _common.save_fig = orig_common
        if orig_mod is not None:
            setattr(mod, "save_fig", orig_mod)
    return True, "ok"


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in FLAGSHIPS:
        t0 = time.time()
        ok, msg = _call_flagship(spec)
        dt = time.time() - t0
        tag = "[ok]" if ok else "[FAIL]"
        print(f"{tag}  {spec.figure_name}  ({dt:.1f}s)  {msg}")
        results.append((spec, ok, msg, dt))

    # Emit the manifest
    manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dpi_png": PAPER_DPI,
        "methods_paper_plan": "docs/plan/06_methods_paper.md",
        "figures": [
            {
                "figure_name": spec.figure_name,
                "source_script": f"scripts/gallery/{spec.source_script}.py",
                "source_function": spec.source_function,
                "section": spec.section,
                "caption": spec.caption,
                "outputs": _EMITTED.get(spec.figure_name, []),
                "ok": ok,
                "message": msg,
            }
            for (spec, ok, msg, _dt) in results
        ],
    }
    manifest_path = OUT / "paper_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[paper] wrote {manifest_path.relative_to(ROOT)}")

    n_ok = sum(1 for (_s, ok, _m, _d) in results if ok)
    print(f"\n=== batch 7 summary: {n_ok}/{len(results)} flagships emitted ===")


if __name__ == "__main__":
    main()
