"""Filter the Phase 3a delta Ye-corrected XP parquet to Ye-OK source_ids.

Mirrors the existing-Stream 3 ad-hoc builder used for
``data/interim/stream3_ye_ok_source_ids.parquet`` (see its sidecar). The
delta sidecar chain is independent from the existing-Stream 3 chain, so
the existing artefact is not touched.

Usage
-----
    python scripts/build_stream3_delta_ye_ok_ids.py

Inputs
------
- ``data/interim/stream3_expansion_delta_source_ids.parquet`` — the
  Phase 3a delta source_id list (full, 454 184 rows expected).
- ``data/interim/xp_sampled_corrected_delta.parquet`` — Ye+2024 flag
  column for the same delta, produced by ``apply_ye2024_xp_delta.py``.

Output
------
- ``data/interim/stream3_delta_ye_ok_source_ids.parquet`` — single
  ``source_id`` column, sorted, ye2024_flag == 0 only. Suitable as the
  ``--source-id-parquet`` input for ``scripts/fetch_ir_photometry.py``.
- Sidecar ``stream3_delta_ye_ok_source_ids.provenance.json``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
DELTA_IDS = REPO_ROOT / "data/interim/stream3_expansion_delta_source_ids.parquet"
YE_DELTA = REPO_ROOT / "data/interim/xp_sampled_corrected_delta.parquet"
OUTPUT = REPO_ROOT / "data/interim/stream3_delta_ye_ok_source_ids.parquet"

YE2024_FLAG_OK = 0  # local constant; identical to arqueogal.data.gaia_xp.YE2024_FLAG_OK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("build_stream3_delta_ye_ok_ids")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    log.info("reading delta source_id list: %s", DELTA_IDS)
    delta_ids = pd.read_parquet(DELTA_IDS, columns=["source_id"])
    delta_ids["source_id"] = delta_ids["source_id"].astype("int64")
    log.info("  %d delta source_ids", len(delta_ids))

    log.info("reading Ye-corrected delta flag column: %s", YE_DELTA)
    ye = pd.read_parquet(YE_DELTA, columns=["source_id", "ye2024_flag"])
    ye["source_id"] = ye["source_id"].astype("int64")
    log.info("  %d Ye-corrected rows", len(ye))

    row_before = len(delta_ids)
    merged = delta_ids.merge(ye, on="source_id", how="inner")
    if len(merged) != row_before:
        log.warning(
            "join dropped %d source_ids (delta %d -> %d); expected full coverage",
            row_before - len(merged), row_before, len(merged),
        )

    ok = merged[merged["ye2024_flag"] == YE2024_FLAG_OK][["source_id"]]
    ok = ok.sort_values("source_id").reset_index(drop=True)
    log.info("  Ye-OK: %d / %d (%.2f%%)", len(ok), len(merged),
             100.0 * len(ok) / max(len(merged), 1))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    ok.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(OUTPUT)
    out_sha = _sha256(OUTPUT)
    log.info("wrote %s (%d rows, sha256=%s…)",
             OUTPUT, len(ok), out_sha[:12])

    prov = Provenance(
        output_file=str(OUTPUT.relative_to(REPO_ROOT)),
        script="scripts/build_stream3_delta_ye_ok_ids.py",
        sources=[
            LocalSource(
                name="Phase 3a delta source_ids (union: uniform + volume-limited)",
                path=str(DELTA_IDS.relative_to(REPO_ROOT)),
                sha256=_sha256(DELTA_IDS),
            ),
            LocalSource(
                name="Ye+2024-corrected delta XP (flag column)",
                path=str(YE_DELTA.relative_to(REPO_ROOT)),
                sha256=_sha256(YE_DELTA),
            ),
        ],
        cuts_applied=["ye2024_flag == YE2024_FLAG_OK (== 0)"],
        corrections=[],
        row_count_before=len(merged),
        row_count_after=len(ok),
        random_seed=None,
        notes=(
            "Delta-only equivalent of stream3_ye_ok_source_ids.parquet. "
            "Used as the input list for scripts/fetch_ir_photometry.py on "
            "the Phase 3a delta; mirrors the existing-Stream 3 filter but "
            "keeps the artefact chain disjoint so the existing 164 314-row "
            "IR file is not touched."
        ),
        extra={
            "ye2024_flag_ok_value": YE2024_FLAG_OK,
            "output_sha256": out_sha,
        },
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
