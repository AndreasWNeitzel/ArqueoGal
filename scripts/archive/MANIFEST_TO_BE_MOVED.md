# Archive Manifest — Scripts for Archival Review

All files listed below match the diagnostic-script patterns (`diagnose_*.py`, `analyze_*.py`, `compare_*.py`) and were developed during methodological work on Pipeline 1 v1 (April 2026). **DO NOT EXECUTE MOVES — this manifest is for review only.** Andreas will approve archival after reading this table.

| Filename | Last Modified | Size (bytes) | Recommended Action | Notes |
|---|---|---|---|---|
| `analyze_hermite_pre_emit.py` | 2026-04-24 18:32 | 20,423 | move-to-archive | Exploratory Hermite preprocessing analysis; not part of v1 release pipeline |
| `compare_stream3_alpha_m_bias.py` | 2026-04-24 18:32 | 5,869 | move-to-archive | Stream 3 alpha/M bias comparison; methodological exploration |
| `diagnose_alpha_m_by_mh_bin.py` | 2026-04-24 18:32 | 6,991 | move-to-archive | Alpha/M diagnostic binned by [M/H]; alpha/M-related v1 variant |
| `diagnose_alpha_m_v2.py` | 2026-04-24 18:32 | 7,441 | move-to-archive | Alpha/M diagnostic v2; superseded or exploratory iteration |
| `diagnose_bias_location.py` | 2026-04-24 18:32 | 12,994 | move-to-archive | Regime B Teff bias spatial analysis; diagnostic only, no release role |
| `diagnose_halt_cells.py` | 2026-04-24 18:32 | 13,013 | move-to-archive | Cell halt-condition diagnostics; validation-phase work |
| `diagnose_per_label_calibration.py` | 2026-04-24 18:32 | 9,664 | move-to-archive | Per-label calibration diagnostics; methodological validation |

**Summary:** 7 scripts identified for archival. All are diagnostic or exploratory in nature. No duplicates by exact name, but `diagnose_alpha_m_by_mh_bin.py` and `diagnose_alpha_m_v2.py` are both alpha/M diagnostic variants; tie-break by deletion of `v2` if one must be chosen (but both may be retained for provenance).

**Action:** Review this manifest, confirm file retention policy, then resubmit explicit move command (e.g., `git mv scripts/analyze_hermite_pre_emit.py scripts/archive/analyze_hermite_pre_emit.py`). All moves should be in a single commit with message "chore: archive diagnostic scripts to scripts/archive/".
