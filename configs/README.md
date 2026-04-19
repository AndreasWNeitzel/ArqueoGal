# Configuration Files

YAML configs for reproducible experiments. One file per model variant or training run.

## Layout

```
configs/
├── main/            # configs for main-pipeline deliverables (frozen during sprints)
│   ├── xp_abundances_baseline.yaml
│   ├── population_classifier_fire2.yaml   # Subtask 5.1 hare-and-hounds
│   └── population_classifier_real.yaml    # Subtask 5.2 D-Cat-d
└── experimental/    # configs for experimental-arm exploration (pre-promotion)
    └── xp_abundances_extended_labels.yaml
```

Main/experimental separation mirrors `src/arqueogal/*/main|experimental/`. Cross-config
references (a main config `includes:` an experimental one) are not allowed.

## Convention

- One config per experiment or model variant. Name files `<pipeline>_<variant>.yaml`.
- Store random seeds, hyperparameter grids, and feature lists here — NOT in code.
- Paths are relative to the repo root or resolved via `$ARQUEOGAL_DATA`.
- Every training run records the exact config into its checkpoint
  (see `arqueogal.xp_abundances.main` checkpoint schema).
