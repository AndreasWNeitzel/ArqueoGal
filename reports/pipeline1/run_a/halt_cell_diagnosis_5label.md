# Halt-cell diagnosis — 5-label z-scored main pipeline
Source ckpt: `models/main/xp_abundances/20260419_nogit_859afab_finetune_5label/xp_abundances_main_finetune_5label_seed0_best.pt`  
Val stars total: **41851**  
Halt cells (err > 30 %): `[4, 15, 28, 34, 49]`  
Over 15 % (warning): `[39, 59]`  

Axis edges (quantile-binned on val):

- Teff:  [4441.489990234375, 4657.88818359375, 4820.589111328125]  → 4 bins
- log g: [2.051452398300171, 2.4012203216552734, 2.6347821950912476]  → 4 bins
- [M/H]: [-0.42502500116825104, -0.22641000151634216, -0.031748004257678986]  → 4 bins

Grid is 4×4×4 = 64 nominal cells.

## Per-cell breakdown (halt + warn)

| cell | severity | Teff bin | log g bin | [M/H] bin | n | edge? | per-label Var(z) (Te/lg/M/αM/MgH) | per-label E[z] (same) | median \|b\| | median A_V | OOD frac | metal-poor frac |
|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| 4 | HALT | Teff < 4441.490 | 2.051 ≤ log g < 2.401 | [M/H] < -0.425 | 56 | edge | +0.60 / +2.58 / +1.68 / +1.26 / +2.65 | -0.46 / +0.62 / -0.53 / +0.04 / -0.56 | 23.3° | 0.19 | 0.00 | 0.05 |
| 15 | HALT | Teff < 4441.490 | log g > 2.635 | [M/H] > -0.032 | 29 | edge | +2.02 / +2.19 / +0.59 / +1.65 / +0.93 | -0.36 / +1.13 / +0.47 / -0.07 / +0.36 | 14.2° | 0.12 | 0.00 | 0.00 |
| 28 | HALT | 4441.490 ≤ Teff < 4657.888 | log g > 2.635 | [M/H] < -0.425 | 33 | edge | +0.89 / +3.50 / +2.93 / +1.81 / +5.29 | -0.59 / +2.20 / -1.04 / -0.28 / -1.23 | 35.4° | 0.09 | 0.00 | 0.00 |
| 34 | HALT | 4657.888 ≤ Teff < 4820.589 | log g < 2.051 | -0.226 ≤ [M/H] < -0.032 | 15 | edge | +0.65 / +0.22 / +0.15 / +0.25 / +0.20 | +1.00 / -0.01 / +0.39 / -0.41 / +0.31 | 1.6° | 0.38 | 0.00 | 0.00 |
| 49 | HALT | Teff > 4820.589 | log g < 2.051 | -0.425 ≤ [M/H] < -0.226 | 15 | edge | +1.09 / +0.47 / +0.34 / +0.13 / +0.30 | +1.41 / -0.14 / +0.74 / -0.92 / +0.44 | 2.5° | nan | 0.00 | 0.00 |
| 39 | warn | 4657.888 ≤ Teff < 4820.589 | 2.051 ≤ log g < 2.401 | [M/H] > -0.032 | 122 | edge | +0.69 / +0.49 / +0.44 / +0.37 / +0.68 | +0.22 / -0.64 / +0.34 / -0.23 / +0.25 | 12.0° | 0.15 | 0.00 | 0.00 |
| 59 | warn | Teff > 4820.589 | 2.401 ≤ log g < 2.635 | [M/H] > -0.032 | 82 | edge | +0.50 / +0.27 / +0.41 / +0.26 / +0.70 | +0.82 / +0.39 / +1.03 / -0.59 / +0.99 | 5.3° | 0.23 | 0.00 | 0.00 |

## Summary

- Halt cells at grid edges: **5 / 5**
- Halt cells with n < 50 val stars: **4 / 5**
- Total val stars in halt cells: **148 / 41851** (0.35 %)
