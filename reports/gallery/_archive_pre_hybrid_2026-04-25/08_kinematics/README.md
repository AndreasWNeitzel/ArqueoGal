# 08 — Kinematics

**What this shows.** Orbital actions (J_R, J_z, L_z), orbital energy E, and
eccentricity ε from galpy/agama with `MWPotential14`. Used as downstream
features by Starfold (population classification, separate repo); not
consumed by Pipeline 1.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | e_lz_plane.png | Gaia-Enceladus plane; substructure = Sausage signature. | batch 3 |
| 02 | action_diagram.png | J_R vs J_z (log-log); disc sits at bottom-left, halo/thick-disc upper-right. | batch 3 |
| 03 | ecc_lz.png | Eccentricity vs L_z; retrograde ε-peak = accreted candidates. | batch 3 |
| 04 | orbit_families_fraction.png | Disc / halo / retrograde fractions vs [M/H] bin. | batch 3 |

## Failure modes
- If E is quantised at discrete values, the integrator converged onto closed
  orbits — fine, but the assumed potential is approximate; results depend on
  the choice of `MWPotential14`.
- L_z > 3000 kpc km/s stars are likely hyper-velocity; check the `ecc=1` tail.
