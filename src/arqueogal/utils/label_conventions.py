"""
Canonical LaTeX label conventions for ArqueoGal methods-paper figures.

All labels use raw strings (r'...') and follow astronomy convention:
italic for variables (Teff, logg), Roman for unit and abbreviation labels,
square brackets for abundance ratios.

Usage:
    from arqueogal.utils.label_conventions import LABELS, TEFF, MG_FE
    ax.set_xlabel(LABELS['Teff'])  # or ax.set_xlabel(TEFF)
"""

# Atmospheric parameters
TEFF = r'$T_{\rm eff}$ [K]'
LOGG = r'$\log g$'
TEFF_RES = r'$\Delta T_{\rm eff}$ [K]'
LOGG_RES = r'$\Delta \log g$'
TEFF_SIGMA = r'$\sigma_{T_{\rm eff}}$ [K]'
LOGG_SIGMA = r'$\sigma_{\log g}$'

# Metallicity and chemistry
M_H = r'$[\mathrm{M/H}]$'
FE_H = r'$[\mathrm{Fe/H}]$'
MG_H = r'$[\mathrm{Mg/H}]$'
ALPHA_M = r'$[\alpha/\mathrm{M}]$'
ALPHA_FE = r'$[\alpha/\mathrm{Fe}]$'
MG_FE = r'$[\mathrm{Mg/Fe}]$'
M_H_RES = r'$\Delta[\mathrm{M/H}]$'
FE_H_RES = r'$\Delta[\mathrm{Fe/H}]$'
MG_H_RES = r'$\Delta[\mathrm{Mg/H}]$'
ALPHA_M_RES = r'$\Delta[\alpha/\mathrm{M}]$'
M_H_SIGMA = r'$\sigma_{[\mathrm{M/H}]}$'
ALPHA_M_SIGMA = r'$\sigma_{[\alpha/\mathrm{M}]}$'

# Magnitudes and photometry
G_MAG = r'$G$ [mag]'
BP_RP = r'$G_{\rm BP} - G_{\rm RP}$ [mag]'
G_BP = r'$G_{\rm BP}$ [mag]'
G_RP = r'$G_{\rm RP}$ [mag]'
J_MAG = r'$J$ [mag]'
H_MAG = r'$H$ [mag]'
K_MAG = r'$K_s$ [mag]'
W1_MAG = r'$W_1$ [mag]'
W2_MAG = r'$W_2$ [mag]'

# Astrometry and kinematics
PARALLAX = r'$\varpi$ [mas]'
PMRA = r'$\mu_{\alpha^*}$ [mas/yr]'
PMDEC = r'$\mu_\delta$ [mas/yr]'
DISTANCE = r'$d$ [kpc]'
V_PHI = r'$v_\phi$ [km/s]'
V_R = r'$v_R$ [km/s]'
V_Z = r'$v_z$ [km/s]'
R_GAL = r'$R_{\rm gal}$ [kpc]'
Z_GAL = r'$z$ [kpc]'
J_R = r'$J_R$'
L_Z = r'$L_z$'
J_Z = r'$J_z$'

# Asteroseismology
NU_MAX = r'$\nu_{\rm max}$ [$\mu$Hz]'
DELTA_NU = r'$\Delta\nu$ [$\mu$Hz]'

# Extinction
A_V = r'$A_V$ [mag]'
E_BV = r'$E(B-V)$ [mag]'

# Galactic latitude and longitude
GAL_LAT = r'$b$ [deg]'
GAL_LON = r'$\ell$ [deg]'

# Convenience dictionary
LABELS = {
    'Teff': TEFF,
    'logg': LOGG,
    'M/H': M_H,
    'Fe/H': FE_H,
    'Mg/H': MG_H,
    'alpha/M': ALPHA_M,
    'alpha/Fe': ALPHA_FE,
    'Mg/Fe': MG_FE,
    'G': G_MAG,
    'BP-RP': BP_RP,
    'parallax': PARALLAX,
    'distance': DISTANCE,
    'numax': NU_MAX,
    'Av': A_V,
    'b': GAL_LAT,
    'l': GAL_LON,
}
