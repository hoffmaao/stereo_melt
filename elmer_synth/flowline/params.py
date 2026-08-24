# All model/numerical parameters are set here.
# Forked from vendor/linear-shelf-melt/nonlinear-model/params.py (E1b deltas:
# through-flow u0, ocean-fixed melt anomaly at xc, longer run, save cadence,
# variable-prefactor hook bfac).

import os

# Model parameters
n = float(os.environ.get("E1B_GLEN_N", "4.0"))  # 1.0 = Newtonian (eta = eta0 everywhere)
# A0 units are Pa^-n s^-1, so a fixed number is only meaningful at ONE n. The
# hardcoded n=4 default gives eta_eff ~ 5.8e14 at the E1b background strain
# rate (0.012 /yr); for other n set E1B_A0 explicitly — e.g. n=3 wants
# A0 = edot^(1-n)/(2*eta_target)^n = 8.60e-25 for eta_eff = 1e14 at background.
A0 = float(os.environ.get("E1B_A0", str(1e-32 if n != 1.0 else 5.0e-15)))


B0 = A0**(-1/n)                    # Ice hardness (Pa s^{1/n})
B = (2**((n-1.0)/(2*n)))*B0        # "2*Viscosity" constant in weak form (Pa s^{1/n})
rm2 = 1 + 1.0/n - 2.0              # Exponent in weak form: r-2

rho_i = 917.0                      # Density of ice
rho_w = 1020.0                     # Density of water
g = 9.81                           # Gravitational acceleration
eta0 = 1e14                        # viscosity at zero deviatoric stress

L = 40*1000.0                      # Length of the domain
H = 500.0                          # Height of the domain
sea_level = H*(rho_i/rho_w)        # Sea level elevation.
z_max = 0.9*H                      # Maximum channel height
t_r = 2*eta0/(rho_i*g*H)           # viscous relaxation time scale

# Numerical parameters
# rm2 = 0 (n=1) makes the regularization exponent moot: (s+eps_v)^0 = 1.
# eps_v pins eta(0) = eta_cap. With the default cap (= eta0) an A0 calibrated
# to eta_eff = eta0 AT background puts the background right at the knee, so the
# medium behaves half-Newtonian there — a real Glen experiment must raise
# E1B_ETA_CAP well above the background secant viscosity (e.g. 1e16).
eta_cap = float(os.environ.get("E1B_ETA_CAP", str(eta0)))
eps_v = (2*eta_cap/B)**(2.0/rm2) if rm2 != 0.0 else 1e-30

# Mesh parameters
Nx = int(L/100)                    # Number of elements in x direction
Nz = int(H/100)                    # Number of elements in z direction

# Time-stepping parameters (env-overridable for smoke tests)
yr = 3.154e7
t_f = float(os.environ.get("E1B_TF_YR", "200"))*yr   # spin-up + saturated response
nt = int(os.environ.get("E1B_NT", "4000"))           # dt = 0.05 yr; upwind CFL u_front*dt/dx ~ 0.7
dt = t_f/nt                        # Timestep size

save_vtk = False                   # Flag for saving solutions in VTK format
save_every = 5                     # save surface velocity fields every N steps

# E1b through-flow + forcing parameters
u0 = 600.0/yr                      # inflow speed (m/s); alpha = u0*t_r/H ~ 1.69
m0 = float(os.environ.get("E1B_M0_MYR", "5.0"))/yr  # peak basal melt (m/s), m > 0 = melt; 0 = control run
stdev = float(os.environ.get("E1B_STDEV_M", 10.0*H/3.0))  # melt-anomaly width (m)
xc = -10*1000.0                    # anomaly center (ocean-fixed; 10 km from inflow)
melt_kind = os.environ.get("E1B_MELT", "gauss")       # gauss | cos (zero-mean single-|k| cosine)
lam_m = float(os.environ.get("E1B_LAM_M", "1500.0"))  # cosine wavelength (m)

def bfac(x):
    # spatial prefactor multiplier on the hardness B in eta(); UFL-compatible.
    # Uniform for the E1b pilot; R-runs override this (shear margins, sutures).
    return 1.0
