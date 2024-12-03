import numpy as np
import matplotlib.pyplot as plt

# number of iterations in Monte-Carlo simulation
nmc = 1000

# Derivative sensor parameters
mu0 = np.pi * 4e-7         # permeability of vacuum H/m
mu_r = 50000               # relative permeability of Vitroperm500
mu = mu0 * mu_r            # permeability of the material in H/m
Ac = 18.0e-4               # cross section of core in m^2
n2 = 20000                 # turns of pickup coil
la = 1.2e-3                # air gaps length in m
lc = 0.3                   # magnetic path in m

# nominal proportionality factor of derivative sensor Vs/A = H
kds = mu * Ac * n2 / (lc + 2 * mu_r * la)

# TOLERANCE
# first rough estimation of tolerances to refine with Vacuumschmeltz
tol_la = 0.1             # relative tolerance of la ~ 1 mm
tol_lc = 0.001           # relative tolerance of lc ~ 300 mm
tol_Ac = 0.05              # relative tolerance of Ac ~ cm^2

# adding tolerances
la_tol = la * (1 + (2 * np.random.rand(nmc) - 1) * tol_la)
lc_tol = lc * (1 + (2 * np.random.rand(nmc) - 1) * tol_lc)
Ac_tol = Ac * (1 + (2 * np.random.rand(nmc) - 1) * tol_Ac)
kds_tol = mu * n2 * Ac_tol / (lc_tol + 2 * mu_r * la_tol)

# plotting
plt.figure()
plt.boxplot(kds_tol)
plt.grid(True)
plt.ylabel('kds tol distribution')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
plt.axhline(y=kds, color='k', linestyle='--')

# estimation of relative tolerance of kds
tol_kds = np.std(kds_tol) / kds

# UNCERTAINTY
# first rough estimation of uncertainties to refine with Vacuumschmeltz
std_la = 0.01            # relative deviation of la ~ 0.1 mm
std_lc = 0.0001          # relative deviation of lc ~ 30 mm
std_Ac = 0.005             # relative deviation of Ac ~ 0.1 cm^2

# adding tolerances
la_std = la * (1 + (2 * np.random.rand(nmc) - 1) * std_la)
lc_std = lc * (1 + (2 * np.random.rand(nmc) - 1) * std_lc)
Ac_std = Ac * (1 + (2 * np.random.rand(nmc) - 1) * std_Ac)
kds_std = mu * n2 * Ac_std / (lc_std + 2 * mu_r * la_std)

# plotting
plt.figure()
plt.boxplot(kds_std)
plt.grid(True)
plt.ylabel('kds values')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
plt.axhline(y=kds, color='k', linestyle='--')

# estimation of relative deviation of kds
s_kds = np.std(kds_std) / kds

plt.show()