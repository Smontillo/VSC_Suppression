import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
#====================================

#====================================
# PHYSICAL CONSTANTS
#====================================
fstoau = 41.341          # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06    # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05 
temp = 300 / autoK 
beta = 1/temp

#====================================
# FUNCTIONS
#====================================

def Jeff(w,wc): # Effective Spectral density J_eff(w)
    Γ = 1/τc
    ΓQ = 2 * λQ / γQ + (2 * N * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * N * wc * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return N * Cj**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )


#====================================
# SYSTEM PARAMETERS
#====================================

# Q_mode
# =========================
N = 1000  
Cj = 4.7 * cmtoau/((1836 * N)**0.5)         # Qj - R0 coupling
wQ = 1189.7 * cmtoau                  # RPV frequency
γQ = 6000 * cmtoau                    # Phonon bath characteristic frequency
λQ = 0.147 * cmtoau                   # Bath reorganization energy     
# =========================

# Cavity
# =========================
τc = 500 * fstoau                     # Cavity lifetime
wc = 1189.7 * cmtoau
ΩR = 100 * cmtoau
ηc = ΩR / np.sqrt(2 * N * wc) 

#====================================
# Fig S4
#====================================

data = np.loadtxt('./Jeff_0.txt')
w = data[:,0]
J = data[:,1]
J_an = Jeff(w * cmtoau,wc)

fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(w,J * 1e7, lw = 4, alpha = 0.7, c = 'r', label = 'Numerical')
ax.plot(w,J_an * 1e7, lw = 2, alpha = 0.8, ls = '--', c = 'black', label = 'Analytical')
ax.set_xlim(1060,1330)
ax.set_ylim(-0.1,4)
ax.set_xlabel(r'$\omega\ (cm^{-1})$')
ax.set_ylabel(r'$J_\mathrm{eff}\ (10^{-7}\ \mathrm{a.u.})$')
ax.set_yticks([0, 2, 4])
ax.legend(loc=1, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_S4.pdf', dpi = 500, bbox_inches='tight')
plt.close()