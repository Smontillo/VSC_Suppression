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
# Fig S8
#====================================

data = np.loadtxt('./Jeff_0.txt')
std = [0.1, 0.3, 0.5, 1]
frac = [10,1, 2,' ' ]
w = data[:,0]

fig, ax = plt.subplots(figsize = (4.5,4.5))
color = ['b', 'blueviolet', 'green', 'red']

for k in range(len(std)):
    if k == 1:
        continue
    J = data[:,k+1]
    if k == 3:
        ax.plot(w,J * 1e7, lw = 2, color = f'{color[k]}', alpha = 0.9, label = r'$0 - \pi$')
    else:
        ax.plot(w,J * 1e7, lw = 2, color = f'{color[k]}', alpha = 0.9, label = r'$0 - \pi$/' f'{frac[k]}')
ax.set_xlim(1060,1330)
ax.set_ylim(-0.1,4.7)
ax.set_xlabel(r'$\omega\ (cm^{-1})$')
ax.set_ylabel(r'$J_\mathrm{eff}\ (10^{-7}\ \mathrm{a.u.})$')
ax.set_yticks([0, 2, 4])
ax.legend(title =r'$\varphi$', loc=1, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_S8.pdf', dpi = 500, bbox_inches='tight')
plt.close()