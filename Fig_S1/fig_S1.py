import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
#====================================

# CONSTANTS
#====================================
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05 
wc = 1189.7 * cmtoau

#====================================
# Fig. S1a
#====================================
dat = np.loadtxt('flux_freq_Rabi_60.txt')
time = dat[:,0]
freq = [500, 700, 900, 1000, 1100, 1130, 1160, 1175, 1189, 1210, 1250, 1270, 1330, 1400, 1500, 1600, 1700]

index = [2,4, 5, 6,8,10,12, 13]
fig, ax = plt.subplots(figsize = (4.5,4.5))

for n in index:
    ax.plot(time/1000, dat[:,n+1]*fstoau*1E6, lw = 2, alpha = 1, label = f'{np.round(freq[n],1)}')   
ax.set_ylim(3,3.8)
ax.legend(title =r"${\omega}_c$ (cm$^{-1}$)", loc=0, frameon = False, fontsize = 14, handlelength=1, title_fontsize = 15, labelspacing = 0.2, ncol = 2)
ax.set_ylabel('$k$ $(x10^{-6}$ $fs^{-1})$')

ax.set_xlabel('Time (ps)')
ax.set_xlim(-0.1,10)

plt.savefig('./Fig_S1a.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. S1b
#====================================
dat = np.loadtxt('flux_Rabi_wc_1189.txt')
time = dat[:,0]
eta = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065]
ΩR = np.array([(2*wc)**0.5 * eta[k] for k in range(len(eta))])/cmtoau

index = [0,2,4, 5, 6,8,10,12]
fig, ax = plt.subplots(figsize = (4.5,4.5))
for n in index:
    ax.plot(time/1000, dat[:,n+1]*fstoau*1E6, lw = 2, alpha = 1, label = f'{int(ΩR[n])}')   
ax.set_xlim(-0.1,10)
ax.set_ylim(1.7,4.1)
ax.legend(title =r"${\Omega}_R$ (cm$^{-1}$)", loc=0, frameon = False, fontsize = 14, handlelength=1, title_fontsize = 15, labelspacing = 0.2, ncol = 2)
ax.set_ylabel('$k$ $(x10^{-6}$ $fs^{-1})$')
ax.set_xlabel('Time (ps)')

plt.savefig('./Fig_S1b.pdf', dpi = 500, bbox_inches='tight')
plt.close()