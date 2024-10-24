import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
# plt.rcParams['font.size'] = 19
# plt.rcParams['font.family'] = "times"
#==================================

#====================================
# PHYSICAL CONSTANTS
#====================================
cmtoau = 4.556335e-06 
#====================================

#====================================
# GET RATES AND WQ DISTRIBUTIONS
#====================================
std_wQ = np.array([0, 10, 20, 30, 40])#, 100, 200]) # Standard deviation of wQ in cm-1
ntasks = len(std_wQ)
rate = np.zeros(ntasks)

fig, ax = plt.subplots(figsize = (4.5,4.5))
for k in range(ntasks):
    data = np.loadtxt(f'data/wQ_{ntasks - 1 - k}.txt', dtype = complex)
    rate[k] = np.loadtxt(f'data/rate_{k}.txt')                                         # Get rate from files
    plt.hist(np.real(data[:])/cmtoau, bins = 100, label = f'{std_wQ[ntasks - 1 - k]}', alpha = 1) # Plot histogram of wQ distribution

plt.legend(title = r'$\sigma$ (cm$^{-1}$)', frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
plt.ylim(0,30)
ax.set_xlabel(r'$\omega$ (cm$^{-1}$)', fontsize = 22)
plt.savefig('images/wQ_dist.png', dpi = 500, bbox_inches='tight')
plt.close()

np.savetxt('rates.txt', np.c_[std_wQ,rate])

#====================================
# GET EFFECTIVE SPECTRAL DENSITY
#====================================
freq = len(data)
Jeff = np.zeros((freq,ntasks))
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax = plt.subplots(figsize = (4.5,4.5))
for k in range(ntasks):
    data = np.real(np.loadtxt(f'data/Jeff_{k}.txt', dtype = complex))
    peaks, _ = find_peaks(data[:,1] * 1E7, height=0.1)
    UP = peaks[1]
    LP = peaks[0]
    print((data[UP,0] - data[LP,0])/80, r'$\Omega_R$')
    print(1 + 2*std_wQ[k]**2/80**2, r'$\sigma$')
    print('=================')
    ax.plot(data[:,0], data[:,1] * 1E7, lw = 2.5, label = f'{std_wQ[k]}', alpha = 0.8, color = f'{color[k]}')

ax.set_xlim(1000,1400)
ax.set_ylim(-0.1,3.1)
ax.set_ylabel(r'$J_{{eff}}$ ($10^{-7}$ a.u.)', fontsize = 22)
ax.set_xlabel(r'$\omega$ (cm$^{-1}$)', fontsize = 22)
plt.legend(title = r'$\sigma$ (cm$^{-1}$)', frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
plt.savefig('images/Jeff.png', dpi = 500, bbox_inches='tight')
plt.close()