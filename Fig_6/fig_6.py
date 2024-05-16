import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 19})
plt.rcParams.update({'font.family': "times"})
#====================================
# CONSTANTS
#====================================
fstoau = 41.341          # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06    # 1 cm^-1 = 4.556335e-06 a.u.
wQ = 1189.7 * cmtoau

#====================================
# Fig. 6a
#====================================
w_scan = np.linspace(600, 1800, 200)
dat = np.loadtxt('./rates_FGR.txt')
ηs = [0.002, 0.0035, 0.005]
ΩR = [int(2*wQ/(2*wQ)**0.5 * x/cmtoau) for x in ηs]

color = ['b', 'green', 'red']

fig, ax = plt.subplots(figsize = (4.5,4.5))#, dpi = 500)
ax.axhline(1, c = 'black', lw = 3, alpha = 0.5, label = '0')
ax.axvline(wQ/cmtoau, ls = '--', alpha = 1, c = 'black')
for k in range(len(ηs)):
    data2 = np.loadtxt(f'1D_{ηs[k]}.txt')
    ax.plot(w_scan,data2, color = f'{color[k]}', lw = 3, alpha = 0.5, label = f'{ΩR[k]}')
    ax.plot(dat[:,0],dat[:,k+2], color = f'{color[k]}', lw = 2, alpha = 1, ls = '--')

ax.text(1211,0.757, '$\omega_Q$', fontsize = 20)
ax.legend(title ='$\Omega_R$ (cm$^{-1}$)', loc=3, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
ax.set_xlim(700,1700)
ax.set_ylim(0.75, 1.01)
ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])
ax.set_ylabel('$k/k_0$', fontsize = 22)
ax.set_xlabel('$\omega_c$ (cm$^{-1}$)', fontsize = 22)

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '--',label='Single mode', c='black')
ax2.plot(np.NaN, np.NaN, ls= '-',label='Multimode', c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(loc=(0.55,0.095), frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_6a.pdf', dpi = 500, bbox_inches='tight')

#====================================
# Fig. 6b
#====================================
w_scan = np.linspace(600, 1740, 200)
ηs = [0.002, 0.0035, 0.005]
ΩR = [int(2*wQ/(2*wQ)**0.5 * x/cmtoau) for x in ηs]

color = ['blue', 'green', 'red']
fig, ax = plt.subplots(figsize = (4.5,4.5))#, dpi = 500)

ax.axhline(1, c = 'black', lw = 3, alpha = 0.5, label = '0')
ax.axvline(wQ/cmtoau, ls = '--', alpha = 1, c = 'black')
for k in range(len(ηs)):
    data2 = np.loadtxt(f'2D_{ηs[k]}.txt')
    ax.plot(w_scan,data2, color = f'{color[k]}', lw = 3, alpha = 0.7, label = f'{ΩR[k]}')

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '',label='$\omega_Q$', c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(loc= (0.33,0.01), frameon = False, fontsize = 20, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

ax.set_xlim(700,1700)
ax.set_ylim(0.78, 1.01)
ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])
ax.set_ylabel('$k/k_0$', fontsize = 22)
ax.set_xlabel('$\omega_c$ (cm$^{-1}$)', fontsize = 22)

plt.savefig('./Fig_6b.pdf', dpi = 500, bbox_inches='tight')