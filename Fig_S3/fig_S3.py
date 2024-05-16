import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
cmtoau = 4.556335e-06 
fstoau = 41.341                           # 1 fs = 41.341 a.u.
pstoau = 41.341 * 1000                           # 1 fs = 41.341 a.u.
#====================================

#====================================
# Functions
#====================================
def exp_decay(x, Γ):
    PE = 0.5 # Population at equilibrium
    PI = 0.99669031 # Initial population
    return (PI - PE) * np.exp(-x * Γ) + PE

#====================================
# Population data
#====================================
pop_in_cav = np.loadtxt('./pop_in_cav.txt')
pop_out_cav = np.loadtxt('./pop_out_cav.txt')

#====================================
# Fig. S3a
#====================================
time = pop_in_cav[:,0]/1000

fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(time,pop_in_cav[:,2], ls = '-', lw = 3, c = 'b', alpha = 0.5, label = r"$|{\nu}_R⟩$")
ax.plot(time,pop_out_cav[:,2], ls = '--', lw = 2, c = 'b')
ax.plot(time,pop_in_cav[:,3], ls = '-', lw = 3, c = 'r', alpha = 0.5, label = r"$|{\nu}_2⟩$")
ax.plot(time,pop_out_cav[:,3], ls = '--', lw = 2, c = 'r')
ax.plot(time,pop_in_cav[:,4], ls = '-', lw = 3, c = 'g', alpha = 0.5, label = r"$|{\nu}_3⟩$")
ax.plot(time,pop_out_cav[:,4], ls = '--', lw = 2, c = 'g')
ax.plot(time,pop_in_cav[:,5], ls = '-', lw = 3, c = 'darkorange', alpha = 0.5, label = r"$|{\nu}_4⟩$")
ax.plot(time,pop_out_cav[:,5], ls = '--', lw = 2, c = 'darkorange')
ax.set_xlim(-0.05,5)
ax.set_yticks([0.0, 0.25, 0.5])
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Population')
ax.legend(loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2, ncol = 1)

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '-',label='Inside Cavity', c='black')
ax2.plot(np.NaN, np.NaN, ls= '--',label='Outside Cavity', c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(loc= (0.5,0.1), frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_S3a.pdf', dpi = 500, bbox_inches='tight')
plt.close()


#====================================
# Fig. S3b
#====================================
time = pop_in_cav[:,0]/1000

fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(time,pop_in_cav[:,2], ls = '-', lw = 3, alpha = 0.5, c = 'b')
ax.plot(time,pop_out_cav[:,2], ls = '--', lw = 2, c = 'b')
ax.plot(time,pop_in_cav[:,3], ls = '-', lw = 3, alpha = 0.5, c = 'r')
ax.plot(time,pop_out_cav[:,3], ls = '--', lw = 2, c = 'r')
ax.plot(time,pop_in_cav[:,4], ls = '-', lw = 3, alpha = 0.5, c = 'g')
ax.plot(time,pop_out_cav[:,4], ls = '--', lw = 2, c = 'g')
ax.plot(time,pop_in_cav[:,5], ls = '-', lw = 3, alpha = 0.5, c = 'darkorange')
ax.plot(time,pop_out_cav[:,5], ls = '--', lw = 2, c = 'darkorange')
ax.set_xlim(-0.05,1.5)
ax.set_ylim(-0.001,0.04)
ax.set_yticks([0.0, 0.02, 0.04])
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Population')

plt.savefig('./Fig_S3b.pdf', dpi = 500, bbox_inches='tight')
plt.close()
#====================================
# Fig. S3c
#====================================
data = np.loadtxt('./pop_Cj_0.1_wc_1200_Rabi_60cm.txt')
time = data[:,0]
pop = data[:,1]

rates, covariance = curve_fit(exp_decay, time, pop, p0 = [6E-4])

fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(time/1000,pop, lw = 3.3, label = 'HEOM', c = 'dodgerblue')
ax.plot(time/1000,exp_decay(time,rates), lw = 2, ls = '--', label = 'Fitting', c = 'r')
ax.set_ylabel('Population')
ax.set_xlabel(r'Time (ps)')
ax.set_yticks([0.5, 0.75, 1])
ax.legend(loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2, ncol = 1)

plt.savefig('./Fig_S3c.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. S3d
#====================================
wQ = 1385.98 * cmtoau
wc = 1385.98 * cmtoau
data = np.loadtxt('./pop_Cj_0.1_wc_1385_Rabi_100cm.txt')
time = data[0:,0]
pop = data[0:,1]

rates, covariance = curve_fit(exp_decay, time, pop, p0 = [6E-4])

fig, ax = plt.subplots(figsize = (4.5,4.5))

# ax.plot(time,pop[:,0])
ax.plot(time/1000,pop, lw = 3.3, label = 'HEOM', c = 'dodgerblue')
ax.plot(time/1000,exp_decay(time,rates), lw = 2, ls = '--', label = 'Fitting', c = 'black')
ax.set_ylabel('Population')
ax.set_xlabel(r'Time (ps)')
ax.set_yticks([0.5, 0.75, 1])
ax.legend(loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2, ncol = 1)

plt.savefig('./Fig_S3d.pdf', dpi = 500, bbox_inches='tight')
plt.close()