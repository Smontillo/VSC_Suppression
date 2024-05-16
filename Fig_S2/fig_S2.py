import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
cmtoau = 4.556335e-06 
fstoau = 41.341                           # 1 fs = 41.341 a.u.
pstoau = 41.341 * 1000                           # 1 fs = 41.341 a.u.
#====================================

#====================================
# Population data
#====================================
out_cavity = np.loadtxt('./DW_pop_Rabi_0.txt')
in_cavity = np.loadtxt('./DW_pop_Rabi_100.txt')

#====================================
# Fig. S2a
#====================================
fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,3], ls = '--', lw = 2, c = 'g')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,3], c = 'g', label = r"$|{\nu}'_L⟩$")
ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,2], ls = '--', lw = 2, c = 'r')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,2], c = 'r', label = r"$|{\nu}'_R⟩$")
ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,1], ls = '--', lw = 2, c = 'b')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,1], c = 'b', label = r'$|\nu_R⟩$')
ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,4], ls = '--', lw = 2, c = 'gray')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,4], c = 'gray', label = r'$|\nu_4⟩$')

ax.set_xlim(-0.1,10)
ax.set_ylim(-0.0001,0.035)
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Population')
ax.legend(loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2, ncol = 1)
ax.set_xticks([0, 5, 10])

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '-',label='Inside Cavity', c='black')
ax2.plot(np.NaN, np.NaN, ls= '--',label='Outside Cavity', c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(loc= (0.5,0.1), frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_S2a.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. S2b
#====================================

fig, ax = plt.subplots(figsize = (4.5,4.5))#, dpi = 500)

ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,3], ls = '--', lw = 2, c = 'g')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,3], c = 'g', label = r"$|{\nu}'_R⟩$")
ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,2], ls = '--', lw = 2, c = 'r')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,2], c = 'r', label = r"$|{\nu}'_L⟩$")
ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,1], ls = '--', lw = 2, c = 'b')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,1], c = 'b', label = r'$|\nu_R⟩$')
ax.plot(out_cavity[:,0]/pstoau, out_cavity[:,4], ls = '--', lw = 2, c = 'gray')
ax.plot(in_cavity[:,0]/pstoau, in_cavity[:,4], c = 'gray', label = r'$|\nu_R⟩$')
ax.set_xlim(-0.02,1.5)
ax.set_ylim(-0.0001,0.0058)
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Population')
ax.set_yticks([0.001, 0.003, 0.005])

plt.savefig('./Fig_S2b.pdf', dpi = 500, bbox_inches='tight')
plt.close()