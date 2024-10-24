import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
cmtoau = 4.556335e-06 
fstoau = 41.341                           # 1 fs = 41.341 a.u.
pstoau = 41.341 * 1000                    # 1 fs = 41.341 a.u.
#====================================

#====================================
# Functions
#====================================


#====================================
# Fig. S4a
#====================================
x    = np.loadtxt('./data/ADW_VDW.dat')[:,0]     # Position coordinate
VDW  = np.loadtxt('./data/ADW_VDW.dat')[:,1:-1]  # Asymmetric DW eigenstates
ADW  = np.loadtxt('./data/ADW_VDW.dat')[:,-1]        # Asymmetric DW potential
EDW  = np.loadtxt('./data/ADW_EDW.dat')          # Asymmetric DW eigenenergies

fig, ax = plt.subplots(figsize = (4.5,4.5), dpi = 150)
ax.plot(x, VDW[:,0]*2E3 + (EDW[0]-EDW[0]), lw = 2, color = '#9b59b6')
ax.fill_between(x,VDW[:,0]*2E3 + (EDW[0]-EDW[0]), y2 = (EDW[0]-EDW[0]), color = '#9b59b6', alpha = 0.3)
# =====================================================================================================
ax.plot(x, VDW[:,1]*2E3 + (EDW[1]-EDW[0]), lw = 2, color = '#2980b9')
ax.fill_between(x,VDW[:,1]*2E3 + (EDW[1]-EDW[0]), y2 = (EDW[1]-EDW[0]),  alpha = 0.3, color = '#2980b9')
# =====================================================================================================
ax.plot(x, -VDW[:,2]*2E3 + (EDW[2]-EDW[0]), lw = 2, color = '#27ae60')
ax.fill_between(x,-VDW[:,2]*2E3 + (EDW[2]-EDW[0]), y2 = (EDW[2]-EDW[0]),  alpha = 0.3, color = '#27ae60')
# =====================================================================================================
ax.plot(x, -VDW[:,3]*2E3 + (EDW[3]-EDW[0]), lw = 2, color = '#e74c3c')
ax.fill_between(x,-VDW[:,3]*2E3 + (EDW[3]-EDW[0]), y2 = (EDW[3]-EDW[0]),  alpha = 0.3, color = '#e74c3c')
# =====================================================================================================
ax.plot(x, VDW[:,4]*2E3 + (EDW[4]-EDW[0]), lw = 2, color = '#e67e22')
ax.fill_between(x,VDW[:,4]*2E3 + (EDW[4]-EDW[0]), y2 = (EDW[4]-EDW[0]),  alpha = 0.3, color = '#e67e22')
# =====================================================================================================
ax.plot(x, VDW[:,5]*2E3 + (EDW[5]-EDW[0]), lw = 2, color = '#34495e')
ax.fill_between(x,VDW[:,5]*2E3 + (EDW[5]-EDW[0]), y2 = (EDW[5]-EDW[0]),  alpha = 0.3, color = '#34495e')
# =====================================================================================================
ax.plot(x,(ADW-EDW[0]), c = 'black', lw = 4)
ax.text(80,110, r'$|\nu_0\rangle$', fontsize = 15)
ax.text(-95,1120, r'$|\nu_1\rangle$', fontsize = 15)
ax.text(80,1450, r'$|\nu_2\rangle$', fontsize = 15)
ax.text(-95,1960, r'$|\nu_3\rangle$', fontsize = 15)
ax.text(80,2580, r'$|\nu_4\rangle$', fontsize = 15)
ax.text(-95,3200, r'$|\nu_5\rangle$', fontsize = 15)
ax.set_ylim(-750,3500)
ax.set_xlim(-105,105)
ax.set_xlabel('$R_0$ (a.u.)')
ax.set_ylabel('Energy (cm$^{-1}$)')
ax.set_xticks([-100, -50, 0, 50, 100])
plt.savefig('./Fig_S4a.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()
#====================================

#====================================
# Fig. S4b
#====================================
in_dia = np.loadtxt('./data/pop_in_cav_180cm.dat')
out_dia = np.loadtxt('./data/pop_out_cav.dat')

fig, ax = plt.subplots(figsize = (4.5,4.5), dpi =150)

ax.plot(out_dia[:,0], out_dia[:,1], ls = '--', lw = 1.5, color = '#9b59b6')
ax.plot(out_dia[:,0], out_dia[:,3], ls = '--', lw = 1.5, color = '#27ae60')
ax.plot(out_dia[:,0], out_dia[:,4], ls = '--', lw = 1.5, color = '#e74c3c')
ax.plot(out_dia[:,0], out_dia[:,5], ls = '--', lw = 1.5, color = '#e67e22')
ax.plot(out_dia[:,0], out_dia[:,6], ls = '--', lw = 1.5, color = '#34495e')
ax.plot(in_dia[:,0], in_dia[:,1], ls = '-', lw = 2, color = '#9b59b6', label = r'$|\nu_0\rangle$')
ax.plot(in_dia[:,0], in_dia[:,3], ls = '-', lw = 2, color = '#27ae60', label = r'$|\nu_2\rangle$')
ax.plot(in_dia[:,0], in_dia[:,4], ls = '-', lw = 2, color = '#e74c3c', label = r"$|\nu_3\rangle$")
ax.plot(in_dia[:,0], in_dia[:,5], ls = '-', lw = 2, color = '#e67e22', label = r"$|\nu_4\rangle$")
ax.plot(in_dia[:,0], in_dia[:,6], ls = '-', lw = 2, color = '#34495e', label = r'$|\nu_5\rangle$')
ax.set_xlabel(r'Time (ps)')
ax.set_ylabel(r'Population')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 0.006)
ax.set_yticks([0, 0.002, 0.004, 0.006])
ax.legend(loc=0, frameon = False, fontsize = 11, handlelength=1, title_fontsize = 15, labelspacing = 0, ncol = 5, columnspacing = 0.5)

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '-', label='Inside Cavity', c='black')
ax2.plot(np.NaN, np.NaN, ls= '--',label='Outside Cavity', c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(loc = (0.65,0.1), frameon = False, fontsize = 9)

plt.savefig('./Fig_S4b.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()

#====================================
# Fig. S4c
#====================================
out_LR = np.loadtxt('./data/R_P_pop_out_cav.dat')
in_LR = np.loadtxt('./data/R_P_pop_in_cav_180cm.dat')

fig, ax = plt.subplots(figsize = (4.5,4.5), dpi = 150)

ax.plot(out_LR[:,0], out_LR[:,1], ls = '--', lw = 1.5, color = '#2980b9')
ax.plot(in_LR[:,0], in_LR[:,1], ls = '-', lw = 2, color = '#2980b9')
ax.set_xlabel(r'Time (ps)')
ax.set_ylabel(r'Population')
ax.set_xlim(0, 2.5)
ax.set_ylim(0.991, 1)
ax.set_yticks([0.995, 1.0])
ax.text(0.1, 0.9915, 'Reactant', fontsize = 20)

plt.savefig('./Fig_S4c.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()

#====================================
# Fig. S4d
#====================================
fig, ax = plt.subplots(figsize = (4.5,4.5), dpi = 150)

ax.plot(out_LR[:,0], out_LR[:,2], ls = '--', lw = 1.5, color = '#e74c3c')
ax.plot(in_LR[:,0], in_LR[:,2], ls = '-', lw = 2, color = '#e74c3c')
ax.set_xlabel(r'Time (ps)')
ax.set_ylabel(r'Population')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 0.009)
ax.text(0.15, 0.008, 'Product', fontsize = 20)

plt.savefig('./Fig_S4d.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()