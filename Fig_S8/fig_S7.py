import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
cmtoau = 4.556335e-06 
fstoau = 41.341                           # 1 fs = 41.341 a.u.
#====================================

def inv(R,a,b,c):
    return c + b/(1+a*R**2)

#====================================
# Population data
#====================================
data_FGR = np.loadtxt('./rate_FGR.txt')
data_HEOM = np.loadtxt('./rate_HEOM.txt')
data_Ebb = np.loadtxt('./Ebbesen_exp.txt')

kD = 5.9467918061933605e-08 # Rate constant of DW without Q at eta_s 0.1
k0 = 8.974686702058101812E-8 # Q + Rxn rates
wQ = 1189.7 * cmtoau
wc = 1189.7 * cmtoau
#====================================
# Fig. S7a
#====================================
ΩR_FGR = data_FGR[:,0]
rate_FGR = data_FGR[:,1]

ppar_FGR, pcov_FGR = curve_fit(inv,ΩR_FGR,rate_FGR/k0,p0 = [0,0, kD/k0])

fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(ΩR_FGR[::8],rate_FGR[::8]/k0, ls = ' ', marker = 'o', markersize = 8, label = 'FGR', alpha = 1)
ax.plot(ΩR_FGR,inv(ΩR_FGR,*ppar_FGR), c = 'r', lw = 2, ls = '--' , label = 'Fitting')
ax.set_xlim(30,150)
ax.set_ylim(0.7,1.05)
ax.set_xlabel(r'$\Omega_R\ (\mathrm{cm}^{-1})$')
ax.set_ylabel(r'$k/k_0$')
ax.set_yticks([0.8, 0.9, 1])
ax.legend(frameon = False, fontsize = 17, handlelength=1.5, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_S7a.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. S7b
#====================================
η_HEOM = data_HEOM[:,0] 
rate_HEOM = data_HEOM[:,1]

ppar_HEOM, pcov_HEOM = curve_fit(inv,η_HEOM[3:],rate_HEOM[3:]/k0, p0 = [2.5E-4,0.43, kD/k0])

fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(η_HEOM[3:]* (2 * wc)**0.5/cmtoau , rate_HEOM[3:]/k0, lw = 3, ls = ' ', alpha = 1, marker = 'o', markersize = 8, label = 'HEOM')
ax.plot(η_HEOM* (2 * wc)**0.5/cmtoau ,(inv(η_HEOM,*ppar_HEOM)), lw = 2, c = 'r', ls = '--', label = 'Fitting')
ax.set_xlim(30,150)
ax.set_ylim(0.7,1.05)
ax.set_xlabel(r'$\Omega_R\ (\mathrm{cm}^{-1})$')
ax.set_ylabel(r'$k/k_0$')
ax.set_yticks([0.8, 0.9, 1])
ax.legend(frameon = False, fontsize = 17, handlelength=1.5, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_S7b.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. S7c
#====================================
rate_eb = data_Ebb[:,1]
ΩR_eb = data_Ebb[:,0]
ΩR = data_FGR[:,0]
ppar_eb, c = curve_fit(inv,ΩR_eb,rate_eb, p0 = [ 3E-4, 1E-1,2])

fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(ΩR,inv(ΩR, *ppar_eb), lw = 2, ls = '--', c = 'r', label = 'Fitting')
ax.plot(ΩR_eb, rate_eb, ls = ' ', marker = 'o', markersize = 8, label = 'Exp. Data')
ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_ylabel('$k/k_0$')
ax.set_ylim(0.0,1.05)
ax.set_xlim(50,105)
ax.set_xlabel(r'$\Omega_R$ $(cm^{-1})$')
ax.legend(frameon = False, fontsize = 17, handlelength=1.5, title_fontsize = 15, labelspacing = 0.2)
plt.savefig('./Fig_S7c.pdf', dpi = 500, bbox_inches='tight')
plt.close()
