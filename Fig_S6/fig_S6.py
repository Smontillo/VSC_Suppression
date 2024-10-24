import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
cmtoau = 4.556335e-06 
fstoau = 41.341                           # 1 fs = 41.341 a.u.
pstoau = 41.341 * 1000                    # 1 fs = 41.341 a.u.
autoK = 3.1577464e+05 
#====================================

#====================================
# Functions
#====================================
def Jeff(ω, ωc, ωQ, CQ, λQ, γQ, ηc, τc):
    ΓQ = 2 * λQ / γQ + (2 * ωc**3 * ηc**2 * 1/τc) / ((ωc**2 - ω**2)**2 + (ω * 1/τc)**2) 
    R  = (2 * ωc * ηc**2 * ω**2) / ((ωc**2 - ω**2)**2 + (ω * 1/τc)**2) * (ω**2 - ωc**2 + (1/τc)**2)
    return CQ**2 / 2 * ω * ΓQ / ((ωQ**2 - ω**2 + R)**2 + (ω * ΓQ)**2)

def coth(x):
    return np.cosh(x)/np.sinh(x)

def DrudeL(λ, γ, ω):
    return 2 * λ * γ * ω / (ω**2 + γ**2)

def sigma2(λv, γv, εz, β):
    ω_cut  = np.arange(1E-15, γv, 0.001 * cmtoau)
    int_f  = DrudeL(λv, γv, ω_cut) * coth(β * ω_cut / 2)
    return εz**2 / np.pi * np.trapz(int_f, ω_cut)
 
def Kernel(ω, ω0, σ2):
    return 1/np.pi * σ2**0.5/((ω - ω0)**2 + σ2)

def k_FGR(Δx, ω, ωc, ωQ, CQ, λQ, γQ, ηc, τc, β):
    n = 1 / (np.exp(β * ω) - 1)
    return 2 * Δx**2 * n * Jeff(ω, ωc, ωQ, CQ, λQ, γQ, ηc, τc)

def K_VSC(Δx, σ2, ω, ω0, ωc, ωQ, CQ, λQ, γQ, ηc, τc, β):
    k = k_FGR(Δx, ω, ωc, ωQ, CQ, λQ, γQ, ηc, τc, β)
    L = Kernel(ω, ω0, σ2)
    return np.trapz(k * L, ω)

#====================================
# Parameters
#====================================
# Asymmetric DW parameters
# =============================
ω0 = 1192.0871712890928 * cmtoau 
ωb = 1030 * cmtoau
ηv = 0.1
γv = 200 * cmtoau
λv = ηv * γv * ωb / 2
εz = 12.5055003773874 
Δx = -9.202435816404238
T  = 300 / autoK
β  = 1 / T 
σ2 = sigma2(λv, γv, εz, β)

# =============================

# Rate Promoting Mode parameters
# =============================
ωQ = 1175 * cmtoau
γQ = 6000 * cmtoau
λQ = 6.7E-7
CQ = 5E-7
# =============================

# Cavity Parameters
# =============================
τc = 500 * fstoau
k0 = 3.0278788513352e-06
kQ = 3.72153865e-06

#====================================
# Fig. S4a
#====================================
x    = np.loadtxt('./data/ADW_VDW_diabats.dat')[:,0]     # Position coordinate
VDW  = np.loadtxt('./data/ADW_VDW_diabats.dat')[:,1:-1]  # Asymmetric DW eigenstates
ADW  = np.loadtxt('./data/ADW_VDW_diabats.dat')[:,-1]        # Asymmetric DW potential
EDW  = np.loadtxt('./data/ADW_EDW_diabats.dat')          # Asymmetric DW eigenenergies

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
ax.plot(x, VDW[:,3]*2E3 + (EDW[3]-EDW[0]), lw = 2, color = '#e74c3c')
ax.fill_between(x,VDW[:,3]*2E3 + (EDW[3]-EDW[0]), y2 = (EDW[3]-EDW[0]),  alpha = 0.3, color = '#e74c3c')
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
plt.savefig('./Fig_S6a.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()
#====================================

#====================================
# Fig. S4b
#====================================
fs_45 = np.loadtxt('./data/rate_wc_1175_RS_45cm.dat')
fs_70 = np.loadtxt('./data/rate_wc_1175_RS_70cm.dat')
fs_100 = np.loadtxt('./data/rate_wc_1175_RS_100cm.dat')

# FGR
ω      = np.arange(1E-5, 4000, 0.02) * cmtoau
points = 300
nωc    = np.linspace(800, 2000, points) * cmtoau
ΩR     = np.array([45, 70, 100]) * cmtoau
nηc     = ΩR / (2 * ωQ)**0.5
fs     = np.zeros((points, len(nηc)))

# Define α
ηc = 0
ωc = 1 * cmtoau
ks = (K_VSC(Δx, σ2, ω, ω0, ωc, ωQ, CQ, λQ, γQ, ηc, τc, β) * fstoau)
α  = (kQ - k0) / ks
print('α -> ', α)

for h in range(len(nηc)):
    ηc = nηc[h]
    for k in range(points):
        ωc = nωc[k]
        fs[k,h]  = (K_VSC(Δx, σ2, ω, ω0, ωc, ωQ, CQ, λQ, γQ, ηc, τc, β) * fstoau * α + k0)  / kQ


fig, ax = plt.subplots(figsize = (4.5,4.5), dpi =150)
ax.axhline(1, ls = '--', lw = 1, c = 'black')
# FGR data
ax.plot(nωc / cmtoau, fs[:,0], lw = 4, alpha = 0.7, color = '#e74c3c', label = f'{int(ΩR[0]/cmtoau)}')
ax.plot(nωc / cmtoau, fs[:,1], lw = 4, alpha = 0.7, color = '#3498db', label = f'{int(ΩR[1]/cmtoau)}')
ax.plot(nωc / cmtoau, fs[:,2], lw = 4, alpha = 0.7, color = '#27ae60', label = f'{int(ΩR[2]/cmtoau)}')
# Simulation data
ax.plot(fs_45[:,0], fs_45[:,1] * fstoau/kQ , ls = '--', lw = 1, marker = 'o', color = '#e74c3c')
ax.plot(fs_70[:,0], fs_70[:,1] * fstoau/kQ , ls = '--', lw = 1, marker = 'o', color = '#2980b9')
ax.plot(fs_100[:,0], fs_100[:,1] / kQ, ls = '--', lw = 1, marker = 'o', color = '#27ae60')

ax.set_xlabel(r'$\omega_c$ (cm$^{-1}$)')
ax.set_ylabel(r'$k/k_0$')
ax.set_xlim(800, 1600)
ax.set_ylim(0.89, 1.015)
ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls = '--', lw = 1, marker = 'o',label='HEOM', c='black')
ax2.plot(np.NaN, np.NaN, ls= '-',label='FGR', c='black')
ax2.get_yaxis().set_visible(False)

ax.set_yticks([0.90, 0.94, 0.96, 0.98, 1.0])
ax.set_yticks([0.90, 0.95, 1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])

ax.legend(loc=3, frameon = False, title = r'$\Omega_R$ (cm$^{-1}$)', fontsize = 11, handlelength=1, title_fontsize = 11, labelspacing = 0.2, ncol = 1)
ax2.legend(loc=4, frameon = False, fontsize = 10)

plt.savefig('./Fig_S6b.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.show()
