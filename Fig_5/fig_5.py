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

#====================================
# PHYSICAL CONSTANTS
#====================================
fstoau = 41.341          # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06    # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05 
temp = 300 / autoK 
beta = 1/temp
R = 8.31446261815324 * 1/1000# kJ/(K*mol)
kB = 1.380649E-23 #J/K
h = 6.62607015E-34 # J.s

#====================================
# FUNCTIONS
#====================================

def Jeff(w,wc): # Effective Spectral density J_eff(w)
    Γ = 1/τc
    ΓQ = 2 * λQ / γQ + (2 * N * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * N * wc * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return N * Cj**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )

def J_spec(w,lambda_, gamma): # Drude - Lorentz Spectral density
    return 2*lambda_*gamma*w/(w**2 + gamma**2)

def k_FGR(w,wc,beta): # FGR rate
    J = Jeff(w, wc)
    n = 1/(np.exp(beta * w)-1)
    return 2 * Δx**2 * J * n

def sigma(): # Sigma square
    w_cut = np.arange(1E-15,γv,0.001*cmtoau)
    dw_cut = w_cut[1] - w_cut[0]
    f = J_spec(w_cut,λv,γv) * np.cosh(beta*w_cut/2)/np.sinh(beta*w_cut/2)
    return εz**2/np.pi * trapz(f,w_cut,dw_cut)

def K_VSC(w,wc,beta): # Convolution k_FGR @ Gaussian
    k = k_FGR(w,wc,beta)
    G = 1/np.pi * σ2**0.5/((w-w0)**2 + σ2) # Lorentzian Broadening function
    dw = w[1] - w[0]
    return trapz(k*G,w,dw)

def lineal(x,m,b): # Linear function
    return x * (m) + b 
#====================================
# SYSTEM PARAMETERS
#====================================
# Reaction coordinate
# =========================
w0 = 1189.7 * cmtoau                  # Transition frequency
M = 1836                              # Proton mass
wb = 1000 * cmtoau                    # Transition barrier frequency

# Reaction coordinate phonon bath
# =========================
cut_off = 200                         # Phonon bath cut-off frequency
ηv = 0.1                              # Friction parameter                      
γv = 200 * cmtoau                     # Bath characteristic frequency
λv = ηv * γv * wb/2               # Reorganization energy
εz = 9.386744774386742                
Δx = 9.140954405510243                # Transition dipole
σ2 = sigma()                          # Variance square
# =========================

# Q_mode
# =========================
Cj = 4.7 * cmtoau/(1836**0.5)         # Qj - R0 coupling
wQ = 1189.7 * cmtoau                  # RPV frequency
γQ = 6000 * cmtoau                    # Phonon bath characteristic frequency
λQ = 0.147 * cmtoau                   # Bath reorganization energy       
# =========================

# Cavity
# =========================
τc = 500 * fstoau                     # Cavity lifetime

# HEOM rates
# =========================
kD = 6.205e-08                        # Reaction coordinate + bath rate (No RPV coupling)
k0 = 9.077e-08                        # Outside cavity rate
# =========================

#====================================
# Fig. 5a
#====================================
w = np.arange(0.001, 3000, 1) * cmtoau
wc = wQ
points = 1000
kappas = np.zeros(points)
nwc = np.linspace(600,2000,points) * cmtoau
fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.axvline(wQ/cmtoau, ls = '--', alpha = 1, c = 'black')

color = ['black', 'b', 'green', 'red', 'purple']
ΩR = [0, 45, 79, 114]
N = 1E4
etas = [ΩR[k]/(2 * wc * N)**0.5 * cmtoau for k in range(len(ΩR))]
for n in range(len(etas)):
    Cj = 5E-7/N**0.5
    ηc = etas[n]
    for i in range(points):
        wc = nwc[i]
        kappas[i] = (K_VSC(w,wc,beta))* 0.7 + kD
    ax.plot(nwc/cmtoau, (kappas/k0) , color = f'{color[n]}', ls = '-', lw = 3, label = f'{np.round(ΩR[n],0)}', alpha = 0.7)

ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])
ax.set_xlim(700,1700)
ax.set_ylim(0.72, 1.01)

ax.set_xlabel('$\omega_c$ (cm$^{-1}$)', fontsize = 22)
ax.set_ylabel('$k/k_0$', fontsize = 22)

ax.legend(title =r"${\Omega}_R$ (cm$^{-1}$)", loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
ax.text(1210,0.74,r"$\omega_Q$", fontsize = 22)

plt.savefig('./Fig_5a.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 5b
#====================================
wc = wQ
etas = [0, 0.2E-4, 0.4E-4, 0.76E-4]
Ns = np.arange(0.1,1E4,1)
ΩR = np.array([(2*wc)**0.5 * etas[k] for k in range(len(etas))])
points = len(Ns)
kappas = np.zeros((points, len(etas)))

fig, ax = plt.subplots(figsize = (4.5,4.5))

for k in range(len(etas)):
    for n in range(len(Ns)):
        N = Ns[n]
        Cj = 5E-7/N**0.5
        ηc = etas[k]
        kappas[n,k] = (K_VSC(w,wc,beta)) * 0.7 + kD
    ax.plot((Ns)**0.5, (kappas[:,k]/k0) , color = f'{color[k]}', ls = '-', alpha = 0.7, lw = 3, label = f'{np.round(ΩR[k]/cmtoau,1)}')

ax.set_xlim(700,1700)
ax.set_ylim(0.72, 1.01)
ax.set_yticks([0.8,0.9,1.0])
ax.set_xlim(0, 100)
ax.set_xlabel('$\sqrt{N}$ ', fontsize = 22)

ax.legend(title =r"${\Omega}_R/\sqrt{N}$ (cm$^{-1}$)", ncol = 2, loc=3, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

plt.savefig('./Fig_5b.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 5c
#====================================
ε = [0, 0.5, 1, 2]
data_Cj = np.loadtxt(f'data_fig5c.txt')
w = data_Cj[:,0]

fig, ax = plt.subplots(figsize = (4.5,4.5))
color = [ 'b', 'green', 'red', 'blueviolet']

plt.axhline(1, lw = 3, alpha = 0.7, c = 'black')
ax.axvline(1187, ls = '--', alpha = 1, c = 'black')
for k in range(len(ε)):
    J = data_Cj[:,k+1]
    ax.plot(w,J, lw = 3, color = f'{color[k]}', alpha = 0.7, label = f'{ε[k]}')
ax.set_xlim(700,1700)
ax.set_ylim(0.73,1.01)
ax.set_xlabel('$\omega_c$ (cm$^{-1}$)', fontsize = 22)
ax.set_ylabel('$k/k_0$', fontsize = 22)
ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])
ax.legend(title ='$\epsilon $', loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
plt.savefig('./Fig_5c.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 5d
#====================================
data_dis = np.loadtxt('./data_fig5d.txt')
Cj = data_dis[:,0]
phi = data_dis[:,1]
rates = data_dis[:,2]
len_3D = int(len(Cj)**0.5)

vmin = min(rates)
vmax = max(rates)

std_cQ = np.linspace(0,3,len_3D)
std_dip = np.linspace(0,1,len_3D)
X, Y = np.meshgrid(std_cQ, std_dip) 

Cj = np.array(Cj).reshape(len_3D,len_3D)
phi = np.array(phi).reshape(len_3D,len_3D)
rates = np.array(rates).reshape(len_3D,len_3D)

fig, ax = plt.subplots(figsize = (4.5,4.5))

img = ax.imshow(rates.T,extent=(np.amin(std_cQ), np.amax(std_cQ), np.amin(std_dip), np.amax(std_dip)),origin='lower', aspect='auto', cmap='seismic', interpolation='bicubic')
plt.xlim(0,2)
plt.xlabel('$\epsilon$', fontsize = 22)
plt.ylabel(r'$\varphi\ (rad)$', fontsize = 22)
plt.xticks([1,2], fontsize = 19)
plt.yticks([0, 1/4, 1/2, 3/4, 1],['0', r'$\frac{π}{4}$', r'$\frac{π}{2}$', r'$\frac{3π}{4}$', r'π'], fontsize = 20)
bar = fig.colorbar(img, ticks = [0.8, 0.9, 1])
bar.ax.set_ylabel('$k/k_0$', fontsize = 22)
bar.ax.tick_params(labelsize=19)
plt.savefig('./Fig_5d.pdf', dpi = 500, bbox_inches='tight')
plt.close()