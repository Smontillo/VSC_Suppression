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

# =========================
#    FUNCTIONS
# =========================
def J_spec(w,lambda_, gamma):
    return 2*lambda_*gamma*w/(w**2 + gamma**2)

# FGR rate
def k_FGR(w,beta,J):
    n = 1/(np.exp(beta * w)-1)
    k = 2 * Δx**2 * J * n
    return k

# Sigma squared
def sigma():
    w_cut = np.arange(1E-15,γv,0.001*cmtoau)
    dw_cut = w_cut[1] - w_cut[0]
    f = J_spec(w_cut,λv,γv) * np.cosh(beta*w_cut/2)/np.sinh(beta*w_cut/2)
    return εz**2/np.pi * np.trapz(f,w_cut,dw_cut)

# Gaussian function
def Kernel(w):
    l = 1/np.pi * σ2**0.5/((w-w0)**2 + σ2)
    return l

# Convolution k_FGR @ Gaussian
def K_VSC(w,beta,J):
    k = np.real(k_FGR(w,beta,J))
    G = np.real(Kernel(w)) 
    dw = w[1] - w[0]
    return np.trapz(k*G,dx = dw) 

def Jeff(w,wc): # Effective Spectral density J_eff(w)
    Γ = 1/τc
    ΓQ = 2 * λQ / γQ + (2 * N * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * N * wc * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return N * Cj**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )


#====================================
# SYSTEM PARAMETERS
#====================================
# System
# =========================
cut_off = 200 #cm-1
w0 = 1189.7 * cmtoau 
M = 1
wb = 1000 * cmtoau

# Bath 
# =========================
cQ = 5E-7
wQ = 1189.678284945453 * cmtoau
ηv = 0.1
γv = 200 * cmtoau
λv = ηv * M * γv * wb/2
temp = 300 / autoK 
beta = 1/temp
εz = 9.386744774386742 
Δx = 9.140954405510243
σ2 = sigma()
# =========================

# Q_mode
# =========================
N = 1
Cj = 4.7 * cmtoau/((1836 * N)**0.5)         # Qj - R0 coupling
wQ = 1189.7 * cmtoau                  # RPV frequency
γQ = 6000 * cmtoau                    # Phonon bath characteristic frequency
λQ = 0.147 * cmtoau                   # Bath reorganization energy     
# =========================

# Cavity
# =========================
kD = 6.204982961140102545e-08 # Rate constant of DW without Q at eta_s 0.1
k0 = 9.076763853644350798e-08 # Q + Rxn rates                  # Cavity lifetime
wc = 1189.7 * cmtoau
ΩR = 100 * cmtoau
ηc = ΩR / np.sqrt(2 * N * wc) 

#====================================
# Fig S12
#====================================
w = np.arange(1E-5,5000,0.015) * cmtoau
life = np.loadtxt('./Lifetime_Rabi_100.txt')
color = ['black','b', 'green', 'red', 'g']

points = 200
kappas = np.zeros(points)
times = np.linspace(50,2000,points) * fstoau

for n in range(len(times)):
    τc = float(times[n])
    kappas[n] = (K_VSC(w,wc,beta)) * 0.7 + kD

fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(times/fstoau, (kappas/k0) , ls = '-', lw = 3, alpha = 0.6, c = 'blue', label = 'FGR')


ax.plot(life[:,0], life[:,1]/k0, label = 'HEOM', marker = 'o')
ax.axvline(1/(50*cmtoau*fstoau), ls = '--', c = 'black', lw = 1, alpha = 0.7)
ax.set_xlabel(r'$\tau_c$ (fs)')
ax.set_ylabel(r'$k/k_0$')
ax.set_ylim(0.75,0.95)
ax.set_ylim(0.73,1.02)

ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([0, 500, 1000, 1500, 2000])
ax.legend(loc=1, frameon = False, fontsize = 14)

plt.savefig('./Fig_S12.pdf', dpi = 500, bbox_inches='tight')
plt.close()
