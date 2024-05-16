import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 19})
plt.rcParams.update({'font.family': "times"})
#====================================

#====================================
# PHYSICAL CONSTANTS
#====================================
fstoau = 41.341          # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06    # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05 
temp = 300 / autoK 
beta = 1/temp

#====================================
# FUNCTIONS
#====================================

def Jeff(w,wc): # Effective Spectral density J_eff(w)
    Γ = 1/τc
    ΓQ = 2 * λQ / γQ + (2 * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * wc * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return Cj**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )

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
# FIG 2a. 
#====================================
w = np.arange(1E-5,5000,0.05) * cmtoau
wc = wQ
color = ['black', 'blue', 'green', 'red']
etas = [0, 0.0025, 0.0035, 0.005]                      # Light - matter coupling scan values
ΩR = [int(2*wc/(2*wQ)**0.5 * x/cmtoau) for x in etas]  # Rabi Splitting

fig, ax = plt.subplots(figsize = (4.5,4.5))

for k in range(len(etas)):
    ηc = etas[k]
    ax.plot(w/cmtoau, Jeff(w, wc)*1E7, ls = '-', lw = 2.5, c = f'{color[k]}', alpha = 0.7, label = f'{ΩR[k]}')

ax.set_xlim(1050,1360)
ax.set_xlabel('$\omega$ $(cm^{-1})$')
ax.set_ylabel('$J_{\mathrm{eff}} (x10^{-7}a.u.$)')
ax.legend(title ='$\Omega_R$ (cm$^{-1}$)', loc=1, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
ax.set_xticks([1100, 1200, 1300])

plt.savefig('./Fig_2a.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# FIG 2b. 
#====================================
points = 200
rates = np.zeros(points)
nwc = np.linspace(500,2000,points) * cmtoau # Cavity frequency scan values

IR_data = np.loadtxt('./IR_data_wQ1189_RPV.txt')
k_35 = np.loadtxt('k_wc_scan_etac_0.0035.txt')
ηc = 0.0035
wQ = 1187.7 * cmtoau
for i in range(points):
    wc = nwc[i]
    rates[i] = (K_VSC(w,wc,beta)) * 0.7 + kD

fig, ax1 = plt.subplots(figsize = (4.5,4.5))
color = 'tab:red'
ax1.set_xlabel('$\omega$ (cm$^{-1}$)')
ax1.set_ylabel('Intensity', color=color, fontsize = 22)
ax1.plot(IR_data[:,0], IR_data[:,1], color=color, lw = 3, linestyle = '-',label = 'IR spectra', zorder = 20)
ax1.invert_yaxis()
ax1.set_yticks([0.0,0.5,1.0])
ax1.set_xticks([600, 800, 1000, 1200, 1400, 1600, 1800])
ax1.set_xticks([600, 900, 1200, 1500, 1800])
ax1.fill_between(IR_data[:,0], 0, IR_data[:,1], color = color, alpha = 0.5)

ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(labelcolor='linecolor', loc = 3, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('$k/k_0$', color=color, fontsize = 22)  # we already handled the x-label with ax1
ax2.plot(k_35[:,0], k_35[:,1]/k0, color=color, lw = 3, linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '5', label = 'HEOM', zorder = 10)
ax2.plot(nwc/cmtoau, rates/k0, lw = 3, alpha = 0.7, label = 'FGR', zorder = 5)
ax2.legend(labelcolor='linecolor', loc = 4, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
ax2.tick_params(axis='y', labelcolor=color)
plt.xlim(600,1800)

ax1.set_zorder(1)
ax1.set_facecolor("none")
ax2.set_zorder(2)
ax2.set_facecolor("none")

plt.savefig("./Fig_2b.pdf", dpi = 500, bbox_inches='tight')
plt.close()


#====================================
# FIG 2c. 
#====================================
points = 200
rates = np.zeros(points)
nwc = np.linspace(500,2000,points) * cmtoau # Cavity frequency scan values

# HEOM data
#====================================
k_25 = np.loadtxt('./k_wc_scan_etac_0.0025.txt')
k_35 = np.loadtxt('./k_wc_scan_etac_0.0035.txt')
k_5 = np.loadtxt('./k_wc_scan_etac_0.005.txt')

fig, ax = plt.subplots(figsize = (4.5,4.5))
color = ['black','blue', 'green', 'red'] 
etas = [0, 0.0025, 0.0035, 0.005]           # Light - matter coupling scan values

ΩR = [int(2*wc/(2*wQ)**0.5 * x/cmtoau) for x in etas]
for n in range(len(etas)):
    ηc = float(etas[n])
    for i in range(points):
        wc = nwc[i]
        rates[i] = (K_VSC(w,wc,beta)) * 0.7 + kD
    ax.plot(nwc/cmtoau, (rates/k0) , ls = '-', lw = 3, c = f'{color[n]}', alpha = 0.6, label = f'{ΩR[n]}')

ax.axvline(wQ/cmtoau, ls = '--', c = 'black', lw = 2)
ax.plot(k_25[:,0],k_25/k0, c = color[1], linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '6')
ax.plot(k_35[:,0],k_35/k0, c = color[2], linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '6')
ax.plot(k_5[:,0],k_5/k0, c = color[3], linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '6')

ax.set_xlim(700,1700)
ax.set_ylim(0.76, 1.01)
ax.set_xlabel('$\omega_c$ (cm$^{-1}$)', fontsize = 22)
ax.set_ylabel('$k/k_0$', fontsize = 22)

ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])


ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '', marker = 'o',label='HEOM', c='black')
ax2.plot(np.NaN, np.NaN, ls= '-',label='FGR', c='black')
ax2.get_yaxis().set_visible(False)
ax.legend(title ='$\Omega_R$ (cm$^{-1}$)', loc=0, frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
ax2.legend(loc=4, frameon = False, fontsize = 14)
plt.savefig('./Fig_2c.pdf', dpi = 500, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.close()
