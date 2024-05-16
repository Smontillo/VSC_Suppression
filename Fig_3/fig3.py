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
    ΓQ = 2 * λQ / γQ + (2 * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * wc * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return cQ**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )

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
cQ = 4.7 * cmtoau/(1836**0.5)         # Qj - R0 coupling
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
# Fig. 3b
#====================================
w = np.arange(1E-5,5000,0.05) * cmtoau
wc = 1189.7 * cmtoau
# Temperature variation of rate from HEOM
#====================================
k0 = np.loadtxt('./temp_scan_rate_k0.txt') 
kD = np.loadtxt('./temp_scan_rate_kD.txt')
k_25 = np.loadtxt('./temp_scan_rate_eta_25.txt')
k_35 = np.loadtxt('./temp_scan_rate_eta_35.txt')
k_5 = np.loadtxt('./temp_scan_rate_eta_5.txt')

T = k0[:,0]  # Temperature scan values

# Eyring type plot
#====================================
etas_T = [0, 0.0025, 0.0035, 0.005]   # Light-matter coupling values
ΩR_FGR_T = [int(2*wc/(2*wQ)**0.5 * x/cmtoau) for x in etas_T]

# Temperature dependent rates from HEOM and FGR at various light - matter coupling values
kT_HEOM = np.zeros((len(k0[:,0]),4))
kT_HEOM[:,0], kT_HEOM[:,1], kT_HEOM[:,2], kT_HEOM[:,3] = k0[:,1], k_25[:,1], k_35[:,1], k_5[:,1]

kT_FGR = np.zeros((len(T), len(etas_T)))
for k in range(len(T)):
    temp = T[k] / autoK 
    beta = 1/temp
    for i in range(len(etas_T)):
        ηc = float(etas_T[i])
        kT_FGR[k,i] = ((K_VSC(w,wc,beta)) * 0.7 + kD[k,1])

# ln(k/T) vs 1/T
Eyring_HEOM = np.array([[np.log(kT_HEOM[n,k]/T[n] * autoK) for k in range(len(etas_T))] for n in range(len(T))])
Eyring_FGR = np.array([[np.log(kT_FGR[n,k]/T[n] * autoK) for k in range(len(etas_T))] for n in range(len(T))])

# Linear fitting and extraction of slope and intercept
fit_param_HEOM = np.zeros((len(etas_T),2))
fit_param_FGR = np.zeros((len(etas_T),2))

fit_param_HEOM[:,0] = [curve_fit(lineal,1/T * autoK,Eyring_HEOM[:,k])[0][0] for k in range(len(etas_T))]  # Slope
fit_param_HEOM[:,1] = [curve_fit(lineal,1/T * autoK,Eyring_HEOM[:,k])[0][1] for k in range(len(etas_T))]  # Intercept
fit_param_FGR[:,0] = [curve_fit(lineal,1/T * autoK,Eyring_FGR[:,k])[0][0] for k in range(len(etas_T))]    # Slope
fit_param_FGR[:,1] = [curve_fit(lineal,1/T * autoK,Eyring_FGR[:,k])[0][1] for k in range(len(etas_T))]    # Intercept

# Eyring eq. plotting
colors = ['black', 'blue', 'green', 'red']
fig, ax = plt.subplots(figsize = (4.5,4.5))
for k in range(len(etas_T)):
    ax.plot(1/T,lineal(1/T * autoK,fit_param_HEOM[k,0],fit_param_HEOM[k,1]), c = f'{colors[k]}', lw = 1)
    ax.plot(1/T, Eyring_HEOM[:,k], label = f'{ΩR_FGR_T[k]}', c = f'{colors[k]}', linestyle = '', marker = 'o')
    ax.plot(1/T, Eyring_FGR[:,k], c = f'{colors[k]}', linestyle = '', marker = 'o', markersize = 10, fillstyle = 'none')

plt.ylim(-9.68, -8.95)
ax.set_xlabel('$1/T (K^{-1}$)', fontsize = 22)
ax.set_ylabel("$ln(k/T)$", fontsize = 22)
ax.set_xticks([0.0031,0.0033,0.0035])

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '', marker = 'o',label='HEOM', c='black')
ax2.plot(np.NaN, np.NaN,label='FGR', c='black', linestyle = '', marker = 'o', fillstyle = 'none')
ax2.get_yaxis().set_visible(False)
ax.legend(title ='$\Omega_R$ (cm$^{-1}$)', loc=0, frameon = False, fontsize = 15, handlelength=0.1, title_fontsize = 15, labelspacing = 0.2)
ax2.legend(loc=3, frameon = False, fontsize = 15, handletextpad=0.01)

plt.savefig('./Fig_3b.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 3a
#====================================

#====================================
# CONSTRUCTION OF DD(G) DATA 
#====================================
# Rates at diff temp, for the diff couplings
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
Ts = np.array([280, 290, 300, 310, 320, 330])/ autoK  # Temperatures
kD = np.loadtxt('bare_DW/fs.txt')                     # Bare DW rates
k0 = np.loadtxt('eta_0/fs.txt')                       # DW - RPV rates
coup =  ['01', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65']# Couplings
rates_HEOM_T = np.zeros((len(Ts), len(coup)))
for j in range(len(coup)):
    var = np.loadtxt(f'eta_{coup[j]}/fs.txt') # Temporal variable
    rates_HEOM_T[:,j] = var[:,1]

ΩR_HEOM_T = np.zeros(len(coup))
for h in range(len(coup)):
    ηc = float(coup[h]) * 1E-4
    ΩR_HEOM_T[h] = (2/wQ)**0.5 * wc * ηc /cmtoau
ηc_FGR = np.linspace(1E-5,0.0065,50)
ΩR_FGR_T = (2/wQ)**0.5 * wc * ηc_FGR /cmtoau

Eyring_FGR = np.zeros((len(Ts), len(ηc_FGR)))
slope_FGR = np.zeros(len(ηc_FGR))
intercept_FGR = np.zeros(len(ηc_FGR))

Eyring_HEOM = np.zeros((len(Ts), len(coup)))
slope_HEOM = np.zeros(len(coup))
intercept_HEOM = np.zeros(len(coup))

for i in range(len(ηc_FGR)):
    for n in range(len(Ts)):
        temp = Ts[n] 
        beta = 1/temp
        ηc = float(ηc_FGR[i]) 
        Eyring_FGR[n,i] = np.log(((K_VSC(w,wc,beta)) * 0.7 + kD[n,1])/Ts[n]) # FGR Eyring plot

    # FGR linear fitting
    popt, pcov = curve_fit(lineal,1/(Ts * autoK),Eyring_FGR[:,i])
    slope_FGR[i] = popt[0]
    intercept_FGR[i] = popt[1]

    # HEOM linear fitting
for i in range(len(coup)):
    Eyring_HEOM[:,i] = np.log(rates_HEOM_T[:,i]/Ts) # HEOM Eyring plot
    popt, pcov = curve_fit(lineal,1/(Ts * autoK),Eyring_HEOM[:,i])
    slope_HEOM[i] = popt[0]
    intercept_HEOM[i] = popt[1]

dH_FGR = - slope_FGR * R
dS_FGR = (intercept_FGR - np.log(kB/h)) * R
dG_FGR = (dH_FGR - dH_FGR[0]) - (dS_FGR - dS_FGR[0]) * 300

dH_HEOM = - slope_HEOM * R
dS_HEOM = (intercept_HEOM - np.log(kB/h)) * R
dG_HEOM = (dH_HEOM - dH_HEOM[0]) - (dS_HEOM - dS_HEOM[0]) * 300


# RATE AS A FUNCTION OF RABI SPLITTING
#====================================
temp = 300 / autoK 
beta = 1/temp
kD = 6.205e-08                        # Reaction coordinate + bath rate (No RPV coupling)
k0 = 9.077e-08                        # Outside cavity rate
data_HEOM = np.loadtxt('./rates_HEOM_Rabi_scan.txt')
points = 200
rates_FGR = np.zeros(points)
rates_HEOM = data_HEOM[:,1]
etac = np.linspace(0.0015,0.0065,points)               # Light matter coupling scan values

ΩR_HEOM = 2 * wc / (2 * wQ)**0.5 * data_HEOM[:,0]
ΩR_FGR = 2 * wc / (2 * wQ)**0.5 * etac

for i in range(points):
    ηc = float(etac[i])
    rates_FGR[i] = (K_VSC(w,wc,beta)) * 0.7 + kD

# PLOTTING
#====================================
fig, ax1 = plt.subplots(figsize = (4.5,4.5))
color = 'tab:blue'
ax1.set_xlabel('$\Omega_R$ (cm$^{-1}$)')
ax1.set_ylabel('$k/k_0$', color=color, fontsize = 22) 
ax1.plot(np.array(ΩR_FGR)/cmtoau, rates_FGR/k0 , ls = '-', lw = 3, c = color, alpha = 0.5)
ax1.plot(np.array(ΩR_HEOM[3:])/cmtoau, rates_HEOM[3:]/k0, c = color, linestyle = '', marker = 'o', fillstyle= 'full', markersize = '6')
ax1.set_yticks([0.8,0.9, 1.0])

ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.73,1.02)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('$\Delta(\Delta G^\u2021$) (kJ/mol)', color=color, fontsize = 22)  # we already handled the x-label with ax1
ax2.plot(ΩR_FGR_T[11:], dG_FGR[11:] , ls = '-', lw = 3, c = color, alpha = 0.5)
ax2.plot(ΩR_HEOM_T[3:], dG_HEOM[3:], c = color, linestyle = '', marker = 'o', fillstyle= 'full', markersize = '6')

ax2.set_yticks([0.0,0.25,0.50, 0.75])
ax2.set_xticks([30, 60, 90, 120, 150])
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
ax3.plot(np.NaN, np.NaN, ls= '', marker = 'o',label='HEOM', c='black')
ax3.plot(np.NaN, np.NaN, ls= '-',label='FGR', c='black')
ax3.get_yaxis().set_visible(False)
ax3.legend(loc=7, frameon = False, fontsize = 14)

plt.savefig('./Fig_3a.pdf', dpi = 500, bbox_inches='tight')
plt.close()