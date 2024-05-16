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

#====================================
# FUNCTIONS
#====================================
def DW(R, ωDW, E_b, m = 1836):
    Vx = -(m * ωDW**2 / 2) * R**2 + (m**2 * ωDW**4 / (16 * E_b)) * R**4 
    return  Vx - np.min(Vx)

def T(N,Δx,m=1836):
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        A[i, i] = np.pi**2 / 3
        for j in range(1, N - i):
            A[i, i+j] = 2 * (-1)**j / j**2
            A[i+j, i] = A[i, i+j]
    A = A / (2 * m * Δx**2)
    return A

def DVR(N,Δx,Pot):
    H = np.diag(Pot) + T(N,Δx,m=1836)
    EgEn, EgVe = np.linalg.eigh(H)
    return EgEn, EgVe 

def exp_decay(x, Γ):
    PE = 0.5 # Population at equilibrium
    PI = 0.99669031 # Initial population
    return (PI - PE) * np.exp(-x * Γ) + PE
#====================================

#====================================
# Fig. 4a
#====================================
Eb = 2300 * cmtoau
wDW = 1500 * cmtoau
xgrid = 1001
Lx = 3
x = np.linspace(-Lx,Lx,xgrid)
Δx = x[1] - x[0]
Vx = DW(x,wDW,Eb) 
nDW = 4

EDW, VDW = DVR(xgrid,Δx,Vx)
Normx = np.trapz(VDW[:,0].conjugate() * VDW[:,0],x,Δx)
VDW = VDW/(Normx)**0.5
VDW = -VDW

diabats = np.zeros((xgrid,nDW))
diabats[:,0] = -(VDW[:,0] - VDW[:,1])/2**0.5
diabats[:,1] = -(VDW[:,0] + VDW[:,1])/2**0.5
diabats[:,2] = (VDW[:,2])
diabats[:,3] = (VDW[:,3])

E0 = (EDW[0]+EDW[1])/2
E_diabats = np.zeros((nDW))
E_diabats[0] = E_diabats[1] = (EDW[0]+EDW[1])/2 
E_diabats[2] = EDW[2]
E_diabats[3] = EDW[3]


fig, ax = plt.subplots(figsize = (4.5,4.5))

ax.plot(x,diabats[:,0]*3E2 + (E_diabats[0]-E0)/cmtoau, lw = 3, label = r'$|ν_L \rangle$')
ax.fill_between(x,diabats[:,0]*3E2 + (E_diabats[0]-E0)/cmtoau, y2 = (E_diabats[0]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,diabats[:,1]*3E2 + (E_diabats[1]-E0)/cmtoau, lw = 3, label = r'$|ν_R \rangle$')
ax.fill_between(x,diabats[:,1]*3E2 + (E_diabats[1]-E0)/cmtoau, y2 = (E_diabats[1]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,diabats[:,2]*3E2 + (E_diabats[2]-E0)/cmtoau, lw = 3, label = r"$|\nu_2 \rangle$")
ax.fill_between(x,diabats[:,2]*3E2 + (E_diabats[2]-E0)/cmtoau, y2 = (E_diabats[2]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,diabats[:,3]*3E2 + (E_diabats[3]-E0)/cmtoau, lw = 3, label = r"$|\nu_3 \rangle$")
ax.fill_between(x,diabats[:,3]*3E2 + (E_diabats[3]-E0)/cmtoau, y2 = (E_diabats[3]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,(DW(x,wDW,Eb)-E0)/cmtoau, c = 'black', lw = 4)
ax.set_ylim(-E0/cmtoau -50,2500)
ax.set_xlim(-1.8,1.8)

ax.text(-1.75,110,r"$|ν_L \rangle$", fontsize = 22)
ax.text(-1.75,1100,r"$|ν_2 \rangle$", fontsize = 22)
ax.text(1.2,110,r"$|ν_R \rangle$", fontsize = 22)
ax.text(-1.75,1670,r"$|ν_3 \rangle$", fontsize = 22)
ax.set_xlabel('$R_0$ (a.u.)', fontsize = 22)
ax.set_ylabel('Energy (cm$^{-1}$)', fontsize = 22)

plt.savefig('./Fig_4a.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 4b
#====================================

# Outside cavity data
#====================================
data_out = np.loadtxt('./rates_outside_cav.txt')
time_out = data_out[:,0]
pop_out = data_out[:,1]
param, cov = curve_fit(exp_decay, time_out, pop_out, p0 = [6E-4])
k0 = param[0]

# Cavity Freq. scan
#====================================
data_wc = np.array(np.loadtxt('./rates_wc_scan.txt'))
time_wc = data_wc[:,0]
pop_wc = data_wc[:,1:]
freq = [1100, 1150, 1200, 1250, 1300, 1350, 1386, 1400, 1450, 1500, 1550, 1600]
points = len(freq)
rates_wc = np.zeros(points)

for k in range(len(freq)):
    parameters, covariance = curve_fit(exp_decay, time_wc, pop_wc[:,k], p0 = [6E-4])
    rates_wc[k] = parameters[0]

fig, ax = plt.subplots(figsize = (4.5,1))
ax.plot(freq,rates_wc/k0, ls = '-', marker = 'o', alpha = 0.7)
ax.set_ylim(0.97, 1.03)
ax.set_xlabel('$\omega_c$ (cm$^{-1}$)', fontsize = 22)
ax.set_ylabel('$k/k_0$', fontsize = 22)
ax.set_yticks([1])
ax.set_xticks([1150, 1300, 1450, 1600])
plt.savefig('./Fig_4b.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 4c
#====================================
wQ = 1385.98 * cmtoau
wc = wQ 
etas = np.array([0.001, 0.0015, 0.002, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0055])
ΩR = 2*wc/(2*wQ)**0.5 * np.array(etas)

data_ηc = np.array(np.loadtxt('./rates_etac_scan.txt'))
time_ηc = data_ηc[0:,0]
pop_ηc = data_ηc[0:,1:]
points = len(etas)
rates_ηc = np.zeros(points)

for k in range(points):
    parameters, covariance = curve_fit(exp_decay, time_ηc, pop_ηc[:,k], p0 = [6E-4])
    rates_ηc[k] = parameters[0]

fig, ax = plt.subplots(figsize = (4.5,1))#, dpi = 500)
ax.plot(ΩR/cmtoau, rates_ηc/k0, ls= '-', marker = 'o', alpha = 0.7)
ax.set_ylim(0.97, 1.03)
ax.set_xlabel('$\Omega_R$ (cm$^{-1}$)', fontsize = 22)
ax.set_ylabel('$k/k_0$', fontsize = 22)
ax.set_yticks([1])
ax.set_xticks([25, 50, 75, 100, 125])

plt.xlabel('$\Omega_R$ (cm$^{-1}$)', fontsize = 22)
plt.ylabel('$k/k_0$', fontsize = 22)
plt.savefig('./Fig_4c.pdf', dpi = 500, bbox_inches='tight')
plt.close()