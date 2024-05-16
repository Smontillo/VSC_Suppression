import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 19})
plt.rcParams.update({'font.family': "times"})
cmtoau = 4.556335e-06 
#====================================

#====================================
# FUNCTIONS
#====================================
def DW(R, ωDW, E_b, m = 1836):
    Vx = -(m * ωDW**2 / 2) * R**2 + (m**2 * ωDW**4 / (16 * E_b)) * R**4 
    return  Vx - np.min(Vx)

def HO(R, ωHO, ms):
    return ms * ωHO**2 * R**2/2  

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

#====================================
# Fig. 1b
#====================================
xgrid = 1001
Lx = 60
x = np.linspace(-Lx,Lx,xgrid) 
ws = 1189 * cmtoau
En_Osc = (0 + 1/2) * 1189#E0/cmtoau
HO = ws**2*x**2/2
x_range = [13.6, 23.5, 30.7]

fig, ax = plt.subplots(figsize = (3.5,4.5))#, dpi = 500)

ax.hlines((0 + 1/2) * 1189 - En_Osc, -x_range[0], x_range[0], color = 'cornflowerblue', lw = 3, alpha = 0.8)
ax.hlines((1 + 1/2) * 1189 - En_Osc, -x_range[1], x_range[1], color = 'cornflowerblue', lw = 3, alpha = 0.6)
ax.hlines((2 + 1/2) * 1189 - En_Osc, -x_range[2], x_range[2], color = 'cornflowerblue', lw = 3, alpha = 0.4)
ax.plot(x,(HO/cmtoau-En_Osc), lw = 4, c = 'royalblue', alpha = 1)
ax.set_xlim(-65,65)
ax.set_ylim(-750,1800)
ax.set_xticks([-50,0,50])
ax.tick_params(left = True , labelleft = False) 
ax.tick_params(right = True , labelright = False) 
ax.set_xlabel(r'$\mathcal{Q}$ (a.u.)')
ax.text(30,1290,r"$\omega_Q$", fontsize = 40)
plt.ylabel('Energy ($cm^{-1}$)')
plt.savefig('./Fig_1b.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 1c
#====================================
xgrid = 1001
Lx = 60
x = np.linspace(-Lx,Lx,xgrid) 
ws = 1189 * cmtoau
En_Osc = (0 + 1/2) * 1189#E0/cmtoau
HO = ws**2*x**2/2
x_range = [13.6, 23.5, 30.7]

fig, ax = plt.subplots(figsize = (3.5,4.5))#, dpi = 500)

ax.hlines((0 + 1/2) * 1189 - En_Osc, -x_range[0], x_range[0], color = 'blueviolet', lw = 3, alpha = 0.8)
ax.hlines((1 + 1/2) * 1189 - En_Osc, -x_range[1], x_range[1], color = 'blueviolet', lw = 3, alpha = 0.6)
ax.hlines((2 + 1/2) * 1189 - En_Osc, -x_range[2], x_range[2], color = 'blueviolet', lw = 3, alpha = 0.4)
ax.plot(x,(HO/cmtoau-En_Osc), lw = 4, c = 'darkviolet', alpha = 1)
ax.set_xlim(-65,65)
ax.set_ylim(-750,1800)
ax.set_xticks([-50,0,50])
ax.tick_params(left = True , labelleft = False) 
ax.tick_params(right = True , labelright = False) 
ax.set_xlabel(r'$\mathcal{Q}$ (a.u.)')
ax.text(30,1290,r"$\omega_Q$", fontsize = 40)
plt.ylabel('Energy ($cm^{-1}$)')
plt.savefig('./Fig_1c.pdf', dpi = 500, bbox_inches='tight')
plt.close()

#====================================
# Fig. 1d
#====================================
Eb = 2250 * cmtoau
wDW = 1000*cmtoau
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

dia = np.zeros((xgrid,nDW))
dia[:,0] = (VDW[:,0] + VDW[:,1])/2**0.5
dia[:,1] = (VDW[:,0] - VDW[:,1])/2**0.5
dia[:,2] = -(VDW[:,2] + VDW[:,3])/2**0.5
dia[:,3] = -(VDW[:,2] - VDW[:,3])/2**0.5

E0 = (EDW[0]+EDW[1])/2
Edia = np.zeros((nDW))
Edia[0] = Edia[1] = (EDW[0]+EDW[1])/2 
Edia[2] = Edia[3] = (EDW[2]+EDW[3])/2 

fig, ax = plt.subplots(figsize = (6.5,4.5))
ax.plot(x,dia[:,1]*3E2 + (Edia[1]-E0)/cmtoau, lw = 3, label = r'$|ν_R \rangle$')
ax.fill_between(x,dia[:,1]*3E2 + (Edia[1]-E0)/cmtoau, y2 = (Edia[1]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,dia[:,0]*3E2 + (Edia[0]-E0)/cmtoau, lw = 3, label = r'$|ν_L \rangle$')
ax.fill_between(x,dia[:,0]*3E2 + (Edia[0]-E0)/cmtoau, y2 = (Edia[0]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,dia[:,2]*3E2 + (Edia[2]-E0)/cmtoau, lw = 3, label = r"$|\tilde{ν}_L^ \rangle$")
ax.fill_between(x,dia[:,2]*3E2 + (Edia[2]-E0)/cmtoau, y2 = (Edia[2]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,dia[:,3]*3E2 + (Edia[3]-E0)/cmtoau, lw = 3, label = r"$|\tilde{ν}_R^ \rangle$")
ax.fill_between(x,dia[:,3]*3E2 + (Edia[3]-E0)/cmtoau, y2 = (Edia[3]-E0)/cmtoau,  alpha = 0.3)
ax.plot(x,(DW(x,wDW,Eb)-E0)/cmtoau, c = 'black', lw = 4)
ax.set_ylim(-750,1800)
ax.set_xlim(-2.2,2.2)
ax.text(-2.1,110,r"$|ν_L \rangle$", fontsize = 22)
ax.text(-2.1,970,r"$|ν'_L \rangle$", fontsize = 22)
ax.text(1.6,110,r"$|ν_R \rangle$", fontsize = 22)
ax.text(1.6,970,r"$|ν'_R \rangle$", fontsize = 22)
plt.xlabel('$R_0$ (a.u.)')
plt.ylabel('Energy (cm$^{-1}$)')
plt.savefig('./Fig_1d.pdf', dpi = 500, bbox_inches='tight')
plt.close()