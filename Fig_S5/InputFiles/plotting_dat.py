#!/software/anaconda3/2020.11/bin/python
#SBATCH -p gpu-debug
#SBATCH --job-name=plot   # create a name for your job
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=5G         # memory per cpu-core
#SBATCH -t 1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err

import numpy as np
import matplotlib.pyplot as plt
from gen_input import parameters, DW
from scipy.optimize import curve_fit

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcaltoau = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.
# ==============================================================================================

x = parameters.x
EDW = parameters.EDW
m = parameters.mDW
wDW = parameters.wDW
Nstates = parameters.Nstates
VDW = parameters.VDW
nc = parameters.nHO
nDW = parameters.nDW
mDW = 1
print(nc, nDW)

def rate_func(X, kf, kb):
    x1, x2 = X
    return kf * x1 - kb * x2 

def modify_rho(rhot):
    mod_rho = rhot[:,1:]
    return mod_rho

def diats(rhot):
    rho_DW = np.zeros((points,nDW+1))
    
    for line in range(points):
        c3, c4 = 0.96886815, 0.2475773
        cp3, cp4 = -0.24751267,  0.96888496
        rho_line = rhot[line,:]
        rho = rho_line.reshape(nc*nDW,nc*nDW)
        rho_tensor= rho.reshape([nc, nDW, nc, nDW])

        # Get the DW populations
        trace = np.trace(rho_tensor, axis1=0, axis2=2)

        rho_DW[line,0] = np.real(trace[0,0])
        rho_DW[line,1] = np.real(trace[1,1])
        rho_DW[line,2] = np.real(trace[2,2])
        rho_DW[line,3] = np.real(c3**2 * trace[3,3] + c3 * c4 * trace[3,4] + c3 * c4 * trace[4,3] + c4**2 * trace[4,4])
        rho_DW[line,4] = np.real(cp3**2 * trace[3,3] + cp3 * cp4 * trace[3,4] + cp3 * cp4 * trace[4,3] + cp4**2 * trace[4,4])
        rho_DW[line,5] = np.real(trace[5,5])
        rho_DW[line,-1] = np.real(np.trace(trace))

    return rho_DW


def get_left_overlap(state1, state2):
    P = 0.0
    dx = x[1] - x[0]
    center = int(len(VDW[:, 0])/2) +1
    dat = VDW[:center, state1].conjugate() * VDW[:center, state2]
    P += np.trapz(dat,x[:center],dx)
    return np.real(P)

def get_right_overlap(state1, state2):
    P = 0.0
    dx = x[1] - x[0]
    center = int(len(VDW[:, 0])/2) +1
    dat = VDW[center:, state1].conjugate() * VDW[center:, state2]
    P += np.trapz(dat,x[center:],dx)
    return np.real(P)

def left_pop(rhot):
    rho_DW = np.zeros((points,nDW,nDW), dtype = 'complex')
    
    for line in range(points):
        rho_line = rhot[line,:]
        rho = rho_line.reshape(nDW*nc,nDW*nc)
        rho_tensor= rho.reshape([nc, nDW, nc, nDW])

        # Get the DW populations
        rho_DW[line,:,:] = np.trace(rho_tensor, axis1=0, axis2=2)

    PRt = np.zeros(points, dtype = 'complex')
    for i in range(nDW):
        for j in range(nDW):
            PRt[:] += get_left_overlap(i,j) * rho_DW[:,i,j]
    return np.real(PRt)

def right_pop(rhot):
    rho_DW = np.zeros((points,nDW,nDW), dtype = 'complex')
    
    for line in range(points):
        rho_line = rhot[line,:]
        rho = rho_line.reshape(nDW*nc,nDW*nc)
        rho_tensor= rho.reshape([nc, nDW, nc, nDW])

        # Get the DW populations
        rho_DW[line,:,:] = np.trace(rho_tensor, axis1=0, axis2=2)

    PRt = np.zeros(points, dtype = 'complex')
    for i in range(nDW):
        for j in range(nDW):
            PRt[:] += get_right_overlap(i,j) * rho_DW[:,i,j]
    return np.real(PRt)

def fit_rate(time, rhot):
    points = len(time)
    pL = left_pop(rhot)
    pR = right_pop(rhot)
    print(pL[-1], 'pl')
    PL = np.zeros(points - 1)
    PR = np.zeros(points - 1)
    for k in range(1, points):
        PL[k-1] = np.trapz(pL[:k], time[:k])
        PR[k-1] = np.trapz(pR[:k], time[:k])
    X = (PL, PR)
    popt, pcov = curve_fit(rate_func, X, pR[1:], bounds = (0, [1E-7, 1E-7]))
    print('Forward rate ->', popt[0] * fstoau, ' fs^-1')
    return popt[0]

freq = [700, 800, 900, 1000, 1100, 1175, 1200, 1300, 1400, 1500, 1600]
kappas = np.zeros((len(freq), 1))
k0 = 3.72153865e-06
# ======================================== #

points = len(np.loadtxt(f'summary/{freq[0]}.dat')[:,0])
pops   = np.zeros((len(points), nDW))

# ==========================================================
# Vibrational States Populations
# ==========================================================
for n in range(len(freq)):
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    rhot = modify_rho(rhot)
    pops[:,n] = diats(rhot)

np.savetxt('./populations.dat', np.c_[time/(1000 * fstoau), pops])


# ==========================================================
# Reactant (left) and Product (right) Populations
# ==========================================================
points = len(np.loadtxt(f'summary/{freq[0]}.dat')[:,0])
popL   = np.zeros((len(points), nDW))
popR   = np.zeros((len(points), nDW))
for n in range(len(freq)):
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)
    popL[:,n] = left_pop(rhot)
    popR[:,n] = right_pop(rhot)

np.savetxt('R_P_populations.dat', np.c_[time/(1000 * fstoau), popL, popR])

# ==========================================================
# Asymmetrical Double Well Eigenstates, coordinates and potential
# ==========================================================
np.savetxt('./ADW_VDW.dat', np.c_[x, VDW[:,:6], DW(x,mDW,wDW)/cmtoau])