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
m = parameters.mDW
wDW = parameters.wDW
Nstates = parameters.Nstates
VDW = parameters.VDW
nc = parameters.nHO
nDW = parameters.nDW

print(nc, nDW)

plt.plot(x, VDW[:,0:5])
plt.savefig('images/wf.png')
plt.close()

def modify_rho(rhot):
    mod_rho = rhot[:,1:]
    return mod_rho

def traces(rhot):
    rho_DW = np.zeros((points,nDW+1))
    rho_c = np.zeros((points,nc+1))
    
    for line in range(points):
        rho_line = rhot[line,:]
        rho = rho_line.reshape(nc*nDW,nc*nDW)
        rho_tensor= rho.reshape([nc, nDW, nc, nDW])

        # Get the DW populations
        trace = np.trace(rho_tensor, axis1=0, axis2=2)
        for i in range(nDW):
            rho_DW[line,i] = np.real(trace[i,i])
        rho_DW[line,-1] = np.real(np.trace(trace))
    
    # Get the Cav populations
        trace = np.trace(rho_tensor, axis1=1, axis2=3)
        for i in range(nc):
            rho_c[line,i] = np.real(trace[i,i])
        rho_c[line,-1] = np.real(np.trace(trace))

    return rho_DW, rho_c

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

def rate(rhot, time):
    dt = time[1] - time[0]
    PRt = left_pop(rhot)
    rate = np.zeros((len(PRt) - 1), dtype = float)
    for n in range(1, len(rate)):
        dPr = ((PRt[n] - PRt[n-1])/dt)
        rate[n] = np.real(dPr/(1.0 - 2.0*PRt[n]))
    return np.real(rate), np.real(PRt)

def pop(rhot):
    rho_DW = np.zeros((points,nDW,nDW), dtype = 'complex')
    
    for line in range(points):
        rho_line = rhot[line,:]
        rho = rho_line.reshape(nc*nDW,nc*nDW)
        rho_tensor = rho.reshape([nc, nDW, nc, nDW])

        # Get the DW populations
        rho_DW[line,:,:] = np.trace(rho_tensor, axis1=0, axis2=2)

    PRt = np.zeros((points,nDW), dtype = 'complex')
    num = range(4)
    for i in num[:2]:
        for j in num[:2]:
            PRt[:,1] += get_right_overlap(i,j) * rho_DW[:,i,j]

    for i in num[2:]:
        for j in num[2:]:
            PRt[:,2] += get_left_overlap(i,j) * rho_DW[:,i,j]
            PRt[:,3] += get_right_overlap(i,j) * rho_DW[:,i,j]
    return np.real(PRt)

freq = [500, 700, 900, 1000, 1100, 1130, 1160, 1189, 1210, 1250, 1270, 1400, 1500, 1600]
freq = [200, 500, 1000, 1189, 1270, 1600]
kappas = np.zeros(len(freq))

# ======================================== #

for n in range(len(freq)):
    print(freq[n])
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rho_DW, rho_c = traces(rhot)
    print(np.shape(rho_c))
    plt.plot(time/fstoau,rho_c[:,0], label = f'{freq[n]}')

plt.legend()
plt.savefig(f'images/ground_SM', dpi=300, bbox_inches='tight')
plt.close()

for n in range(len(freq)):
    print(freq[n])
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rho_DW, rho_c = traces(rhot)
    plt.plot(time/fstoau,rho_c[:,1], label = f'{freq[n]}')

plt.ylim(0.006, 0.012)
plt.axhline(0.0081352831967567, ls = '--', c = 'black', lw = 0.5, label = 'Outside Cav')
plt.legend()
plt.savefig(f'images/first_SM', dpi=300, bbox_inches='tight')
plt.close()

for n in range(len(freq)):
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rho_DW, rho_c = traces(rhot)

    dia = pop(rhot)

    plt.plot(time/fstoau,dia[:,1], label = f'{freq[n]}')

plt.legend()
plt.savefig(f'images/ground_dia.png', dpi=300, bbox_inches='tight')
plt.close()
# ======================================== #
