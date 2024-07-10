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
from gen_input import parameters

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
nt = parameters.nt
nskip = parameters.nskip

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
    PRt[:,4] = rho_DW[:,4,4]
    return np.real(PRt)


freq = [700, 800, 900, 1000, 1050, 1100, 1130, 1160, 1189, 1220, 1250, 1300, 1350, 1400, 1500, 1600, 1700]
kappas = np.zeros(len(freq))

# ======================================== #
# Plot the population of the RPV mode
fig, axs = plt.subplots(nc-1, 1, sharex=True)
# Remove vertical space between axes
fig.subplots_adjust(hspace=0)
for n in range(len(freq)):
    print(freq[n])
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rho_DW, rho_c = traces(rhot)
    print(np.shape(rho_c))
    for i in range(nc-1):
        axs[i].plot(time/fstoau,rho_c[:,i+1], label = f'{freq[n]}')
plt.savefig(f'images/pop_c_Sts', dpi=300, bbox_inches='tight')
plt.close()

# Plot the populations of the DW diabats
fig, axs = plt.subplots(nDW-1, 1, sharex=True)
fig.subplots_adjust(hspace=0)
for n in range(len(freq)):
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rho_DW, rho_c = traces(rhot)

    dia = pop(rhot)

    for i in range(nDW-1):
        axs[i].plot(time/fstoau,dia[:,i+1], label = f'{freq[n]}')

plt.savefig(f'images/diabats', dpi=300, bbox_inches='tight')
plt.close()
# ======================================== #

# Get and plot the flux-side function
n_dat = nt/nskip
flux_side = np.zeros((n_dat,len(freq)))

for n in range(len(freq)):
    print(n)
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rho_DW, rho_c = traces(rhot)

    rates, left_P = rate(rhot, time)
    kappas[n] = rates[-1]
    flux_side[:,n] = rates[:n_dat]

    plt.plot(time[1:]/fstoau,rates*fstoau, label = f'{freq[n]}')
    plt.ylabel('$\kappa$')
    plt.xlabel('Steps')
plt.ylim(0.3E-5,0.4E-5)
plt.legend()
plt.savefig(f'images/kappa.png', dpi=300, bbox_inches='tight')
plt.close()

# Get and plot the rates
w = np.zeros(len(freq))
for i in range(len(freq)):
    w[i] = float(freq[i])
plt.scatter(w, kappas/kappas[0], c = 'r')
plt.xlim(700,1700)
plt.legend()
plt.savefig(f'images/Rates.png', dpi=300, bbox_inches='tight')
plt.close()

# Get and plot the left-side population of the DW 
for n in range(len(freq)):
    rhot = np.loadtxt(f'summary/{freq[n]}.dat')
    time = rhot[:,0]
    points = len(time)
    rhot = modify_rho(rhot)

    rates, left_P = rate(rhot, time)

    plt.plot(time/fstoau, left_P, label = f'{freq[n]}')
    plt.ylabel('Population')
    plt.xlabel('Time (fs)')
plt.legend()
plt.savefig(f'images/left.png', dpi=300, bbox_inches='tight')
plt.close()