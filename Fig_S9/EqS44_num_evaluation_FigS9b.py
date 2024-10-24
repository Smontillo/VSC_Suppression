#!/software/anaconda3/2020.11/bin/python
#SBATCH -p preempt
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --job-name=sbatcharray
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1-00:00:00
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err

import sys, os
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
import numpy as np
import matplotlib.pyplot as plt
from random import random
import numba as nb
import time
import itertools 
from mpi4py import MPI

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array


# ================= global ====================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fstoau = 41.341374575751                  # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

wQ = 1189.7 * cmtoau
wc = 1189.7 * cmtoau
Nq = 1000
print('=============================')
# =========================
# Parallelization
# =========================
nrank = int(TASKID)                        # kind of JOD ID for a job
size = 100                       # total number of processors available

# =========================
#    FUNCTIONS
# =========================
@nb.jit(nopython=True, fastmath=True)
def J_spec(w,lambda_, gamma):
    return 2*lambda_*gamma*w/(w**2 + gamma**2)

# FGR rate
@nb.jit(nopython=True, fastmath=True)
def k_FGR(w,beta,J):
    n = 1/(np.exp(beta * w)-1)
    k = 2 * Δx**2 * J * n
    return k

# Sigma squared
@nb.jit(nopython=True, fastmath=True)
def sigma(w):
    w_cut = np.arange(1E-15,γv,0.001*cmtoau)
    dw_cut = w_cut[1] - w_cut[0]
    f = J_spec(w_cut,λv,γv) * np.cosh(beta*w_cut/2)/np.sinh(beta*w_cut/2)
    return εz**2/np.pi * np.trapz(f,w_cut,dw_cut)

# Gaussian function
@nb.jit(nopython=True, fastmath=True)
def Kernel(w):
    l = 1/np.pi * σ2**0.5/((w-w0)**2 + σ2)
    return l

# Convolution k_FGR @ Gaussian
@nb.jit(nopython=True, fastmath=True)
def K_VSC(w,beta,J):
    k = np.real(k_FGR(w,beta,J))
    G = np.real(Kernel(w)) 
    dw = w[1] - w[0]
    return np.trapz(k*G,dx = dw) 

@nb.jit(nopython=True, fastmath=True)
def Jeff_f(w,wc,ηc):
    CQ = 5E-7/Nq**0.5
    Γ = 1/τc
    ΓQ = 2 * λQ / γQ + (2 * Nq * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * wc * Nq * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return Nq*CQ**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )

@nb.jit(nopython=True, fastmath=True)
def P(x):
    return - x**2 + 2 * lam_Q * x / (1.0j * gam_Q)

@nb.jit(nopython=True, fastmath=True)
def L(x):
    return - x**2 - 1.0j * alpha * x

@nb.jit(nopython=True, fastmath=True)
def Psi(x):
    return wc**2 * L(x) / (wc**2 + L(x))

@nb.jit(nopython=True, fastmath=True)
def dev_scan(std_comb, TaskArray, nrank, omega_s, Psis, Ps, Nq, WQ, dw):
    task_comb = TaskArray
    comb_len = len(task_comb)
    rates = np.zeros((len(Rabis),comb_len))
    # RPV - Rxn distribution 
    for n in range(comb_len):
        print('Job number ', TaskArray[n], 'of rank ', nrank)
        stdQ = std_comb[TaskArray[n],0] * (cQ)
        dist = np.random.normal(cQ, stdQ, Nq)
        alp = (cQ/np.sum(dist) * Nq**0.5)
        Cq = dist.astype(np.complex64)*alp
    # RPV - cavity distribution 
        for r in range(len(Rabis)):
            etac = Rabis[r] / np.sqrt(2 * wc * Nq) 
            v = np.sqrt(2 / wc) * etac * np.ones((Nq), dtype = np.complex64)
            Mat_in = np.outer(v,v)
            Jeff = np.zeros(len(omega_s))
        
        # Creation of the Jeff
            for wi in range(len(omega_s)):
                Mat = Mat_in * Psis[wi] + np.diag(WQ + Ps[wi]) 
                Mat_inv = np.linalg.inv(Mat)
                K = (Cq.T @ Mat_inv @ Cq)
                Jeff[wi] = np.real(np.imag(K))
        # Rate constant calculation
            reorg = np.real(np.trapz(Jeff/omega_s,omega_s, dw))
            Jeff *= reorg_an/reorg
            rates[r,n] = (K_VSC(np.real(omega_s),beta,np.real(Jeff)) * 0.7 + kD)/k0
    return rates

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
# =========================

# Q_mode
# =========================
mQ = 1
wQ = 1189.7 * cmtoau 
γQ = 6000 * cmtoau 
λQ = 6.70E-7 
# =========================

# Cavity
# =========================
τc = 500 * fstoau
kD = 6.204982961140102545e-08 # Rate constant of DW without Q at eta_s 0.1
k0 = 9.076763853644350798e-08 # Q + Rxn rates

cQ = 4.7 * cmtoau/(1836**0.5) #1.48 * cmtoau # / np.sqrt(Nq)
lam_Q = 0.147 * cmtoau
gam_Q = 6000 * cmtoau
alpha = 1. / (500 * fstoau)
reorg_an = 6.582079423930413e-09
Rabis = np.linspace(0,150,50) * cmtoau
#====================================
#   COUPLING DEVIATIONS
#====================================
# Compilation
N_dummy = 1
omega_dummy = np.array(np.arange(1, 1.2, 0.1) * cmtoau, dtype = np.complex64)
dw_dummy = omega_dummy[1] - omega_dummy[0]
σ2 = sigma(omega_dummy)
WQ_dummy = wQ**2 * np.ones(N_dummy, dtype = np.complex64)
Psis_dummy = Psi(omega_dummy)
Ps_dummy = P(omega_dummy)

std_cQ_dummy = np.linspace(0,3,1)
std_dip_dummy = np.linspace(0,1,1)
std_comb_dummy = np.array(list(itertools.product(std_cQ_dummy, std_dip_dummy)))
Ntasks_dummy = 1
nrank_dummy = 1
Task_dummy = np.array([0])
ini_time = time.time()
rates = dev_scan(std_comb_dummy, Task_dummy, nrank_dummy, omega_dummy,Psis_dummy,Ps_dummy, N_dummy, WQ_dummy, dw_dummy)
fin_time = time.time()
print("Time = ",fin_time - ini_time)
#====================================
# Running
#====================================
omega_s = np.array(np.arange(1100, 1280, 0.8) * cmtoau, dtype = np.complex64)
dw = omega_s[1] - omega_s[0]
σ2 = sigma(omega_s)
WQ = wQ**2 * np.ones(Nq, dtype = np.complex64)
Psis = Psi(omega_s)
Ps = P(omega_s)
#====================================
std_cQ = np.array([0.1, 0.5, 1, 2])

tot_Tasks = len(std_cQ)
NTasks = tot_Tasks//size
NRem = tot_Tasks - (NTasks*size)
TaskArray = [i for i in range(nrank * NTasks , (nrank+1) * NTasks)]
for i in range(NRem):
    if i == nrank: 
        TaskArray.append((NTasks*size)+i)
TaskArray = np.array(TaskArray)
ini_time = time.time()
Jeff = dev_scan(std_cQ, TaskArray, nrank, omega_s, Psis, Ps, Nq, WQ, dw)
fin_time = time.time()
print("Time again = ",fin_time - ini_time)

np.savetxt(f'./rate_{nrank}.txt', np.c_[Rabis,rates])
