#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --job-name=sbatcharray
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1-00:00:00
#SBATCH --output=job.out
#SBATCH --error=job.err

import sys, os
from scipy.signal import find_peaks
#-------------------------
import numpy as np
import matplotlib.pyplot as plt
from random import random
import numba as nb
import time
import itertools 
from mpi4py import MPI

# ================= global ====================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fstoau = 41.341374575751                  # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

wQ = 1189.7 * cmtoau
wc = 1189.7 * cmtoau
Nq = 2000
print('=============================')

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
    # trapezoid = dw * ((conv[0] + conv[-1])/2 + np.sum(conv[1:-1]))
    return np.trapz(k*G,dx = dw) #trapezoid 

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
def dev_scan(std_comb, TaskArray, omega_s, Psis, Ps, Nq, dw, samples):
    task_comb = TaskArray
    comb_len = len(task_comb)
    rates = np.zeros(comb_len)
    Jeff_array = np.zeros((len(omega_s), comb_len))
    # RPV - Rxn distribution 
    for n in range(comb_len):
        dist_Cq = np.zeros(Nq)
        dist_dip = np.zeros(Nq)
        dist_wQ = np.zeros(Nq)
        stdWQ = std_comb[TaskArray[n]] 
        dist_Cq += cQ #np.sort(np.random.normal(cQ, stdQ, Nq))
        # dist_dip += np.sort(np.real(np.random.uniform(0, stdD, Nq)))
        for m in range(samples):
            dist_wQ += np.sort(np.random.normal(wQ, stdWQ, Nq))
        # np.random.shuffle(dist_Cq)
        # np.random.shuffle(dist_dip)
        dist_wQ /= samples
        WQ = dist_wQ**2 * np.ones(Nq, dtype = np.complex64)
        # dist_dip /= samples
        alp = 1#(cQ/np.sum(dist_Cq) * Nq**0.5)
        Cq = dist_Cq.astype(np.complex64)*alp
    # RPV - cavity distribution    
        cos_dist = np.cos(dist_dip)
        etac = Rabi / np.sqrt(2 * wc * np.sum(cos_dist**2)) 
        v = np.sqrt(2 / wc) * etac * np.ones((Nq), dtype = np.complex64) * cos_dist
        Mat_in = np.outer(v,v)
        Jeff = np.zeros(len(omega_s))
    # Creation of the Jeff
        for wi in range(len(omega_s)):
            Mat = Mat_in * Psis[wi] + np.diag(WQ + Ps[wi]) 
            Mat_inv = np.linalg.inv(Mat)
            K = (Cq.T @ Mat_inv.astype(np.complex64) @ Cq)
            Jeff[wi] = np.real(np.imag(K))
    # Rate constant calculation
        reorg = np.real(np.trapz(Jeff/omega_s,omega_s, dw))
        Jeff *= reorg_an/reorg
        Jeff_array[:,n] = Jeff
        rates[n] = (K_VSC(np.real(omega_s),beta,np.real(Jeff)) * 0.7 + k0)/kQ
    return rates, Jeff_array, dist_wQ
    # return Jeff_array, task_comb

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
# cQ = 5E-7
# =========================

# Cavity
# =========================
τc = 500 * fstoau
k0 = 6.204982961140102545e-08 # Rate constant of DW without Q at eta_s 0.1
kQ = 9.076763853644350798e-08 # Q + Rxn rates

cQ = 4.7 * cmtoau/(1836**0.5) #1.48 * cmtoau # / np.sqrt(Nq)
lam_Q = 0.147 * cmtoau
gam_Q = 6000 * cmtoau
alpha = 1. / (500 * fstoau)
reorg_an = 6.582079423930413e-09
Rabi = 80 * cmtoau
#====================================
#   COUPLING DEVIATIONS
#====================================
# Compilation
N_dummy = 1
omega_dummy = np.array(np.arange(1, 1.2, 0.1) * cmtoau, dtype = np.complex64)
dw_dummy = omega_dummy[1] - omega_dummy[0]
σ2 = sigma(omega_dummy)
WQ_dummy = wQ**2 * np.ones(N_dummy, dtype = np.complex64)
Psis_dummy = Psi(omega_dummy) * np.ones(N_dummy, dtype = np.complex64)
Ps_dummy = P(omega_dummy) * np.ones(N_dummy, dtype = np.complex64)
sample_dummy = 1

std_cQ_dummy = np.linspace(0,3,1)
std_dip_dummy = np.linspace(0,1,1)
std_comb_dummy = [0] #np.array(list(itertools.product(std_cQ_dummy, std_dip_dummy)))
Ntasks_dummy = 1
nrank_dummy = 1
Task_dummy = np.array([0])
ini_time = time.time()
rates = dev_scan(std_comb_dummy, Task_dummy, omega_dummy,Psis_dummy,Ps_dummy, N_dummy, dw_dummy, sample_dummy)
fin_time = time.time()
print("Time = ",fin_time - ini_time)
#====================================
# Running
#====================================
omega_s = np.array(np.arange(1000, 1400, 1.5) * cmtoau, dtype = np.complex64)
dw = omega_s[1] - omega_s[0]
σ2 = sigma(omega_s)
# WQ = wQ**2 * np.ones(Nq, dtype = np.complex64)
Psis = Psi(omega_s)
Ps = P(omega_s)
#====================================
len_3D = 60
std_wQ = np.array([50]) * cmtoau
# std_dip = np.linspace(0,1,len_3D)
std_comb = std_wQ
sample = 8000
tot_Tasks = len_3D**2
TaskArray = [0]
TaskArray = np.array(TaskArray)
ini_time = time.time()
rates, Jeff, dist_wQ = dev_scan(std_comb, TaskArray, omega_s, Psis, Ps, Nq, dw, sample)
fin_time = time.time()
print("Time again = ",fin_time - ini_time)

# print(Jeff)
peaks, _ = find_peaks(np.real(Jeff[:,0]/max(Jeff[:,0])), height = 0.5)
LP = peaks[0]
UP = peaks[1]
print(np.real(omega_s[UP] - omega_s[LP])/cmtoau, 'Rabi Splitting')
print(rates, 'Rates')

plt.plot(omega_s/cmtoau,Jeff)
plt.savefig('Jeff.png', dpi = 500)
plt.close()

plt.hist(dist_wQ/cmtoau, bins = 100)
plt.savefig('wQ.png', dpi = 500)
plt.close()