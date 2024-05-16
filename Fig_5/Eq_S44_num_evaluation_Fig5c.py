import numpy as np
import matplotlib.pyplot as plt
from random import random
import numba as nb
import time

# ================= global ====================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fstoau = 41.341374575751                  # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

wQ = 1189.7 * cmtoau
Nq = 1000
# =========================


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
def sigma():
    w_cut = np.arange(1E-15,γv,0.001*cmtoau)
    dw_cut = w_cut[1] - w_cut[0]
    f = J_spec(w_cut,λv,γv) * np.cosh(beta*w_cut/2)/np.sinh(beta*w_cut/2)
    return εz**2/np.pi * np.trapz(f,w_cut,dw_cut)

# Gaussian function
@nb.jit(nopython=True, fastmath=True)
def Kernel():
    l = 1/np.pi * σ2**0.5/((w-w0)**2 + σ2)
    return l

# Convolution k_FGR @ Gaussian
@nb.jit(nopython=True, fastmath=True)
def K_VSC(w,beta,J):
    k = np.real(k_FGR(w,beta,J))
    G = np.real(Kernel()) 
    dw = w[1] - w[0]
    return np.trapz(k*G,dx = dw)

@nb.jit(nopython=True, fastmath=True)

@nb.jit(nopython=True, fastmath=True)
def P(x):
    return - x**2 + 2 * lam_Q * x / (1.0j * gam_Q)

@nb.jit(nopython=True, fastmath=True)
def L(x):
    return - x**2 - 1.0j * alpha * x

@nb.jit(nopython=True, fastmath=True)
def Psi(x, wc):
    return wc**2 * L(x) / (wc**2 + L(x))

@nb.jit(nopython=True, fastmath=True)
def InvMat(M):
    return np.linalg.inv(M)

@nb.jit(nopython=True, fastmath=True)
def Jeff_num(omega_s, Mat_in, Cj, wc):
    Jeff = np.zeros(len(omega_s), dtype = np.complex_)
    for wi in range(len(omega_s)):
        ws = omega_s[wi]
        Mat = Mat_in * Psi(ws, wc)
        wQ2 = np.diag(WQ + P(ws)) 
        Mat = Mat + wQ2
        Mat_inv = InvMat(Mat)
        K = (Cj.T @ Mat_inv @ Cj)
        Jeff[wi] = np.imag(K)/2
    return Jeff
# =========================

# System
# =========================
cut_off = 200 #cm-1
w0 = 1189.7 * cmtoau 
M = 1
wb = 1000 * cmtoau

# Bath 
# =========================
ηv = 0.1
γv = 200 * cmtoau
λv = ηv * M * γv * wb/2
temp = 300 / autoK 
beta = 1/temp
εz = 9.387
Δx = 9.141
σ2 = sigma()
# =========================

# Q_mode
# =========================
mQ = 1
wQ = 1189.7 * cmtoau # 1189.678284945453
γQ = 6000 * cmtoau 
λQ = 6.70E-7 
# =========================

# Cavity
# =========================
τc = 500 * fstoau
k0 = 6.204982961140102545e-08 # Rate constant of DW without Q at eta_s 0.1
kQ = 9.076763853644350798e-08 # Q + Rxn rates

Cj = 4.7 * cmtoau/(1836**0.5) #1.48 * cmtoau # / np.sqrt(Nq)
lam_Q = 0.147 * cmtoau
gam_Q = 6000 * cmtoau
alpha = 1. / (500 * fstoau)
reorg_en = 6.582079423930413e-09

#====================================
#   RPV - RXN COUPLINGS
#====================================
dist = np.zeros((Nq), dtype = np.complex_)
ε = 0.1
std = ε * Cj  # Cj disorder is given by normal distribution with σ = ε ⋅ 0.1
dist = (np.random.normal(Cj, std, Nq))
alp = Cj/np.real(sum(dist)) * Nq**0.5
Cj = np.array(dist*alp, dtype = np.complex_) # total reorg must be constant
#====================================

omega_s = np.array(np.arange(500, 1900, 0.1) * cmtoau, dtype = np.complex_)
w = omega_s
dw = omega_s[1] - omega_s[0]
WQ = wQ**2 * np.ones(Nq, dtype = np.complex_)

#====================================
#   LIGHT - MATTER COUPLING SET UP
#====================================
Rabi = 100 * cmtoau
etac = Rabi / np.sqrt(2 * Nq * wQ)
ηc = etac
omegas = np.linspace(700,1700,150) * cmtoau
kappas = np.zeros(len(omegas))
kappas_an = np.zeros(len(omegas))

for k in range(len(omegas)):
    wc = omegas[k]
    v = np.sqrt(2 / wc) * etac * np.ones((Nq), dtype = np.complex_)
    Mat_in = np.outer(v,v)
    Jeff = np.real(Jeff_num(omega_s,Mat_in,Cj,wc))
#====================================

#====================================
#   MAKE THE REORG CONSTANT
#====================================
    reorg = np.real(np.trapz(Jeff/omega_s,omega_s, dw))
    Jeff = Jeff * reorg_en/reorg
    reorg = np.real(np.trapz(Jeff/omega_s,omega_s, dw))
#====================================
#   RATE CALCULATION
#====================================
    kappas[k] = (K_VSC(np.real(omega_s),beta,np.real(Jeff)) * 0.7 + k0)/kQ

plt.axhline(1, c = 'black', alpha = 0.5)
plt.plot(omegas/cmtoau, kappas, ls = '-', lw = 3, label = 'Numerical')
plt.xlim(700,1700)
plt.legend(frameon = False)
plt.savefig('rates.png', dpi = 500)
plt.close()

np.savetxt('rates.txt', np.c_[Rabi/cmtoau, kappas])