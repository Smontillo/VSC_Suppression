import numpy as np
import matplotlib.pyplot as plt
from random import random
import numba as nb
import time

conv = 27.211397                            # 1 a.u. = 27.211397 eV
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
fstoau = 41.341374575751   

wQ = 1189.7 * cmtoau # 1189.678284945453
γQ = 6000 * cmtoau 
λQ = 6.70E-7 
cQ = 5E-7
# =========================

# Cavity
# =========================
τc = 500 * fstoau

# ================= global ====================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fstoau = 41.341374575751                  # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

wQ = 1189.7 * cmtoau
wc = 1189.7 * cmtoau
Nq = 1000

@nb.jit(nopython=True, fastmath=True)
def Jeff_f(w,wc):
    Γ = 1/τc
    ηc = etac
    ΓQ = 2 * λQ / γQ + (2 * Nq * wc**3 * ηc**2 * Γ) / ((wc**2 - w**2)**2 + (w*Γ)**2)
    P = (2 * wc * Nq * ηc**2 * w**2) / ((wc**2 - w**2)**2 + (w * Γ)**2) * (w**2 - wc**2 + Γ**2)
    return np.sum(Cq**2)/(2*wQ**2) * wQ**2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2 )

@nb.jit(nopython=True, fastmath=True)
def delta(m, n):
    return 1 if m == n else 0

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
def InvMat(M):
    return np.linalg.inv(M)

cQ = 4.7 * cmtoau/(1836**0.5) #1.48 * cmtoau # / np.sqrt(Nq)
lam_Q = 0.147 * cmtoau
gam_Q = 6000 * cmtoau
Rabi = 100 * cmtoau
etac = Rabi / np.sqrt(2 * Nq * wc) * 1
alpha = 1. / (500 * fstoau)

print('=============================')
print(etac, 'eta_c')


# Cq coupling disorder
Cq = np.zeros((Nq), dtype = np.complex_)
dist = np.zeros((Nq), dtype = np.complex_)
std = 0.5 * cQ
for i in range(Nq):
    dist[i] = (np.random.normal(cQ, std))
alp = cQ/np.real(sum(dist)) * Nq**0.5
Cq = dist*alp 
print(np.real(sum(Cq)), 'total coupling')


v = np.sqrt(2 / wc) * etac #* np.ones((Nq), dtype = np.complex_)
omega_s = np.arange(1000, 1400, 0.1) * cmtoau
dw = omega_s[1] - omega_s[0]
Jeff_list = np.zeros(len(omega_s), dtype = np.complex_)

# Mat_in = np.outer(v,v)
WQ = wQ**2 * np.ones(Nq, dtype = np.complex_)
Jeff = np.zeros(len(omega_s), dtype = np.complex_)
totTime = 0.0

for wi in range(len(omega_s)):
    w = omega_s[wi]
    wQ2 = (wQ**2 + P(w) + Nq*Psi(w)*v**2) 
    st_time = time.time()
    K = (Cq.T @ Cq)/wQ2
    Jeff[wi] = np.imag(K)/2
    ed_time = time.time()
    totTime += (ed_time - st_time)

J_an = Jeff_f(omega_s,wc)
reorg_an = np.real(np.trapz(J_an/omega_s,omega_s, dw))

reorg = np.real(np.trapz(Jeff/omega_s,omega_s, dw))

Jeff = Jeff * reorg_an/reorg

reorg = np.real(np.trapz(Jeff/omega_s,omega_s, dw))

J_an = Jeff_f(omega_s,wc)

np.savetxt('Jeff_approx.txt', np.c_[np.real(omega_s/cmtoau),np.real(Jeff),np.real(J_an)])