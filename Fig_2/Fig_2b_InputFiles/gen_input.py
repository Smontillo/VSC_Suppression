from scipy import integrate
import json
import numpy as np
import armadillo as arma
from bath_gen_Drude_PSD import generate

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# DVR parameters
Ngrid = 1001
L = 100.0
R = np.linspace(- L, L, Ngrid)
dx = R[1] - R[0]

# Model parameters
"""
the model Arkajit used, see orginal paper: JCP 140, 174105 (2014)
"""

m_s = 1
omega_b = 1000 * cm_to_au
E_b = 2250 * cm_to_au

'''
    DISCREATE VARIABLE REPRESENTATION
'''
def kinetic(Ngrid, M, dx):
    A = np.zeros((Ngrid, Ngrid), dtype=float)
    for i in range(Ngrid):
        A[i, i] = np.pi**2 / 3
        for j in range(1, Ngrid - i):
            A[i, i+j] = 2 * (-1)**j / j**2
            A[i+j, i] = A[i, i+j]
    A = A / (2 * M * dx**2)
    return A

def potential(Ngrid, R, m_s, omega_b, E_b):
    def V(R):
        return - (m_s * omega_b**2 / 2) * R**2 + (m_s**2 * omega_b**4 / (16 * E_b)) * R**4   
    B = np.zeros((Ngrid, Ngrid), dtype=float)
    for i in range(Ngrid):
        B[i, i] = V(R[i])
    return B

def diagonalization(Ngrid, dx, R, m_s, omega_b, E_b):
    H = kinetic(Ngrid, m_s, dx) + potential(Ngrid, R, m_s, omega_b, E_b)
    eigenvalue, eigenvec = np.linalg.eig(H)
    return eigenvalue, eigenvec

# ==============================================================================================

# get the vibrational eigen states and sort ascendingly
eigenvalue, eigenvec = diagonalization(Ngrid, dx, R, m_s, omega_b, E_b)
eigenvec = eigenvec
ordered_list = sorted(range(len(eigenvalue)), key=lambda k: eigenvalue[k])

temp1 = np.zeros((len(eigenvec[:, 0]), len(eigenvec[0, :])), dtype = complex)
temp2 = np.zeros((len(eigenvalue)), dtype=float)

for count in range(len(eigenvalue)):
    temp1[:, count] = eigenvec[:, ordered_list[count]]
    temp2[count] = eigenvalue[ordered_list[count]]
eigenvec = temp1
eigenvalue = temp2
eigenvec[:, 0] = - eigenvec[:, 0]

del temp1
del temp2
del ordered_list

# ==============================================================================================
#                                       Auxiliary functions     
# ==============================================================================================

def creation(quanta):
  size = quanta 
  a = np.zeros((size,size), dtype = 'complex')
  b = np.array([(x+1)**0.5 for x in range(size)])
  np.fill_diagonal(a[1:], b)
  return a

def get_Rx(NStates):
    R_s = np.zeros((NStates, NStates), dtype=complex)
    for i in range(NStates):
        for j in range(NStates):
            for k in range(len(eigenvec[:, i])):
                R_s[i,j] += eigenvec[k, i].conjugate() * R[k] * eigenvec[k, j]
    return np.real(R_s)

def get_Rs():
    return (1/(2*wHO))**0.5 * (a_dag+a).real

def get_Rs2(NHO):
    Ry = 2*(a_dag@a) + np.identity(NHO, dtype = 'complex')
    return ((1/(2*wHO)) * Ry).real

def get_Hs(NStates, lambda_1, lambda_2):
    Rx = get_Rx(NStates)
    Rx2 = np.dot(Rx,Rx)
    Rs = get_Rs()
    Rs2 = get_Rs2(NHO)
    IDW = np.identity(NStates)
    IHO = np.identity(NHO)

    HDW = np.kron(IHO,np.diag(eigenvalue[:NStates])) 
    HHO = np.kron(np.diag(EHO), IDW)
    Hint = cQ * np.kron(Rs,Rx)
    Hreorg = (lambda_1 + cQ**2/(2*wHO**2)) * np.kron(IHO,Rx2) + lambda_2 * np.kron(Rs2,IDW)
    return HDW + HHO + Hint + Hreorg

def get_Qs1(NStates):  # R = R_ij |v_i >< v_j|
    return get_Rx(NStates)

def get_Qs2(NHO):  # mu = R
    return get_Rs()

def get_Qs3(NStates):  # mu = Rs
    return get_Rs()

def get_rho0(NStates):
    IDW = np.identity(NStates)
    Rs = get_Rs()
    rho_x = np.zeros((NStates, NStates), dtype = complex)
    rho_x[0,0] = 1.0  
    rho_s = np.zeros((NHO, NHO), dtype = complex)   
    rho_s[0,0] = 1
    
    μ = np.kron(Rs,IDW)
    rho0 = np.kron(rho_s,rho_x)
    return (μ @ rho0)  # <μ(t)μ(0)> 

def gen_jw(w, omega_c, lam, Gamma):
    return 2 * lam * omega_c**2 * Gamma * w / ((w**2 - omega_c**2)**2 + (Gamma * w)**2)

NHO = 3
mHO = 1
wHO = 1189.7 * cm_to_au
a_dag = creation(NHO)
a = np.conjugate(a_dag.T)
EHO = np.diag(wHO*(a_dag@a))
cQ = 5E-7 * 1

# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================
class parameters:

    # ===== DEOM propagation scheme =====
    dt = 0.1 * fs_to_au
    nt = 40000 
    nskip = 5

    lmax = 10
    nmax = 1000000
    ferr = 1.0e-07

    # ===== number of system states =====
    NStates = 4                     # DW number of states
    NHO = NHO                       # Harmonic oscillator states
    wHO = wHO                       # Harmonic oscillator frequency
    cQ = cQ                         # RPV - Rxn coupling

    # ===== Cavity parameters =====
    omega_c = 1189.7 * cm_to_au     # cavity frequency. Note that in this model, the energy gap is around 1140 cm^-1
    eta_c = 0.0025                  # light-matter-coupling strength. Set as 0 when cavity is turned off

    # ===== Drude-Lorentz model =====
    temp    = 300 / au_to_K         # temperature
    nmod    = 3                     # number of dissipation modes

    # Bath I parameters, Drude-Lorentz model
    gamma_1   = 200 * cm_to_au                      # bath characteristic frequency
    ratio = 0.1                                     # the value of etas / omega_b, tune it from 0.02 to 2.0
    lambda_1 = ratio * m_s * omega_b * gamma_1 / 2  # reorganization energy

    # Bath II parameters, Drude-Lorentz model
    gamma_2   = 6000 * cm_to_au                     # bath characteristic frequency
    lambda_2  = 6.70E-7                             # reorganization energy

    # PSD scheme
    pade    = 1                                     # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsd    = 3                                     # number of Pade terms

    # Bath III parameters, Brownian Oscillator
    tau_c = 2000 * fs_to_au                         # bath relaxation time
    gamma_3 = 1. / tau_c                            # bath characteristic frequency  
    lambda_3 = eta_c**2 * omega_c                   # reorganization energy    

    eigenvec = eigenvec
    x = R
    lam = np.array([lambda_1, lambda_2, lambda_3])
    gam = np.array([gamma_1, gamma_2, gamma_3])   
    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    IHO = np.identity(NHO)
    IDW = np.identity(NStates)
    Qs1 = np.kron(IHO,get_Qs1(NStates))     # DW - Drude_Lorentz bath coupling operator
    Qs2 = np.kron(get_Qs2(NStates),IDW)     # HO - Drude_Lorentz bath coupling operator
    Qs3 = np.kron(get_Qs2(NStates),IDW)     # HO - Cavity coupling operator
    rho0 = get_rho0(NStates)
    R_1 = Qs1
def Jw_w(x):
    omega_c, lambda_2, gamma_2 = parameters.omega_c, parameters.lambda_2, parameters.gamma_2
    return gen_jw(x, omega_c, lambda_2, gamma_2) / (np.pi * x)


# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # cavity
    omega_c = parameters.omega_c
    # bath
    temp = parameters.temp
    nmod = parameters.nmod
    lambda_1 = parameters.lambda_1
    gamma_1 = parameters.gamma_1
    lambda_2 = parameters.lambda_2
    gamma_2 = parameters.gamma_2
    lambda_3 = parameters.lambda_3
    gamma_3 = parameters.gamma_3
    lam = parameters.lam
    gam = parameters.gam
    pade = parameters.pade
    npsd = parameters.npsd
    w = np.linspace(0.00001 * omega_c, 10 * omega_c, 10000000)
    y = Jw_w(w)
    RE_c = integrate.trapz(y, w)
    print(RE_c)
    # system
    NStates = parameters.NStates
    hams = get_Hs(NStates, lambda_1, lambda_2, RE_c = 0.0) # Cavity is not coupled with system
    rho0 = parameters.rho0
    # system-bath
    Qs1 = parameters.Qs1
    Qs2 = parameters.Qs2
    Qs3 = parameters.Qs3
    # DEOM
    dt = parameters.dt
    nt = parameters.nt
    nskip = parameters.nskip
    lmax = parameters.lmax
    nmax = parameters.nmax
    ferr = parameters.ferr

# ==============================================================================================================================
    # hidx
    ini['hidx']['trun'] = 0
    ini['hidx']['lmax'] = lmax
    ini['hidx']['nmax'] = nmax
    ini['hidx']['ferr'] = ferr

	# bath PSD
    ini['bath']['temp'] = temp
    ini['bath']['nmod'] = nmod
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd
    ini['bath']['jomg'] = [{"jdru":[(lam[i], gam[i])]} for i in range(nmod-1)] 
                                                                               
    jomg = ini['bath']['jomg']
    nind = 0
    for m in range(nmod - 1):       # one mode is treated by PFD
        try:
            ndru = len(jomg[m]['jdru'])
        except:
            ndru = 0
        try:
            nsdr = len(jomg[m]['jsdr'])
        except:
            nsdr = 0
        nper = ndru + 2 * nsdr + npsd
        nind += nper
                                                                               
    etal_1, etar_1, etaa_1, expn_1, delr_1 = generate (temp, npsd, pade, jomg)

	# bath II with PSD
    r = 1 #-1
    ini['bath']['temp'] = temp                                                  
    ini['bath']['nmod'] = nmod
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd + r
    ini['bath']['jomg'] = [{"jsdr":[(lambda_3, omega_c, gamma_3)]}] 
                                                                                
    jomg = ini['bath']['jomg']
    etal_2, etar_2, etaa_2, expn_2, delr_2 = generate (temp, npsd + r, pade, jomg)

    mode = np.zeros((nind + 2 + npsd + 1), dtype = int)
    mode[(pade+npsd):nind] = 1
    
    for i in range(nind, nind + npsd + 3):
        mode[i] = 2
    print(nind, 'nind')

    delr = np.append(delr_1, delr_2)
    etal = np.append(etal_1, etal_2)
    etar = np.append(etar_1, etar_2)
    etaa = np.append(etaa_1, etaa_2)
    expn = np.append(expn_1, expn_2)

    arma.arma_write(mode, 'inp_mode.mat')
    arma.arma_write(delr, 'inp_delr.mat')
    arma.arma_write(etal, 'inp_etal.mat')
    arma.arma_write(etar, 'inp_etar.mat')
    arma.arma_write(etaa, 'inp_etaa.mat')
    arma.arma_write(expn, 'inp_expn.mat')

    # two dissipation modes
    qmds = np.zeros((nmod, NStates*NHO, NStates*NHO), dtype = complex)
    qmds[0,:,:] = Qs1                           # the electron-phonon interaction
    qmds[1,:,:] = Qs2                           # the HO - Drude_Lorentz bath coupling operator
    qmds[2,:,:] = 0.0                           # Cavity is decouppleced from the system

    arma.arma_write (hams,ini['syst']['hamsFile'])
    arma.arma_write (qmds,ini['syst']['qmdsFile'])
    arma.arma_write (rho0,'inp_rho0.mat')

    # real time dynamics
    jsonInit = {"deom":ini,
                "spec":{
                    "w1max": 3600 * cm_to_au,
                    "nt1": nt,
                    "dt": dt,
                    "nk": nskip,
                    "staticErr": 1e-05,
                    "rho0File": "inp_rho0.mat",
                    "sdipFile": "inp_sdip.mat",
                    "pdipFile": "inp_pdip.mat",
                    "bdipFile": "inp_bdip.mat"
                },
            }

# ==============================================================================================================================
# ==============================================================================================================================

    # dipoles
    sdip = Qs2
    arma.arma_write(sdip,'inp_sdip.mat')

    pdip = np.zeros((nmod,NStates*NHO,NStates*NHO),dtype=float)
    pdip[1,:,:] = np.identity(NStates*NHO)
    arma.arma_write(pdip,'inp_pdip.mat')

    bdip = np.zeros(nmod * len(expn),dtype=complex)
#    bdip[0]=-complex(5.00000000e-01,8.66025404e-01)
#    bdip[1]=-complex(5.00000000e-01,-8.66025404e-01)
#    bdip[2]=-complex(7.74596669e+00,0.00000000e+00)
    arma.arma_write(bdip,'inp_bdip.mat')

    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4) 
