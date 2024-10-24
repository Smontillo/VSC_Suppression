from scipy import integrate, linalg
import json
import numpy as np
import armadillo as arma
import matplotlib.pyplot as plt
from bath_gen_Drude_PSD import generate

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.
fstocm = 3.33333E4
kB = 3.1668E-6   # au/K
# Model parameters

def DW(x,m,wDW):
    m = 1
    # Eb = 2250 * cmtoau
    Eb = 2730 * cmtoau
    c  = 2.27817E-8 
    V  = -(wDW**2 / 2) * x**2 + (wDW**4 / (16 * Eb)) * x**4 - c * x**3
    return V - min(V)

def T(x,m):
    dx = x[1] - x[0]
    N = len(x)
    K = np.pi/dx
    Kin = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Kin[i,j] = K**2/3 * (1 + 2/N**2)
            else:
                Kin[i,j] = 2*K**2/N**2 * (-1)**(j-i)/(np.sin(np.pi * (j-i)/N))**2 
    return 1/(2*m) * Kin

def DVR(x,m,wDW):
    V = DW(x,m,wDW)
    V = np.diag(V)
    K = T(x,m)
    E, V = np.linalg.eigh(V+K)
    return E, V

def creation(quanta):
  size = quanta 
  a = np.zeros((size,size), dtype = 'complex')
  b = np.array([(x+1)**0.5 for x in range(size)])
  np.fill_diagonal(a[1:], b)
  return a.T

def get_Hs(nDW,nHO,lam_Rxn, lam_NM, lam_cav):
    R1     = get_Rx(nDW)
    R2     = np.dot(R1,R1)
    IDW    = np.identity(nDW, dtype = 'complex')
    IHO    = np.identity(nHO, dtype = 'complex')
    H_DW   = np.kron(IHO,np.diag(EDW[:nDW]))
    H_HO   = (np.kron(EHO,IDW))
    H_int  = Qc/(2*mHO*wHO)**0.5 * np.kron((a + a_dag), get_R(nDW))  
    H_bath = lam_Rxn * R2 + (lam_NM + lam_cav) * get_Ry2(nHO) 
    H_self = Qc**2/(2*mHO*wHO**2) * R2
    return H_DW + H_bath  + H_HO + H_int + H_self 

def get_rho0():
    beta = 1/temp
    rho_DW = np.zeros((nDW,nDW), dtype = 'complex')
    rho_HO = np.zeros((nHO,nHO), dtype = 'complex')

    # Reactant well
    rho_DW[1,1] = 1.0 + 0j
    Z = np.exp(-beta * (EHO))

    Z = np.trace(Z)

    rho_HO = (np.exp(-beta * np.diag(EHO))/Z)
    return np.kron(np.diag(rho_HO),rho_DW)

def get_Rx(nDW):
    IHO = np.identity(nHO, dtype = 'complex')
    pos_DW = np.zeros((nDW,nDW), dtype = 'complex')
    for j in range(nDW):
        for i in range(nDW):
            avg_pos = VDW[:,j].conjugate() * x * VDW[:,i]
            pos_DW[j,i] = np.trapz(avg_pos,x,dx)
    return np.kron(IHO,pos_DW)

def get_R(nDW):
    pos_DW = np.zeros((nDW,nDW), dtype = 'complex')
    for j in range(nDW):
        for i in range(nDW):
            avg_pos = VDW[:,j].conjugate() * x * VDW[:,i]
            pos_DW[j,i] = np.trapz(avg_pos,x,dx)
    return pos_DW

def get_Ry(nHO):
    Id_DW = np.identity(nDW, dtype = 'complex')
    Ry = (a_dag+a)
    return (1/(2*wHO*mHO))**0.5 * np.kron(Ry,Id_DW)

def get_Ry2(nHO):
    Id_DW = np.identity(nDW, dtype = 'complex')
    Ry = 2*(a_dag@a) + np.identity(nHO, dtype = 'complex') + (a_dag @ a_dag) + (a @ a)
    return (1/(2*wHO*mHO)) * np.kron(Ry,Id_DW)

# Bath-matter coupling
def get_Qx(nDW):  # R = R_ij |v_i >< v_j|
    return get_Rx(nDW)

def get_Qy(nHO):  # R = R_ij |v_i >< v_j|
    return get_Ry(nHO)

def gen_jw(w, omega_c, lam, Gamma):
    return 2 * lam * omega_c**2 * Gamma * w / ((w**2 - omega_c**2)**2 + (Gamma * w)**2)

def Jw_w(x):
    wc, lam_cav, gam_cav = parameters.wc, parameters.lam_cav, parameters.gam_cav
    return gen_jw(x, wc, lam_cav, gam_cav) / (np.pi * x)


# ==============================================================================================
# Create the DW and HO potentials
# ==============================================================================================
nDW = 6
mDW = 1
wDW = 1030 * cmtoau
N = 1024
L = 200.0
x = np.linspace(-L,L,N)
dx = x[1] - x[0]

EDW,VDW = DVR(x,mDW,wDW)
Normx = np.trapz(VDW[:,0].conjugate() * VDW[:,0],x,dx)
VDW = VDW/(Normx)**0.5
VDW = np.array(VDW, dtype = 'complex')
VDW[:,1] = -VDW[:,1]
VDW[:,0] = -VDW[:,0]
VDW[:,5] = -VDW[:,5]

nHO = 3 
mHO = 1
wHO = 1175 * cmtoau
a = creation(nHO)
a_dag = np.conjugate(a.T)
EHO = wHO * (a_dag@a)

Qc = 5 * 1E-7
temp = 300 / autoK  
# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================
class parameters:
    # ===== DEOM propagation scheme =====
    dt = 0.1 * fstoau
    t = 5000 * fstoau   # plateau time as 20ps for HEOM
    nt = int(t / dt)
    nskip = 50

    lmax = 8
    nmax = 1000000
    ferr = 1.0e-07 
    # ===== number of system states =====
    x = x
    wDW = wDW
    mDW = mDW
    nDW = nDW
    nHO = nHO
    Nstates = nDW * nHO 

    # ===== Drude-Lorentz model =====
    temp = 300 / autoK                             # temperature
    nmod = 3                                        # number of dissipation modes (C-Q-R)

    rho0 = get_rho0() 

    # Bath I (Rxn), Drude-Lorentz model
    gam_Rxn = 200 * cmtoau                      # bath characteristic frequency
    ratio = 0.1                                  # the value of etas / omega_b, tune it from 0.02 to 2.0
    lam_Rxn = ratio * wDW * gam_Rxn/2  #* mDW * gam_Rxn           # reorganization energy
 
    # Bath II (Cavity), Drude-Lorentz model
    Qc = Qc
    gam_NM = 6000 * cmtoau
    lam_NM = 6.70E-7

    # Bath III parameters, Brownian Oscillator
    wc = 1000 * cmtoau
    tau_c = 500 * fstoau                           # bath relaxation time
    gam_cav = 1. / tau_c                        # bath characteristic frequency
    ΩR = 180 * cmtoau 
    eta_c = ΩR / (2 * wHO)**0.5
    lam_cav = eta_c**2 * wc        # reorganization energy 

    lam = np.array([lam_Rxn, lam_NM, lam_cav])
    gam = np.array([gam_Rxn, gam_NM, gam_cav])

    # PSD scheme
    pade    = 1                          # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsd    = 3                      # number of Pade terms

    # ===== Get the subspace information ===== 
    # eigenvalue = eigenvalue
    VDW = VDW 
    EDW = EDW 

    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    # There are three bath ---> DW bath, HO bath and Cavity 'bath'
    # DW --> x, HO --> y
    Qsx = get_Qx(nDW) * 1
    Qsy = get_Qy(nHO) * 1
    Qsc = get_Qy(nHO) * 1 

    print(eta_c)
    
# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # bath
    temp = parameters.temp
    nmod = parameters.nmod
    lam = parameters.lam
    gam = parameters.gam
    pade = parameters.pade
    npsd = parameters.npsd
    wc = parameters.wc
    w = np.linspace(0.00001 * wc, 10 * wc, 10000000)
    y = Jw_w(w)
    RE_c = integrate.trapz(y, w)
    # system
    Nstates = parameters.Nstates
    hams = get_Hs(nDW,nHO, lam[0], lam[1], RE_c)
    rho0 = parameters.rho0
    # system-bath
    Qsx = parameters.Qsx
    Qsy = parameters.Qsy
    Qsc = parameters.Qsc
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
    for m in range(nmod-1):       # one mode is treated by PFD
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

    # bath III with PSD
    ini['bath']['temp'] = temp                                                  
    ini['bath']['nmod'] = nmod
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd + 1
    ini['bath']['jomg'] = [{"jsdr":[(lam[2], wc, gam[2])]}] 
                                                                                
    jomg = ini['bath']['jomg']
    etal_2, etar_2, etaa_2, expn_2, delr_2 = generate (temp, npsd + 1, pade, jomg)


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
    qmds = np.zeros((nmod, Nstates, Nstates), dtype = complex)
    qmds[0,:,:] = Qsx * 1                    # the electron-phonon interaction (DW)
    qmds[1,:,:] = Qsy * 1                    # the electron-phonon interaction (HO)
    qmds[2,:,:] = Qsc * 1

    arma.arma_write (hams,ini['syst']['hamsFile'])
    arma.arma_write (qmds,ini['syst']['qmdsFile'])
    arma.arma_write (rho0,'inp_rho0.mat')

    jsonInit = {"deom":ini,
                "rhot":{
                    "dt": dt,
                    "nt": nt,
                    "nk": nskip,
					"xpflag": 1,
					"staticErr": 0,
                    "rho0File": "inp_rho0.mat",
                    "sdipFile": "inp_sdip.mat",
                    "pdipFile": "inp_pdip.mat",
					"bdipFile": "inp_bdip.mat"
                },
            }

# ==============================================================================================================================
# ==============================================================================================================================

    # dipoles
    sdip = np.zeros((2,2),dtype=float)
    arma.arma_write(sdip,'inp_sdip.mat')

    pdip = np.zeros((nmod,2,2),dtype=float)
    pdip[0,0,1] = pdip[0,1,0] = 1.0
    arma.arma_write(pdip,'inp_pdip.mat')

    bdip = np.zeros(3,dtype=complex)
#    bdip[0]=-complex(5.00000000e-01,8.66025404e-01)
#    bdip[1]=-complex(5.00000000e-01,-8.66025404e-01)
#    bdip[2]=-complex(7.74596669e+00,0.00000000e+00)
    arma.arma_write(bdip,'inp_bdip.mat')

    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4) 
