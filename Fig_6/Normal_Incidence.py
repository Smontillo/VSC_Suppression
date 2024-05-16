import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from scipy import integrate
from matplotlib.pyplot import MultipleLocator, tick_params
plt.rcParams.update({'font.size': 19})
plt.rcParams.update({'font.family': "times"})

lw = 3.0
axis_size = 28
unitlen = 7
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':15
        }

# axis label size
lsize = 30             
txtsize = 36
# tick length
lmajortick = 15
lminortick = 5

fig, ax = plt.subplots(figsize=(2.0 * unitlen, 1.0 * unitlen), dpi = 512, sharey = 'row')
fig.subplots_adjust(wspace = 0.0) 

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================
#                                       Plotting Fig 6a     
# ==============================================================================================

plt.subplot(1,2,1)

# reaction coordinate and cavity parameters
wc = 1190 * cm_to_au                        # default cavity frequency
w0 = 1190 * cm_to_au                        # system vibration energy
mu_0 = 9.14                                 # molecular transition dipole moment
beta = 1052.6                               # temperature T = 300 K
temp = 1. / beta
tau_c = 500 * fs_to_au                      # cavity perpendicular lifetime
tau_0 = 3.33333333333e0 * fs_to_au          # cavity in-plane lifetime

# spectator mode parameters
wQ = 1189.7 * cm_to_au                      # spectator mode frequency
γQ = 6000 * cm_to_au                        # characteristic frequency
λQ = 6.70E-7                                # bath reorganization energy
cQ = 5E-7                                   # coupling strength with the RxN

# rescaling factor of FGR (here associated with thermal distribution)
rescaling = 0.7
n_0 = 1 / (np.exp(beta * w0) - 1) * rescaling
print('n_0(w0) = ', n_0)

# bare double-well rate
k_0 = 5.946954192406803128e-08

# mathematical function, coth(x)
def coth(x):                                
    return 1 / np.tanh(x)

# Lorentzian lineshape
def Aw(x, x0, sigma_2):                  # gaussian distribution, with center x0 and variance sigma_2
    return (1./np.pi) * np.sqrt(sigma_2) / ((x - x0)**2 + sigma_2)

# ===== Auxiliary functions =====
Ngrids = int(1e4)               # grid points for integration, at least 1e4
eps = 1e-12                     # avoid divergence for 1D case, should be small enough
Npoints = 200                   # number of data points, the more the slower

# 1D spectral density
def gen_jw(w, wcav, ηc, τc):
    
    Γ = 1 / τc
    lamc_square = 2 * ηc**2 * wcav
    
    def DOS(x): # 1D DOS
        Gamma_parallel = (np.sqrt(np.abs(x**2 - wcav**2)) / x) / tau_0
        return ((x / np.sqrt(np.abs(x**2 - wcav**2)))) * Γ * np.exp(- x / temp) / (Γ + Gamma_parallel)
    
    w_int = np.logspace(np.log(wcav + eps), np.log(5 * wcav), Ngrids, base = np.e)

    norm = integrate.trapz(DOS(w_int), w_int)

    def gammaQ(x):
        return DOS(x) * (0.5 * lamc_square * x**2 * Γ) / ((x**2 - w**2)**2 + (w * Γ)**2)
    
    ΓQ_1 = integrate.trapz(gammaQ(w_int), w_int)
    ΓQ = (2 * λQ / γQ) + (ΓQ_1 / norm)

    def Rw(x):
        return DOS(x) * (0.5 * lamc_square * w**2) / ((x**2 - w**2)**2 + (w * Γ)**2) * (w**2 - x**2 + Γ**2)
    
    P = integrate.trapz(Rw(w_int), w_int)
    P = P / norm

    return cQ**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2)

# 2D spectral density
def gen_jw_2(w, wcav, ηc, τc):
    
    Γ = 1 / τc
    lamc_square = 2 * ηc**2 * wcav

    def DOS(x): # 2D DOS
        Gamma_parallel = (np.sqrt(np.abs(x**2 - wcav**2)) / x) / tau_0
        return x * Γ * np.exp(- x / temp) / (Γ + Gamma_parallel)
    
    w_int = np.logspace(np.log(wcav), np.log(5 * wcav), Ngrids, base = np.e)

    norm = integrate.trapz(DOS(w_int), w_int)

    def gammaQ(x):
        return DOS(x) * (0.5 * lamc_square * x**2 * Γ) / ((x**2 - w**2)**2 + (w * Γ)**2)
    
    ΓQ_1 = integrate.trapz(gammaQ(w_int), w_int)
    ΓQ = (2 * λQ / γQ) + (ΓQ_1 / norm)

    def Rw(x):
        return DOS(x) * (0.5 * lamc_square * w**2) / ((x**2 - w**2)**2 + (w * Γ)**2) * (w**2 - x**2 + Γ**2)
    
    P = integrate.trapz(Rw(w_int), w_int)
    P = P / norm

    return cQ**2/2 * ΓQ * w / ((wQ**2 - w**2 + P)**2 + (w * ΓQ)**2)

# the molecular bath spectral density function, J_v(w)
def Drude(x):
    lam = 83.7 * cm_to_au / 1836
    gam = 200 * cm_to_au
    return (2 * lam * gam * x / (x**2 + gam**2)) * coth(beta * x / 2)

"""
sigma^2 = (1 / pi) \int_0^{\infty} dw J_v (w) coth(beta w / 2)
"""

# to get the variance
Rij = 9.87
wi = np.linspace(1e-10, 200 * cm_to_au, 10000000)     # for intergration. Better to be larger (at least 10^3)
y = Drude(wi)
sigma_2 = integrate.trapz(y, wi)
# sigma_2 = (0.01 * cm_to_au)**2 # 
sigma_2 = Rij**2 * sigma_2 / (np.pi)
print("sigma_2 = ", np.sqrt(sigma_2) / cm_to_au, '\t cm^-1')

# get convoluted k_1
wc_scan_2 = np.linspace(0.1 * wc, 3 * wc, 10000)      # for intergration, Better to be larger (at least 10^5)
dwc = wc_scan_2[1] - wc_scan_2[0]
def intergrant(x):
    return 2 * mu_0**2 * gen_jw(x, wc, 0.0, tau_c) * Aw(x, w0, sigma_2) * n_0
y = intergrant(wc_scan_2)
k_1 = integrate.trapz(y, wc_scan_2)
print("k_1 = ", k_1)

# discretizing data points
w_scan = np.linspace(600 * cm_to_au, 1800 * cm_to_au, Npoints)

x_axis = w_scan / cm_to_au
plt.plot(x_axis, [1.0] * len(x_axis), linewidth = lw, color = 'black', label = r"outside cavity")

spd = []
etac = 0.002

for z in w_scan:
    temp_rate = 0
    for x0 in wc_scan_2:
        temp_rate += 2 * mu_0**2 * gen_jw(x0, z, etac, tau_c) * Aw(x0, w0, sigma_2) * n_0 * dwc
    spd.append((k_0 + temp_rate) / (k_0 + k_1))

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'blue', label = r'$\eta_\mathrm{c} = 2.0 \times 10^{-3}$')

np.savetxt("1D_0.002.txt", spd)

spd = []
etac = 0.0035

for z in w_scan:
    temp_rate = 0
    for x0 in wc_scan_2:
        temp_rate += 2 * mu_0**2 * gen_jw(x0, z, etac, tau_c) * Aw(x0, w0, sigma_2) * n_0 * dwc
    spd.append((k_0 + temp_rate) / (k_0 + k_1))

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'green', label = r'$\eta_\mathrm{c} = 3.5 \times 10^{-3}$')

np.savetxt("1D_0.0035.txt", spd)

spd = []
etac = 0.005

for z in w_scan:
    temp_rate = 0
    for x0 in wc_scan_2:
        temp_rate += 2 * mu_0**2 * gen_jw(x0, z, etac, tau_c) * Aw(x0, w0, sigma_2) * n_0 * dwc
    spd.append((k_0 + temp_rate) / (k_0 + k_1))

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'red', label = r'$\eta_\mathrm{c} = 5.0 \times 10^{-3}$')

np.savetxt("1D_0.005.txt", spd)

plt.text(1360, 0.832, r'$\omega_\mathrm{0}$', color = 'black', size = 20)
plt.text(1270, 0.8, r'$1190\ \mathrm{cm}^{-1}$', color = 'black', size = 20)

# x and y range of plotting 
x1, x2 = 600, 1740
y1, y2 = 0.74, 1.05     # y-axis range: (y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(300)
x_minor_locator = MultipleLocator(60)
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.02)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

ax.vlines([1190], 0.6, 1.05, linestyles = 'dashed', colors = ['black'], lw = 3.0) 

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'$\omega_\mathrm{c}~ (\mathrm{cm}^{-1})$', size = txtsize)
ax.set_ylabel(r'$k / k_0$', size = txtsize)
ax.legend(frameon = False, loc = 'lower left', prop = font_legend, markerscale = 1)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 6b     
# ==============================================================================================

plt.subplot(1,2,2)

# data points discretization
w_scan = np.linspace(x1 * cm_to_au, x2 * cm_to_au, Npoints)

plt.plot(x_axis, [1.0] * len(x_axis), linewidth = lw, color = 'black', label = r"outside cavity")

spd = []
etac = 0.002

for z in w_scan:
    temp_rate = 0
    for x0 in wc_scan_2:
        temp_rate += 2 * mu_0**2 * gen_jw_2(x0, z, etac, tau_c) * Aw(x0, w0, sigma_2) * n_0 * dwc
    spd.append((k_0 + temp_rate) / (k_0 + k_1))

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'blue', label = r'$2.0 \times 10^{-3}$')

np.savetxt("2D_0.002.txt", spd)

spd = []
etac = 0.0035

for z in w_scan:
    temp_rate = 0
    for x0 in wc_scan_2:
        temp_rate += 2 * mu_0**2 * gen_jw_2(x0, z, etac, tau_c) * Aw(x0, w0, sigma_2) * n_0 * dwc
    spd.append((k_0 + temp_rate) / (k_0 + k_1))

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'green', label = r'$3.5 \times 10^{-3}$')

np.savetxt("2D_0.0035.txt", spd)

spd = []
etac = 0.005

for z in w_scan:
    temp_rate = 0
    for x0 in wc_scan_2:
        temp_rate += 2 * mu_0**2 * gen_jw_2(x0, z, etac, tau_c) * Aw(x0, w0, sigma_2) * n_0 * dwc
    spd.append((k_0 + temp_rate) / (k_0 + k_1))

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'red', label = r'$5.0 \times 10^{-3}$')

np.savetxt("2D_0.005.txt", spd)

plt.text(1360, 0.832, r'$\omega_\mathrm{0}$', color = 'black', size = 20)
plt.text(1270, 0.8, r'$1190\ \mathrm{cm}^{-1}$', color = 'black', size = 20)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = 5)

ax.vlines([1190], 0.6, 1.05, linestyles = 'dashed', colors = ['black'], lw = 3.0) 

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params('x', labelsize = 30, which = 'both', direction = 'in')
plt.tick_params('y', labelsize = 0, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'$\omega_\mathrm{c}$ ($\mathrm{cm}^{-1}$)', size = txtsize)
# ax.set_ylabel(r'$k/k_0$', size = 36)
# ax.legend(loc = 'lower left', frameon = False, prop = font_legend)
plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)




plt.savefig("figure_rate_v3.pdf", bbox_inches='tight')