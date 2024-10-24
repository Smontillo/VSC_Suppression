import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
#==================================

#====================================
# PHYSICAL AND RATE CONSTANTS
#====================================
cmtoau = 4.556335e-06
k0 = 6.204982961140102545e-08 # Rate constant of DW without Q at eta_s 0.1

#====================================
# DATA LOADING AND PROCESSING
#====================================
ntask = 250
std_wQ = [0, 10, 20, 30, 40]                    # wQ standard deviation
wc_scan = np.linspace(700, 1700,150)            # wc scan frequencies
tot_task = len(std_wQ) * len(wc_scan) 
Ntask = tot_task//ntask                         # Tasks per processor

rates = np.zeros((len(std_wQ), len(wc_scan)))   # Reaction rate data
freq = np.zeros((len(std_wQ), len(wc_scan)))    # Frequency data
run_dat = np.zeros(tot_task)
freq_dat = np.zeros(tot_task)

# RETRIEVE ALL DATA FROM FILES
#====================================
for k in range(ntask):
    data = np.array(np.loadtxt(f'data/rate_{k}.txt'))
    run_dat[(k) * Ntask: (k + 1) * Ntask] = data[:,-1]
    freq_dat[(k) * Ntask: (k + 1) * Ntask] = data[:,1]

out_rate = np.loadtxt('../outside_cavity/rates.txt')    # Load outside cavity rate data

fig, ax = plt.subplots(figsize = (4.5,4.5))
color = [ 'b', 'green', 'red', 'blueviolet', 'darkorange']
ax.axhline(1, c = 'black', alpha = 0.6, lw = 3)
# Divide data by standard deviation and plot
for k in range(len(std_wQ)):
    rates[k,:] = run_dat[k * len(wc_scan) : (k+1) * len(wc_scan)] 
    freq[k,:] = freq_dat[k * len(wc_scan) : (k+1) * len(wc_scan)] 
    ax.plot(freq[k,:], (rates[k,:] * 0.7 + k0)/(out_rate[k,-1] * 0.7 + k0), label = std_wQ[k], lw = '3', alpha = 0.9, color = f'{color[k]}')

ax.set_yticks([0.8,0.9,1.0])
ax.set_xticks([800, 1000, 1200, 1400, 1600])
ax.set_ylim(0.72, 1.01)
plt.xlim(700,1700)
plt.legend(title = r'$\sigma$ (cm$^{-1}$)', frameon = False, fontsize = 15, handlelength=1, title_fontsize = 15, labelspacing = 0.2)
plt.ylabel(r'$k/k_0$', fontsize = 22)
plt.xlabel(r'$\omega_c$ (cm$^{-1}$)', fontsize = 22)
plt.savefig('images/rate.png', dpi = 500, bbox_inches='tight')
plt.close()
