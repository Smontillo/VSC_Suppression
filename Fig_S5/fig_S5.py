import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = "times"
cmtoau = 4.556335e-06 
fstoau = 41.341                           # 1 fs = 41.341 a.u.
pstoau = 41.341 * 1000                    # 1 fs = 41.341 a.u.
#====================================

#====================================
# Functions
#====================================
# Define the function to maximize (we negate it because minimize does minimization by default)
def objective(c, m, n):
    return -(c[0]**2 * (m - n) +  2 * c[0] * c[1] * p + n)

# Define the constraint (must return 0 at the solution)
def constraint(x):
    return x[0]**2 + x[1]**2 - 1 

#====================================
# Data
#====================================
EDW = np.loadtxt('./data/ADW_EDW.dat')
VDW = np.loadtxt('./data/ADW_VDW.dat')
x  = VDW[:,0]
ν3 = -VDW[:,4]
ν4 = VDW[:,5]
points = len(x)
half = int(points/2)

#====================================
# Left Localized State
#====================================
m = np.trapz(ν3[:half]*ν3[:half],x[:half])
n = np.trapz(ν4[:half]*ν4[:half],x[:half])
p = np.trapz(ν3[:half] * ν4[:half], x[:half])

# Initial guess for the variables
x0 = np.array([0.5, 0.5])

# Define the constraint as a dictionary
cons = {'type': 'eq', 'fun': constraint}

# Perform the optimization
solution = minimize(objective, x0,args = (m,n), constraints=cons)

# Output the results
print('Maximized function value:', -solution.fun)
print('Optimal values of x and y:', solution.x)
print('Norm:', solution.x[0]**2 + solution.x[1]**2)
Ca, Cb = solution.x[0], solution.x[1]

#====================================
# Right Localized State
#====================================
m = np.trapz(ν3[half:]*ν3[half:],x[half:])
n = np.trapz(ν4[half:]*ν4[half:],x[half:])
p = np.trapz(ν3[half:] * ν4[half:], x[half:])

# Initial guess for the variables
x0 = np.array([0.5, 0.5])

# Define the constraint as a dictionary
cons = {'type': 'eq', 'fun': constraint}

# Perform the optimization
solution = minimize(objective, x0,args = (m,n), constraints=cons)

# Output the results
print('Optimal values of x and y:', solution.x)
print('Norm:', solution.x[0]**2 + solution.x[1]**2)
Cap, Cbp = solution.x[0], solution.x[1]

νL = Ca * ν3 + Cb * ν4
νR = Cap * ν3 + Cbp * ν4
#====================================
E3, E4, E1 = EDW[3], EDW[4], EDW[1]
EL = E3 * Ca**2 + E4 * Cb**2        # Energy of |ν_L⟩ state
ER = E3 * Cap**2 + E4 * Cbp**2      # Energy of |ν_R⟩ state
Es = E3 * Ca * Cap + E4 * Cb * Cbp  # Tunnelling splitting ⟨ν_R|H|ν_L⟩

print('νL - ν1 ->', EL - E1)
print('#==================#')
print('νR - ν1 ->', ER - E1)
print('#==================#')
print('Tunnelling splitting', Es)
#====================================

#====================================
# Fig. S5a
#====================================
fig, ax = plt.subplots(2,1,figsize=(4.5,4.5), sharex = True, dpi = 150)
fig.subplots_adjust(hspace=0)
ax[0].plot(x, ν3*2E3 + EDW[3], lw = 1.5, c = '#e74c3c', ls = '--', alpha = 0.7, label = r'$|\nu_3\rangle$')
ax[0].plot(x, νL*2E3 + EL, lw = 2.5, c = '#e74c3c', label = r'$|\nu_L\rangle$')
ax[1].plot(x, ν4*2E3 + EDW[4], lw = 1.5, c = '#e67e22', ls = '--', alpha = 0.7, label = r'$|\nu_4\rangle$')
ax[1].plot(x, νR*2E3 + ER, lw = 2.5, c = '#e67e22', label = r'$|\nu_R\rangle$')
ax[0].legend(frameon=False, fontsize = 11, handlelength=1, title_fontsize = 11, labelspacing = 0.2, ncol = 1)
ax[1].legend(frameon=False, fontsize = 11, handlelength=1, title_fontsize = 11, labelspacing = 0.2, ncol = 1)
ax[0].set_xlim(-105,105)
ax[1].set_xlim(-105,105)
ax[1].set_xlabel('$R_0$ (a.u.)')
ax[0].set_yticks([2800, 3200])
ax[1].set_yticks([3000, 3400])
fig.text(-0.1,0.34,'Energy (cm$^{-1}$)', rotation = 'vertical')
plt.savefig('./Fig_S5a.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()

#====================================
# Fig. S5b
#====================================
in_dia = np.loadtxt('./data/pop_in_cav_180cm_diabats.dat')
out_dia = np.loadtxt('./data/pop_out_cav_diabats.dat')

fig, ax = plt.subplots(figsize = (4.5,4.5), dpi =150)

ax.plot(out_dia[:,0], out_dia[:,1], ls = '--', lw = 1.5, color = '#9b59b6')
ax.plot(out_dia[:,0], out_dia[:,3], ls = '--', lw = 1.5, color = '#27ae60')
ax.plot(out_dia[:,0], out_dia[:,4], ls = '--', lw = 1.5, color = '#e74c3c')
ax.plot(out_dia[:,0], out_dia[:,5], ls = '--', lw = 1.5, color = '#e67e22')
ax.plot(out_dia[:,0], out_dia[:,6], ls = '--', lw = 1.5, color = '#34495e')
ax.plot(in_dia[:,0], in_dia[:,1], ls = '-', lw = 2, color = '#9b59b6', label = r'$|\nu_0\rangle$')
ax.plot(in_dia[:,0], in_dia[:,3], ls = '-', lw = 2, color = '#27ae60', label = r'$|\nu_2\rangle$')
ax.plot(in_dia[:,0], in_dia[:,4], ls = '-', lw = 2, color = '#e74c3c', label = r"$|\nu'_L\rangle$")
ax.plot(in_dia[:,0], in_dia[:,5], ls = '-', lw = 2, color = '#e67e22', label = r"$|\nu'_R\rangle$")
ax.plot(in_dia[:,0], in_dia[:,6], ls = '-', lw = 2, color = '#34495e', label = r'$|\nu_5\rangle$')
ax.set_xlabel(r'Time (ps)')
ax.set_ylabel(r'Population')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 0.006)
ax.set_yticks([0, 0.002, 0.004, 0.006])
ax.legend(loc=0, frameon = False, fontsize = 11, handlelength=1, title_fontsize = 15, labelspacing = 0, ncol = 5, columnspacing = 0.5)

ax2 = ax.twinx()
ax2.plot(np.NaN, np.NaN, ls= '-', label='Inside Cavity', c='black')
ax2.plot(np.NaN, np.NaN, ls= '--',label='Outside Cavity', c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(loc = (0.65,0.009), frameon = False, fontsize = 9)

plt.savefig('./Fig_S5b.pdf', dpi = 500, bbox_inches = 'tight', facecolor='white')
plt.close()

#====================================
# Save for Fig. S6
#====================================
np.savetxt('../Fig_S6/data/ADW_VDW_diabats.dat', np.c_[VDW[:,:4], νL, νR, VDW[:,6:]])
EDW[3] = EL
EDW[4] = ER
np.savetxt('../Fig_S6/data/ADW_EDW_diabats.dat', EDW)