import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341374575751                  # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================

# Load data
data = np.loadtxt("resp1st.w1", dtype = float)
data2 = np.loadtxt("resp1st_im.w", dtype = float)

# Normalize data
max = np.argmax(data2)
np.savetxt('IR_data.txt', np.c_[data/cm_to_au,data2/data2[max]])

# Plot data
fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(data / cm_to_au, data2/data2[max], lw = 3, c = 'r')
ax.set_xlim(600, 1800)
ax.set_ylim(0.0, 1.1)

ax.set_xlabel('$\omega$ $(cm^{-1})$')
ax.set_ylabel('Intensity')

plt.savefig("IR.png", dpi = 500, bbox_inches='tight')