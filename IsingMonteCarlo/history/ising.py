# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys

H_s = list(np.linspace(-0.20, 0.20, 80))
T_Tcs = list(np.linspace(0.2, 1.8, 80))

# energies = []
spins   = []

for h in H_s:
    # res_e = []
    res_s = []
    print(".", end="")
    content = os.popen("echo '{0} {1} {2} {3} {4} {5}' |  ./IsingMonteCarlo.exe".format(h,1,0.2*2.268, 1.2*2.268, 80, 200000)).readlines() # removed "\n"
    spin = map(lambda x:float(x[:-1]), content)
    # res_e.append(energy)    # energies.append(res_e)
    spin.append(spin)

# energy_array = np.array(energy)
spins_array = np.array(spins)
plt.imshow(spins_array, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
plt.xticks(range(80),[None if x % 20 != 0 else "{:.4f}".format(H_s[x]) for x in range(80)])
plt.yticks(range(80),[None if x % 10 != 0 else "{:.4f}".format(T_Tcs[::-1][x]) for x in range(80)])
plt.colorbar()
plt.ylabel("$T$")
plt.xlabel("$\\frac{H}{J}$")
plt.show()
