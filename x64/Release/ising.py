# # To add a new cell, type '#%%'
# # To add a new markdown cell, type '#%% [markdown]'
# import os
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt
# import os, sys

# H_s = list(np.linspace(-0.20, 0.20, 80))
# T_Tcs = list(np.linspace(0.2, 1.8, 80))

# # energies = []
# spins   = []

# for h in (H_s):
#     # res_e = []
#     # res_s = []
#     print("echo '{0} {1} {2} {3} {4} {5}' | .\IsingMonteCarlo.exe".format(h,1,0.2*2.268, 1.8*2.268, 5, 200000))#.readlines() # removed "\n"
#     # spin = list(map(lambda x:float(x[:-1]), content))
#     # res_e.append(energy)    # energies.append(res_e)
#     # spin.append(spin)

# # energy_array = np.array(energy)
# spins_array = np.array(spins)
# plt.imshow(spins_array, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
# # plt.xticks(range(80),[None if x % 20 != 0 else "{:.4f}".format(H_s[x]) for x in range(80)])
# # plt.yticks(range(80),[None if x % 10 != 0 else "{:.4f}".format(T_Tcs[::-1][x]) for x in range(80)])
# plt.colorbar()
# plt.ylabel("$T$")
# plt.xlabel("$\\frac{H}{J}$")
# plt.show()
M = [0.994049,
0.993042,
0.990631,
0.982666,
0.977539,
0.965302,
0.946228,
0.919647,
0.878998,
0.815857,
0.503845,
0.062103,
0.010254,
0.042511,
0.018951,
0.010773,
0.004608,
0.005920,
0.011353,
0.006195]
T = [ 1.134000,
1.247400,
1.360800,
1.474200,
1.587600,
1.701000,
1.814400,
1.927800,
2.041200,
2.154600,
2.268000,
2.381400,
2.494800,
2.608200,
2.721600,
2.835000,
2.948400,
3.061800,
3.175200,
3.288600]
plt.plot(T, M)
plt.ylim(0,1)
plt.show()