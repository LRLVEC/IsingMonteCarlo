import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm

def invoke_sync(H1, H2, nH, T1, T2, nT, cycles=200000, binary="IsingMonteCarlo.exe"):
    content = os.popen("{} {:.6f} {:.6f} {} {:.6f} {:.6f} {} {}".format(binary, H1, H2, nH, T1, T2, nT, cycles))
    a = np.zeros([nH+1, nT+1], np.float)
    with tqdm(total=(nH+1)*(nT+1)) as pbar:
        for i in range(nH+1):
            for j in range(nT, -1, -1):
                k = content.readline()[:-1]
                # print(k)
                a[i][j] = list(map(float, k.split(" ")))[-1]
                pbar.update(1)
    return np.transpose(a)

def invoke_async(H1, H2, nH, T1, T2, nT, cycles=200000, binary="IsingMonteCarlo.exe"):
    content = os.popen("{} {:.6f} {:.6f} {} {:.6f} {:.6f} {} {}".format(binary, H1, H2, nH, T1, T2, nT, cycles))
    while True:
        i = content.readline()
        if i == "":
            return
        H, T, M = list(map(float, i[:-1].split(" ")))
        yield (H, T, M)

a = invoke_sync(0.0, 0.2, 40, 0.06, 0.26, 80, cycles=1000000)
np.save("a.npy", a)
plt.imshow(a, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
plt.xticks(range(10),[None if x % 20 != 0 else "{:.4f}".format(np.linspace(0.0, 0.2, 40)[x]) for x in range(10)])
plt.yticks(range(10),[None if x % 10 != 0 else "{:.4f}".format(np.linspace(0.060, 0.260, 80)[x]) for x in range(10)])
plt.colorbar()
plt.ylabel(r"$\frac{T}{T_c}$")
plt.xlabel(r"$\frac{H}{J}$")
plt.savefig("rewind3200.svg",)
plt.show()