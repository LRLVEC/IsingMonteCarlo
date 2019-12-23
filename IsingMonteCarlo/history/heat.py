#!/usr/bin/env python3

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
def init_lattice(n):
    lattice = np.random.choice([1, -1], size=(n, n))
    return lattice
def deltaE(S0, Sn, J, H):
    return 2 * S0 * (H + J * Sn)
def heated_ising_2d(
    n=200,
    nsteps=500000,
    H=0,
    J=1
):
    '''Ising Model Simulator. If count_spins = True, only flipping behavior of 1 site is studied.'''
    '''The temperature increases from 0.5 Tc to 1.5 Tc.'''
    lattice = init_lattice(n)
    energy = 0
    energies = []
    spins = []
    spin = np.sum(lattice)

    for step in tqdm(range(nsteps)):

        T = ( 0.5 + float(step)/nsteps ) * 2.268

        i = np.random.randint(n)
        j = np.random.randint(n)

        # Periodic Boundary Condition
        Sn = lattice[(i - 1) % n, j] + lattice[(i + 1) % n, j] + \
             lattice[i, (j - 1) % n] + lattice[i, (j + 1) % n]

        dE = deltaE(lattice[i, j], Sn, J, H)

        if dE < 0 or np.random.random() < np.exp(-dE/T):
            lattice[i, j] = -lattice[i, j]
            energy += dE
            energies.append(energy)
         # Note that the spin is collected at every step
            spin += 2*lattice[i, j]
        spins.append(spin)
    return lattice, energies, spins
import sys
cv2.namedWindow("spins", cv2.WINDOW_AUTOSIZE)
heated_ising_2d(n=16 if len(sys.argv)==1 else int(sys.argv[1]), nsteps=1000000, J=-1)
cv2.destroyAllWindows()
