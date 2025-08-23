#!/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt 
import sys 
sys.path.append("./utils/")
from ising_sethna import IsingModel
from pathlib import Path

def magnetization(lattice):
    spins = 2 * lattice - 1  # Convert spins from 0/1 to 1/-1
    return np.mean(spins)

def energy(lattice, H=0.0):
    J = 1 
    spins = 2 * lattice - 1  
    E = -J * np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)))
    E_total = E - H * np.sum(spins)
    return E_total

def main():
    N = 100  # Lattice size
    H = 0.0  # External field
    # I will use 2.0 and 2.8 instead of 1.5 and 3.5
    T_below = 2.0
    T_critical = 2.269185314213022
    T_above = 2.8
    n_equil_sweeps = 10_000   # Sweeps to reach equilibrium
    n_measure_sweeps = 10_000 * 13 # Sweeps for averaging
    
    temperatures = np.linspace(T_below, T_above, 50)

    for i, T in enumerate(temperatures):
        print(f"Simulation {i+1}/{len(temperatures)}", end="\r", flush=True)
        # print(f"Simulating lattice at T = {T:.2f}", end="\n", flush=True)
        dirname = Path(__file__).resolve().parent / f"data/{N}/lattices/{T:.2f}"
        dirname.mkdir(parents=True, exist_ok=True)

        model = IsingModel(N=N, T=T, H=H)
        # Random intilization of the model
        # model.lattice[:] = 1

        print("\nThermlizing...")
        for _ in range(n_equil_sweeps):
            model.SweepMetropolis(nTimes=1)
        print("Sampling...")
        for sweep in range(n_measure_sweeps):
            model.SweepMetropolis(nTimes=1)
            if sweep % 13 == 0: # To avoid autocorrelation
                filename = dirname / f"lattice_sweep_{sweep % n_measure_sweeps}.npy"
                np.save(filename, model.lattice)

if __name__ == "__main__":
    main()
