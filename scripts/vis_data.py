import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Compute the magentization 
def magnetization(lattice):
    spins = 2 * lattice - 1
    return np.mean(spins)
def main():
    # Directory path where the data is stored
    base_dir = "./lattices"
    figs = Path(__file__).resolve().parent / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    temp_dirs = sorted(os.listdir(base_dir))  
    all_mags = []

    T = np.linspace(2.0, 2.8, 50)
    for t_str in temp_dirs:
        file_list = sorted(glob.glob(os.path.join(base_dir, t_str, "lattice_sweep_*.npy")))

        mags = []
        for f in file_list:
            lattice = np.load(f)  # shape: (L, L)
            mags.append(magnetization(lattice))

        all_mags.append(mags)

    all_mags = np.vstack(all_mags)  # shape: (num_temps, num_samples)

    # Compute mean magnetization
    mean_mags = all_mags.mean(axis=1)

    # Compute susceptibility: χ = N^2 / T * ( <M^2> - <M>^2 )
    N = all_mags.shape[1]  # number of spins per lattice? Actually should be lattice size L^2
    L = np.load(glob.glob(os.path.join(base_dir, temp_dirs[0], "lattice_sweep_*.npy"))[0]).shape[0]
    N_spins = L**2

    mean_M2 = np.mean(all_mags**2, axis=1)
    susceptibility = N_spins / T * (mean_M2 - mean_mags**2)

    # Plot
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.scatter(T, mean_mags, color="darkblue", label=r'$<M>$')
    plt.xlabel(r'T')
    plt.ylabel(r'$<M>$')
    plt.legend()
    plt.title("Magnetization vs Temperature")

    plt.subplot(1,2,2)
    plt.scatter(T, susceptibility, color="green", label=r'$\chi$')
    plt.xlabel(r'T')
    plt.ylabel(r'$\chi$')
    plt.legend()
    plt.title("Susceptibility vs Temperature")

    plt.tight_layout()
    plt.savefig(figs/"m_chi.png")
    plt.show()
if __name__ == "__main__":
    main()