import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
sys.path.append("./utils/")
from ising_sethna import IsingModel

def main():
    with st.sidebar.expander("Parameters", expanded=True):
        system_size = st.slider("System size (N x N)", 10, 100, 10)
        H = st.slider("External Field (H)", -1.0, 1.0, 0.0)
        n_sweeps = st.slider("Number of Sweeps", 10, 1000, 10)

    st.title(f"Ising Model ({system_size}x{system_size})")
    T_below = 1.5
    T_critical = 2.269185314213022
    T_above = 3.5
    # n_sweeps = st.sidebar.slider("Number of Sweeps", 10, 1000, 200)
    # speed = st.sidebar.slider("Update speed (seconds)", 0.01, 1.0, 0.1)

    ising_below = IsingModel(N=system_size, T=T_below, H=H)
    ising_critical = IsingModel(N=system_size, T=T_critical, H=H)
    ising_above = IsingModel(N=system_size, T=T_above, H=H)

    labels = ["below T_c", "at Tc", "above Tc"]
    isings = [ising_below, ising_critical, ising_above]

    def magnetization(lattice):
        return np.mean(lattice)  

    def energy(lattice, H=0.0):
        J = 1 
        spins = lattice 
        E = -J * np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)))
        E_total = E - H * np.sum(spins)
        # N = lattice.shape[0]
        # J = 1
        # E = 0
        # for i in range(N):
        #     for j in range(N):
        #         s = lattice[i, j]
        #         neighbors = lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N]
        #         E += -J * s * neighbors
        # E_total = E - H * lattice.sum()
        return E_total

    mag_data = np.zeros((len(labels), n_sweeps))
    sus_data = np.zeros((len(labels), n_sweeps))
    specheat_data = np.zeros((len(labels), n_sweeps))
    energy_data = np.zeros((len(labels), n_sweeps))
    progress_bar = st.progress(0)
    progress_text = st.empty()

    plot_lattice_container = st.empty()
    plot_obs_container = st.empty()


    for sweep in range(n_sweeps):
        progress_bar.progress((sweep + 1) / n_sweeps)
        progress_text.text(f"Sweep {sweep + 1} of {n_sweeps}")
        for idx, ising in enumerate(isings):
            ising.SweepMetropolis(nTimes=1)

        # Plot lattices
        fig_lattice, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, label, ising in zip(axs, labels, isings):
            ax.imshow(np.array(ising.lattice), cmap='gray', vmin=0, vmax=1)
            ax.set_title(label)
            ax.axis('off')
        plt.tight_layout()
        plot_lattice_container.pyplot(fig_lattice)
        plt.close(fig_lattice)

        # Calculate observables and update data
        for idx, ising in enumerate(isings):
            lattice_np = np.array(ising.lattice)
            T = ising.GetTemperature()
            M = magnetization(lattice_np)
            E = energy(lattice_np, H)

            mag_data[idx, sweep] = M
            energy_data[idx, sweep] = E

            if sweep > 0:
                chi = (ising.N**2) * np.var(mag_data[idx, :sweep+1]) / T
                C = (ising.N**2) * np.var(energy_data[idx, :sweep+1]) / (T**2)
            else:
                chi = 0
                C = 0

            sus_data[idx, sweep] = chi
            specheat_data[idx, sweep] = C

        # Plot observables
        fig_obs, axs_obs = plt.subplots(3, 1, figsize=(8, 10))
        x = np.arange(sweep + 1)

        for ax, data_array, title in zip(axs_obs,
                                        [mag_data, sus_data, specheat_data],
                                        ["Magnetization", "Susceptibility", "Specific Heat"]):
            ax.set_title(title)
            ax.grid(True)
            ax.set_xlim(0, n_sweeps)
            for idx, label in enumerate(labels):
                ax.plot(x, data_array[idx, :sweep+1], label=label)
            ax.set_xlabel("Number of Sweeps")
            ax.legend()

        plt.tight_layout()
        plot_obs_container.pyplot(fig_obs)
        plt.close(fig_obs)

if __name__ == "__main__":
    main()