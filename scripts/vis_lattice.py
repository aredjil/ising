import numpy as np 
import matplotlib.pyplot as plt 
"""
    This script viuslizes the lattice configuration at three different temperatures
"""
def main():
    Ts = [2.00, 2.26, 2.80]
    titles = [r'$T < T_c$', r'$T \approx T_c$', r'$T > T_c$']
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9, 3))
    ax = ax.flatten()
    i = 0 
    for  T in Ts:
        data = np.load(f"./data/50/lattices/{T:.2f}/lattice_sweep_0.npy")
        ax[i].imshow(data, cmap="Grays")
        ax[i].set_title(titles[i])
        ax[i].axis("off")
        i+=1
    plt.savefig("./figs/lattice_50.png")
    plt.show()
if __name__ =="__main__":
    main()