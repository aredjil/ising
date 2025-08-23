import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    # Ns = [20, 30, 40, 50]
    Ns = [50]
    markers = [
        "s", 
        "o", 
        "^", 
        "*"
    ]
    colors =[
        'blue', 
        'red', 
        'green',
        'darkblue'
    ]
    i = 0
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    ax = ax.flatten()
    for N in Ns:
        base_dir = f"./data/{N}/lattices"
        figs = Path(__file__).resolve().parent / "figs"
        figs.mkdir(parents=True, exist_ok=True)

        T_critical = 2.269185314213022

        temp_dirs = sorted(os.listdir(base_dir))  
        T = np.linspace(2.0, 2.8, 50)


        all_lattices = []
        samples_per_T = []  # keep track of number of samples per temperature

        for t_str in temp_dirs:
            file_list = sorted(glob.glob(os.path.join(base_dir, t_str, "lattice_sweep_*.npy")))
            sampled_files = file_list
            samples_per_T.append(len(sampled_files))
            
            for f in sampled_files:
                lattice = np.load(f).astype(np.uint8)
                all_lattices.append(lattice.flatten())

        all_lattices = np.array(all_lattices)
        print("All lattices shape:", all_lattices.shape)

        pca = PCA(n_components=10)
        lattices_pca = pca.fit_transform(all_lattices)

        # Split PCA results per temperature
        # pc1_split = np.split(lattices_pca[:,0], np.cumsum(samples_per_T)[:-1])
        # pc2_split = np.split(lattices_pca[:,1], np.cumsum(samples_per_T)[:-1])

        # # Average per temperature
        # mean_pc1_per_T = np.array([arr.mean() for arr in pc1_split])
        # mean_pc2_per_T = np.array([arr.mean() for arr in pc2_split])

        # # Scatter plot colored by temperature
        # temps_for_samples = np.concatenate([np.full(n, T[i]) for i, n in enumerate(samples_per_T)])

        # pc1 = lattices_pca[:,0]
        # pc1 = pc1.reshape(50, -1).mean(axis=1) / N
        # pc2 = np.abs(lattices_pca[:,1])
        # pc2 = pc2.reshape(50, -1).mean(axis=1) / N
        # ax[i].scatter(T, pc1, label=f"{N}", marker=markers[i], facecolors='none', edgecolors=colors[i])
        # ax[i].set_title(f"{N}x{N} Ising lattice")
        # ax[i].set_ylabel(r'$<p_1> / N$')
        # ax[i].set_xlabel("T")

        # ax[i].legend(loc="best", edgecolor="black", title="N")
        
        # plt.figure(figsize=(6,4))
        # plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
        
        # pca.explained_variance_ratio_,color="red")
        # plt.title(f"Explained Variance Ratio - {N}x{N}")
        # plt.xlabel("Principal Component")
        # plt.ylabel("Explained Variance Ratio")
        # plt.grid(True)
        # plt.savefig(f"./figs/explained_variance_{N}.png")
        # plt.show()
        # plt.close()
        i+=1
    # plt.tight_layout()
    # plt.savefig(f"./figs/pc1.png")
    # plt.show()
if __name__ == "__main__":
    main()

