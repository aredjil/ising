import networkx as nx
import matplotlib.pyplot as plt
"""
    This code generates the figure of the first neighbors interaction,
"""
def main():

    # Size of the lattice
    N = 3

    # Create a 2D lattice graph
    G = nx.grid_2d_graph(N, N)

    # Relabel nodes for display
    labels = {(i, j): f"$S_{{{i},{j}}}$" for i, j in G.nodes()}

    # Position nodes in a grid
    pos = {(i, j): (j, -i) for i, j in G.nodes()}

    # Define node colors: red for the middle node, lightblue for others
    node_colors = ["red" if (i, j) == (1, 1) else "lightblue" for i, j in G.nodes()]

    # Define edge colors: green if connected to middle node, black otherwise
    middle = (1, 1)
    edge_colors = [
        "red" if middle in edge else "black" for edge in G.edges()
    ]

    # Draw graph
    plt.figure(figsize=(6,6))
    nx.draw(
        G, pos,
        labels=labels,
        with_labels=True,
        node_size=1200,
        node_color=node_colors,
        font_size=10,
        edgecolors="black",
        linewidths=2,
        edge_color=edge_colors,
        width=2
    )
    # plt.title(f"{N}x{N} Ising Lattice with middle node highlighted")
    plt.savefig("./figs/first_neighbors.png")
    plt.show()
if __name__ =="__main__":
    main()
