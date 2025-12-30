import os
import re
import copy
import ase.io
import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from dscribe.descriptors import EANN
from descriptor import Descriptors

#plt.style.use(['nature', 'ieee', 'high-vis', 'retro'])
plt.rcParams['font.family'] = "Arial"
plt.figure(figsize=(20, 14))


def load_config():
    """Load parameters from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Network formation from descriptor clusters.")

    # Arguments to configure the species, cutoff radius, height, and threshold
    parser.add_argument('--species', type=str, nargs='+', default=[
                        "Cu", "Ti", "O"], help="List of species (default: ['Cu', 'Ti', 'O'])")
    parser.add_argument('--r_cut', type=float, default=6.0,
                        help="Cutoff radius for SOAP descriptor (default: 6.0)")
    parser.add_argument('--height', type=float, default=10.5,
                        help="Height threshold (default: 10.5)")
    parser.add_argument('--threshold', type=float, default=0.0,
                        help="Threshold for energy (default: 0.0)")

    return parser.parse_args()


def different_str(desp_cluster, limit=0.02):
    dist_ = np.linalg.norm(desp_cluster, axis=1)
    desp_i = []
    index_all = []
    for i in range(len(dist_)):
        if len(index_all) == 0:
            index_all.append(i)
            desp_i.append(dist_[i])
        else:
            dist_i_other = np.min(dist_[i] - np.array(desp_i))
            if abs(dist_i_other) >= limit:
                index_all.append(i)
                desp_i.append(dist_[i])
    return index_all


def index_i_j(index_i, index_j, index_i_all, index_j_all):
    index_ini = None
    for i in range(len(index_i_all)):
        if index_i_all[i] == index_i and index_j_all[i] == index_j:
            index_ini = i
            break
    return index_ini


def formation_nx(desp_cluster, filename="network_cluster.out", limit=0.02, threshold=0):
    nodes = different_str(desp_cluster, limit=limit)

    index_i_all, index_j_all, energy_f, energy_b = read_pathway(
        filename=filename)

    G = nx.Graph()
    G.add_nodes_from(list(set(index_i_all + index_j_all)))

    for i in range(len(nodes) - 1):
        index_i = nodes[i]
        # Only traverse the upper triangle to avoid duplicate edges
        for j in range(i + 1, len(nodes)):
            index_j = nodes[j]
            index_ts = index_i_j(index_i, index_j, index_i_all, index_j_all)
            if index_ts is not None:
                weight = energy_f[index_ts]
                print(weight, index_i, index_j)
                if weight > threshold:  # Add edge if the weight exceeds the threshold
                    G.add_edge(index_i, index_j, weight=weight)

    pos = nx.spring_layout(G, k=1)  # Spring layout for node positioning

    edge_colors = [d['weight'] for u, v, d in G.edges(data=True)]
    edge_cmap = plt.get_cmap('coolwarm')  # Colormap for edge weights
    norm = plt.Normalize(min(edge_colors), max(
        edge_colors))  # Normalize edge weights
    edge_colors = [edge_cmap(1 - norm(c))
                   for c in edge_colors]  # Map edge weights to colors

    nx.draw(G, pos, with_labels=True, node_color="#F4A261",
            node_size=4000, font_size=40, width=1)
    edge_labels = {(u, v): f"{d['weight']:.2f}"
                   for u, v, d in G.edges(data=True) if d['weight'] > 0.01}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=40)

    plt.savefig("network_x.tif", dpi=64)
    plt.savefig("network_x.svg", format="svg")


def read_pathway(filename="network_cluster.out"):
    index_i = []
    index_j = []
    energy_f = []
    energy_b = []
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i >= 1:
                line_i = line.strip().split()
                index_i.append(int(line_i[0]))
                index_j.append(int(line_i[1]))
                energy_f.append(float(line_i[-2]))
                energy_b.append(float(line_i[-1]))
    return index_i, index_j, energy_f, energy_b


if __name__ == "__main__":
    # Load configuration from command line arguments
    args = load_config()

    atoms = ase.io.read("meta_all.db", index=":")

    index_all = []
    if not os.path.exists("desp.npy"):
        for i in range(len(atoms)):
            chemical_symbols = atoms[i].get_chemical_symbols()
            positions = atoms[i].get_positions()
            index = [j for j in range(len(positions))
                     if positions[j][2] >= args.height]
            index_all.append(index)

        eann = EANN(species=args.species, r_cut=args.r_cut, periodic=True)
        desp_ = Descriptors(atoms=atoms, desp_type=eann, index=index_all)
        output = np.array(desp_.similarity_all())
        desp_cluster = np.sqrt(2 - 2 * output)

        np.save("desp.npy", desp_cluster)
    else:
        desp_cluster = np.load("desp.npy")

    dist_ = np.linalg.norm(desp_cluster, axis=1)
    formation_nx(desp_cluster, limit=args.threshold)
