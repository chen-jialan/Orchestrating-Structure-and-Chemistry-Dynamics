import ase.io
import itertools
import copy
import numpy as np
import os
import argparse


def new_positions_all(positions, index):
    positions_index = positions[index, :]
    permutations = list(itertools.permutations(positions_index))
    all_arrays = []
    for perm in permutations:
        permuted_columns = np.array(perm)
        new_array = np.copy(positions)
        new_array[index, :] = permuted_columns
        all_arrays.append(new_array)
    return all_arrays


def dis_min_positions(positions1, all_array):
    if len(all_array) <= 1:
        return all_array[0]
    else:
        dis_ini = 100
        positions_final = all_array[0]
        for i in range(len(all_array)):
            dis = np.linalg.norm(positions1 - all_array[i])
            if dis <= dis_ini:
                dis_ini = dis
                positions_final = all_array[i]
                print(dis_ini)
        return positions_final


def neigh_dis(atoms1, atoms2, metal="Cu", distance_limit=4):
    positions1 = atoms1.get_positions()
    positions2 = atoms2.get_positions()
    chemical_symbols1 = atoms1.get_chemical_symbols()
    chemical_symbols2 = atoms2.get_chemical_symbols()

    # Find the index of atoms of the given metal in atoms2
    metal_index = [i for i in range(
        len(chemical_symbols1)) if metal in chemical_symbols2[i]]

    index_all = [metal_index]
    positions2_copy = copy.deepcopy(positions2)

    for index_i in index_all:
        if len(index_i) != 0:
            all_array = new_positions_all(positions2_copy, index_i)
            min_positions = dis_min_positions(positions1, all_array)
            positions2_copy = min_positions

    atoms2_copy = copy.deepcopy(atoms2)
    atoms2_copy.set_positions(positions2_copy)

    # Check the distance between atoms1 and atoms2
    distance = np.linalg.norm(positions1 - positions2_copy)
    print(f"Distance between atoms: {distance}")
    if distance <= distance_limit:
        return atoms1, atoms2_copy
    else:
        return None, None


def load_config():
    """Load parameters from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Structure processing with configurable parameters.")
    parser.add_argument('--metal', type=str, default='Cu',
                        help="Metal type to look for in atoms (default: Cu)")
    parser.add_argument('--distance_limit', type=float, default=4.0,
                        help="Distance limit for atoms to be considered (default: 4.0)")
    return parser.parse_args()


if __name__ == "__main__":
    # Load configuration from command line arguments
    args = load_config()

    atoms = ase.io.read("meta_all.db", index=":")
    metal = args.metal  # Set the metal from command line argument
    # Set the distance limit from command line argument
    distance_limit = args.distance_limit

    filename = "network_cluster.out"

    if os.path.exists(filename):
        os.remove(filename)

    if os.path.exists("neb_data"):
        os.system("rm -r neb_data/")

    # Create the output file and add headers
    f = open(filename, "w")
    f.write("%s  %s  \n" % ("index_i".ljust(10),
                            "index_j".ljust(10)))
    f.close()

    number_str = 0
    for i in range(len(atoms)-1):
        for j in range(i+1, len(atoms)):
            atoms1 = atoms[i]
            atoms2 = atoms[j]
            atoms1, atoms2 = neigh_dis(
                atoms1, atoms2, metal=metal, distance_limit=distance_limit)

            if atoms1 is not None and atoms2 is not None:
                # Create directories and save trajectory files
                number_str += 1
                if os.path.exists(f"./data{number_str}"):
                    os.system(f"rm -r ./data{number_str}")
                os.mkdir(f"./data{number_str}")
                ase.io.write(f"./data{number_str}/ini.traj", atoms1)
                ase.io.write(f"./data{number_str}/final.traj", atoms2)

                # Write to the output file
                f = open(filename, "a")
                f.write("%s  %s \n" % (str(i).ljust(10), str(j).ljust(10)))
                f.close()
