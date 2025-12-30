import os
import numpy as np
import copy
import argparse
from ase.io import read, write
from ase.db import connect
from ase.ga.data import DataConnection


def neigh_dis(positions1, positions2):
    """Compute minimum distance between atoms in two structures."""
    x = positions1.shape[0]
    dist_min_all = []
    for i in range(x):
        positions_x_i = positions1[i]
        min_dis = np.min(np.linalg.norm(positions2 - positions_x_i, axis=1))
        dist_min_all.append(min_dis)
    return np.linalg.norm(np.array(dist_min_all))


def setup_directories():
    """Setup necessary directories and remove old ones."""
    if os.path.exists('./meta_all.db'):
        os.remove('./meta_all.db')
    if os.path.exists('./pos'):
        os.system('rm -r ./pos')
    os.mkdir('./pos')


def write_energy_data(energy_copy, index_copy):
    """Write sorted energy data to a file."""
    with open('./energy_ga_all', "w") as f1:
        for i in range(len(energy_copy)):
            f1.write('%s %s\n' % (energy_copy[i], index_copy[i]))


def select_structures(da, energy_copy, limit_meta, limit_energy, limit_struture):
    """Select and write structures based on energy and structural similarity."""
    atom_all = []
    a = 0
    energy_write = min(energy_copy)

    with open('./energy', "w") as f:
        for i in range(len(energy_copy)):
            if abs(energy_copy[i] - min(energy_copy)) <= limit_meta:
                atoms = da[i]
                if len(atom_all) == 0:
                    atom_all.append(atoms)
                    write('POSCAR-final', atoms, format='vasp', vasp5='True')
                    os.system('mv POSCAR-final pos/POSCAR-%s' % a)
                    f.write('%s %s\n' % (a, energy_copy[i]))
                    a += 1
                else:
                    j = 1
                    a2_add_position = atoms.get_positions()
                    for atom_e in atom_all:
                        a1_position = atom_e.get_positions()
                        different_str = neigh_dis(a2_add_position, a1_position)
                        print(different_str)
                        if different_str <= limit_struture:
                            j = 0
                            break
                    if j == 1 and abs(energy_copy[i] - energy_write) >= limit_energy:
                        print('add structure %s' % i)
                        write('pos/POSCAR-%s' % a, atoms, format='vasp', vasp5='True')
                        atom_all.append(atoms)
                        f.write('%s %s\n' % (a, energy_copy[i]))
                        energy_write = energy_copy[i]
                        a += 1
    os.system('mv energy pos/.')
    return atom_all


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Structure selection based on energy and geometry.")
    
    parser.add_argument('--limit_meta', type=float, default=0.5, help="Limit for energy difference compared to most stable")
    parser.add_argument('--limit_energy', type=float, default=0.0, help="Minimum energy difference for considering new structures")
    parser.add_argument('--limit_structure', type=float, default=1.5, help="Limit for structural similarity")

    args = parser.parse_args()

    # Set up directories
    setup_directories()

    # Read data
    da = read("all_candidates.traj", index=":")

    # ------------------parameters----------------
    limit_meta = args.limit_meta  # External control
    limit_energy = args.limit_energy  # External control
    limit_struture = args.limit_structure  # External control

    # Get energies and indices
    energy = [stru.get_potential_energy() for stru in da]
    index_atom = [i for i in range(len(da))]

    # Sort energies and track corresponding indices
    energy_copy = copy.deepcopy(energy)
    energy_copy.sort()
    index_copy = [index_atom[energy.index(e)] for e in energy_copy]

    # Write energy data
    write_energy_data(energy_copy, index_copy)

    # Select structures based on energy and structural similarity
    atom_all = select_structures(da, energy_copy, limit_meta, limit_energy, limit_struture)

    # Write to database
    db = connect('meta_all.db')
    for atoms in atom_all:
        db.write(atoms, relaxed=True)


if __name__ == "__main__":
    main()

