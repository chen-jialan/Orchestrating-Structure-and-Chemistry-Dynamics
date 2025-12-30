import os
import sys
import numpy as np
import ase.io.vasp
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms
from ase.build import fcc111
import argparse


def load_config():
    """Load parameters from command line arguments."""
    parser = argparse.ArgumentParser(description="Generate and optimize atomic clusters.")

    # Arguments to configure the atom type and number of atoms
    parser.add_argument('--atomtype', type=int, required=True, help="Atomic number of the atom type (e.g., 47 for Ag)")
    parser.add_argument('--number', type=int, required=True, help="Number of Rh atoms in the cluster")
    parser.add_argument('--db_file', type=str, default='gadb.db', help="Database file name (default: 'gadb.db')")
    parser.add_argument('--poscar_file', type=str, default="POSCAR_base", help="POSCAR file to read for the slab (default: 'POSCAR_base')")

    return parser.parse_args()


if __name__ == "__main__":
    # Load configuration from command line arguments
    args = load_config()

    # Read the slab from the POSCAR file
    if not os.path.exists(args.poscar_file):
        print(f"Error: {args.poscar_file} does not exist.")
        sys.exit(1)

    atomtype = args.atomtype
    number = args.number
    db_file = args.db_file

    # Remove the database file if it already exists
    if os.path.exists(f'./{db_file}'):
        os.remove(f'./{db_file}')
    
    # Read the slab structure
    slab = ase.io.vasp.read_vasp(args.poscar_file)

    # Define the volume in which the adsorbed cluster is optimized
    pos = slab.get_positions()
    cell = slab.get_cell()

    # Define corner position (p0) and spanning vectors (v1, v2, v3)
    p0 = np.array([max(pos[:, 0])/3, max(pos[:, 1])/3, max(pos[:, 2])+1.6])
    v1 = cell[0, :] * 0.6
    v2 = cell[1, :] * 0.6
    v3 = cell[2, :]
    v3[2] = 3

    # Define the composition of the atoms to optimize, where `atomtype` corresponds to the element to use
    atom_numbers = number * [atomtype]

    # Define the closest distance two atoms of a given species can be to each other
    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                        ratio_of_covalent_radii=0.75)

    # Create the starting population
    sg = StartGenerator(slab, atom_numbers, blmin, box_to_place_in=[p0, [v1, v2, v3]])

    # Generate the starting population with a size proportional to `number`
    population_size = int(1.5 * number)
    starting_population = [sg.get_new_candidate() for i in range(population_size)]

    # Create the database to store information
    d = PrepareDB(db_file_name=db_file,
                  simulation_cell=slab,
                  stoichiometry=atom_numbers)

    # Add the unrelaxed candidates to the database
    for a in starting_population:
        d.add_unrelaxed_candidate(a)

    print(f"Generated and stored {population_size} candidates in the database '{db_file}'")

