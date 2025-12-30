import ase.io
import os
import numpy as np
import argparse


def load_config():
    """Load parameters from command line arguments."""
    parser = argparse.ArgumentParser(
        description="NEB barrier calculation and energy extraction.")

    # Add arguments for number of images, maximum force (fmax), and energy threshold
    parser.add_argument('--number_images', type=int, default=4,
                        help="Number of intermediate images (default: 4)")
    parser.add_argument('--fmax', type=float, default=0.15,
                        help="Maximum force for optimization (default: 0.15)")
    parser.add_argument('--energy_threshold', type=float, default=1e-10,
                        help="Energy threshold for comparison (default: 1e-10)")

    return parser.parse_args()


if __name__ == "__main__":
    # Load configuration from command line arguments
    args = load_config()

    # Set up directories
    fold = "neb_data"
    if os.path.exists(fold):
        os.system(f"rm -r {fold}")
    os.mkdir(fold)

    atoms = ase.io.read("neb.traj", index=":")

    # Find the point where energy is stable
    number_images = 0
    energy1 = atoms[0].get_potential_energy()
    for i in range(1, len(atoms)):
        number_images += 1
        energy2 = atoms[i].get_potential_energy()
        if abs(energy1 - energy2) <= args.energy_threshold:
            break

    # Extract energy values for the final images
    energy_all = []
    for i in range(len(atoms) - number_images, len(atoms)):
        energy = atoms[i].get_potential_energy()
        energy_all.append(energy)

    # Adjust energies
    energy_all = [energy - energy_all[0] for energy in energy_all]
    energy_intermidia = [energy_all[i] for i in range(1, len(energy_all) - 1)]

    print(energy_all)

    # Find the maximum energy in the intermediate images
    if min(energy_intermidia) > 0:
        max_energy_index = energy_all.index(max(energy_intermidia))
        atoms_ts = atoms[len(atoms) - number_images + max_energy_index]

        # Calculate forces and check if the optimization is done
        forces = atoms_ts.get_forces()
        print(np.max(np.linalg.norm(forces, axis=1)), max_energy_index)

        if np.max(np.linalg.norm(forces, axis=1)) <= args.fmax:
            print("done")
            # Write the results to a file
            f = open(f"{fold}/barrier.txt", "w")
            ini_energy = energy_all[0]
            final_energy = energy_all[-1]
            ts_energy = max(energy_intermidia)
            energy_forward = ts_energy - ini_energy
            energy_backward = ts_energy - final_energy

            f.write(f"{str(round(ini_energy, 5)).ljust(10)} {str(round(ts_energy, 5)).ljust(10)} "
                    f"{str(round(final_energy, 5)).ljust(10)} {str(round(energy_forward, 5)).ljust(10)} "
                    f"{str(round(energy_backward, 5)).ljust(10)}\n")
            f.close()

            # Save the final atoms
            atoms_final = [
                atoms[len(atoms) - number_images], atoms_ts, atoms[-1]]
            ase.io.write(f"{fold}/ts.traj", atoms_final)
