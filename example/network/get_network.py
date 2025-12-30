import os
import re
import copy
import ase.io
import argparse


def load_config():
    """Load parameters from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and write network cluster data.")

    # Arguments to configure the input/output files and threshold values
    parser.add_argument('--filename_pattern', type=str, default='ts',
                        help="Pattern to match filenames (default: 'ts')")
    parser.add_argument('--barrier_file', type=str, default='barrier',
                        help="Path to the barrier data file (default: 'barrier')")
    parser.add_argument('--index_file', type=str, default='index.txt',
                        help="Path to the index file (default: 'index.txt')")
    parser.add_argument('--output_file', type=str, default='network_cluster.out',
                        help="Path to the output file (default: 'network_cluster.out')")

    return parser.parse_args()


if __name__ == "__main__":
    # Load configuration from command line arguments
    args = load_config()

    # Get all filenames with the pattern "ts"
    filename_all = [file_i for file_i in os.listdir(
        "./") if args.filename_pattern in file_i]

    # Extract numbers from the filenames for sorting
    number = [int(re.split(r'[^\d]+', file_i)[1]) for file_i in filename_all]
    number_copy = copy.deepcopy(number)
    number_copy.sort()
    index_number = [number.index(i) for i in number_copy]
    file_all = [filename_all[i] for i in index_number]

    # Read the barrier data
    ef_all = []
    eb_all = []
    with open(args.barrier_file, "r") as f:
        for line in f.readlines():
            line_i = line.strip().split()
            ef_all.append(float(line_i[-2]))
            eb_all.append(float(line_i[-1]))

    # Read the index data
    index_i_all = []
    index_j_all = []
    with open(args.index_file, "r") as f:
        for line in f.readlines():
            line_i = line.strip().split()
            index_i_all.append(int(line_i[-2]))
            index_j_all.append(int(line_i[-1]))

    # Write the output to the specified file
    with open(args.output_file, "w") as f:
        f.write(f"{'index_i'.ljust(10)}\t{'index_j'.ljust(10)}\t{'energy'.ljust(10)}\t{'forward'.ljust(10)}\t{'backward'.ljust(10)}\n")
        for i, index_ii in enumerate(number_copy):
            print(i, index_ii - 1)
            index_i_i = index_i_all[index_ii - 1]
            index_i_j = index_j_all[index_ii - 1]
            Ef = ef_all[i]
            Eb = eb_all[i]
            energy = ase.io.read(file_all[i], index="0").get_potential_energy()
            if Eb > 0:
                f.write(
                    f"{str(index_i_i).ljust(10)}\t"
                    f"{str(index_i_j).ljust(10)}\t"
                    f"{str(round(energy, 5)).ljust(10)}\t"
                    f"{str(round(Ef, 5)).ljust(10)}\t"
                    f"{str(round(Eb, 5)).ljust(10)}\n"
                )
