import csv
import os
import re
import ase.io
import numpy as np


def find_similar_times(csv_file):
    """
    This function reads the CSV file and compares each row with the last row (energy values).
    It returns the rows where the differences in energy values are small (less than 1e-3).
    """
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    if len(data) < 2:  # At least the header and one data row are needed
        print("CSV file has insufficient data.")
        return [], None, []

    # Extract headers and the index of 'Times' column
    headers = data[0]
    time_index = headers.index('Times')

    # Extract the data and convert to floats where possible
    rows = []
    times = []
    for row in data[1:]:  # Skip header row
        try:
            row_data = [
                float(cell) if i >= time_index else cell for i, cell in enumerate(row)
            ]
            rows.append(row_data)
            times.append(float(row[time_index]))
        except ValueError:
            continue  # Skip rows with invalid data

    if not rows:
        print("No valid data rows found.")
        return [], None, []

    last_row = rows[-1]  # Last row (excluding the 'Times' column)
    # Find similar rows by comparing values of all columns (except 'Times')
    similar_times = []
    for i, row in enumerate(rows[:-1]):  # Skip the last row
        all_columns_similar = True
        # Compare all columns except the 'Times' column
        for col_idx in range(time_index + 1, len(row)):
            if abs(row[col_idx] - last_row[col_idx]) >= 1e-3:
                all_columns_similar = False
                break

        if all_columns_similar:
            # i+1 because we skipped the header row
            similar_times.append((i+1, times[i]))

    # Extract energy values from the matched rows
    energy_values = np.array([row[2:] for row in rows[:len(similar_times)]])

    return similar_times, times[-1], energy_values


def get_species():
    """
    Extracts initial and final species from 'network_cluster.out' file.
    Returns a sorted list of unique species.
    """
    initial = []
    final = []
    barrier_forward = []
    barrier_backward = []

    with open("network_cluster.out", "r") as f:
        for a, line in enumerate(f.readlines()):
            if a >= 1:  # Skip the header
                line_i = line.strip().split()
                initial.append(int(line_i[0]))
                final.append(int(line_i[1]))
                barrier_forward.append(float(line_i[-2]))
                barrier_backward.append(float(line_i[-1]))

    return list(set(initial + final))


def rmsd(atoms, metal="Cu"):
    """
    Calculate RMSD between positions of a specific atom type (default "Cu").
    Returns RMSD for all atoms in the system.
    """
    positions = atoms[0].get_positions()
    chem = atoms[0].get_chemical_symbols()
    metal_index = [i for i, chem_i in enumerate(chem) if chem_i == metal]
    p1 = positions[metal_index]

    rmsd_all = []
    for i in range(1, len(atoms)):
        atoms_i = atoms[i]
        pos_i = atoms_i.get_positions()
        chem = atoms_i.get_chemical_symbols()
        metal_index = [i for i, chem_i in enumerate(chem) if chem_i == metal]
        p2 = pos_i[metal_index]

        rmsd = np.sqrt(np.sum(np.linalg.norm(
            p1 - p2, axis=1)) / len(metal_index))
        rmsd_all.append(rmsd)

    return np.array(rmsd_all)


# Main execution
if __name__ == "__main__":
    # Read atoms from the database
    atoms = ase.io.read("meta_all.db", index=":")
    metal = "Cu"
    species_all = get_species()

    # Find the folder with '5cu_' in the name
    filename = [file_i for file_i in os.listdir("./") if "5cu_" in file_i][0]

    # Get numeric folders from the directory
    items = os.listdir(f'{filename}/.')
    # Match numbers with a decimal point (e.g., 300.0)
    pattern = re.compile(r'^\d+\.\d+$')
    numeric_folders = [item for item in items if pattern.match(item)]

    # Read RMSD data
    rmsd_all = rmsd(atoms, metal=metal)

    # Process each folder
    for folder in sorted(numeric_folders):
        T = folder
        #print(f"Processing temperature: {T}")

        # Path to the CSV file
        csv_file = f'{filename}/{T}/coverage.csv'
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} does not exist.")
            continue

        # Find similar times based on the energy comparison
        result, last_time, coverage = find_similar_times(csv_file)

        # Calculate RMSD sum weighted by coverage
        #rmsd_sum = np.sum(rmsd_all[species_all] * coverage[species_all])

        if result:
            for row_num, time_val in result:
                print(f"temperature: {T}, Time: {time_val} ")
                break
        else:
            print("No matching rows found.")
