import csv
import os
import re
import ase.io
import numpy as np


def find_similar_times(csv_file, energy_all, species_all):
    """
    This function reads the CSV file and compares each row with the last row (energy values).
    It returns the rows where the differences in energy values are small (less than 1e-3).
    """
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    if len(data) < 2:  # At least the header and one data row are needed
        print("CSV file has insufficient data.")
        return [], []

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
        return [], []

    last_row = rows[-1]  # Last row (excluding the 'Times' column)
    energy_index = energy_all[species_all]

    # List to store times where energy differences are small
    similar_times = []
    energy_coverage = []

    for i, row in enumerate(rows[:-1]):  # Skip the last row
        all_columns_similar = True
        for col_idx in range(time_index, len(row)):
            if abs(row[col_idx] - last_row[col_idx]) >= 1e-3:
                all_columns_similar = False
                break

        # Calculate the energy for the current row
        energy_i = np.sum(energy_index * np.array(row[2:]))
        energy_coverage.append(energy_i)

        if all_columns_similar:
            # i+1 because we skipped the header row
            similar_times.append((i+1, times[i]))

        print(times[i], energy_i)

    return similar_times, energy_coverage


def ini_energy(atoms):
    """
    This function computes the energy difference between the minimum energy
    and each of the atoms' energy.
    """
    energy_all = [atoms_i.get_potential_energy() for atoms_i in atoms]
    energy_min = min(energy_all)
    energy_all = [energy_i - energy_min for energy_i in energy_all]
    return np.array(energy_all)


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


def get_numeric_folders(path):
    """
    Retrieves all folders or files matching the numeric pattern like 300.0, 310.0 etc.
    """
    items = os.listdir(path)
    # Match numbers with a decimal point (e.g., 300.0)
    pattern = re.compile(r'^\d+\.\d+$')
    numeric_folders = [item for item in items if pattern.match(item)]
    return sorted(numeric_folders)


# Main execution
if __name__ == "__main__":
    # Read atoms from database
    atoms = ase.io.read("meta_all.db", index=":")
    energy_all = ini_energy(atoms)

    # Get the species list
    species_all = get_species()

    # Find the folder with '5cu_' in the name
    filename = [file_i for file_i in os.listdir("./") if "5cu_" in file_i][0]

    # Get numeric folders from the directory
    numeric_folders = get_numeric_folders(f'{filename}/.')

    for folder in numeric_folders:
        if abs(float(folder) - 300) <= 0.001:  # Matching the folder name close to 300.0
            T = folder
            print(f"Processing temperature: {T}")

            # Path to the CSV file
            csv_file = f'{filename}/{T}/coverage.csv'
            if not os.path.exists(csv_file):
                print(f"CSV file {csv_file} does not exist.")
                continue

            # Find similar times based on the energy comparison
            result, coverage = find_similar_times(
                csv_file, energy_all, species_all)

            if result:
                for row_num, time_val in result:
                    print(f"{T}, Time: {time_val}")
                break
            else:
                print("No similar rows found")
            break
