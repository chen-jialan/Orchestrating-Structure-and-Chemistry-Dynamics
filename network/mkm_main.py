# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:19:49 2023

@author: Jia-Lan Chen
"""

import math
from microkinetic import MicroKinetic
import copy
import numpy as np
import os
import argparse


def load_config():
    """Load parameters from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Microkinetic simulation parameters.")

    # Arguments for different simulation parameters
    parser.add_argument('--pressure', type=float, default=1e5,
                        help="Pressure in Pascals (default: 1e5)")
    parser.add_argument('--temperature_start', type=float, default=300,
                        help="Start temperature in Kelvin (default: 300)")
    parser.add_argument('--temperature_final', type=float, default=500,
                        help="Final temperature in Kelvin (default: 500)")
    parser.add_argument('--temperature_point', type=float,
                        default=10, help="Temperature step size (default: 10)")
    parser.add_argument('--method', type=str, default="LSODA",
                        help="Method for solving ODEs (default: LSODA)")
    parser.add_argument('--coverage_run', type=bool, default=True,
                        help="Whether to run coverage (default: True)")
    parser.add_argument('--order_run', type=bool, default=False,
                        help="Whether to run order (default: False)")
    parser.add_argument('--drc_run', type=bool, default=False,
                        help="Whether to run drc (default: False)")
    parser.add_argument('--apparent_energy_run', type=bool, default=False,
                        help="Whether to run apparent energy (default: False)")
    parser.add_argument('--t_max', type=float, default=1e0,
                        help="Maximum time for simulation (default: 1e0)")
    parser.add_argument('--dt', type=float, default=1e2,
                        help="Time step size (default: 1e2)")
    parser.add_argument('--filename', type=str, default="5cu",
                        help="Filename for the output (default: '5cu')")

    return parser.parse_args()


def input_data():
    initial = []
    final = []
    barrier_forward = []
    barrier_backward = []

    with open("network_cluster.out", "r") as f:
        a = -1
        for line in f.readlines():
            a += 1
            if a >= 1:
                line_i = line.strip().split()
                initial.append(line_i[0])
                final.append(line_i[1])
                barrier_forward.append(float(line_i[-2]))
                barrier_backward.append(float(line_i[-1]))

    species = copy.deepcopy(initial)
    species.extend(final)
    species = list(set(species))
    species = [spe for spe in species]
    species.sort()
    print(species)

    coverage = np.random.uniform(0.1, 1.0, len(species))
    coverage = coverage / np.sum(coverage)
    print(coverage)

    element_step = []
    for i in range(len(initial)):
        element_i = [[initial[i]], [final[i]], math.pow(10, 13), math.pow(
            10, 13), barrier_forward[i], barrier_backward[i]]
        element_step.append(element_i)
    return species, coverage, element_step


if __name__ == "__main__":
    # Load configuration from command line arguments
    args = load_config()

    species, coverage, element_step = input_data()

    # Cleaning up any previous data
    os.system(f"rm -r {args.filename}*")

    # Parameters for simulation
    pressure = args.pressure
    temperature_start = args.temperature_start
    temperature_final = args.temperature_final
    temperature_point = int(
        (temperature_final - temperature_start) / args.temperature_point) + 1
    t_max = args.t_max
    dt = args.dt

    coverage_run = args.coverage_run
    order_run = args.order_run
    drc_run = args.drc_run
    apparent_energy_run = args.apparent_energy_run

    # Method is RK45, RK23, DOP853, Radau, BDF, LSODA
    method = args.method

    # Initialize the MicroKinetic class with the provided parameters
    mk = MicroKinetic(
        species=species,
        coverage=coverage,
        t_max=t_max,
        temperature_start=temperature_start,
        temperature_final=temperature_final,
        temperature_point=temperature_point,
        reaction_species=[],
        product_species=[],
        pressure=pressure,
        element_step=element_step,
        method=method,
        coverage_run=coverage_run,
        order_run=order_run,
        drc_run=drc_run,
        apparent_energy_run=apparent_energy_run,
        filename=args.filename,
        dt=dt,
        ssia_calc=False,
        atol=1e-6,
        rtol=1e-6,
    )

    # Perform the post-analysis
    mk.post_analysis()
