# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys

import numpy as np
import sparse as sp
from scipy.sparse import coo_matrix
from ase import Atoms

from dscribe.descriptors.descriptorlocal import DescriptorLocal
from dscribe.ext import EANNWrapper
import dscribe.utils.geometry
import copy


class EANN(DescriptorLocal):
    """Implementation of Atom-Centered Symmetry Functions.

    Notice that the species of the central atom is not encoded in the output,
    only the surrounding environment is encoded. In a typical application one
    can train a different model for each central species.

    For reference, see:
        "Atom-centered symmetry functions for constructing high-dimensional
        neural network potentials", Jörg Behler, The Journal of Chemical
        Physics, 134, 074106 (2011), https://doi.org/10.1063/1.3553717
    """

    def __init__(
        self,
        r_cut,
        r_cut_min=2,
        orbital=2,
        nwave=10,
        alpha=1.00,
        species=None,
        periodic=False,
        sparse=False,
        dtype="float64",
    ):
        """
        Args:
            r_cut (float): The smooth cutoff value in angstroms. This cutoff
                value is used throughout the calculations for all symmetry
                functions.
            ortital (int): L
            nwave (int) : number of wave
            alpha (double) : parameter of gaussian function
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical species as low as possible is
                preferable.
            periodic (bool): Set to true if you want the descriptor output to
                respect the periodicity of the atomic systems (see the
                pbc-parameter in the constructor of ase.Atoms).
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        self.eann_wrapper = EANNWrapper()
        super().__init__(periodic=periodic, sparse=sparse, dtype=dtype)

        # Setup
        self.species = species
        self.orbital = orbital
        self.nwave = nwave
        self.alpha = alpha
        self.r_cut = r_cut
        self.r_cut_min = r_cut_min
        np.random.seed(5)
        if self.species != None:
            self.c = np.random.random((len(species), nwave))
            self.rs = np.ones(len(species))
        else:
            ValueError("the species is none, please add it")
        #self.eann_wrapper = EANNWrapper()

    def create(
        self, system, centers=None, n_jobs=1, only_physical_cores=False, verbose=False
    ):
        """Return the EANN output for the given systems and given centers.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            centers (list): Indices of the atoms to use as EANN centers. If no
                centers are defined, the output will be created for all atoms in
                the system. When calculating output for multiple systems,
                provide the centers as a list for each system.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the used cpus
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | sparse.COO: The EANN output for the given
            systems and centers. The return type depends on the
            'sparse'-attribute. The first dimension is determined by the amount
            of centers and systems and the second dimension is determined by
            the get_number_of_features()-function. When multiple systems are
            provided the results are ordered by the input order of systems and
            their centers.
        """
        # Validate input / combine input arguments
        if isinstance(system, Atoms):
            system = [system]
            centers = [centers]
        if centers is None:
            inp = [(i_sys,) for i_sys in system]
        else:
            inp = list(zip(system, centers))

        # Determine if the outputs have a fixed size
        n_features = self.get_number_of_features()
        static_size = None
        if centers is None:
            n_centers = len(inp[0][0])
        else:
            first_sample, first_pos = inp[0]
            if first_pos is not None:
                n_centers = len(first_pos)
            else:
                n_centers = len(first_sample)

        def is_static():
            for i_job in inp:
                if centers is None:
                    if len(i_job[0]) != n_centers:
                        return False
                else:
                    if i_job[1] is not None:
                        if len(i_job[1]) != n_centers:
                            return False
                    else:
                        if len(i_job[0]) != n_centers:
                            return False
            return True

        if is_static():
            static_size = [n_centers, n_features]

        # Create in parallel
        output = self.create_parallel(
            inp,
            self.create_single,
            n_jobs,
            static_size,
            only_physical_cores,
            verbose=verbose,
        )

        return output

    def create_single(self, system, centers=None):
        """Creates the descriptor for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            centers (iterable): Indices of the atoms around which the EANN
                will be returned. If no centers defined, EANN will be created
                for all atoms in the system.

        Returns:
            np.ndarray: The EANN output for the given system and centers. The
            first dimension is given by the number of centers and the second
            dimension is determined by the get_number_of_features()-function.
        """
        # Check if there are types that have not been declared
        self.check_atomic_numbers(system.get_atomic_numbers())
        number_atoms = system.get_atomic_numbers()
        system_copy = copy.deepcopy(system)
        # Create C-compatible list of atomic indices for which the EANN is
        # calculated
        calculate_all = False
        if centers is None:
            calculate_all = True
            indices = np.arange(len(system))
        else:
            indices = centers

        # If periodicity is not requested, and the output is requested for all
        # atoms, we skip all the intricate optimizations that will make things
        # actually slower for this case.
        if calculate_all and not self.periodic:
            n_atoms = len(system)
            all_pos = system.get_positions()
            dmat = dscribe.utils.geometry.get_adjacency_matrix(
                self.r_cut, all_pos, all_pos
            )
        # Otherwise the amount of pairwise distances that are calculated is
        # kept at minimum. Only distances for the given indices (and possibly
        # the secondary neighbours if G4 is specified) are calculated.
        else:
            # Create the extended system if periodicity is requested. For EANN only
            # the distance from central atom needs to be considered in extending
            # the system.
            if self.periodic:
                system = dscribe.utils.geometry.get_extended_system(
                    system, self.r_cut, return_cell_indices=False
                )

            # First calculate distances from specified centers to all other
            # atoms. This is already enough for everything else except G4.
            n_atoms = len(system)
            all_pos = system.get_positions()
            # print(indices)
            central_pos = all_pos[indices]
            dmat_primary = dscribe.utils.geometry.get_adjacency_matrix(
                self.r_cut, central_pos, all_pos
            )

            # Create symmetric full matrix
            col = dmat_primary.col
            row = [
                indices[x] for x in dmat_primary.row
            ]  # Fix row numbering to refer to original system
            data = dmat_primary.data
            print(data.shape)
            #print(f"col {col}")
            #print(f"row {row}")
            #print(f"data {data}")

            # dmat_primary_min = dscribe.utils.geometry.get_adjacency_matrix(
            #    self.r_cut_min, central_pos, all_pos
            # )
            # Create symmetric full matrix
            #col_min = dmat_primary_min.col
            # row_min = [
            #    indices[x] for x in dmat_primary_min.row
            # ]  # Fix row numbering to refer to original system
            #data_min = dmat_primary_min.data
            ##print(f"col_min {col_min}")
            ##print(f"row_min {row_min}")
            ##print(f"data_min {data_min}")

            #print(data.shape, data_min.shape)
            #print(col.shape, col_min.shape)
            #index_in_r_min = []
            #data_list = list(data)
            #data_min_list = list(data_min)
            # for i in range(len(data_list)):
            #    if data_list[i] in data_min_list and data_list[i] > 0:
            #        #        print(data_list[i])
            #        index_in_r_min.append(i)
            #print(f"index = {index_in_r_min}")
            #col = np.delete(col, index_in_r_min)
            # row = [item for index, item in enumerate(
            #    row) if index not in index_in_r_min]
            #data = np.delete(data, index_in_r_min)
            #data = data.reshape(-1, 1)
            # print(data.shape)
            print(data.shape, col.shape, len(col))
            dmat = coo_matrix((data, (row, col)), shape=(n_atoms, n_atoms))
            dmat_lil = dmat.tolil()
            dmat_lil[col, row] = dmat_lil[row, col]

            if self.orbital > 0:
                neighbour_indices = np.unique(col)
                neigh_pos = all_pos[neighbour_indices]
                dmat_secondary = dscribe.utils.geometry.get_adjacency_matrix(
                    self.r_cut, neigh_pos, neigh_pos
                )
                col = [
                    neighbour_indices[x] for x in dmat_secondary.col
                ]  # Fix col numbering to refer to original system
                row = [
                    neighbour_indices[x] for x in dmat_secondary.row
                ]  # Fix row numbering to refer to original system
                dmat_lil[row, col] = np.array(dmat_secondary.data)

            dmat = dmat_lil.tocoo()
        # Get adjancency list and full dense adjancency matrix
        neighbours = dscribe.utils.geometry.get_adjacency_list(dmat)
        neighbours_list = [neighbours[i] for i in range(len(number_atoms))]
        dmat_dense = np.full(
            (n_atoms, n_atoms), sys.float_info.max
        )   # The non-neighbor values are treated as "infinitely far".
        dmat_dense[dmat.col, dmat.row] = dmat.data
        #dmat_dense_list = dmat_dense.tolist()
        # dmat_dense_list = [dmat_dense_list[i]
        #                   for i in range(len(number_atoms))]
        # print(indices.tolist())
        # print(dmat_dense)
        # print(dmat_dense.tolist())
        # print(dmat_dense.tolist())
        # print(indices.tolist())
        # Calculate EANN with C++
        if type(indices) != list:
            if type(indices) == range:
                indices = [i for i in indices]
            else:
                indices = indices.tolist()
        output = np.array(
            self.eann_wrapper.create(
                system.get_positions().tolist(),
                system.get_atomic_numbers().tolist(),
                dmat_dense.tolist(),
                neighbours,
                indices,
            ),
            dtype=np.float64,
        )
        # output = np.array(
        #    self.eann_wrapper.create(
        #        system_copy.get_positions().tolist(),
        #        system_copy.get_atomic_numbers().tolist(),
        #        dmat_dense_list,
        #        neighbours_list,
        #        indices.tolist(),
        #    ),
        #    dtype=np.float64,
        # )

        return output

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        return self.nwave

    def validate_derivatives_method(self, method, attach):
        if not attach:
            raise ValueError(
                "EANN derivatives can only be calculated with attach=True."
            )
        return super().validate_derivatives_method(method, attach)

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        """Used to check the validity of given atomic numbers and to initialize
        the C-memory layout for them.

        Args:
            value(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        self._set_species(value)
        #print(self._atomic_numbers.tolist(), self.eann_wrapper.__dict__)
        self.eann_wrapper.atomic_numbers = self._atomic_numbers.tolist()

    @property
    def r_cut(self):
        return self.eann_wrapper.r_cut

    @r_cut.setter
    def r_cut(self, value):
        """Used to check the validity of given radial cutoff.

        Args:
            value(float): Radial cutoff.
        """
        if value <= 0:
            raise ValueError("Cutoff radius should be positive.")
        self.eann_wrapper.r_cut = value

    @property
    def nwave(self):
        return self.eann_wrapper.nwave

    @nwave.setter
    def nwave(self, value):
        """Used to check the validity of given nwave.

        Args:
            value(int): nwave.
        """
        if value <= 0:
            raise ValueError(" nwave should be positive.")
        self.eann_wrapper.nwave = value

    @property
    def orbital(self):
        return self.eann_wrapper.orbital

    @orbital.setter
    def orbital(self, value):
        """Used to check the validity of given orbital.

        Args:
            value(int): orbital
        """
        if value < 0:
            raise ValueError("Orbital should be positive.")
        self.eann_wrapper.orbital = value

    @property
    def alpha(self):
        return self.eann_wrapper.alpha

    @alpha.setter
    def alpha(self, value):
        """used to check the validity of given alpha.
        args:
            alpha(numpy): alpha
        """
        self.eann_wrapper.alpha = value

    @property
    def rs(self):
        return self.eann_wrapper.rs

    @alpha.setter
    def rs(self, value):
        """used to check the validity of given alpha.
        args:
            alpha(numpy): alpha
        """
        self.eann_wrapper.rs = value

    @property
    def c(self):
        return self.eann_wrapper.c

    @c.setter
    def c(self, value):
        """used to check the validity of given alpha.
        args:
            alpha(numpy): alpha
        """
        self.eann_wrapper.c = value
