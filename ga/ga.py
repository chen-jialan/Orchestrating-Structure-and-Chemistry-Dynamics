from random import random
from ase.io import write
#from gpaw import PW, Mixer, FermiDirac
from gpaw.new.ase_interface import GPAW
from ase.optimize import LBFGS
from ase.optimize import GPMin
from ase.parallel import world
# import ase.dft.kpoints
# from ase.calculators.vasp import Vasp
# from ase.calculators.eann import EANN
import numpy as np
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
#from ase.ga.standardmutations import PermutationMutation
from ase.ga.standardmutations import RotationalMutation
import os
import ase.io.vasp
#import tranfer
from ase.calculators.singlepoint import SinglePointCalculator
#from ase.calculators.manual_written import Manual_written
from ase.ga.convergence import GenerationRepetitionConvergence
from ase.io import read, write


#def cal_vasp(a):
#    path = os.getcwd()
#    calculator = 'calculator'
#    path_calculator = path + '/' + calculator
#    if not os.path.exists(path_calculator):
#        os.mkdir(path_calculator)
#    os.chdir(path_calculator)
#    # ---------------poscar------------------------
#    ase.io.write('POSCAR', a, format='vasp', vasp5='True')
#    # ---------------incar, kpoints---------------------
#    os.system('cp ../INCAR ../KPOINTS ../tranfer.py .')
#    # -------------potcar auto get-----------------------
#    os.system('potcar.sh  $(head -n 6  POSCAR | tail -n 1)')
#    # ------------------------if or not you change in running vasp------------------
#    os.system('mpirun -np $NPROCS --hostfile $PBS_NODEFILE $VASP_EXE')
#    if not os.path.exists("./data"):
#        os.system("mkdir ../data")
#    l = len(os.listdir("../data"))
#    os.system("cp OUTCAR ../data/OUTCAR-%s" % l
#              )
#    #os.system('mpirun -np $NPROCS --hostfile $PBS_NODEFILE $VASP_EXE > p')
#    a_new_position = read('CONTCAR').get_positions()
#    energy = tranfer.Energy_VASP()
#    a.set_positions(a_new_position)
#    a.set_calculator(SinglePointCalculator(a, energy=energy))
#    a.get_potential_energy()
#    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
#    os.chdir(path)
#    return a


def set_calc(i=0):
    params = {'convergence': {'density': 1e-03, 'energy': 1e-4},
          'kpts': {'size': (1,1,1), 'gamma': True},
          'setups':{'Ti': ':d,4.2'},
          'random': True,
          'mode': {'ecut': 400, 'name': 'pw', 'force_complex_dtype': False},
          'occupations': {'name': 'fermi-dirac', 'width': 0.1},
          #'mixer': {'method': 'pulay', 'backend': 'noreset'},
          'txt': f'./result/data_{i}.txt',
          'parallel': {'gpu': True},
          'xc': 'PBE'}
    return GPAW(**params)
    #return GPAW(mode=PW(400),
    #            txt=f'./result/data_{i}.txt',
    #            kpts={'size': (1,1,1), 'gamma': True},
    #            symmetry={'point_group': False},
    #            xc='PBE',
    #            #eigensolver="rmm-diis",
    #            parallel={'gpu': True},
    #            convergence={'energy': 1e-4,
    #                        'eigenstates': 5e-3,
    #                         'density': 1e-3,
    #                         'bands': 'occupied'},
    #            mixer=Mixer(0.1, 5, 50),
    #            #occupations=FermiDirac(0.2),
    #            #spinpol=True,
    #            maxiter=60)


if __name__ == "__main__":
    # Change the following four parameters to suit your needs
    population_size = 12
    crossover_probability = 0.9
    mutation_probability = 0.5
    # ------------------------iterations of GA----------------------
    iterations = 300

    # Initialize the different components of the GA
    da = DataConnection('gadb.db')
    # ------------------parameters of distance and if or not the same structures-------------------
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types,
                                        ratio_of_covalent_radii=0.7)

    comp = InteratomicDistanceComparator(n_top=n_to_optimize,
                                         pair_cor_cum_diff=0.015,
                                         pair_cor_max=0.7,
                                         dE=0.02,
                                         mic=False)
    # --------------------parameters of crossover and mutation----------------------------
    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
    mutations = OperationSelector([1., 5., 1.],
                                  [MirrorMutation(blmin, n_to_optimize),
                                   RattleMutation(blmin, n_to_optimize),
                                   # PermutationMutation(n_to_optimize),    # use if there is not single component (such as CuOx)
                                   RotationalMutation(blmin, n_to_optimize)])

    # Relax all unrelaxed structures (e.g. the starting population)
    number_str = len([file_i for file_i in os.listdir("./result") if "data" in file_i])
    # --------start--------------------------------
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        number_str += 1
        print('Relaxing starting candidate {0}'.format(a.info['confid']))
        calc_gpaw = set_calc(i=number_str)
        a.calc = calc_gpaw
        #dyn = LBFGS(a,trajectory=f'./result/opt{number_str}.traj')
        dyn = GPMin(a,trajectory=f'./result/opt{number_str}.traj')
        dyn.run(fmax=0.3, steps=100)
        a.get_potential_energy()
        a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
        da.add_relaxed_step(a)

    # create the population
    population = Population(data_connection=da,
                            population_size=population_size,
                            comparator=comp,
                            logfile='log.txt',)
    # -------------------------convergence of ga, however in many times it is not used ----------------
    cc = GenerationRepetitionConvergence(population, 3)
    # test n_to_test new candidates
    if os.path.exists('./allcon'):
        os.system('rm -r allcon')
    os.system('mkdir ./allcon')
    if os.path.exists('./step'):
        os.remove('./step')

    # --------------------ga crossover and mutation------------------------
    for i in range(iterations):
        if cc.converged():
            print('converged')
            break
        print('Now starting configuration number {0}'.format(i))
        with open('step', 'a') as f:
            f.write('Now starting configuration number %s\n' % format(i))
        if random() < crossover_probability:
            a1, a2 = population.get_two_candidates()
            a3, desc = pairing.get_new_individual([a1, a2])
            if a3 is None:
                continue
            da.add_unrelaxed_candidate(a3, description=desc)
            # Check if we want to do a mutation
            if random() < mutation_probability:
                a3_mut, desc = mutations.get_new_individual([a3])
                if a3_mut is not None:
                    da.add_unrelaxed_step(a3_mut, desc)
                    a3 = a3_mut
        else:
            a1, a2 = population.get_two_candidates()
            if random() < 1/2:
                a3_mut, desc = mutations.get_new_individual([a1])
            else:
                a3_mut, desc = mutations.get_new_individual([a2])
            if a3_mut is not None:
                da.add_unrelaxed_step(a3_mut, desc)
                a3 = a3_mut
        # Relax the new candidate
        #a3 = cal_vasp(a3)
        number_str += 1
        calc_gpaw = set_calc(i=number_str)
        a3.calc = calc_gpaw
        #dyn = LBFGS(a3,trajectory=f'./result/opt{number_str}.traj')
        dyn = GPMin(a3,trajectory=f'./result/opt{number_str}.traj')
        dyn.run(fmax=0.1, steps=100)
        a3.info['key_value_pairs']['raw_score'] = -a3.get_potential_energy()
        da.add_relaxed_step(a3)
        population.update()
        if i % 10 == 9:
            write('all_candidates%s.traj' % i, da.get_all_relaxed_candidates())
            os.system('mv all_candidates%s.traj allcon/.' % i)

    write('all_candidates.traj', da.get_all_relaxed_candidates())
