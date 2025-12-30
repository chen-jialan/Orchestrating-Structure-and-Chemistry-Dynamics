import ase.io
from ase.units import Pascal, m
from ase.optimize import BFGS
from ase.neb import interpolate
from ase.dyneb import DyNEB
from gpaw import PW, Mixer, FermiDirac, GPAW


def make_calculator(index=0):
    params = {'convergence': {'density': 1e-03, 'energy': 1e-4},
              'kpts': {'size': (1, 1, 1), 'gamma': True},
              'setups': {'Ti': ':d,4.2'},
              'random': True,
              'mode': {'ecut': 400, 'name': 'pw', 'force_complex_dtype': False},
              'occupations': {'name': 'fermi-dirac', 'width': 0.1},
              'txt': f'./data_{index}.txt',
              'xc': 'PBE'}
    return GPAW(**params)


if __name__ == "__main__":
    # Load configuration from command line arguments
    number = 4
    fmax = 0.05
    steps = 200
    # Create the band of images, attaching a calc to each.
    initial = ase.io.read('ini.traj')
    initial.set_pbc([True, True, False])
    final = ase.io.read('final.traj')
    final.set_pbc([True, True, False])

    # Create the list of images with the specified number of intermediate images
    images = [initial]
    for index in range(number):
        images += [initial.copy()]
    images += [final]

    # Attach the calculator to each image
    for index in range(number + 2):
        images[index].calc = make_calculator(index)
        images[index].get_potential_energy()

    # Create and relax the DyNEB.
    neb = DyNEB(images, climb=True, scale_fmax=2, method='spline')
    neb.interpolate('idpp')

    # Perform optimization with user-specified fmax and steps
    opt = BFGS(neb, logfile='neb.log', trajectory='neb.traj')
    opt.run(fmax=fmax, steps=steps)
