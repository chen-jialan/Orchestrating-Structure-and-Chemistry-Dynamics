import ase.io
from ase.units import Pascal, m
from ase.optimize import BFGS
from ase.neb import interpolate
from ase.dyneb import DyNEB
from gpaw import PW, Mixer, FermiDirac, GPAW


def make_calculator(index=0):
    params = {'convergence': {'density': 1e-03, 'energy': 1e-4},
          'kpts': {'size': (1,1,1), 'gamma': True},
          'setups':{'Ti': ':d,4.2'},
          'random': True,
          'mode': {'ecut': 400, 'name': 'pw', 'force_complex_dtype': False},
          'occupations': {'name': 'fermi-dirac', 'width': 0.1},
          #'mixer': {'method': 'pulay', 'backend': 'noreset'},
          'txt': f'./data_{index}.txt',
          #'parallel': {'gpu': True},
          'xc': 'PBE'}
    return GPAW(**params)


# Create the band of images, attaching a calc to each.
initial = ase.io.read('ini.traj')
initial.set_pbc([True, True, False])
final = ase.io.read('final.traj')
final.set_pbc([True, True, False])
images = [initial]
number = 4
for index in range(number):
    images += [initial.copy()]
images += [final]

for index in range(number+2):
    images[index].calc = make_calculator(index)
    images[index].get_potential_energy()

# interpolate(images,'idpp')

# Create and relax the DyNEB.
neb = DyNEB(images, climb=True, scale_fmax=1, method='spline')
neb.interpolate('idpp')
opt = BFGS(neb, logfile='neb.log', trajectory='neb.traj')
opt.run(fmax=0.1,steps=200