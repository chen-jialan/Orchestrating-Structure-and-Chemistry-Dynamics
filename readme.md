# The Framework to **Orchestrate Structure and Chemistry Dynamics for Cluster Catalysis**

This repository provides a computational framework to investigate the **Orchestrating Structure and Chemistry Dynamics** in cluster-based catalysis. It integrates several modules for modeling, simulation, and analysis of catalytic behavior using GA and microkinetic modelling.

## 📌 Key Components

1. **Genetic Algorithm (GA)**: Used for global optimization in metastable state ensemble.
2. **Isomer Transform  Calculations**:  For identifying isomer forward and backward barrier energy.
3. **Isomer Network Generation:** For identifying isomer network.
4. **Microkinetic Modeling (MKM)**: For simulating isomer transform kinetics and equilibrium time.

##  👥 Developers

Jia-Lan Chen (jlchen20@mail.ustc.edu.cn)

Advisors: Jin-Xun Liu (jxliu86@ustc.edu.cn) and Wei-Xue Li (wxli70@ustc.edu.cn)

## 🧪 Methods

- **GA**: Integrated with ASE for global structure optimization.
- **Isomer Transform algorithm**: the connection of pairing isomers
- **MKM**: solve equilibrium time and coverage distribution by ordinary differential equations

## 📁 Contents

|   Folder   |                     Description                      |
| :--------: | :--------------------------------------------------: |
| `dataset/` | Isomers and transition state of network in this work |
|   `ga/`    |         Metastable structure search using GA         |
|   `neb/`   | Calculation of transition state of isomer transform  |
| `network/` |    Transition state search using  DyNEB with GPAW    |
| `example/` |              Example of this framework               |
| `dscribe/` |                  Atomic descriptor                   |

---

## ⚙️ Module Details

#### ✅ Searching metastable structures and pairing isomers by GA (`ga`)

These simulations are used to search metastable structures and pairing isomers. 

- **Genetic algorithm (GA)**

Used to optimize catalyst structures via evolutionary principles.

```python
python3 initial.py \
  --atomtype 29 \ 
  --number 8  \
   --db_file gadb.db \
  --poscar_file POSCAR_base 
```

Where the parameters are below:

1.   atomtype:  Atomic number of the atom type (e.g., 47 for Ag)
2.   number: Number of Rh atoms in the cluster
3.   db_file: Database file name (default: 'gadb.db')
4.   poscar_file: POSCAR file to read for the slab (default: 'POSCAR_base')



- **Metastable Ensemble**

Used to identify the metastable ensemble.

```python
python3 allga.py

python3 meta_all.py \
	--limit_meta 0.01 \
	--limit_energy 1 \
    --limit_structure 0.15
```

Where the parameters are below:

1.   limit_meta:  Limit for energy difference compared to most stable
2.   limit_energy: Minimum energy difference for considering new structures
3.   limit_structure: Limit for structural similarity



- **Pairing Isomers**

Used to identify the pairing isomers to connect the networks.

```python
python3 pair_neb.py \
	--metal Cu \
	--distance_limit 4
```

Where the parameters are below:

1.   metal:  Metal type to look for in atoms (default: Cu)
2.   distance_limit: Distance limit for atoms to be considered (default: 4.0)

#### **✅ ** Identification of transition state of isomer transition (`neb`)** 

This step involves identifying the transition state of isomer transition.

- **Transition state search**

  Identifies transition state of isomers.

```bash
gpaw -P 96 python neb.py
```

- **NEB barrier calculation**

  Obtain NEB barrier energy.

```python
python3 get_energy.py \
    --number_images 4 \
  --fmax 0.1 \
  --energy_threshold 0.01 \
```

Where the parameters are below:

1.   number_images:  Number of intermediate images (default: 4)
2.   fmax: Maximum force for optimization (default: 0.15)
3.   energy_threshold: Energy threshold for comparison (default: 1e-10)

#### **✅ Network (`network/`)**

Networks of isomers are obtained by **networkx**.

**Network**

Obtain networks and forward and backward barrier energy of isomers.

```python
python3 get_network.py \
    --filename_pattern 'ts' \
    --barrier_file 'barrier' \
    --index_file 'index.txt' \
    --output_file 'network_cluster.out'
```

Where the parameters are below:

1.  filename_pattern:  Pattern to match filenames (default: 'ts')
2.  barrier_file: Path to the barrier data file (default: 'barrier')
3.  index_file: Path to the index file (default: 'index.txt'), user are found in pair_neb.py to generate "network_cluster.out" file.
4. output_file: Path to the output file (default: 'network_cluster.out')

**MKM**

The change of distribution during the time.

```python
python3 mkm_main.py
  --pressure 1e5   \
  --temperature_start 300 \
  --temperature_final 500 \
  --temperature_point 10 \
  --method LSODA  \
  --coverage_run True \
  --order_run False  \
  --drc_run False   \
  --apparent_energy_run False \
  --t_max 1e0 \
  --dt 1e2 \
  --filename '5cu'
```

Where the parameters are below:

1.  pressure: Pressure in Pascals (default: 1e5)
2.  temperature_start: Start temperature in Kelvin (default: 300)
3.  temperature_final: Final temperature in Kelvin (default: 500)
4.  temperature_point:  Temperature step size (default: 10)
5.  method: Method for solving ODEs (default: LSODA)
6.  coverage_run: Whether to run coverage (default: True)
7.  order_run: Whether to run order (default: False)
8.  drc_run: Whether to run drc (default: False)
9.  apparent_energy_run: Whether to run apparent energy (default: False)
10.  t_max: Maximum time for simulation (default: 1e0)
11.  dt: Time step size (default: 1e2)
12.  filename: Filename for the output (default: '5cu')

## ⚙️ User guidance

User can employed the code in this procession.

1. GA
2. metastable structures
3. pairing isomers
4. transition state search 
5. network generation
6. Microkinetic modelling

User are employed by the code below:

```python3
python3 ga.py
python3 meta_str.py
python3 pair_neb.py
gpaw -P 96 python neb.py
python3 get_network.py
python3 mkm_main.py
python3 get_equ.py
```

## 📦 Dependencies



|                        Package                        |                         Purpose                          | Version |
| :---------------------------------------------------: | :------------------------------------------------------: | :-----: |
|         [ASE](https://wiki.fysik.dtu.dk/ase/)         |         Structure handling & GA/M-GCMC interface         | 3.22.1  |
|           [Networkx](https://networkx.org/)           |              Study of the complex networks               |   2.4   |
|              [NumPy](https://numpy.org/)              |                   Numerical operations                   | 1.20.3  |
|         [Matplotlib](https://matplotlib.org/)         |                         Plotting                         |  3.4.3  |
|         [Pandas](https://pandas.pydata.org/)          |                  Data handling & export                  |  1.3.3  |
|              [SciPy](https://scipy.org/)              |                    Polynomial fitting                    |  1.7.1  |
| [Dscribe](https://singroup.github.io/dscribe/latest/) | Atomic structures into fixed-size numerical fingerprints |  2.1.0  |



## 📚 References

1. Vilhelmsen, L. B.; Hammer, B. A genetic algorithm for first principles global structure optimization of supported nano structures. *J. Chem. Phys.* **2014**, 141 (4), 044711.
2. Lindgren, P.; Kastlunger, G.; Peterson, A. A. Scaled and Dynamic Optimizations of Nudged Elastic Bands. *J. Chem. Theory Comput.* **2019**, 15 (11), 5787-5793.
3. Kolsbjerg, E. L.; Groves, M. N.; Hammer, B. An automated nudged elastic band method. *J. Chem. Phys.* **2016**, 145 (9), 094107. 