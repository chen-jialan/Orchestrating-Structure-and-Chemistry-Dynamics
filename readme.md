# The Framework to **Orchestrate Structure and Chemistry Dynamics for Cluster Catalysis**

This repository provides a computational framework to investigate the **Structure and Chemistry Dynamics** in cluster-based catalysis. It integrates modules for global optimization, transition state searching, and kinetic analysis to model the behavior of catalytic clusters under realistic conditions.

## 📌 Key Components

1. **Genetic Algorithm (GA)**: Global optimization to identify the metastable state ensemble.
2. **Isomer Transform Calculations**: Identification of forward and backward activation barriers between isomers.
3. **Isomer Network Generation**: Construction of complex isomer transition networks using graph theory.
4. **Microkinetic Modeling (MKM)**: Simulation of isomer transformation kinetics and system equilibrium time.

## 🧪 Methods & Workflow

1. **GA Search**: Integrated with **ASE** for first-principles global structure optimization.
2. **Isomer Transform**: Algorithmic pairing and connection of isomers.
3. **MKM**: Solving Ordinary Differential Equations (ODEs) to determine equilibrium time and coverage distribution.

## 📁 Repository Structure

| **Folder** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `ga/`      | Metastable structure search using Genetic Algorithm          |
| `neb/`     | Transition state (TS) calculations for isomer transformation |
| `network/` | Network generation and analysis using **NetworkX** and **DyNEB** |
| `mkm/`     | Microkinetic modeling scripts                                |
| `dataset/` | Pre-calculated isomers and transition states used in this work |
| `example/` | Step-by-step tutorials and example files                     |
| `dscribe/` | Atomic descriptors for structural similarity analysis        |

---

## ⚙️ Module Details & Usage

### 1. Searching Metastable Structures (`ga/`)

Used to search for the global minimum and relevant metastable isomers.

**Initialize GA:**

```python
python3 initial.py --atomtype 29 --number 8 --db_file gadb.db --poscar_file POSCAR_base
```

- `--atomtype`: Atomic number (e.g., `29` for Cu, `47` for Ag).
- `--number`: Number of atoms in the cluster.
- `--poscar_file`: The slab substrate structure.

**Identify Metastable Ensemble:**

```python
python3 allga.py
python3 meta_all.py --limit_meta 0.01 --limit_energy 1 --limit_structure 0.15
```

**Pair Isomers:**

Bash

```python
python3 pair_neb.py --metal Cu --distance_limit 4.0
```

### 2. Transition State Identification (`neb/`)

Calculates the energy barriers between paired isomers.

**Run NEB (with GPAW):**

```bash
gpaw -P 96 python neb.py
```

**Extract Barriers:**

```python
python3 get_energy.py --number_images 4 --fmax 0.1 --energy_threshold 0.01
```

### 3. Network & Microkinetic Modeling (`network/`)

Constructs the isomer network and simulates the time-dependent population.

**Generate Network:**

```python
python3 get_network.py --filename_pattern 'ts' --barrier_file 'barrier' --index_file 'index.txt'
```

**Run Microkinetic Simulation:**

```python
python3 mkm_main.py --pressure 1e5 --temperature_start 300 --temperature_final 500 --t_max 1e0
```

## 🚀 Quick Start Guide

To run a complete simulation, follow this sequence:

1. **Optimization**: `python3 ga.py` & `python3 meta_str.py`
2. **Pairing**: `python3 pair_neb.py`
3. **TS Search**: `gpaw -P 96 python neb.py`
4. **Network**: `python3 get_network.py`
5. **Kinetics**: `python3 mkm_main.py` & `python3 get_equ.py`

------

## 📦 Dependencies

| **Package**                                    | **Version** | **Purpose**                       |
| ---------------------------------------------- | ----------- | --------------------------------- |
| [ASE](https://wiki.fysik.dtu.dk/ase/)          | 3.22.1      | Structure handling & GA interface |
| [NetworkX](https://networkx.org/)              | 2.4         | Complex network analysis          |
| [Dscribe](https://singroup.github.io/dscribe/) | 2.1.0       | Atomic structural fingerprints    |
| [NumPy](https://numpy.org/)                    | 1.20.3      | Numerical operations              |
| [SciPy](https://scipy.org/)                    | 1.7.1       | ODE solving & Fitting             |

------

## 👥 Developers & Contact

- **Jia-Lan Chen**: [jlchen20@mail.ustc.edu.cn](mailto:jlchen20@mail.ustc.edu.cn)
- **Advisors**:
  - Jin-Xun Liu ([jxliu86@ustc.edu.cn](mailto:jxliu86@ustc.edu.cn))
  - Wei-Xue Li ([wxli70@ustc.edu.cn](mailto:wxli70@ustc.edu.cn))

## 📚 References

1. Vilhelmsen, L. B.; Hammer, B. A genetic algorithm for first principles global structure optimization of supported nano structures. *J. Chem. Phys.* **2014**, 141 (4), 044711.
2. Lindgren, P.; Kastlunger, G.; Peterson, A. A. Scaled and Dynamic Optimizations of Nudged Elastic Bands. *J. Chem. Theory Comput.* **2019**, 15 (11), 5787-5793.
3. Kolsbjerg, E. L.; Groves, M. N.; Hammer, B. An automated nudged elastic band method. *J. Chem. Phys.* **2016**, 145 (9), 094107. 
4.  Zhang, Y. L.; Hu, C.; Jiang, B. Embedded Atom Neural Network Potentials: Efficient and Accurate Machine Learning with a Physically Inspired Representation. *J. Phys. Chem. Lett.* **2019**, 10 (17), 4962-4967.