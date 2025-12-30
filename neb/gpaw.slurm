#!/bin/bash

##SBATCH -p TEST_INTEL_8581    
#SBATCH -p INTEL_8581          
#SBATCH -J gpaw
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=96        
#SBATCH -o out_%j.log
#SBATCH -e error_%j.log

ulimit -c unlimited
ulimit -l unlimited
ulimit -s unlimited

export OMPI_MCA_btl_openib_allow_ib=1

source /home/tjzs171a10599/intel/oneapi/setvars.sh  --force

XC=/home/tjzs171a10599/bin/libxc5.2/
export C_INCLUDE_PATH=$XC/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$XC/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$XC/lib:$LD_LIBRARY_PATH

export GPAW_SETUP_PATH=/home/tjzs171a10599/software/gpaw/gpaw-setups-24.11.0
conda activate gpaw

gpaw -P 96 python neb.py   > q

