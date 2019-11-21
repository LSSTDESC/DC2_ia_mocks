#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30
#SBATCH --nodes=8
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

module load python/3.7-anaconda-2019.07 PrgEnv-gnu
srun python tidal_generator.py --snapshot=/global/projecta/projectdirs/lsst/groups/CS/cosmoDC2/plarsen_tmp/STEP247/m000.mpicosmo.247 > log
