#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30
#SBATCH --nodes=16
#SBATCH --tasks-per-node=16
#SBATCH --constraint=haswell

module load python/3.7-anaconda-2019.10 PrgEnv-gnu openmpi
srun python tidal_generator.py > log
