#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=4
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

module load python/3.7-anaconda-2019.07 PrgEnv-gnu
srun python tidal_generator.py
