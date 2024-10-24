#!/bin/bash
#SBATCH -p debug
#SBATCH --job-name=plot   # create a name for your job
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=5G         # memory per cpu-core
#SBATCH -t 1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err

# srun python Transmission_coeff.py
srun python plotting_dat.py