#!/bin/bash
#SBATCH -p debug
#SBATCH -o output.log
#SBATCH --mem=60GB
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=T_nRPV
#SBATCH --open-mode=append


srun python3 fig_4.py
# srun python runMPIjob.py