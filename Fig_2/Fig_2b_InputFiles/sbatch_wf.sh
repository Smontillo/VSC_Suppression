#!/bin/bash
#SBATCH -p debug
#SBATCH --output=plot.out
#SBATCH --error=plot.err
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -N 1

python3 plotting_dat.py

