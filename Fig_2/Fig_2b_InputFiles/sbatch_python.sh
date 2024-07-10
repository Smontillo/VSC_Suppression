#!/bin/bash
#SBATCH -p debug
#SBATCH --output=python.out
#SBATCH --error=python.err
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -N 1

python3 gen_input.py

