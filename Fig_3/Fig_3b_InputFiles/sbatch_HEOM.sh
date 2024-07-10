#!/bin/bash
#SBATCH -p gpu
#SBATCH -o output.log
#SBATCH --mem=60GB
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=et25
#SBATCH --open-mode=append

time /scratch/smontill/HEOM/bin/rhot ./input.json

