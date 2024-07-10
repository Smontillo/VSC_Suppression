#!/bin/bash
#SBATCH -p action
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err
##SBATCH -o output.log
#SBATCH --mem=60GB
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=uSM
#SBATCH --open-mode=append

time /scratch/smontill/HEOM/bin/1d-resp ./input.json

