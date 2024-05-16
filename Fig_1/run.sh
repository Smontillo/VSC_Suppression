#!/bin/bash
#SBATCH -p debug
#SBATCH --job-name=myJob                    # create a name for your job
#SBATCH --ntasks=1                         # total number of tasks
#SBATCH --cpus-per-task=1                       # cpu-cores per task
#SBATCH --mem-per-cpu=5G                    # memory per cpu-core
#SBATCH -t 1:00:00                           # total run time limit (HH:MM:SS)
#SBATCH --output=myJobOutput.out
#SBATCH --error=myJobError.err

srun python fig_1.py
# srun python runMPIjob.py