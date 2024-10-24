#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH -o output_multipar.log
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import os, sys
import subprocess
import time
import numpy as np
from pathlib import Path

try:
    os.rmdir("/scratch/smontill/HEOM/VSC/Jeff_numerical/wQ_disorder/tmpdir")
except:
    print("No folder") 

NARRAY = str(249) # number of jobs
filename = "job"

manual = 0
JOBIDnum = 22951491
ARRAYJOBIDnum = 22951498

if(manual==1):
    JOBID = str(JOBIDnum)
    ARRAYJOBID = str(ARRAYJOBIDnum)
else:
    JOBID = str(os.environ["SLURM_JOB_ID"]) # get ID of this job

    Path("tmpdir").mkdir(parents=True, exist_ok=True) # make temporary directory for individual job files
    os.chdir("tmpdir") # change to temporary directory
    command = str("sbatch --array [0-" + NARRAY + "] ../sbatcharray.py") # command to submit job array

    open(filename,'a').close()

    t0 = time.time()

    ARRAYJOBID = str(subprocess.check_output(command, shell=True)).replace("b'Submitted batch job ","").replace("\\n'","") # runs job array and saves job ID of the array

    t1 = time.time()
    print("Job ID: " + JOBID)
    print("Array time: ",t1-t0)

    os.chdir("..") # go back to original directory

# Gather data     

# psi = np.loadtxt("tmpdir_" + JOBID + "/psi_" + ARRAYJOBID + "_" + str(1) + ".txt", dtype = np.complex) # load first job to get parameters
# steps = int(len(psi[0,:])) # number of printed steps (NSteps//NSkip) times number of single-run trajectories
# psi = np.zeros((len(psi[:,0]),steps*int(NARRAY)), dtype = psi.dtype) # initialize zeros matrix using parameters

# for i in range(int(NARRAY)):
#     psi[:,i*steps : (i+1)*steps] = np.loadtxt("tmpdir_" + JOBID + "/psi_" + ARRAYJOBID + "_" + str(i+1) + ".txt", dtype = np.complex) # append each line with next trajectory(s)

# psiFile = np.savetxt(f"./psi_{filename}.txt",psi)

# t1 = time.time()
# print("Total time: ", t1-t0)

# os.system("rm -rf tmpdir_" + JOBID) # delete temporary folder (risky since this removes original job file data)