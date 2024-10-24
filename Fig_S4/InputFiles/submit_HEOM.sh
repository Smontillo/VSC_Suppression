for i in 500 700 800 900 1000 1100 1175 1200 1300 1400 1500 1600
do

cd $i

cat << Eof > sbatch_HEOM.sh
#!/bin/bash
#SBATCH -p standard
#SBATCH -o output.log
#SBATCH --mem=10GB
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=80cm
#SBATCH --open-mode=append

time /scratch/smontill/HEOM/bin/rhot ./input.json

Eof

chmod +x sbatch_HEOM.sh
sbatch sbatch_HEOM.sh
cd ../

done
