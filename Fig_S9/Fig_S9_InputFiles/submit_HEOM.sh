for i in 50 100 200 400 600 800 1000 1200 1400 1600 1800 2000 
do

cd $i

cat << Eof > sbatch_HEOM.sh
#!/bin/bash
#SBATCH -p standard
#SBATCH -o output.log
#SBATCH --mem=60GB
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=et25
#SBATCH --open-mode=append

time /scratch/smontill/HEOM/bin/rhot ./input.json

Eof

chmod +x sbatch_HEOM.sh
sbatch sbatch_HEOM.sh
cd ../

done
