for i in 700 800 900 1000 1050 1100 1130 1160 1189 1220 1250 1300 1350 1400 1500 1600 1700
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
