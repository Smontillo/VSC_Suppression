for j in 0.0010 0.0015 0.0020 0.0025 0.0030 0.0035 0.0040 0.0045 0.0050 0.0055 0.0060
do

cd $j

for i in 280 290 300 310 320 330
do

cd $i

cat << Eof > sbatch_HEOM.sh
#!/bin/bash
#SBATCH -p standard
#SBATCH -o output.log
#SBATCH --mem=60GB
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=T15
#SBATCH --open-mode=append

time /scratch/smontill/HEOM/bin/rhot ./input.json

Eof

chmod +x sbatch_HEOM.sh
sbatch sbatch_HEOM.sh
cd ../

done

cd ../
done